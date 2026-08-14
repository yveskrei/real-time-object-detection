# `model-optimization` — architecture and reference

The model toolchain. Everything else in this repository *consumes* models; this is the
only project that *produces* them. It takes a PyTorch checkpoint, wraps it so the graph
emits exactly the tensor the runtime clients know how to decode, exports it to ONNX, and
documents the TensorRT step that turns that ONNX into a GPU engine.

This is a Python project, not Rust — `uv`-managed, Python ≥ 3.11
(`model-optimization/pyproject.toml:6`, `.python-version`). It runs offline, by hand, when
a model changes. Nothing at runtime imports it.

All paths in this document are relative to the repository root unless stated otherwise.

---

## 1. Where this sits in the system

```
  <checkpoint>.pt
        │   export_model.py  --model-type {YOLOV9,DINOV3,YOLO26}
        ▼
  <stem>-fp32.onnx                       models/onnx/
        │   trtexec --fp16               (run inside the Triton container)
        ▼
  <name>.engine                          models/trt/
        │   copy + rename by hand
        ▼
  triton_models/<model name>/1/model.plan   ← GPU
  triton_models/<model name>/1/model.onnx   ← CPU
        │
        ▼
  loaded at startup by client-object-detection / client-image-retrieval /
  image-retrieval-search, which generate the Triton model configuration at
  runtime rather than committing a config.pbtxt
```

The last hop is what makes this project matter to the clients. Each client builds a Triton
model configuration in `InferenceModel::build_model_config` and pushes it with
`repository_model_load`, declaring `dims` taken straight from `output_shape` in its
`secrets/config.yaml`. Triton validates that declaration against the real signature of the
artifact in the model repository. So the shape this exporter bakes into the ONNX graph and
the shape a client declares have to agree, or the model refuses to load.

Two defaults line up for the same reason. `--input-name` defaults to `images` and
`--output-name` to `output` (`export_model.py:156`, `:162`), which is exactly what every
client config declares (`input_name: images`, `output_name: output`). And the
device-derived precision defaults in the clients — FP32 on CPU, FP16 on GPU — match what
this toolchain produces at each stage: the ONNX export is FP32, and the TensorRT step is
run with `--fp16`.

| Model type | Wrapper output | Consumed by |
| --- | --- | --- |
| `YOLOV9` | raw head, `[4 + classes, anchors]` | `processing/yolo.rs` → `postprocess_yolov9` |
| `YOLO26` | `[max_det, 6]` = `[x1, y1, x2, y2, score, class_id]` | `processing/yolo.rs` → `postprocess_yolo26` |
| `DINOV3` | CLS token, `[embedding_dim]` | `processing/dino.rs` → `postprocess` |

---

## 2. Module map

| File | Responsibility |
| --- | --- |
| `export_model.py` | CLI, argument validation, and the ONNX export itself. |
| `export_utils.py` | The `ModelType` enum and one loader per architecture. Each loader returns an `nn.Module` already wrapped so `forward()` yields a single tensor. |
| `README.md` | The commands, in brief. This document is the longer form. |
| `pyproject.toml` | Dependency pins — see §8. |

---

## 3. The export pipeline

`main()` (`export_model.py:113`) parses arguments, dispatches on `--model-type` to build a
wrapped model, constructs a dummy input, and hands both to `export_model()`.

```
main()                                              export_model.py:113
 ├─ ModelType[args.model_type]                                   :180
 ├─ dispatch to a loader                                         :183-208
 │    DINOV3 → requires --dino-type and --model-source-code      :184-187
 │    YOLOV9 → requires --model-source-code                      :195-196
 │    YOLO26 → needs neither; loaded from the installed package  :202-206
 ├─ dummy_input = torch.randn(1, *input_shape)                   :211-215
 └─ export_model(...)                                            :221
      ├─ move model + input to cuda when available               :26-29
      ├─ one forward pass to learn the real output shape         :34-35
      ├─ reject a wrapper that returns more than one tensor      :37-41
      ├─ torch.onnx.export → <stem>-fp32.onnx                    :51-66
      ├─ optional onnxslim pass                                  :74-83
      ├─ pin the static output dims into the graph               :90-100
      ├─ onnx.checker.check_model                                :103-107
      └─ report the file size                                    :110-111
```

Points worth holding onto:

**The output filename is derived, not given.** It is `Path(--model-path).stem` plus
`-fp32.onnx` (`export_model.py:47`, `:222`), so `yolov9-e.pt` becomes `yolov9-e-fp32.onnx`
— which is how `models/onnx/` is named today. `--output-name` is unrelated: it names the
output *tensor* inside the graph, not the file.

**Exactly one file comes out, and it is FP32.** There is no FP16 ONNX export path;
precision is chosen later, at the TensorRT step, with `trtexec --fp16` (§6). The FP16
artifacts in `models/trt/` come from there.

**Only the batch axis is dynamic.** `dynamic_axes` marks axis 0 of both the input and the
output as `batch_size` (`:59-62`); every other dimension is fixed at trace time. That is
the property §5 relies on.

**The export runs on GPU when one is present** (`:26`), purely for speed — the traced graph
is identical either way. Everything is cast to `.float()` before tracing (`:35`, `:52-53`).

**Opset 18, constant folding on, legacy tracer.** `dynamo=False` (`:65`) keeps the classic
TorchScript-based exporter rather than the newer dynamo path.

---

## 4. The three loaders

Each loader in `export_utils.py` exists to solve the same problem: these models do not
natively return one clean tensor, and the exporter requires exactly that
(`export_model.py:37-41`). The wrapper is where that reduction happens.

### `get_yolov9_model` (`export_utils.py:10`)

Puts the YOLOv9 source tree on `sys.path` (`:24`) — the checkpoint pickles references to
its own classes, so the code has to be importable — then `torch.load(..., weights_only=False)`
(`:27`) and takes `model['model']`. `YOLOV9Wrapper.forward` returns `output[0]` (`:21`),
discarding the auxiliary heads and keeping the detection tensor.

Requires `--model-source-code` pointing at a checkout of the YOLOv9 repository.

### `get_dinov3_model` (`export_utils.py:32`)

Builds the architecture through `torch.hub.load(..., source='local', pretrained=False)`
(`:50-55`) against a local DINOv3 checkout, then loads the weights separately with
`torch.load(..., weights_only=True)` (`:58-62`). Splitting construction from weight-loading
is what lets a local checkout and a downloaded `.pth` be paired freely.

`DINOV3Wrapper.forward` calls `forward_features(x)` and returns `output['x_norm_clstoken']`
(`:42-47`) — the CLS token, i.e. the embedding vector. This is the tensor the retrieval
clients store in Elasticsearch.

Requires both `--model-source-code` and `--dino-type` (e.g. `dinov3_vitb16`).

### `get_yolo26_model` (`export_utils.py:70`)

The only loader that needs no source tree: it imports `ultralytics` (`:90`) from the
installed package. Loads with `YOLO(model_path).model.float().eval()`, which picks EMA
weights when present, then `.fuse()` (`:93-94`) to fold batch-norm into the preceding
convolutions.

It then reconfigures the detection head for export (`:97-105`):

| Assignment | Why |
| --- | --- |
| assert `head.end2end` | YOLO26 heads are NMS-free; a head without it would need suppression the runtime does not apply |
| `head.export = True` | emit a bare tensor, and make TopK's `k` a graph constant |
| `head.format = 'onnx'` | select the ONNX output convention |
| `head.dynamic = False` | bake anchors as constants for a fixed H×W |
| `head.max_det = max_det` | the `--max-det` value, fixed into the graph |
| `head.shape = None` | drop the cached shape so anchors regenerate |

`YOLO26Wrapper.forward` reshapes to `(-1, max_det, 6)` (`:88`). Each row is
`[x1, y1, x2, y2, score, class_id]`, xyxy in letterboxed-input pixels, sorted by score
descending and **not** confidence-filtered — the consumer applies its own threshold
(`:74-77`). That contract is exactly what `postprocess_yolo26` implements on the client
side, and why `--max-det 300` produces the `output_shape: [300, 6]` the client configs
declare.

---

## 5. Output shape pinning

The most consequential twenty lines in the project (`export_model.py:85-100`).

After tracing, the graph's declared output signature can carry symbolic placeholders
instead of real numbers — `[batch_size, Concatoutput_dim_1, Concatoutput_dim_2]` rather
than `[batch_size, 84, 8400]`. ONNX shape inference cannot always derive them: a
`view()`/`reshape()` against the symbolic batch axis, which YOLOv9's head performs, leaves
the trailing dimensions unresolved.

Triton reads those placeholders as `[-1, -1, -1]`, compares them against the `dims` in the
model configuration the client uploaded, and refuses to load the model. The numbers were
never wrong; only the declared signature was under-specified.

The fix is to take the shape from the reference forward pass at `:34-35` and write it into
the graph's output dims (`:97-98`). This is only sound because the batch axis is the sole
dynamic one, so every dimension being pinned is genuinely static. Deriving it from a real
forward pass rather than hardcoding keeps it correct across class counts, input sizes and
`--max-det` values.

A guard at `:91-95` compares the declared rank against what the model returned and leaves
the graph untouched on a mismatch rather than writing dimensions into the wrong slots.

**`--simplify` interacts with this.** `onnxslim` (`:74-83`) is a structural pass — constant
folding and dead-node removal — that must not change the numbers, but it *can* freeze
dynamic axes. It runs before pinning, and its failure path is non-fatal: the exception is
caught, the unsimplified graph is reloaded, and the export continues (`:81-83`). Re-verify
batching after using it.

---

## 6. CLI reference

```bash
uv run export_model.py --model-type {YOLOV9,DINOV3,YOLO26} --model-path MODEL.pt [...]
```

| Flag | Required | Default | Effect |
| --- | --- | --- | --- |
| `--model-type` | yes | — | Selects the loader. Choices come from `ModelType._member_names_` (`:121`), so the enum is the single source of truth. |
| `--model-path` | yes | — | The `.pt` checkpoint. Its stem also names the output file. |
| `--model-source-code` | YOLOV9, DINOV3 | — | Path to the model's source checkout. Not needed for YOLO26. |
| `--dino-type` | DINOV3 | — | Architecture passed to `torch.hub.load`, e.g. `dinov3_vitb16`. |
| `--input-shape` | no | `3,640,640` | CHW, comma-separated. Validated as exactly three dimensions (`:212-213`). |
| `--input-name` | no | `images` | Name of the ONNX input tensor. Must match the clients' `input_name`. |
| `--output-name` | no | `output` | Name of the ONNX output tensor. Must match the clients' `output_name`. |
| `--output-path` | no | `cwd` | Directory for the `.onnx`. |
| `--max-det` | no | `300` | YOLO26 only. Baked into the graph; becomes the first element of the client's `output_shape`. |
| `--simplify` | no | off | Run `onnxslim` over the graph. See §5. |

Requirements are enforced per model type at `:183-208`, before any weights are touched.

---

## 7. ONNX → TensorRT

Run `trtexec` **inside the Triton container**, so the engine is built against the same
TensorRT the server will use — engines are not portable across TensorRT versions, and are
compiled for the specific GPU architecture present at build time.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --shapes=images:1x3x640x640 \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw
```

The three shape profiles are the dynamic batch range the engine will accept, and they are
where this step meets the client configuration: `maxShapes` batch 16 is the clients'
`batch_max_size: 16`, and `optShapes` batch 8 sits inside their
`batch_preferred_sizes`. An engine built with a smaller `maxShapes` than the client's
`batch_max_size` will fail at inference time on a full batch.

The spatial dimensions change per model — `640×640` for the detectors, `512×512` and
`224×224` for the two DINOv3 variants — and must match the `input_shape` the client
declares for that model.

`--fp16` plus the two `IOFormats` flags are what make this the FP16 half of the pipeline,
which is why the clients default to FP16 on GPU.

---

## 8. Placing artifacts and running

### Environment

```bash
cd model-optimization
uv sync
```

Pins (`pyproject.toml:7-14`): `torch==2.7.1`, `onnx==1.18.0`, `onnxruntime-gpu==1.22.0`,
`ultralytics>=8.4.0`, `onnxslim>=0.1.94`, and `setuptools<81`. The exact `torch` and `onnx`
pins matter — opset support and the tracer both move between releases.

### Into the Triton model repository

Triton discovers models by directory layout, and the clients ask for one by the `name` in
their `secrets/config.yaml`:

```
triton_models/<name>/1/model.onnx    # served when DEVICE_TYPE=CPU  (onnxruntime_onnx)
triton_models/<name>/1/model.plan    # served when DEVICE_TYPE=GPU  (tensorrt_plan)
```

Both filenames can sit in the same version directory. The client's generated configuration
sets `default_model_filename` explicitly, so the device decides which one is opened and the
other is simply ignored — one repository serves both modes.

No `config.pbtxt` is needed. The clients generate the configuration at load time, which is
why the exporter's job ends at producing an artifact whose signature matches what they
declare.

### Naming

`models/onnx/` and `models/trt/` hold the intermediate artifacts under their descriptive
names (`yolov9-e-fp32.onnx`, `dinov3_vitb16-fp32-512.onnx`, `yolov9-e-fp16.engine`). The
copy placed under `triton_models/<name>/1/` is renamed to the flat `model.onnx` /
`model.plan` that Triton expects.
