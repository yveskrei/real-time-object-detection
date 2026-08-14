# `client-object-detection` — architecture and reference

Real-time object detection client. It receives decoded video frames from a proprietary
C library (`libclient_video.so`) over FFI, runs YOLO inference against an NVIDIA Triton
Inference Server over gRPC, and hands the resulting bounding boxes back to the same
library so the backend can overlay them on the stream.

All paths in this document are relative to the repository root unless stated otherwise.
Cargo package name is `client` (`Cargo.toml:2`), so the binary produced is
`client-object-detection/target/{debug,release}/client`. Rust **edition 2024**.

---

## 1. Read this first — the two things that stop it from running today

Neither of these is a bug; both are missing artefacts that are deliberately not
committed. The crate will not start without them.

**(a) `client-object-detection/secrets/libclient_video.so` does not exist.**
`.gitignore` ends with `*.so`, so the compiled video library is never committed.
`ClientVideo::new()` (`client-object-detection/src/client_video.rs:110`) `dlopen`s
`secrets/libclient_video.so` relative to the process working directory, and a failure
there aborts `Services::new` and therefore startup. A built copy of the library exists
in the sibling project at `client-real-time/client/secrets/libclient_video.so`
(~11 MB, dated April). Whether that build is ABI-compatible with the symbol set this
crate expects has not been verified here — see §5 for the exact contract. *This
document does not copy the file; that is a deliberate manual step.*

**(b) The configured model is not in the Triton model repository.**
`secrets/config.yaml` requests `name: yolo26x`, but `triton_models/` currently contains
only `dinov3-224/1/model.plan`, `dinov3-512/1/model.plan` and `yolov9-e/1/model.plan`.
Under `DEVICE_TYPE=CPU`, Triton will be asked to load `yolo26x` as `onnxruntime_onnx`
with `default_model_filename: model.onnx`, i.e. it needs
`triton_models/yolo26x/1/model.onnx`; under `DEVICE_TYPE=GPU` it needs
`triton_models/yolo26x/1/model.plan` instead. Both may sit in the same version
directory — the generated `default_model_filename` picks between them. No `config.pbtxt` is required — the client
generates and uploads the model configuration itself (§6).

Three more preconditions, less surprising but equally fatal:

- **Triton must already be up.** `InferenceModel::new` calls `server_ready()` and bails
  if the server is not ready (`src/inference.rs:88-95`). There is no retry loop.
- **Working directory matters.** Both `secrets/config.yaml` (`src/utils/config.rs:246`)
  and `secrets/libclient_video.so` (`src/client_video.rs:23`) are resolved relative to
  the process CWD. Run the binary from the crate root.
- **`DEVICE_TYPE` must be set, and must match the Triton profile you booted.** The
  client is what tells Triton which platform to load, so pointing `DEVICE_TYPE=GPU` at
  the CPU compose profile asks an unaccelerated Triton for a `tensorrt_plan`. The moon
  tasks set both together — `client-detection:cpu` boots `services:triton-cpu` and
  exports `DEVICE_TYPE=CPU` — so this only needs thought when running by hand. The
  variable has no default: unset, the process exits at config load naming it.

---

## 2. Module map

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Four-line entry point: config → services → start → sleep forever. |
| `src/lib.rs` | Module declarations only. Everything is `pub`, so the binary consumes the crate as a library. |
| `src/services.rs` | The `Services` singleton (`OnceCell<Arc<Services>>`) that every other subsystem reaches through. |
| `src/utils/config.rs` | `secrets/config.yaml` deserialization, per-source override + clamping, hardware-name resolution, tracing setup. |
| `src/utils/queue.rs` | `FixedSizeQueue<T>` — bounded queue that drops the oldest item on overflow. |
| `src/utils.rs` | `get_image_raw` helper (currently unused). |
| `src/client_video.rs` | The FFI boundary: `dlopen`, exported-function wrappers, and the four host callbacks. |
| `src/client_video/utils.rs` | Types that cross the boundary (`RawFrame`, `Detection`, `ResultBBOX`) and the `Ownership`-aware C-pointer readers. |
| `src/source.rs` | Per-source queueing, frame skipping, concurrency limiting, and the per-frame orchestration. |
| `src/inference.rs` | Triton gRPC client, runtime model-config generation, load/unload, batching. |
| `src/processing.rs` | Fused preprocessing (resize + letterbox + normalize) and the conversion lookup tables. |
| `src/processing/yolo.rs` | The two output decoders (YOLOv9, YOLO26), NMS, and the per-frame pre/infer/post pipeline. |
| `src/statistics.rs` | Per-source counters and their 1 s reporter; NVML GPU telemetry. |

---

## 3. Startup chain

```
main()                                        src/main.rs:8
 └─ AppConfig::new()                          src/utils/config.rs:199
      ├─ load_config_file()  reads ./secrets/config.yaml      :244
      ├─ init_logging(local)                                  :260
      ├─ per-source override + clamping                        :207-231
      ├─ DeviceType::from_env()  reads DEVICE_TYPE, required
      ├─ device_type.hardware_name()  (NVML when GPU)          :164
      └─ fill each model's absent precision from the device default
 └─ services::init_services(&cfg, Handle::current())          src/services.rs:24
      ├─ Services::new(...)                                    :51
      │    ├─ InferenceModels::new   → Triton client, server_ready, base request
      │    ├─ SourceProcessors::new  → one SourceProcessor per configured id
      │    │                            (each spawns its consumer task immediately)
      │    ├─ Statistics::new        → NVML init when device_type == GPU
      │    └─ ClientVideo::new       → dlopen secrets/libclient_video.so
      └─ SERVICES.set(Arc::new(services))                      :35
 └─ get_services()?.start(&cfg)                               src/services.rs:77
      ├─ Statistics::start           → source-stats task + GPU-stats blocking thread
      ├─ InferenceModels::start      → unload previous, then load with generated config
      ├─ ClientVideo::init_state(RunMode::Regular)  → SetCallbacks, then SetSettings
      └─ ClientVideo::init_sources   → InitSources  ← frames start flowing here
 └─ sleep(Duration::from_secs(u64::MAX))                      src/main.rs:23
```

**The ordering in `init_services` is deliberate.** `SERVICES.set` happens *before*
`start()` is called (`src/services.rs:34-37`, and note the comment there). Every spawned
task and every FFI callback re-enters through `services::get_services()`, so the global
must be populated before anything can call back into the process. `Services::new` itself
already spawns the per-source consumer tasks, but those only touch the global once a
frame arrives, which cannot happen until `init_sources` runs.

**Shutdown is `SIGINT` with the default handler** (`src/main.rs:22`). No signal handler is
installed, so the process is terminated outright: the `Drop` impls on `Statistics`
(`src/statistics.rs:313`) and `SourceProcessor` (`src/source.rs:286`) do not run, and
`StopSources` is never called on the video library.

### Logging

`init_logging` (`src/utils/config.rs:260`) builds a `tracing_subscriber` registry with:

- `EnvFilter::from_default_env()` — driven by `RUST_LOG`. With `RUST_LOG` unset the
  filter carries no directives, so export something (`RUST_LOG=INFO`) to see output; the
  sibling `client-real-time/run_local.sh` does exactly that.
- a **stdout** layer, always on, JSON-formatted with RFC-3339 UTC timestamps. (The
  inline comment calls it "pretty format"; the code calls `.json()`.)
- a **file** layer, only when `local: true`, writing `logs/app.log` via
  `RollingFileAppender` with `Rotation::NEVER` — the file is appended to, never rolled.

The non-blocking writer guard is intentionally leaked with `std::mem::forget(_guard)`
(`:288`) so the background writer thread stays alive for the process lifetime.

---

## 4. The frame path

```
 libclient_video.so decoder thread
   │  (borrowed *const u8, width*height*3 bytes, RGB24, tightly packed)
   ▼
 ClientVideo::_source_frames_callback              src/client_video.rs:335
   │  copy immediately (Ownership::Borrowed → no free, valid only for this call)
   │  build RawFrame { source_id, data, width, height, pts }
   ▼
 services.runtime().spawn(...)                     src/client_video.rs:370
   │  hand off to tokio so the C callback returns immediately
   ▼
 SourceProcessor::add_to_queue                     src/source.rs:205
   │  if (frames_total + 1) % inf_frame != 0  ──►  frames_total += 1, frame dropped
   │  else stamp `added` and enqueue
   ▼
 FixedSizeQueue<QueuedFrame>  capacity 15          src/utils/queue.rs:55
   │  on overflow: pop_front() (the OLDEST frame) and invoke the drop callback,
   │  which increments frames_failed                src/source.rs:99
   ▼
 per-source consumer loop                          src/source.rs:114
   │  acquire_owned() on Semaphore(15), then recv() from the queue
   │  tokio::spawn a task that holds the permit for the duration
   ▼
 SourceProcessor::process_frame                    src/source.rs:227
   │  frame_queue_time = added.elapsed()
   ▼
 processing::yolo::process_frame                   src/processing/yolo.rs:426
   │  spawn_blocking → preprocess_frame           (resize + letterbox + normalize)
   │  InferenceModel::infer(vec![one_frame])      (gRPC to Triton)
   │  spawn_blocking → postprocess_yolov9 | postprocess_yolo26
   ▼
 SourceProcessor::populate_bboxes → ClientVideo::populate_bboxes
   │  spawn_blocking → PostResults(...)            src/client_video.rs:262
   ▼
 library posts to the backend, then calls _post_results_callback
```

Notes on the shape of this pipeline:

- **`MAX_QUEUE_FRAMES = 15`** (`src/source.rs:19`) is used twice: as the queue capacity
  *and* as the semaphore permit count (`:106`). So at most 15 frames wait, and at most 15
  frames are in flight, per source.
- **The consumer acquires a permit before pulling from the queue** (`:118-121`), so a
  permit is held while blocked on `recv()`. This is the back-pressure mechanism: when 15
  frames are already being processed, nothing is drained, the queue fills, and the
  overflow policy starts discarding the oldest frames.
- **Frame skipping is counted, not queued.** `add_to_queue` only enqueues when
  `(frames_total + 1) % inf_frame == 0`; the skipped frames just bump `frames_total`
  (`:209-222`). With the default `inf_frame: 25`, one frame in 25 reaches inference.
- **Dropping the oldest, not the newest**, is the right call for a real-time overlay:
  a stale frame's boxes are worth less than a fresh frame's.
- **`RawFrame` is wrapped in an `Arc`** as it enters the queue and shared between the
  YOLO pipeline and the results call (`src/source.rs:211`, `:240`, `:249`), so the RGB
  buffer is copied exactly once — at the FFI boundary.

---

## 5. FFI contract with `libclient_video.so`

The library is `dlopen`ed once via `libloading` into `ClientVideo { library: Library }`
and lives on `Services` for the process lifetime. Symbols are looked up by name on every
call (`library().get(b"…")`), not cached. Every call into the library is wrapped in
`tokio::task::spawn_blocking`, because these are synchronous C calls that may block.

### Functions this crate calls

| Symbol | Rust type | Called from | Purpose |
| --- | --- | --- | --- |
| `SetCallbacks` | `SetCallbacksFn` (`client_video.rs:50`) | `set_callbacks` `:151` | Registers all four host callbacks in one shot. |
| `SetSettings` | `SetSettingsFn` `:56` | `set_settings` `:180` | Takes a `RunMode` as `c_int` — the library's log level. |
| `InitSources` | `InitSourcesFn` `:57` | `start_sources` `:203` | `(*const c_uint, c_int)` array of source ids; starts a decoder per id. |
| `StopSources` | `StopSourcesFn` `:58` | `stop_sources` `:231` | Same signature. **Never called today** (`#[allow(dead_code)]`). |
| `PostResults` | `PostResultsFn` `:59` | `populate_bboxes` `:262` | Returns `c_int`; non-zero is treated as failure. |
| `FreeCPtr` | `FreeCPtrFn` `:65` | `free_c_ptr` `:454` | Releases a library-owned pointer. |

**Ordering requirement:** `init_state` (`:122`) calls `SetCallbacks` and *then*
`SetSettings`, and the doc comment states the reason — `SetCallbacks` boots the
library's global state and `SetSettings` is a no-op until that exists. `InitSources`
must come after both; `Services::start` enforces this by calling `init_state` before
`init_sources` (`src/services.rs:92-101`).

### Callbacks the library invokes

All four are `extern "C" fn` free functions on `ClientVideo`, so they carry no state and
must reach the rest of the process through `services::get_services()`.

| Callback | Line | Behaviour |
| --- | --- | --- |
| `_source_frames_callback` | `:335` | Copies the frame (§4) and spawns onto the tokio runtime. Never blocks. |
| `_source_metadata_callback` | `:391` | Logs `source_name`/`width`/`height`/`fps`. The name pointer is **`Owned`** — read then handed to `FreeCPtr`. |
| `_source_status_callback` | `:411` | Maps the `c_int` through `SourceStatus::from_i32` and logs it. |
| `_post_results_callback` | `:418` | Reads and frees the id array, each id string, and the timestamp array — `N + 2` calls to `FreeCPtr`. Its `tracing::info!` is **commented out** (`:436-442`), so today it only drains and frees. |

### Pointer ownership discipline

`Ownership` (`src/client_video/utils.rs:16`) is the single place that decides whether a
pointer goes back to `FreeCPtr`:

- `Ownership::Borrowed` — valid only for the duration of the call; copy and do not free.
  Used for the frame buffer in `_source_frames_callback`, which is why the copy is
  unconditional and immediate.
- `Ownership::Owned` — the library allocated it and expects it back. `release()`
  (`:22-30`) calls `free_c_ptr` and downgrades a failure to a `warn!` rather than
  propagating.

`get_c_array` (`:104`) copies the whole slice *before* releasing the array pointer,
which matters for arrays of pointers: the elements must outlive their container.
`get_c_string` (`:124`) uses `to_string_lossy().into_owned()`, so invalid UTF-8 becomes
replacement characters rather than an error.

In the other direction, `populate_bboxes` (`src/client_video.rs:262`) allocates
`CString`s for the per-bbox ids and for the JSON body, keeps them alive across the
`PostResults` call, and lets Rust drop them afterwards — the comment at `:301` records
the assumption that the library copies synchronously. Nothing is handed off, and
`FreeCPtr` is not involved on this path.

### Enums

```rust
#[repr(i32)] enum RunMode      { Regular = 0, Debug = 1 }              // client_video.rs:70
#[repr(i32)] enum SourceStatus { Idle = 0, Initializing = 1,
                                 Streaming = 2, Terminating = 3 }      // client_video.rs:83
```

`SourceStatus::from_i32` maps anything unrecognised to `Idle` (`:95-102`).

### The `PostResults` body

`populate_bboxes` sends two things: an array of `results_count` NUL-terminated id
strings, and a single JSON document. Each bbox id appears in both, so the backend can
correlate them.

```json
{
  "stream_id": 1,
  "bboxes": [
    {
      "id": "e5a1…-uuid-v4",
      "pts": 1234567890,
      "top_left_corner": 401234,
      "bottom_right_corner": 452110,
      "class_name": "person",
      "confidence": 0.87
    }
  ]
}
```

`top_left_corner` / `bottom_right_corner` are **flat pixel indices**, `y * width + x`,
not coordinate pairs (`ResultBBOX::corners_coordinates`, `src/client_video/utils.rs:90`).
`pts` is the frame presentation timestamp forwarded verbatim from the frame callback;
the `RawFrame` doc comment (`:34`) states it is a 90 kHz clock. `class_name` comes from
`ResultBBOX::class_name` (`:77`), which spells out COCO ids 0–5 and stringifies the
numeric id for everything else.

---

## 6. Inference against Triton

### The client generates the model configuration

This is the most consequential design decision in the crate. There is no `config.pbtxt`
in the model repository. `InferenceModel::build_model_config`
(`src/inference.rs:193-270`) builds the equivalent JSON at runtime and uploads it as the
`config` parameter of `RepositoryModelLoadRequest` (`load_model`, `:273-297`). Consequences:

- The model repository only needs `<name>/1/model.{onnx,plan}`.
- Batching, warmup, instance count and precision are all driven by
  `secrets/config.yaml`, so changing them needs no repository edit and no Triton restart.
- **`--model-control-mode=explicit` in `services/docker-compose-triton.yml:35` is load-bearing.**
  Triton only accepts `repository_model_load` / `repository_model_unload` requests in
  explicit mode; in `none` (the default) or `poll` mode, the model-control API is
  rejected. Explicit mode also means Triton starts with nothing loaded, which is exactly
  what this design wants — the client is the sole authority on what gets loaded.
  (`--strict-model-config=false` is also set, but is redundant here since the client
  always supplies a full config.)

### What `DeviceType` selects

| | `DeviceType::GPU` | `DeviceType::CPU` |
| --- | --- | --- |
| `platform` | `tensorrt_plan` | `onnxruntime_onnx` (TensorRT has no CPU implementation) |
| `default_model_filename` | `model.plan` | `model.onnx` |
| `instance_group.kind` | `KIND_GPU`, `gpus: [0]` | `KIND_CPU`, **no** `gpus` field |
| `optimization` block | added | omitted |

Stating `default_model_filename` explicitly is what lets a `model.plan` and a
`model.onnx` sit side by side in the same version directory — the platform then picks
its own file. The `optimization` block (`:257-267`) enables input/output pinned memory
and sets `gather_kernel_buffer_threshold: 0`; both are host↔device concerns and are
GPU-only, hence the conditional.

The rest of the generated config is device-independent: `max_batch_size`, one input
(name/dtype/dims) and one output from `ModelConfig`, `dynamic_batching` with the
configured `max_queue_delay_microseconds` and `preferred_batch_size` and
`preserve_ordering: false`, `model_transaction_policy.decoupled: false`, and a single
`model_warmup` entry named `warmup_random` that pushes one full-size random batch.

### Load lifecycle

`InferenceModel::start` (`:133`) unloads any previously loaded instances first
(`:142-147`, logging a `warn!` when the unload succeeds, i.e. when something *was*
loaded), resolves the instance count from
`instances.resolve(hardware_name, model_purpose)` — the purpose is passed down by
`InferenceModels::start` as it iterates the model map — and loads.
Restarting the client therefore always replaces whatever was on the server.

### `infer()`

`infer(Vec<Vec<u8>>) -> Vec<Vec<u8>>` (`:301`) computes the per-sample output size from
`output_shape` × precision size, then takes one of two paths:

- **Fast path** (`:321-357`) when `num_inputs <= batch_max_size`: concatenate, insert the
  batch dimension at the front of the input shape, one `model_infer` await, then slice
  the output blob inline. The comment at `:347-349` explains the choice — for a single
  batch, spawning a blocking task costs more than the slicing saves.
- **Chunked path** (`:358-431`): `chunks(max_batch_size)`, one `tokio::spawn` per chunk,
  output splitting on `spawn_blocking`, results placed directly into pre-allocated slots
  by index (no sort), joined with `try_join_all`.

**Today only the fast path executes.** `processing::yolo::process_frame` calls
`infer(vec![pre_frame])` with exactly one element (`src/processing/yolo.rs:451`), and
`batch_max_size` is 16. The chunked path is there for a future caller that batches
across sources.

Requests go to `triton_config.url`, which in the checked-in config is
`http://localhost:8001` — Triton's **gRPC** port (8000 is HTTP, 8002 metrics).

---

## 7. Preprocessing

`processing::resize_letterbox_and_normalize` (`src/processing.rs:147`) does resize,
letterbox and normalization in **one fused pass** over the output buffer, which is
allocated exactly once at the final precision.

1. `calculate_letterbox(in_h, in_w, target)` (`:119`) scales by the **longer** side —
   `scale = target / max(in_h, in_w)` — and centres the result, so padding lands
   symmetrically on the two shorter edges.
2. Resampling is **nearest neighbour**: `src_y = (y * inv_scale).min(in_h - 1)`, and the
   per-row x offsets are precomputed once into `x_offsets` (`:166-169`) so the inner loop
   is a table lookup rather than a float multiply.
3. Normalization (`/255`) is a **256-entry LUT** lookup, not a division.
4. The whole output is pre-filled with the *already normalized* pad colour
   (`PAD_GRAY_COLOR = 114`, `:12`) before the real pixels are written (`:190-192`), so
   the padded region is never touched twice.
5. Output layout is **planar** — `[R… | G… | B… ]`, each plane `target_h*target_w`
   elements, written through three `from_raw_parts_mut` views of the single buffer
   (`:181-187`). *(The doc comment on `preprocess_frame` in `src/processing/yolo.rs:21`
   writes this as `[RRRBBBGGG]`; the code writes R, G, B in that order.)*

`preprocess_frame` (`src/processing/yolo.rs:22`) is the only caller. It first validates
that `frame.data.len() == height * width * 3`, and derives the target size from the last
element of the configured `input_shape` (`yolo.rs:432-435`), i.e. `640` for
`[3, 640, 640]`.

### Lookup tables

Three `OnceLock`s in `src/processing.rs`, all lazily built on first use:

| Table | Size | Direction | Used by |
| --- | --- | --- | --- |
| `F16_TO_F32_LUT` `:15` | 65 536 × `f32` (256 KiB) | FP16 bits → `f32` | both decoders, when precision is FP16 |
| `F16_LUT` `:17` | 256 × `u16` | `u8` pixel → normalized FP16 bits | preprocessing |
| `F32_LUT` `:19` | 256 × `f32` | `u8` pixel → normalized `f32` | preprocessing |

`create_f16_to_f32_lut` (`:22`) implements the IEEE half decode by hand, including
denormals, infinities and NaN. The two 256-entry tables are small enough to stay
L1-resident, which is the point.

---

## 8. Postprocessing

`ModelType` (`src/utils/config.rs:93`) selects the decoder; the dispatch is a `match` on
`spawn_blocking` in `process_frame` (`src/processing/yolo.rs:478-496`). Both decoders
receive the declared `output_shape` as `[u32; 2]`, so no tensor layout is hardcoded.

### Guards

Two, and only two:

- `process_frame` rejects an `output_shape` that is not exactly 2-dimensional
  (`yolo.rs:470-475`).
- `validate_output_size` (`yolo.rs:120`) checks `results.len() == dims[0] * dims[1] *
  precision_size`. **This is the only thing standing between a mis-declared
  `output_shape` and a decoder walking off the end of the tensor**, since both decoders
  index with `get_unchecked`.
- YOLO26 additionally checks `row_fields == 6` *before* the byte check (`yolo.rs:341`),
  precisely because a declared `[max_det, 7]` against a real 7-wide tensor would pass the
  byte check and then silently misparse every row.

### `YOLOV9` — raw head

Tensor is `[4 + classes, anchors]`, **feature-major**: one anchor's values are strided
across the entire buffer, `stride = anchors`. `postprocess_yolov9` (`yolo.rs:153`) walks
anchors in the outer loop and, for each:

1. Reads `x, y, w, h` at offsets `0, s, 2s, 3s` and converts xywh → xyxy while undoing
   the letterbox in one fused expression: `(x ± w/2 - pad) * inv_scale` (`:203-208`).
2. Argmaxes over `target_classes = features - 4` class scores at `4s + anchor + c*s`.
3. **Confidence-filters before NMS** (`:226`). The comment at `:151` flags this as the
   significant optimization: on a 640×640 v9 head there are ~8 400 anchors, and dropping
   sub-threshold candidates first keeps NMS's input tiny.

`bbox_nms` (`yolo.rs:50`) then runs on the survivors: sort by score descending,
class-aware (different classes never suppress each other, `:72`), inline IoU with an
early intersection test, compacting in place with a write index and a final `truncate`.
It is `O(n · kept)` and marked `#[inline(never)]` with the stated intent of keeping the
instruction cache hot for the caller's main loop.

UUIDs are minted **after** NMS (`:299-304`), so the thousands of suppressed candidates
never pay for one. That is also why `Detection` (`client_video/utils.rs:49`) is `Copy`
and id-free while `ResultBBOX` (`:69`) owns a `String` id.

### `YOLO26` — end-to-end head

Tensor is `[max_detections, 6]`, **row-major**: `[x1, y1, x2, y2, score, class_id]`,
six contiguous values per detection. `postprocess_yolo26` (`yolo.rs:328`):

1. Reads **score first** (`base + 4`) and `continue`s below threshold, so a rejected row
   costs one read (`:376-380`).
2. Boxes are already xyxy in letterboxed pixels, so only `(v - pad) * inv_scale` is
   needed — no xywh conversion, no argmax.
3. `class_id` arrives as an integral float and is `.round().max(0.0) as u32`.
4. Allocates `Vec::with_capacity(max_detections)` once; the vector never grows.

**No NMS is applied, and `nms_iou_threshold` is deliberately not a parameter of this
function.** The head has already deduplicated. Note the documented consequence
(`:326-327`): the head can legitimately emit the same box under more than one class id,
and every such row above the threshold is reported.

The `decode_rows!` macro (`:371`) is what lets the FP16 and FP32 arms share one body
while keeping the element read monomorphic in each arm.

With the checked-in config (`[300, 6]`, FP32) that is 1 800 values in a linear sweep,
versus a v9-e head's stride-walk over hundreds of thousands.

---

## 9. Statistics

Two independent reporters, both started from `Statistics::start`
(`src/statistics.rs:154`).

**Per-source counters** — a tokio task ticking every `SOURCE_STATS_INTERVAL` (1 s,
`:11`). On each tick it iterates every `SourceProcessor`, logs one `inference statistics`
event, and **resets the counters to zero** (`:170`). Every number in that log line is
therefore a one-second window, not a cumulative total. Averages are computed by dividing
the accumulated microsecond totals by `frames_success` (`:287-294`), and stay at `0.00`
when no frame succeeded in the window.

| Counter | Meaning |
| --- | --- |
| `frames_total` | Frames seen: incremented once per skipped frame (`source.rs:220`) **and** once per processed frame (`:138`). |
| `frames_expected` | Incremented at the same site as the processed-frame `frames_total`, so it counts only frames that reached processing. |
| `frames_success` | `process_frame` returned `Ok`. |
| `frames_failed` | `process_frame` returned `Err`, **or** the frame was evicted from a full queue by the drop callback (`source.rs:99`). |

| Timer (µs, averaged) | Measures |
| --- | --- |
| `avg_queue` | From `added` (stamped on enqueue, `source.rs:213`) to the start of `process_frame` — queue wait plus scheduling delay. |
| `avg_pre_proc` | The `spawn_blocking` preprocessing call (`yolo.rs:441-447`). |
| `avg_inference` | `infer()` end to end — gRPC round trip plus Triton's own queueing and execution. |
| `avg_post_proc` | The `spawn_blocking` decode call. |
| `avg_results` | `populate_bboxes` — the `PostResults` FFI call. **Only accumulated when the frame produced at least one bbox** (`source.rs:245`). |
| `avg_processing` | `yolo::process_frame` wall time (pre + inference + post) **plus** the queue time added at `source.rs:260`. It does *not* include `results`, which is measured after `stats.processing` is set. |

**GPU telemetry** — only when `device_type == GPU`. `Statistics::new` (`:133`) initialises
NVML eagerly at startup (`dlopen` of the driver library is expensive and a late failure
would silently cost all telemetry) and treats a failure as fatal. `start()` then
`take()`s the handle and moves it into a `spawn_blocking` thread that loops on
`GPU_STATS_INTERVAL` (1 s, `:12`), sleeping for the remainder of each period. It reports
GPU 0 only: name, UUID, serial, total/used/free memory in MiB, utilization %, and memory
%. On CPU the thread is never spawned and a single `warn!` is logged at construction.

---

## 10. Configuration reference — `secrets/config.yaml`

Deserialized into `AppConfig` (`src/utils/config.rs:186`) by `serde_yaml`. Unknown keys
are ignored; missing keys without a `#[serde(default)]` are a hard parse error.

### Top level

| Key | Type | Effect |
| --- | --- | --- |
| `local` | `bool` | When true, adds the `logs/app.log` file layer. Stdout logging is unconditional. |
| `sources_config` | map | See below. |
| `triton_config.url` | string | Triton gRPC endpoint. Currently `http://localhost:8001`. |
| `inference_config` | map | See below. |

`hardware_name` is a field of `AppConfig` but is `#[serde(skip)]` (`:193`) — it is
resolved at startup, never read from the file.

### `sources_config`

| Key | Type | Effect |
| --- | --- | --- |
| `ids` | `Vec<u32>` | The sources to start. Drives both `SourceProcessors` and the `InitSources` call. |
| `default` | `SourceConfig` | Required. Baseline `inf_frame` / `conf_threshold` / `nms_iou_threshold`. |
| `custom` | `map<u32, SourceConfigOptional>` | Optional per-source overrides; defaults to empty. Every field inside is `Option`. |
| `sources` | — | **Not authored by hand.** `#[serde(default)]`, populated by `AppConfig::new`. |

`AppConfig::new` (`:207-231`) walks `ids`, clones `default`, and applies each present
override **through a range filter**:

| Field | Accepted range | Out-of-range behaviour |
| --- | --- | --- |
| `inf_frame` | `1..=30` | `.filter(...)` yields `None`, so `unwrap_or` silently falls back to the default. No warning. |
| `conf_threshold` | `0.0..=1.0` | same |
| `nms_iou_threshold` | `0.0..=1.0` | same |

Note the asymmetry: the **default** values are *not* clamped — only overrides pass
through the filter. A `default.inf_frame` of 0 would reach the modulo in `add_to_queue`
unchecked. An id listed in `custom` but absent from `ids` is simply never read.

`nms_iou_threshold` is only consulted by the YOLOv9 decoder; under `model_type: YOLO26`
it is parsed, clamped and then unused.

### `inference_config`

The device is **not** configured here — it comes from the `DEVICE_TYPE` environment
variable (see below).

| Key | Type | Effect |
| --- | --- | --- |
| `instances.default` | `u32` | Instance count when the hardware, or the purpose under that hardware, is not listed. |
| `instances.custom` | `map<String, map<ModelPurpose, u32>>` | Keyed first by `hardware_name` — the literal `"CPU"`, or the NVML device name for GPU (e.g. `"NVIDIA GeForce RTX 4090"`) — then by `ModelPurpose`, so the count is tunable per hardware *and* per task. Resolved by `InstancesConfig::resolve` (`:78`), which falls back to `default` when either level is absent. |
| `models` | `map<ModelPurpose, ModelConfig>` | Currently `FrameDetection` is the only `ModelPurpose` variant (`:109`). |

`ModelConfig` (`:14`):

| Field | Notes |
| --- | --- |
| `name` | Triton model name; must match the directory in `triton_models/`. |
| `model_type` | `YOLOV9` or `YOLO26` — selects the decoder. |
| `precision` | Optional. `FP32` or `FP16`; omit to take the device default — FP32 under `DEVICE_TYPE=CPU`, FP16 under `GPU`. Becomes `TYPE_FP32`/`TYPE_FP16` in the generated config, the request datatype, the preprocessing output format, and the decoder's element width. One knob, four consumers — so an explicit value must match the precision the artifact was built at. |
| `input_name` / `output_name` | Tensor names as compiled into the model. |
| `input_shape` | Without the batch dimension, e.g. `[3, 640, 640]`. The **last** element is the letterbox target size. |
| `output_shape` | Must be 2-dimensional. `[4 + classes, anchors]` for YOLOv9; `[max_detections, 6]` for YOLO26. |
| `batch_max_size` | `max_batch_size`, and the fast-path threshold in `infer()`. |
| `batch_max_queue_delay` | `dynamic_batching.max_queue_delay_microseconds`. |
| `batch_preferred_sizes` | `dynamic_batching.preferred_batch_size`. |

### The checked-in configuration

CPU, one instance, one source (id 1), inference on every 25th frame, conf 0.25,
IoU 0.5 (unused under YOLO26), model `yolo26x` FP32 with input `images` `[3,640,640]`
and output `output` `[300,6]`, dynamic batching up to 16 with a 2.5 ms delay. Note there
is no `custom` block under `sources_config` — that is fine, it defaults to empty.

---

## 11. Building and running

### Build

```bash
cd client-object-detection
cargo build --release
```

`.cargo/config.toml` sets `CMAKE_POLICY_VERSION_MINIMUM=3.5` so the build works out of
the box: `triton-client 0.2.0` pulls in `prost-build 0.10.4`, which vendors protobuf and
builds it with CMake, and CMake ≥ 4.0 dropped compatibility with the
`cmake_minimum_required(VERSION < 3.5)` that vendored protobuf declares. The faster
alternative, noted in the file itself, is to install `protoc`
(`dnf install protobuf-compiler`) and set `PROTOC_NO_VENDOR=1`, which skips the
from-source protobuf build entirely.

### Run

The moon workspace maps this crate (`.moon/workspace.yml`:
`client-detection: 'client-object-detection'`), and the entry points are

```bash
moon run client-detection:cpu     # or
moon run client-detection:gpu
```

Each depends on the matching `services:triton-*` task, so Triton is started for you.
Those tasks run `docker compose up --detach --wait`, which blocks on the compose
healthcheck in `services/docker-compose-triton.yml` and exits only once Triton reports ready —
so by the time `cargo run` starts, `server_ready()` is guaranteed to succeed. This is
why the client task needs no readiness polling of its own.

The gating depends on the Triton tasks being **non-persistent**: moon starts
`persistent` tasks concurrently and never waits for one to finish, so a persistent
Triton task would not gate anything. Confirm the shape with
`moon action-graph client-detection:cpu --dot` — it must show `RunTask(services:triton-cpu)`
feeding `RunPersistentTask(client-detection:cpu)`. Two `RunPersistentTask` nodes means
the ordering is broken.

Because Triton runs detached, nothing streams its output and it outlives the client:

```bash
moon run services:triton-logs     # follow Triton's output
moon run services:triton-down     # stop it (names both profiles)
```

### Environment

| Variable | Required | Effect |
| --- | --- | --- |
| `DEVICE_TYPE` | **yes** | `CPU` or `GPU`, case-insensitive. Selects the Triton platform, model filename, instance kind, the `optimization` block, whether NVML runs, and each model's default precision. No default — unset, the process exits at config load naming the variable. Both moon tasks set it to match the Triton they boot. |
| `RUST_LOG` | in practice | `EnvFilter::from_default_env()` carries no directives when unset, so nothing is logged (§Logging). Both moon tasks set `INFO`; a value exported in your shell takes precedence. |

At startup the client logs one line per model reporting the resolved device, hardware
name, precision, and whether that precision came from the config or the device default.

The manual equivalent, if you are not going through moon:

1. Start Triton with the matching profile and wait for health (run from `services/`,
   since the compose file mounts `../triton_models` relative to its own directory):
   `cd services && docker compose -f docker-compose-triton.yml --profile cpu up --detach --wait`
2. Place `libclient_video.so` in `client-object-detection/secrets/` and the model in
   `triton_models/<name>/1/model.{onnx,plan}` (§1).
3. `cd client-object-detection && RUST_LOG=INFO DEVICE_TYPE=CPU ./target/release/client`

The sibling script is worth reading as a template but is **not** directly reusable: it
references `../docker-compose.yml`, which has been split into the three per-service
compose files under `services/`, and it exports `PLAYER_BACKEND_URL`, which this
crate never reads (that variable is consumed by `libclient_video.so`, so it may still be
required by the library — this has not been verified from the Rust side).

---

## 12. Sharp edges

Observations of current behaviour. Several are load-bearing assumptions that hold today;
none are changed by this document.

**Alignment assumptions in the unsafe casts.** Both decoders reinterpret the raw
`Vec<u8>` output as `*const f32` / `*const u16` via `from_raw_parts`
(`yolo.rs:183-185`, `:239-241`, `:405-407`, `:414-416`), and `infer()`'s output slicing
does pointer arithmetic on `output_blob.as_ptr()` (`inference.rs:350`, `:399`). A
`Vec<u8>` is only guaranteed 1-byte aligned; the code assumes the allocator returns
suitably aligned blocks, which it does in practice for allocations of this size.
Likewise `processing.rs:180` and `:217` cast the freshly allocated output `Vec<u8>` to
`*mut u16` / `*mut f32`.

**`infer()`'s fast path does not validate the output blob length** before slicing
(`inference.rs:350-357`): it reads `num_inputs * output_size_per_sample` bytes from
whatever Triton returned. A short blob would be an out-of-bounds read; the
`validate_output_size` check that would catch a size mismatch happens later, in the
decoder.

**`frames_total` counts two different things.** It is incremented once per skipped frame
in `add_to_queue` (`source.rs:220`) and once per processed frame in the consumer
(`:138`), while `frames_expected` is incremented only at the second site (`:141`). So
`frames_total` is "frames the source produced" (minus any that overflowed the queue,
which are only counted as `frames_failed`) and `frames_expected` is "frames that reached
processing".

**`stats.processing` includes queue time.** `source.rs:260` does
`stats.processing += frame_queue_time`, while `:259` also reports the same value as
`stats.queue`. The queue wait is therefore counted twice across the two metrics, and
`avg_processing` is not the sum of the other timers.

**`corners_coordinates` truncates and can overflow.** `client_video/utils.rs:90` casts
`f32` → `u32` with `as`. Rust's float→int `as` saturates, so a negative coordinate
(entirely possible after the letterbox inverse for a box that runs off the frame edge)
becomes `0` rather than a large number — but a large positive coordinate saturates to
`u32::MAX`, and the subsequent `y * frame.width + x` is unchecked `u32` arithmetic that
would panic in a debug build and wrap in release. The function also returns flat pixel
indices, so the caller cannot recover x and y without knowing the frame width.

**`class_name()` only knows COCO 0–5** (`client_video/utils.rs:77`): person, bicycle,
car, motorcycle, airplane, bus. Every other id is stringified as a number, so the
backend receives e.g. `"17"` for a dog.

**The logging guard is leaked.** `std::mem::forget(_guard)` (`config.rs:288`) keeps the
non-blocking appender's worker alive for the process lifetime. Intentional, and the only
reasonable option given that `init_logging` returns nothing.

**`RunMode::Regular` is hardcoded** in `Services::start` (`services.rs:93`). `Debug` is
only reachable by editing the source; there is no config key or environment variable.

**`_post_results_callback` logs nothing.** Its `tracing::info!` is commented out
(`client_video.rs:436-442`), so the callback exists solely to free the library's
pointers. `#[allow(unused_variables)]` on `:417` is what keeps that quiet.

**Dead code kept on purpose.** `#[allow(dead_code)]` sits on `ClientVideo::stop_sources`
(`client_video.rs:230`), the `SourceProcessor` struct (`source.rs:69`), the `Statistics`
struct (`statistics.rs:118`) and `FixedSizeQueue` (`queue.rs:5`). Because `lib.rs`
re-exports everything as `pub`, other unused items are not flagged at all — among them
`utils::get_image_raw`, `AppConfig::is_local`, `InferenceModel::{client,
triton_config, base_request}`, `InferenceModels::models` and `SourceStatus::as_i32`.

**Nothing ever stops cleanly.** `Ctrl+C` kills the process outright (§3), so
`StopSources` is never called and the `Drop` impls that would abort the background tasks
never run. In a long-lived deployment this is the OS's problem, not the crate's.

**Symbols are resolved on every FFI call.** Each wrapper does
`library().get(b"SetCallbacks")` etc. rather than caching a `Symbol`. On the
`populate_bboxes` path that is one `dlsym` per frame with detections. It has not shown up
as a cost, but it is a per-frame cost.
