## Model Conversion
The following folder contains scripts that involve converting the raw `.pt` frame you got(whether its self made or pre-made - e.g. YOLOV9), to a format that is the most performant in a production system. We essentially want to get the fastest inference time, with minimal accuracy loss.

We first need to convert our model from `.pt` to `.onnx`, which is a universal format for machine learning models.
The conversion will result with model_**fp16** and model_**fp32** files, which represent the same model with different datatype formats. <br>
**FP16** Models are essentially equal in accuracy(for inference) and are much lighter and faster to run, and it is recommanded to use. 

Next, we would be converting the `.onnx` file we have to `.engine`, using NVIDIA's TensorRT tool.<br>
TensorRT compiles a model for your specific achitecture(one used at time of compilation), therefore making it very efficient when running on your machine.<br>

## Setting up the environment
```bash
uv sync
```

## Pytorch to Onnx Conversion
The following command is used for converting a model from Pytorch to Onnx:
```bash
# YOLOV9
uv run export_model.py \
  --model-type YOLOV9 \
  --model-path MODEL.pt \
  --model-source-code ./yolov9/

# DINOV3
uv run export_model.py \
  --model-type DINOV3 \
  --model-path MODEL.pt \
  --model-source-code ./dinov3/ \
  --dino-type dinov3_vitb16

# YOLO26X - loaded from the installed `ultralytics` package,
# so no --model-source-code is needed
uv run export_model.py \
  --model-type YOLO26X \
  --model-path models/yolo26x.pt
```

Optional flags:
- `--max-det` (default `300`) - detections per image, baked into the graph (YOLO26X)
- `--simplify` - run `onnxslim` over the exported graph (see below)
- `--output-path` (defaults to `cwd`) - The path to save the exported model

### Output shape annotation
Every export runs the model once before tracing and writes the resulting shape into the ONNX graph's
output signature, so it advertises e.g. `[batch_size, 84, 8400]` rather than
`[batch_size, Concatoutput_dim_1, Concatoutput_dim_2]`.

This matters for Triton. ONNX shape inference cannot always derive the output dims on its own - a
`view()`/`reshape()` against the symbolic batch axis (which YOLOv9's head does) leaves the trailing
dimensions as symbolic placeholders. Triton reads those as `[-1, -1, -1]`, compares them to the
`dims` in your `config.pbtxt`, and refuses to load the model. The numbers were always correct; only
the declared signature was under-specified.

Only the batch axis is dynamic (see `dynamic_axes` in `export_model.py`), so every other dimension is
static by construction and safe to pin. Deriving it from a real forward pass rather than hardcoding
means it stays correct across class counts, input sizes and `--max-det` values.

## Onnx to TensorRT Conversion
The following command is used for converting a model (with support of batch inference).<br>
We would be doing the TensorRT conversion from within the docker image of Triton Server, to ensure its compatibility with the compiled model:
```bash
# YOLOV9/YOLO26
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --optShapes=images:8x3x640x640 \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --shapes=images:1x3x640x640 \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw

# DinoV3 - 512
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --optShapes=images:8x3x512x512 \
    --minShapes=images:1x3x512x512 \
    --maxShapes=images:16x3x512x512 \
    --shapes=images:1x3x512x512 \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw

# DinoV3 - 224
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --optShapes=images:8x3x224x224 \
    --minShapes=images:1x3x224x224 \
    --maxShapes=images:16x3x224x224 \
    --shapes=images:1x3x224x224 \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw
```
