# Object Detection Inference Client
The script is a collection of research & production grade scripts allowing you to deploy a real-time object detection inference client using Nvidia Triton Inference Server.
The client was built to allow scalability, and supports multiple instances running at once(Using Kubernetes/third party tools).

## Architecture

## Extras - TensorRT conversion
TensorRT is a tool used to enhance capabilities of a given model. Works best with ONNX models.<br>
It compiles a model for your specific achitecture(one used at time of compilation).<br>
The following command is used for converting a model(with support of batch inference):
```bash
trtexec \
  --onnx=MODEL.onnx \
  --saveEngine=CONVERTED.engine \
  --optShapes=images:16x3x640x640 \
  --minShapes=images:1x3x640x640 \
  --maxShapes=images:32x3x640x640 \
  --shapes=images:16x3x640x640 \
  --fp16(Or omit for FP32)
```

To test performance of TRT model, use the following command:
```bash
trtexec \
  --loadEngine=CONVERTED.engine \
  --shapes=images:4x3x640x640 \
  --exportTimes=inference_times.json
```

## References
* [Triton Dynamic Batching(Nvidia)](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html#what-is-dynamic-batching)
* [Triton Inference Protocols](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html)