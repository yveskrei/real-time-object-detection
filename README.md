# Object Detection Inference Client
The script is a collection of research & production grade scripts allowing you to deploy a real-time object detection inference client using Nvidia Triton Inference Server.
The client was built to allow scalability, and supports multiple instances running at once(Using Kubernetes/third party tools).

## Extras - Get model best latency/throughput
Using a third party tool, `perf_analyzer`, we iterate over different batch sizes for one model instance. We find the sweet spot of when the model is giving the best latency for the max amount of batch size.
```bash
for b in 1 2 4 8 16 32; do
  perf_analyzer \
    -m <model_name> \
    -b $b \
    --concurrency-range 1:1 \
    --collect-metrics \
    --verbose-csv \
    -f results_batch_${b}.csv
done
```

## Extras - TensorRT conversion
TensorRT is a tool used to enhance capabilities of a given model. Works best with ONNX models.<br>
It compiles a model for your specific achitecture(one used at time of compilation).<br>
The following command is used for converting a model(with support of batch inference)(best to do on triton docker image itself for full TRT compatibility):
```bash
/usr/src/tensorrt/bin/trtexec \
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
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=CONVERTED.engine \
  --shapes=images:4x3x640x640 \
  --exportTimes=inference_times.json
```

We convert the model to TensorRT with the binary found inside of the docker image, in order to ensure compatibility with the host system. Used versions:
- Latest image: `nvcr.io/nvidia/tritonserver:25.06-py3`
- Previous image: `nvcr.io/nvidia/tritonserver:23.08-py3`
## References
* [Triton Dynamic Batching(Nvidia)](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html#what-is-dynamic-batching)
* [Triton Inference Protocols](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html)