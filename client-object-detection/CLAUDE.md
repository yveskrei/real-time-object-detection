# client-object-detection

Real-time object detection client: it takes decoded video frames from `libclient_video.so`
over FFI, runs YOLO inference against NVIDIA Triton over gRPC, and posts the resulting
bounding boxes back through the same library. Per-source queueing, frame skipping and
statistics live in this crate; the Triton model configuration is generated at runtime
rather than committed as a `config.pbtxt`.

**The authoritative reference for this crate is `docs/client-object-detection.md`**
(repo-root-relative). Read it before changing anything here — it covers the startup
chain, the FFI contract, the inference and pre/post-processing pipelines, the full
`secrets/config.yaml` reference, how to run it, and the known sharp edges.
