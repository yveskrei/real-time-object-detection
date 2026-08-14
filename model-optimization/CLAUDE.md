# model-optimization

The model toolchain, and the only project here that produces models rather than consuming
them. It wraps a PyTorch checkpoint so the graph emits exactly one tensor in the layout the
runtime clients decode, exports it to FP32 ONNX, and documents the `trtexec` step that
compiles that ONNX into an FP16 TensorRT engine for the Triton model repository. Python,
`uv`-managed; run by hand when a model changes, never at runtime.

**The authoritative reference for this crate is `docs/model-optimization.md`**
(repo-root-relative). Read it before changing anything here — it covers the export
pipeline, the three per-architecture loaders and the tensor contract each one guarantees,
why the output shape is pinned into the graph, the full CLI reference, the TensorRT step,
and how artifacts are placed under `triton_models/`.

The tensor each loader emits is a contract with the Rust clients: see
`docs/client-object-detection.md` §8 for the YOLO decoders and
`docs/client-image-retrieval.md` §8 for DINOv3, which consume these exports.
