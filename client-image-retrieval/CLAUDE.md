# client-image-retrieval

Real-time image-retrieval ingest client: it takes decoded video frames from
`libclient_video.so` over FFI, runs three Triton models over gRPC — a YOLO detector whose
boxes are used as crop regions, plus two DINOv3 embedders for the whole frame and for
each crop — and writes the resulting 768-dimensional vectors into Elasticsearch. The
video library is used as a frame source only; nothing is posted back through it.

**The authoritative reference for this crate is `docs/client-image-retrieval.md`**
(repo-root-relative). Read it before changing anything here — it covers the startup
chain, the three-model pipeline, the Elasticsearch write path and its dependence on
`elasticsearch-setup.md`, the full `secrets/config.yaml` reference, and how to run it.
It cross-references `docs/client-object-detection.md` for the parts shared byte-for-byte with
`client-object-detection`.
