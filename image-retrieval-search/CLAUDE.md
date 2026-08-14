# image-retrieval-search

The query half of image retrieval: an axum HTTP API that embeds an uploaded image with
DINOv3 on Triton, parks the vector in Redis under a generated UUID with a 120 s TTL, and
serves kNN search over the Elasticsearch index that `client-image-retrieval` writes.
`POST /upload` returns the UUID; `GET /search` reads it back and runs the query. No
video, no FFI — the unit of work is an HTTP request.

**The authoritative reference for this crate is `docs/image-retrieval-search.md`**
(repo-root-relative). Read it before changing anything here — it covers the HTTP surface,
the two-step upload/search flow, the kNN query and its three tuning tiers, the dependence
on `elasticsearch-setup.md`, the full `secrets/config.yaml` reference, and how to run it.
It cross-references `docs/client-object-detection.md` and `docs/client-image-retrieval.md`
for the Triton, preprocessing and statistics machinery shared with those crates.
