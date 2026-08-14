# `image-retrieval-search` — architecture and reference

The **query half** of image retrieval. `client-image-retrieval` is the ingest half: it
embeds video frames and detections with DINOv3 and bulk-writes 768-dimensional vectors
into Elasticsearch. This crate is the other end of that index — an axum HTTP service that
takes an uploaded image, embeds it with the same models on the same Triton server, and
runs a kNN query against the vectors the ingest client wrote.

There is no video, no FFI and no queue here. The unit of work is an HTTP request.

All paths in this document are relative to the repository root unless stated otherwise.
The crate root is `image-retrieval-search`, and the Cargo package name is `client`
(`image-retrieval-search/Cargo.toml:2`), so the binary produced is
`image-retrieval-search/target/{debug,release}/client`. Rust
**edition 2024**. Source paths below are given relative to the crate root.

---

## 1. Relationship to the two client crates

This crate shares its inference and preprocessing skeleton with `client-object-detection`
and `client-image-retrieval`. Those parts are documented once, in the other two
documents, and this one cross-references rather than restates them:

| Area | Covered by |
| --- | --- |
| Triton gRPC client, the runtime-generated model config, `--model-control-mode=explicit`, the `DeviceType` CPU/GPU split, the load/unload lifecycle, `infer()`'s fast and chunked paths | `docs/client-object-detection.md` §6 |
| Letterbox geometry (`calculate_letterbox`), the fused single-pass preprocessor, the planar `[R…|G…|B…]` output layout, the ImageNet per-channel LUTs and the FP16/FP32 conversion tables | `docs/client-image-retrieval.md` §7, `docs/client-object-detection.md` §7 |
| DINOv3 postprocessing — an embedder's output *is* the feature vector, so the whole job is a width conversion | `docs/client-image-retrieval.md` §8 |
| `DeviceType::hardware_name()`, NVML initialisation and the 1 s GPU telemetry thread | `docs/client-object-detection.md` §9 |
| The `OnceCell<Arc<Services>>` singleton and why the global is set before `start()` | `docs/client-object-detection.md` §3 |

The files themselves line up as follows:

| File | Relation to `client-image-retrieval` |
| --- | --- |
| `src/inference.rs` | Same code, formatted differently. `InferenceModels` holds one `InferenceModel` per configured `ModelPurpose`; `InferenceModels::model()` (`inference.rs:63`) is how a handler reaches one by purpose. |
| `src/processing.rs` | The ImageNet half only. `resize_letterbox_and_normalize_imagenet` (`:326`), `calculate_letterbox` (`:298`) and all the LUTs are the same; the plain `x/255` preprocessor is absent, because nothing here runs a detector. Also defines this crate's `RawFrame` and `ResultEmbedding`. |
| `src/processing/dino.rs` | The retrieval crate's file minus `preprocess_bbox` / `process_bboxes`. What remains is `preprocess_frame` (`:17`), `postprocess` (`:47`) and `process_frame` (`:91`). |
| `src/utils/config.rs` | Same shape, retargeted: no `sources_config`, plus `port`, `redis_config` and `search_config`. `ModelType` has one variant, `DINOV3` (`:107`), and `ModelPurpose` has two, both embedders (`:128`). |
| `src/statistics.rs` | Same reporter mechanics and the same NVML code, but one process-wide `ProcessingStats` aggregate instead of per-source counters. See §9. |

Two structural differences follow from having no video source:

**`RawFrame` carries no stream identity.** It is `{ data, height, width }`
(`processing.rs:20-24`) — no `source_id`, no `pts`. A frame here is whatever image the
caller uploaded.

**`ResultEmbedding` is `Encode, Decode`** (`processing.rs:27-30`), not `Serialize`. The
ingest crate serialises the vector into a JSON document; this crate bincodes it into
Redis and then hands `data` to `serde_json` as a query vector.

---

## 2. Module map

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Config → services → start → build the axum `Router` → bind → serve. |
| `src/lib.rs` | Module declarations only; everything is `pub`, so the binary consumes the crate as a library. |
| `src/services.rs` | The `Services` singleton: Elastic, Redis, statistics, inference models. |
| `src/handlers.rs` | Route table, the OpenAPI document, Swagger UI mounting, `/health` and the 404 fallback. |
| `src/handlers/api.rs` | `ApiResponse<T>` — the response envelope and its constructors — plus `AppError` and the `ApiResult<T>` alias. |
| `src/handlers/retrieval/mod.rs` | The two retrieval routes. |
| `src/handlers/retrieval/post.rs` | `POST /upload`: multipart parsing, image decoding, and the `ImageUploadRequest` schema. |
| `src/handlers/retrieval/get.rs` | `GET /search`: query-parameter binding and response shaping. |
| `src/processing/search.rs` | The orchestration layer. Inference, Redis and Elasticsearch meet here; the two handlers are thin wrappers over its two functions. |
| `src/processing.rs` | Fused ImageNet preprocessing and the conversion lookup tables; defines `RawFrame` and `ResultEmbedding`. |
| `src/processing/dino.rs` | DINOv3 preprocess / infer / postprocess for one image. |
| `src/inference.rs` | Triton gRPC client, runtime model-config generation, load/unload, batching. |
| `src/utils.rs` | `parse_image` / `detect_format` — bytes to RGB8. |
| `src/utils/config.rs` | `secrets/config.yaml` deserialization, hardware-name resolution, tracing setup. |
| `src/utils/elastic.rs` | The kNN search request builder and its response unwrapping. |
| `src/utils/redis.rs` | One multiplexed Redis connection, built from `redis_config`. |
| `src/statistics.rs` | The process-wide processing aggregate, its 1 s reporter, and NVML GPU telemetry. |

---

## 3. Startup chain

```
main()                                        src/main.rs:12
 └─ AppConfig::new()                          src/utils/config.rs:229
      ├─ load_config_file()  reads ./secrets/config.yaml       :247
      ├─ init_logging(local)                                   :263
      ├─ DeviceType::from_env()  reads DEVICE_TYPE, required
      ├─ device_type.hardware_name()  (NVML when GPU)          :176
      └─ fill each model's absent precision from the device default
 └─ services::init_services(&cfg)                             src/services.rs:28
      ├─ Services::new(...)                                    :52
      │    ├─ Elastic::new     → transport + client, keeps elastic_config
      │    │                     and the SearchType tier map    :54
      │    ├─ Redis::new       → Client::open, then one
      │    │                     multiplexed connection         :61
      │    ├─ Statistics::new  → NVML init when device_type == GPU  :65
      │    └─ InferenceModels::new → one InferenceModel per configured
      │                             ModelPurpose, each checking
      │                             server_ready()              :69
      └─ SERVICES.set(Arc::new(services))                       :38
 └─ get_services()?.start(&cfg)                               src/services.rs:82
      ├─ Statistics::start        → processing-stats task + GPU-stats thread
      └─ InferenceModels::start   → for each model: unload previous, load with
                                     the generated config       inference.rs:50
 └─ Router::new().merge(handlers::routes())                    src/main.rs:25
      .layer(CorsLayer::permissive())                                 :27
      .layer(TraceLayer::new_for_http())                              :28
      .layer(DefaultBodyLimit::max(50 MB))                            :29
 └─ TcpListener::bind(127.0.0.1:<port>)                               :32-38
 └─ axum::serve(listener, app).await                                  :43
```

`Services::new` builds the four members in order and `init_services` sets the global
*before* `start()` runs (`services.rs:38`), the same discipline as the client crates:
`start()` and every request handler re-enter through `services::get_services()`.

`init_services` bails if `SERVICES` is already populated (`services.rs:29-31`), so it is
callable exactly once.

Unlike the client crates, `main` does not sleep forever — `axum::serve` is the thing that
never returns (`main.rs:43`). Shutdown is `SIGINT` with the default handler; no graceful
shutdown signal is wired into `serve`.

Of the four services, only two prove their dependency is reachable at startup.
`InferenceModel::new` calls `server_ready()` and bails when Triton is not serving
(`inference.rs:101-107`), and `Redis::new` opens a real connection
(`redis.rs:27`). `Elastic::new` builds a lazy `Transport::single_node`
(`elastic.rs:31`), which does not contact the cluster — the first search request is the
first connection attempt. §11 covers how the moon tasks gate all three.

### Logging

Identical to the client crates (`config.rs:263-292`): `EnvFilter::from_default_env()`,
so export `RUST_LOG`; an always-on JSON stdout layer with RFC-3339 UTC timestamps; and a
`logs/app.log` file layer when `local: true`, written through a non-blocking appender
whose guard is deliberately leaked with `std::mem::forget` so the writer thread outlives
the call.

---

## 4. The HTTP surface

axum 0.8 with utoipa 5 for the OpenAPI document and tower-http for the middleware.

`handlers::routes()` (`handlers.rs:45`) builds the whole tree and `main` merges it **at
the root** (`main.rs:26`) — there is no `/api` or `/v1` prefix anywhere:

| Method | Path | Handler |
| --- | --- | --- |
| `POST` | `/upload` | `retrieval::post::upload_image` (`retrieval/mod.rs:13`) |
| `GET` | `/search` | `retrieval::get::search_image` (`retrieval/mod.rs:14`) |
| `GET` | `/health` | `handlers::health` (`handlers.rs:55`) |
| `GET` | `/docs` | Swagger UI (`handlers.rs:50`) |
| `GET` | `/openapi.json` | The generated OpenAPI document (`handlers.rs:51`) |
| any | anything else | `handlers::default` → 404 (`handlers.rs:56`) |

The OpenAPI document is generated from the `#[derive(OpenApi)]` on `APIDoc`
(`handlers.rs:21-43`): the three annotated paths, the `ImageUploadRequest`,
`ImageSearchRequest` and `ModelPurpose` schemas, and two tags — `General` and `Retrieval`
(`handlers.rs:18-19`). `SwaggerUi::new("/docs").url("/openapi.json", openapi)` serves the
UI and the document together, so `/docs` is a live console against the running service.

### The response envelope

Every handler returns `ApiResponse<T>` (`api.rs:10-14`), whose `IntoResponse`
(`api.rs:114-123`) emits the same two-key body for success and failure alike:

```json
{ "message": "…", "data": … }
```

with the HTTP status taken from the `status_code` field. The constructors in
`api.rs:16-112` cover the usual statuses; the handlers use `success_with_message`
(`:26`), `bad_request` (`:67`) and `not_found` (`:79`). `AppError` (`api.rs:126`) wraps
`anyhow::Error` into a 500 with the error's `Display` as the message, and the blanket
`From` impl (`:136-143`) is what makes `?` usable in a handler returning
`ApiResult<T>` (`:146`).

### Middleware and binding

Three layers wrap the whole router (`main.rs:27-29`): `CorsLayer::permissive()`,
`TraceLayer::new_for_http()` — which emits request/response spans through the same
`tracing` subscriber as everything else — and `DefaultBodyLimit::max(50 * 1024 * 1024)`,
a 50 MB cap that applies to the multipart upload body.

The listener binds `127.0.0.1` with the port from `config.yaml` (`main.rs:32-38`), and
the bound address is logged at `main.rs:40`. The loopback address is hardcoded; the port
is the only part configuration controls.

---

## 5. The two-step retrieval flow

This is the shape of the whole crate. Upload and search are two separate HTTP calls, and
a UUID minted server-side is the only thing joining them. Redis holds the embedding in
between.

```
  ┌── caller ────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  POST /upload            multipart: image=<bytes>, model_type=<enum>  │
  ▼                                                                      │
 ImageUpload::from_multipart                          retrieval/post.rs:29
  │  read "image" field as Bytes                                    :36-39
  │  read "model_type" as text, parse::<ModelPurpose>()             :41-47
  │  utils::parse_image  → (rgb8, width, height)                    :56
  │  build RawFrame { data, height, width }                         :58-62
  ▼
 processing::search::upload_image(frame, model_type)   processing/search.rs:20
  │  inference_models().model(model_type)                           :24
  │  dino::process_frame  → (FrameProcessStats, ResultEmbedding)    :26
  │  image_id = Uuid::new_v4()                                      :29
  │  ImageMetadata { embedding, model_type }                        :30-33
  │  bincode::encode_to_vec(config::standard())                     :36
  │  SET retrieval_<uuid> <bytes> EX 120                            :39-48
  │  statistics.processing_stats().accumulate(&stats)               :51-56
  ▼
  {"message": "Image uploaded successfully",
   "data": {"image_id": "<uuid>"}}                     retrieval/post.rs:90-95
  │
  │        ── caller holds the uuid, ≤ 120 s ──
  ▼
  GET /search?image_id=<uuid>&search_type=MEDIUM[&channel_ids=1,2,3]
                              [&timestamp_start=…][&timestamp_end=…]
  ▼
 search_image                                          retrieval/get.rs:66
  │  bind Query<ImageSearchRequest>                                 :67
  │  parse_channel_ids  "1,2,3" → Vec<u32>, 400 if malformed        :30, :74
  │  build SearchMetadata from the three optional filters           :84-88
  ▼
 processing::search::search_image                      processing/search.rs:61
  │  GET retrieval_<uuid>                                           :69-74
  │  None → "Image is not found!"                                   :75
  │  bincode::decode_from_slice → ImageMetadata                     :81-83
  │  elastic().search_disk_bbq(embedding, search_type, metadata)    :86-89
  ▼
  {"message": "Image processed successfully",
   "data": {"count": N,
            "candidates": [{"score": …, "metadata": {…}}]}}   retrieval/get.rs:61-74
```

Points worth holding onto:

- **The UUID is the handle, and the caller owns it.** `Uuid::new_v4()`
  (`search.rs:29`) is minted after the embedding succeeds and returned as `image_id`.
  Nothing else identifies the pending upload.
- **The `retrieval_` prefix is applied server-side, on both sides.** Upload writes
  `format!("retrieval_{}", image_id)` (`search.rs:43`) and search reads
  `format!("retrieval_{}", &image_id)` (`search.rs:72`). The caller only ever sees the
  bare UUID.
- **The TTL is 120 seconds** (`search.rs:45`), set with `SET … EX` via
  `set_ex`. After that the key is gone and the search reports the image as not found
  (`search.rs:75`) — the caller uploads again to get a fresh id. One upload can be
  searched any number of times inside the window, with different `search_type`s and
  different filters, without re-embedding.
- **What is stored is `ImageMetadata { embedding, model_type }`** (`search.rs:14-18`),
  bincode-encoded with `config::standard()` (`search.rs:36`). Both fields derive
  `Encode`/`Decode` — `ResultEmbedding` at `processing.rs:27` and `ModelPurpose` at
  `config.rs:125-127`. Storing the purpose alongside the vector keeps the record
  self-describing; the search path reads back the embedding it needs
  (`search.rs:88`).
- **Search never touches Triton.** The only inference in the crate happens on upload.
  A search is a Redis `GET`, a bincode decode and one Elasticsearch request.
- **The uploaded image is not retained.** Only the 768-float vector goes into Redis; the
  decoded pixels are dropped when `upload_image` returns.

### Error surfaces

Both handlers convert an internal `Err` into a 400 with a fixed message and log the real
error through `tracing::error!` — `"Error uploading image"` (`post.rs:98-103`) and
`"Could not process search"` (`get.rs:77-82`). A malformed multipart body — a missing
`image` or `model_type` field, an unparseable `model_type`, or bytes that
`parse_image` cannot decode — short-circuits in `from_multipart` and yields
`"Could not process input"` (`post.rs:107-109`). The log is where the detail lives; the
response body is deliberately generic.

---

## 6. From bytes to embedding

### Decoding the upload

`utils::parse_image` (`utils.rs:10`) is the entry point. It sniffs the format with
`detect_format` (`utils.rs:20`) and then decodes with
`image::load_from_memory_with_format`, returning `(rgb8_bytes, width, height)` — exactly
the three pieces `RawFrame` needs.

`detect_format` reads magic bytes, requiring at least 8 bytes of input (`utils.rs:21-23`):

| Format | Signature | Line |
| --- | --- | --- |
| JPEG | `FF D8 FF` | `:26` |
| PNG | `89 50 4E 47 0D 0A 1A 0A` | `:31` |
| WebP | `RIFF` … `WEBP` at offset 8 | `:36` |
| GIF | `GIF87a` / `GIF89a` | `:41` |

Anything that matches none of those falls through to `image::guess_format`
(`utils.rs:46`), so the explicit checks are a fast path over the library's own detection
rather than a whitelist. `into_rgb8()` (`utils.rs:17`) drops any alpha channel and
normalises the pixel layout, so what reaches the preprocessor is always tightly packed
interleaved RGB24 — the same layout the ingest crate receives from the video library.

### Preprocess, infer, postprocess

`dino::process_frame` (`dino.rs:91`) is one image end to end, and it is the same function
the ingest crate uses for whole frames:

1. Derive `target_size` from the **last** element of the configured `input_shape`
   (`dino.rs:96-99`) — 512 for `dinov3-512`, 224 for `dinov3-224`.
2. `spawn_blocking` → `preprocess_frame` (`dino.rs:105`), which validates that
   `data.len() == height * width * 3` (`dino.rs:23-30`) and then runs the fused
   letterbox + ImageNet-normalize pass described in `docs/client-image-retrieval.md` §7.
3. `infer(vec![pre_frame])` (`dino.rs:115`) — one input, so always `infer()`'s fast path
   (`inference.rs:330`).
4. `spawn_blocking` → `postprocess` (`dino.rs:122`), then take the first (only) embedding
   out of the batch (`dino.rs:129-132`).

It returns `(FrameProcessStats, ResultEmbedding)` (`dino.rs:143`); `upload_image` feeds
the stats into the global aggregate (`search.rs:51-56`) and keeps the vector.

The vectors are not L2-normalized anywhere, which matches the index mapping's
`"similarity": "cosine"` — Elasticsearch normalizes at index and query time, so raw model
magnitudes are what should be sent, on both the write side and the query side.

### Which embedder, and why the caller picks it

There is no classification step and no detector. The caller *declares* which model to
use, in the `model_type` multipart field, and `ModelPurpose::from_str`
(`config.rs:143-153`) parses it case-insensitively:

| `model_type` | Model | What it searches for |
| --- | --- | --- |
| `FrameEmbedding` | `dinov3-512` | Whole frames similar to the uploaded image. |
| `BBOXEmbedding` | `dinov3-224` | Crops similar to the uploaded image. |

The two purposes exist because the ingest crate writes both kinds of vector into the same
index: one whole-frame embedding per processed frame, plus one embedding per detected
box. Declaring `BBOXEmbedding` asserts that the uploaded image already *is* a crop of the
kind the detector produces — a single object filling the frame — and searches it against
the crop vectors at the resolution they were embedded at. That assertion is why this crate
has no cropping path at all: `preprocess_bbox` and `process_bboxes` exist in
`client-image-retrieval`'s `dino.rs` and are simply absent here. There is no
`FrameDetection` purpose (`config.rs:128-131`), and no YOLO model in the configuration.

Because both purposes are just keys into `inference_config.models`, both models are
loaded onto Triton at startup regardless of which one a given request uses
(`inference.rs:50-61`).

---

## 7. The Elasticsearch search path

`Elastic` (`elastic.rs:12-16`) holds three things: the client, `elastic_config`, and the
`SearchType → SearchConfigOption` map handed to it from `AppConfig::search_config()`
(`services.rs:54-57`). It has exactly one method.

`search_disk_bbq` (`elastic.rs:44`) builds a **top-level `knn` query** — not a
`query.knn`, not a script score — and sends it to `SearchParts::Index(&[index_name])`
(`elastic.rs:121`):

```json
{
  "timeout": "25s",
  "size":  <output_vectors>,
  "knn": {
    "field": "embedding",
    "query_vector": [ … 768 floats … ],
    "k": <output_vectors>,
    "num_candidates": <num_candidates>,
    "visit_percentage": <centriod_visit_percentage>,
    "rescore_vector": { "oversample": <vector_oversample_multiplier> },
    "filter": {                              // present only when a filter was requested
      "bool": { "must": [
        { "terms": { "channel_id": ["1","2"] } },
        { "range": { "timestamp": { "gte": …, "lte": … } } }
      ] }
    }
  },
  "_source": { "excludes": ["embedding"] }
}
```

The tier lookup happens first (`elastic.rs:51-52`): an unknown `search_type` is a hard
error, so every one of `LIGHT`, `MEDIUM`, `HEAVY` must be present in
`search_config`. Everything after that is assembly.

**The filter is optional and additive.** `must_clauses` starts empty (`elastic.rs:58`).
A `terms` clause on `channel_id` is pushed only when `channel_ids` is `Some` *and*
non-empty (`elastic.rs:61-67`). Those ids arrive as a single comma-separated query
parameter — `?channel_ids=1,2,3` — which `parse_channel_ids` (`get.rs:30`) splits and
parses into `Vec<u32>`, skipping empty segments and rejecting anything non-numeric with a
400. A single string rather than a repeated parameter because `axum::extract::Query`
deserializes with `serde_urlencoded`, which has no sequence support. `u32` matches the
`source_id` the ingest side writes into `channel_id`; the field is mapped `keyword`, and
Elasticsearch coerces numeric terms to their string form. A `range` clause on `timestamp` is pushed when at least
one of the two bounds is present, with `gte` and `lte` filled in independently
(`elastic.rs:68-84`) — so a one-sided range is expressible. Only if at least one clause
was built does a `bool.must` wrapper get created (`elastic.rs:86-94`) and attached to the
kNN query as `filter` (`elastic.rs:107-109`); with no filters requested, the key is
absent entirely rather than present and empty. Because the filter sits *inside* the `knn`
clause, it constrains the vector search itself rather than post-filtering its output.

**`_source.excludes: ["embedding"]`** (`elastic.rs:115-117`) keeps the 768-float vector
out of every hit. Ten hits would otherwise carry 7,680 floats of JSON that the caller has
no use for; what comes back is the metadata the ingest side wrote alongside the vector.

**The response is passed through nearly raw.** `search_disk_bbq` returns
`hits.hits` as a `Vec<serde_json::Value>`, defaulting to empty when the path is missing
(`elastic.rs:131-136`), and the handler reshapes each hit into
`{"score": _score, "metadata": _source}` (`get.rs:61-66`), with `count` derived from the
result length (`get.rs:71`). No struct models the document, so whatever fields the
mapping carries reach the caller under `metadata` unchanged.

### The tuning knobs

`SearchConfigOption` (`config.rs:196-202`) has four fields, and they are the crate's main
operational surface — the same query, the same index, three latency/recall points
selected per request by `search_type`.

| Field | Query parameter | What it controls |
| --- | --- | --- |
| `output_vectors` | `k` **and** `size` | How many hits come back. Set in both places (`elastic.rs:99`, `:113`) so the kNN result set and the search result window agree. |
| `num_candidates` | `num_candidates` | The size of the candidate pool the vector search keeps per shard before the top `k` are chosen. Larger pools cost more work and recover more of the true nearest neighbours. |
| `centriod_visit_percentage` | `visit_percentage` | How much of the quantized index the search is allowed to visit — the `bbq_disk` format groups vectors into clusters, and this is the proportion of them examined. The lower bound on how much of the index a query can ignore. |
| `vector_oversample_multiplier` | `rescore_vector.oversample` | The first pass runs against the BBQ-quantized vectors; this multiplies how many of its results are then rescored against fuller-precision vectors. It buys back the accuracy quantization gives away, at the cost of the rescoring pass. |

The checked-in tiers (`secrets/config.yaml:50-65`) hold `output_vectors` at 10 and scale
the other three together:

| | `LIGHT` | `MEDIUM` | `HEAVY` |
| --- | --- | --- | --- |
| `output_vectors` | 10 | 10 | 10 |
| `num_candidates` | 50 | 200 | 1000 |
| `centriod_visit_percentage` | 2 | 10 | 50 |
| `vector_oversample_multiplier` | 1.5 | 4 | 10 |

Read across: every tier returns ten candidates, and the difference is how hard the
cluster works to make those ten the right ten. `LIGHT` looks at 2 % of the index and
oversamples 1.5×; `HEAVY` looks at half of it, keeps twenty times the candidate pool and
rescores ten times as many vectors. The `timeout: "25s"` (`elastic.rs:112`) is fixed
across all three and bounds the server-side work of the heaviest tier.

---

## 8. Relationship to `elasticsearch-setup.md` and to the ingest half

**This crate never creates the index, its mapping, or any document.** It only issues
searches against the name in `elastic_config.index_name` — `embeddings-live` in the
checked-in config. Everything that gives that name meaning is provisioned once, by hand,
with the Kibana Dev Tools commands in `elasticsearch-setup.md`:

| Provisioned by `elasticsearch-setup.md` | What it means for a search |
| --- | --- |
| ILM policy `embeddings-phasing-policy` | Rolls the write index over at 7 days / 50 GB primary shard, then moves it to `warm`, shrinks it to one shard and marks it read-only. Searches keep reaching the rolled-over indices through the alias. |
| Component template `embedding-live-object-template` | The mapping every filter and the kNN query depend on: `timestamp` as `date`/`epoch_millis`, `channel_id` as `keyword`, `embedding` as `dense_vector` with `dims: 768`, `element_type: bfloat16`, `index_options.type: bbq_disk`, `similarity: cosine`. |
| Index template `embedding-live-template` | Binds the component template and the ILM policy to the `embeddings-live-*` pattern, and adds the `embeddings_search` alias. |
| `PUT embeddings-live-000001` with `"embeddings-live": {"is_write_index": true}` | Creates the first concrete index and makes `embeddings-live` the rollover alias. |

So `index_name: embeddings-live` is an **alias**, not an index, and the search targets it
directly (`elastic.rs:121`). ILM adds each newly rolled index to that alias, so the alias
resolves to the whole chain of `embeddings-live-00000N` indices and a query covers the
history rather than only the current write index.

Three properties of the mapping are what make the query in §7 well-formed:

- **`dims: 768`, `similarity: cosine`** — the `query_vector` must be 768 floats, which is
  exactly what both `dinov3-512` and `dinov3-224` emit (`output_shape: [768]` in both
  model blocks). Because the same embedding dimension is shared by both models, a
  `BBOXEmbedding` query vector is structurally valid against frame vectors and vice
  versa; which population a query is *meant* for is the caller's declaration, not
  something the index enforces.
- **`index_options.type: bbq_disk`** — the quantized, disk-backed vector format that
  `visit_percentage` and `rescore_vector.oversample` are parameters of. Hence the method
  name `search_disk_bbq`.
- **`channel_id` as `keyword`, `timestamp` as `epoch_millis`** — the two filter clauses
  target these directly. The ingest side writes `channel_id` from the configured source
  id and `timestamp` from `Utc::now().timestamp_millis()`
  (`docs/client-image-retrieval.md` §9), so `channel_ids` on this side names video
  sources and `timestamp_start`/`timestamp_end` are epoch **milliseconds** on the same
  clock.

The write side is `client-image-retrieval/src/utils/elastic.rs`: a document queue drained
every 2 s into one `_bulk` request, one document per embedding, each carrying
`timestamp`, `channel_id` and `embedding`. Every document this crate can return was
produced there.

---

## 9. Statistics

`src/statistics.rs` keeps the reporter mechanics and the NVML code of the client crates
(`docs/client-object-detection.md` §9) and replaces the per-source counters with a single
process-wide aggregate.

`ProcessingStats` (`statistics.rs:60-68`) is a flat set of `AtomicU64`s — one success
counter and six microsecond totals — owned by `Statistics` behind an `Arc`
(`statistics.rs:110`) and handed out by `processing_stats()` (`statistics.rs:295`).
`accumulate` (`statistics.rs:93`) bumps `frames_success` and adds one
`FrameProcessStats` field by field, with `Ordering::Relaxed` throughout: the numbers are
telemetry, and no other state depends on their ordering.

The reporter (`statistics.rs:149-160`) ticks every `PROCESSING_STATS_INTERVAL` (1 s,
`:10`), logs one `processing statistics` event, and **resets the aggregate to zero**
(`:158`). Every number in that line is therefore a one-second window. Averages divide
each total by `frames_success` and stay at `0.00` when nothing succeeded
(`statistics.rs:272-279`).

| Field | Meaning |
| --- | --- |
| `frames_success` | Uploads whose embedding completed, in the last second. |
| `avg_pre_proc` | The `spawn_blocking` decode-and-normalize call (`dino.rs:102-111`). |
| `avg_inference` | `infer()` end to end — gRPC round trip plus Triton's own queueing and execution (`dino.rs:114-118`). |
| `avg_post_proc` | The FP16→FP32 conversion of the output tensor (`dino.rs:121-134`). |
| `avg_processing` | `dino::process_frame` wall time, stamped at `dino.rs:141`. |
| `avg_queue` | Present in the aggregate; nothing on this path enqueues, so it stays 0. |
| `avg_search` | Present in the aggregate; the Elasticsearch path records no timings, so it stays 0. |

The one accumulation site is `upload_image` (`search.rs:51-56`), so the line describes
embedding work: how many images were embedded in the last second and where the time went
inside each. `search_image` performs no inference and contributes nothing.

GPU telemetry is unchanged: NVML is initialised eagerly in `Statistics::new` and treated
as fatal on a GPU deployment (`statistics.rs:120-131`), and `start()` moves the handle
into a `spawn_blocking` thread that reports GPU 0's name, UUID, serial, memory and
utilization once a second (`statistics.rs:166-185`). On CPU the thread is never spawned
and a single `warn!` is logged at construction.

`Statistics` has a `Drop` impl that clears `is_running` and aborts both tasks
(`statistics.rs:300-311`).

---

## 10. Configuration reference — `secrets/config.yaml`

Deserialized into `AppConfig` (`config.rs:212-225`) by `serde_yaml`, from a path
resolved relative to the process working directory (`config.rs:249`). Unknown keys are
ignored; missing keys without a `#[serde(default)]` are a hard parse error.

### Top level

| Key | Type | Effect |
| --- | --- | --- |
| `local` | `bool` | When true, adds the `logs/app.log` file layer. Stdout logging is unconditional. |
| `port` | `u16` | The TCP port bound on `127.0.0.1` (`main.rs:32-35`). |
| `elastic_config` | map | Cluster URL and the index/alias to search. |
| `triton_config.url` | string | Triton gRPC endpoint. |
| `redis_config` | map | Where the pending embeddings live. |
| `inference_config` | map | Device, instance counts, and the two embedders. |
| `search_config` | `map<SearchType, SearchConfigOption>` | The three tiers of §7. All three keys are needed for all three `search_type` values to work. |

`hardware_name` is a field of `AppConfig` but is `#[serde(skip)]` (`config.rs:223`) —
resolved at startup from `DeviceType::hardware_name()`, never read from the file.

### `elastic_config` and `redis_config`

`ElasticConfig` (`config.rs:55-59`):

| Key | Effect |
| --- | --- |
| `url` | Elasticsearch HTTP endpoint handed to `Transport::single_node` (`elastic.rs:31`). Plain HTTP; no credentials, TLS settings or timeouts are configurable here. |
| `index_name` | The search target (`elastic.rs:121`) — the `embeddings-live` alias (§8). |

`RedisConfig` (`config.rs:204-209`) has three mandatory fields, assembled into one
connection URL at `redis.rs:17-22`:

```
redis://<username>:<password>@<url>
```

so `url` is a `host:port` pair (`localhost:6379` in the checked-in config), not a full
URL. The checked-in credentials — `default` / `redis_admin` — match the
`--requirepass redis_admin` in `services/docker-compose-redis.yml`, whose healthcheck
authenticates with the same password.

`Redis` (`redis.rs:9-12`) holds one `MultiplexedConnection`, built once in `Redis::new`
(`redis.rs:27`). `connection()` (`redis.rs:40-42`) hands out a clone, which is how the
`redis` crate shares a multiplexed connection: clones are cheap handles onto the same
socket, with commands pipelined and responses demultiplexed by the connection's driver
task. Both `set_ex` and `get` (`search.rs:42`, `:71`) go through such a clone.

### `inference_config`

| Key | Type | Effect |
| --- | --- | --- |
| `instances.default` | `u32` | Instance count when the hardware, or the purpose under that hardware, is not listed. |
| `instances.custom` | `map<String, map<ModelPurpose, u32>>` | Keyed first by `hardware_name` — the literal `"CPU"`, or the NVML device name — then by `ModelPurpose`. Resolved by `InstancesConfig::resolve` (`config.rs:79`), falling back to `default` when either level is absent. This two-level map is shared verbatim with the two client crates. |
| `models` | `map<ModelPurpose, ModelConfig>` | Two purposes here, both embedders. |

`ModelConfig` (`config.rs:36-48`) — all fields mandatory:

| Field | Notes |
| --- | --- |
| `name` | Triton model name; must match the directory in `triton_models/`. |
| `model_type` | `DINOV3` is the only variant (`config.rs:107`). |
| `precision` | Optional. `FP32` or `FP16`; omit to take the device default — FP32 under `DEVICE_TYPE=CPU`, FP16 under `GPU`. Becomes `TYPE_FP32`/`TYPE_FP16` in the generated Triton config, the request datatype, the preprocessing output format, and the postprocessing element width — so an explicit value must match the precision the artifact was built at. |
| `input_name` / `output_name` | Tensor names as compiled into the model. |
| `input_shape` | Without the batch dimension. The **last** element is the letterbox target size (`dino.rs:96-99`). |
| `output_shape` | Product × precision size gives the per-sample output byte count (`inference.rs:317-323`). |
| `batch_max_size` | `max_batch_size` in the generated config, and the fast-path threshold in `infer()`. |
| `batch_max_queue_delay` | `dynamic_batching.max_queue_delay_microseconds`. |
| `batch_preferred_sizes` | `dynamic_batching.preferred_batch_size`. |

### The checked-in configuration

`local: true`, `port: 8080`. Elastic at `http://localhost:9200`, index `embeddings-live`.
Triton at `http://localhost:8001` — the **gRPC** port (8000 is HTTP, 8002 metrics). Redis
at `localhost:6379`. One instance of each model. The device comes from `DEVICE_TYPE`,
not from this file.

| | `FrameEmbedding` | `BBOXEmbedding` |
| --- | --- | --- |
| `name` | `dinov3-512` | `dinov3-224` |
| `model_type` | `DINOV3` | `DINOV3` |
| `precision` | *(device default)* | *(device default)* |
| `input_name` / `output_name` | `images` / `output` | `images` / `output` |
| `input_shape` | `[3, 512, 512]` | `[3, 224, 224]` |
| `output_shape` | `[768]` | `[768]` |
| `batch_max_size` | 16 | 16 |
| `batch_max_queue_delay` | 1000 µs | 1000 µs |
| `batch_preferred_sizes` | `[4, 8, 16]` | `[4, 8, 16]` |

These are the same two model blocks the ingest crate uses, which is what makes the query
vectors comparable to the indexed ones: same architecture, same precision, same input
resolution, same 768-dimensional output. Because every request embeds exactly one image,
`infer()` always takes its single-batch fast path and the dynamic-batching settings apply
to whatever Triton coalesces across concurrent requests.

---

## 11. Building and running

### Prerequisites

1. **The two models in the Triton model repository.** Under `DEVICE_TYPE=CPU` the client
   uploads a config declaring `onnxruntime_onnx` and
   `default_model_filename: model.onnx`, so it needs
   `triton_models/{dinov3-512,dinov3-224}/1/model.onnx`. On GPU it needs `model.plan` in
   the same places; `triton_models/` currently holds `dinov3-224/1/model.plan` and
   `dinov3-512/1/model.plan`, so a GPU run is ready and a CPU run needs ONNX exports.
   `model-optimization/` is where those exports are produced. No `config.pbtxt` is
   required — the client generates and uploads the model configuration itself
   (`docs/client-object-detection.md` §6).
2. **The Elasticsearch index templates applied.** Run the four Kibana Dev Tools commands
   in `elasticsearch-setup.md` — ILM policy, component template, index template, and the
   initial `embeddings-live-000001` index with the `embeddings-live` write alias (§8).
   Kibana comes up with the Elastic stack on port 5601.
3. **Vectors in the index.** Searches return what `client-image-retrieval` wrote, so that
   crate needs to have run against the same cluster for a search to find anything.
4. **Working directory is the crate root**, `image-retrieval-search`. Both
   `secrets/config.yaml` (`config.rs:249`) and `logs/app.log` (`config.rs:264`) resolve
   relative to the process CWD, which is why no task here sets `runFromWorkspaceRoot`.
5. **`DEVICE_TYPE` is set, and matches the Triton profile you boot.** The client is what
   tells Triton which platform to load, so pointing `DEVICE_TYPE=GPU` at the CPU profile
   asks an unaccelerated Triton for a `tensorrt_plan`. The moon tasks set both together,
   so this only needs thought when running by hand. The variable has no default: unset,
   the process exits at config load naming it.

### Build

```bash
cd image-retrieval-search
cargo build --release
```

Beyond the shared dependency set, `Cargo.toml` adds the HTTP and storage stack: `axum`
0.8.4 with the `multipart` and `macros` features, `tower-http` 0.6 with `cors` and
`trace`, `utoipa` 5.4 with `axum_extras` plus `utoipa-swagger-ui` 9.0, `redis` 1.0 with
`tokio-comp`, `bincode` 2.0, `uuid` 1.21 with `v4`, and `image` 0.25 for decoding.
`elasticsearch = "9.1.0-alpha.1"` matches the 9.x cluster in
`services/docker-compose-elastic.yml`.

### Run

```bash
moon run retrieval-search:cpu     # or
moon run retrieval-search:gpu
```

Each task depends on the matching `services:triton-*` task plus `services:elastic` and
`services:redis` (`moon.yml:53-56`, `:66-69`), so the whole backing stack is started for
you. All three service tasks run `docker compose up --detach --wait`, which blocks on the
compose healthchecks and exits only once the services report ready — Triton serving, the
Elasticsearch cluster at `wait_for_status=yellow`, and Redis answering an authenticated
`PING`. By the time `cargo run` starts, `server_ready()` and the Redis connection are
guaranteed to succeed, and the cluster is answering queries.

The gating depends on those service tasks being **non-persistent**: moon starts
`persistent` tasks concurrently and never waits for one to finish, so a persistent task
would not gate anything. Confirm the shape with
`moon action-graph retrieval-search:cpu --dot` — it must show `RunTask(services:triton-cpu)`,
`RunTask(services:elastic)` and `RunTask(services:redis)` feeding
`RunPersistentTask(retrieval-search:cpu)`.

Because all three stacks run detached, nothing streams their output and they outlive the
client:

```bash
moon run services:triton-logs
moon run services:elastic-logs
moon run services:redis-logs
moon run services:triton-down
moon run services:elastic-down
moon run services:redis-down
```

### Environment

| Variable | Required | Effect |
| --- | --- | --- |
| `DEVICE_TYPE` | **yes** | `CPU` or `GPU`, case-insensitive. Selects the Triton platform, model filename, instance kind, the `optimization` block, whether NVML runs, and each model's default precision. No default — unset, the process exits at config load naming the variable. Both moon tasks set it to match the Triton they boot. |
| `RUST_LOG` | in practice | Nothing is logged when unset. Both moon tasks set `INFO`; a value exported in your shell takes precedence. |

At startup the client logs one line per model reporting the resolved device, hardware
name, precision, and whether that precision came from the config or the device default.

`image-retrieval-search/run_local.sh` predates both the moon tasks and the compose-file
split — it drives `../docker-compose.yml` with a `client-search` profile, which has since
become the three per-service files under `services/`. The moon tasks are the current
launch path.

### Exercising it

With the service up on the configured port, `/docs` is the quickest way in. The
equivalent by hand:

```bash
# 1. upload — returns {"message": …, "data": {"image_id": "<uuid>"}}
curl -s -X POST http://127.0.0.1:8080/upload \
  -F "image=@query.jpg" \
  -F "model_type=FrameEmbedding"

# 2. search, within 120 s of the upload
curl -s "http://127.0.0.1:8080/search?image_id=<uuid>&search_type=MEDIUM"
```

`search_type` is bound by serde from the query string, so it is matched against the
variant names exactly: `LIGHT`, `MEDIUM`, `HEAVY`.

### What success looks like

On a healthy start you should see, in order: two `Initiated model instances` /
`Successfully initiated model` pairs from `InferenceModels::start`, then
`Server running on http://127.0.0.1:8080` (`main.rs:40`), then a `processing statistics`
line once a second from the reporter — all zeros until the first upload arrives, and
reporting the embedding timings of the last second's uploads after that.
