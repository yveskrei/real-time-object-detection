# `client-image-retrieval` — architecture and reference

Real-time image-retrieval **ingest** client. It receives decoded video frames from the
same proprietary C library (`libclient_video.so`) as `client-object-detection`, runs
three Triton-hosted models over gRPC — one detector and two DINOv3 embedders — and writes
the resulting feature vectors into Elasticsearch, where the query half of the system
searches them.

This is the write side of image retrieval, extracted from the older
`image-retrieval-search/` prototype and grafted onto the FFI / Triton / queue skeleton of
`client-object-detection`. The query half still lives, unconverted, in
`image-retrieval-search/`.

All paths in this document are relative to the repository root unless stated otherwise.
Cargo package name is `client` (`client-image-retrieval/Cargo.toml:2`), so the binary
produced is `client-image-retrieval/target/{debug,release}/client`. Rust
**edition 2024**.

---

## 1. Relationship to `client-object-detection`

The two crates share a skeleton. These files are **byte-identical** between them, and
this document does not re-describe them — read the referenced section of
`docs/client-object-detection.md` instead:

| File | Covered by |
| --- | --- |
| `.cargo/config.toml` | `docs/client-object-detection.md` §11 (the `CMAKE_POLICY_VERSION_MINIMUM` build note) |
| `src/main.rs` | §3 |
| `src/lib.rs` | §2 |
| `src/inference.rs` | §6 — Triton client, runtime config generation, load/unload, `infer()` |
| `src/statistics.rs` | §9 |
| `src/client_video.rs` | §5 — the FFI boundary |
| `src/client_video/utils.rs` | §5 — `RawFrame`, `Detection`, `ResultBBOX`, `Ownership` |

`src/processing/yolo.rs` is the detection crate's file plus one line: a catch-all
`_ => anyhow::bail!("Invalid model type")` arm (`yolo.rs:500`) in the decoder dispatch,
needed now that `ModelType` has a third variant. `src/utils/queue.rs` gains a single
method, `drain()` (`queue.rs:92-95`), used only by the Elasticsearch flusher.

Everything genuinely new is in five files: `src/source.rs`, `src/processing.rs`,
`src/processing/dino.rs`, `src/utils/elastic.rs` and `src/utils/config.rs`.

### The three differences worth knowing before reading any code

**(a) Processing is serial per source, latest frame wins.** `MAX_QUEUE_FRAMES = 1`
(`source.rs:19`), there is no semaphore, and the consumer loop `await`s `process_frame`
inline rather than spawning it (`source.rs:115-123`). One frame is in flight per source
at a time and the queue holds exactly one more; when a new frame arrives while one is
pending, the pending one is evicted by the queue's overflow policy. The detection crate
runs fifteen frames concurrently per source; this one runs one, and always the freshest
available. That is the right shape for a three-model pipeline whose output is an index
entry rather than a live overlay: pipelining would multiply GPU residency for vectors
that only need to be periodically representative.

**(b) YOLO detections are a cropping instruction, not an output.** Nothing is ever sent
back to the video library. `ClientVideo::populate_bboxes` (`client_video.rs:262`) and
`_post_results_callback` (`:418`) still exist, and the callback is still registered at
`SetCallbacks` time (`:166`), but nothing in this crate calls `populate_bboxes`. The FFI
is **input-only** here: frames flow in, nothing flows out. The detector exists solely so
`dino::process_bboxes` knows which regions of the frame to crop and embed.

**(c) The sink is Elasticsearch, not the video library.** A new `Elastic` service
(`src/utils/elastic.rs`) buffers documents and flushes them on a timer with the `_bulk`
API. See §9.

---

## 2. Module map

| File | Responsibility |
| --- | --- |
| `src/main.rs` | Four-line entry point: config → services → start → sleep forever. |
| `src/lib.rs` | Module declarations only. |
| `src/services.rs` | The `Services` singleton. Same shape as the detection crate plus an `Elastic` member. |
| `src/utils/config.rs` | `secrets/config.yaml` deserialization, per-source override + clamping, hardware-name resolution, tracing setup. Adds `ElasticConfig`, `ModelType::DINOV3` and two new `ModelPurpose` variants. |
| `src/utils/elastic.rs` | **New.** Bulk write path into Elasticsearch: document queue plus a background flusher. |
| `src/utils/queue.rs` | `FixedSizeQueue<T>` — bounded queue that drops the oldest item on overflow. Adds `drain()`. |
| `src/utils.rs` | `get_image_raw` helper; module declarations (now including `elastic`). |
| `src/client_video.rs` | The FFI boundary: `dlopen`, exported-function wrappers, the four host callbacks. |
| `src/client_video/utils.rs` | Types that cross the boundary and the `Ownership`-aware C-pointer readers. |
| `src/source.rs` | Per-source queueing, frame skipping, and the three-model per-frame orchestration. |
| `src/inference.rs` | Triton gRPC client, runtime model-config generation, load/unload, batching. |
| `src/processing.rs` | Fused preprocessing (resize + letterbox + normalize), plain and ImageNet-normalized, and the conversion lookup tables. Defines `ResultEmbedding`. |
| `src/processing/dino.rs` | **New.** DINOv3 pre/post-processing, bbox cropping, and the per-frame / per-batch pipelines. |
| `src/processing/yolo.rs` | The two YOLO output decoders, NMS, and the per-frame detect pipeline. |
| `src/statistics.rs` | Per-source counters and their 1 s reporter; NVML GPU telemetry. |

---

## 3. Startup chain

```
main()                                        src/main.rs:8
 └─ AppConfig::new()                          src/utils/config.rs:205
      ├─ load_config_file()  reads ./secrets/config.yaml      :250
      ├─ init_logging(local)                                  :266
      ├─ per-source override + clamping                        :213-237
      └─ device_type.hardware_name()  (NVML when GPU)          :169
 └─ services::init_services(&cfg, Handle::current())          src/services.rs:25
      ├─ Services::new(...)                                    :53
      │    ├─ InferenceModels::new   → one InferenceModel per configured ModelPurpose,
      │    │                            each checking server_ready()      inference.rs:28
      │    ├─ SourceProcessors::new  → one SourceProcessor per configured id
      │    │                            (each spawns its consumer task immediately)
      │    ├─ Elastic::new           → transport + client, spawns the flusher  :64
      │    ├─ Statistics::new        → NVML init when device_type == GPU       :68
      │    └─ ClientVideo::new       → dlopen secrets/libclient_video.so       :72
      └─ SERVICES.set(Arc::new(services))                      :36
 └─ get_services()?.start(&cfg)                               src/services.rs:84
      ├─ Statistics::start           → source-stats task + GPU-stats blocking thread
      ├─ InferenceModels::start      → for each model: unload previous, load with
      │                                 generated config              inference.rs:43
      ├─ ClientVideo::init_state(RunMode::Regular)  → SetCallbacks, then SetSettings
      └─ ClientVideo::init_sources   → InitSources  ← frames start flowing here
 └─ sleep(Duration::from_secs(u64::MAX))                      src/main.rs:23
```

Two things differ from the detection crate's chain, both in `Services::new`.

`InferenceModels::new` iterates `inference_config().models`, which is now a three-entry
map rather than a one-entry map (`inference.rs:31-38`). Each entry gets its own
`InferenceModel` with its own Triton client and its own prebuilt base request, and
`InferenceModels::start` (`:43`) loads all three in sequence, logging one
`Successfully initiated model` line per `ModelPurpose`. Load order follows `HashMap`
iteration order and is not significant — the models are independent as far as Triton is
concerned.

**`Elastic` has no `start()` phase.** `Elastic::new` (`elastic.rs:23`) builds the
transport and client, then immediately moves the client into a detached
`tokio::spawn`ed flusher (`:37-52`) and returns a struct holding only the queue
(`:17-19`). The flusher is therefore ticking before `Services::start` runs. Nothing is
in the queue yet, so the early ticks are no-ops, but it does mean the write path is live
from construction rather than from start.

`SERVICES.set` happening *before* `start()` (`services.rs:36-38`) is load-bearing for
the same reason as in the detection crate: every spawned task and every FFI callback
re-enters through `services::get_services()`.

Shutdown is `SIGINT` with the default handler (`main.rs:22`); no handler is installed.

### Logging

Unchanged from `docs/client-object-detection.md` §3: `EnvFilter::from_default_env()`
(so export `RUST_LOG`), an always-on JSON stdout layer, and a `logs/app.log` file layer
when `local: true` (`config.rs:266-295`).

---

## 4. The frame path

```
 libclient_video.so decoder thread
   │  (borrowed *const u8, width*height*3 bytes, RGB24, tightly packed)
   ▼
 ClientVideo::_source_frames_callback              src/client_video.rs:335
   │  copy immediately, build RawFrame { source_id, data, width, height, pts }
   ▼
 services.runtime().spawn(...)                     src/client_video.rs:370
   ▼
 SourceProcessor::add_to_queue                     src/source.rs:182
   │  if (frames_total + 1) % inf_frame != 0  ──►  frames_total += 1, frame dropped
   │  else stamp `added` and enqueue                            :186-194
   ▼
 FixedSizeQueue<QueuedFrame>  capacity 1           src/source.rs:19, queue.rs:55
   │  on overflow: pop_front() and invoke the drop callback,
   │  which increments frames_failed                            src/source.rs:98-100
   ▼
 per-source consumer loop                          src/source.rs:111-168
   │  recv().await, then process_frame(...).await  — inline, not spawned
   ▼
 SourceProcessor::process_frame                    src/source.rs:204
```

and inside `process_frame`, the three-model fan-out:

```
                    Arc<RawFrame>  (one copy, made at the FFI boundary)
                            │
        ┌───────────────────┴───────────────────┐
        │        futures::try_join!             │        source.rs:228-237
        ▼                                       ▼
 yolo::process_frame                     dino::process_frame
   ModelPurpose::FrameDetection            ModelPurpose::FrameEmbedding
   yolo26x   FP32   in [3,640,640]         dinov3-512  FP16  in [3,512,512]
                    out [300,6]                              out [768]
   whole frame, letterboxed                whole frame, letterboxed + ImageNet
        │                                       │
        │ Vec<ResultBBOX>                       │ ResultEmbedding (768 f32)
        ▼                                       │
 dino::process_bboxes            (only if bboxes is non-empty, source.rs:248)
   ModelPurpose::BBOXEmbedding                  │
   dinov3-224  FP16  in [3,224,224] out [768]   │
   one crop per bbox, expanded 1.5x,            │
   all crops in a single infer() call           │
        │                                       │
        │ Vec<ResultEmbedding>                  │
        └───────────────────┬───────────────────┘
                            ▼
        embeddings = [ frame_embedding, bbox_0 … bbox_{n-1} ]    source.rs:244-262
                            ▼
        SourceProcessor::populate_embeddings(source_id, &embeddings)
                            ▼
        Elastic queue ──(2 s timer)──► one _bulk request ──► index `embeddings-live`
```

Notes on the shape of this pipeline:

- **The detector and the frame embedder run concurrently** (`source.rs:229`). They read
  the same `Arc<RawFrame>` and neither depends on the other's result, so `try_join!`
  overlaps two Triton round trips. `try_join!` is short-circuiting: if either branch
  errors the frame is abandoned.
- **The crop embedder is strictly downstream**, because it needs the boxes. It runs only
  when the detector returned at least one (`source.rs:248`), so a frame with nothing in
  it costs two inferences instead of three.
- **The frame embedding always leads the result set** (`source.rs:245`), with crop
  embeddings appended after it (`:262`).
- **One `RawFrame` copy per frame, made at the FFI boundary.** Three `Arc::clone`s
  (`source.rs:225-226`, `:252`) hand the same buffer to all three stages; the crop path
  copies only the pixels inside each expanded box.
- **Frame skipping is counted, not queued.** `add_to_queue` only enqueues when
  `(frames_total + 1) % inf_frame == 0` (`source.rs:186`); skipped frames just bump
  `frames_total` (`:197-199`). With the checked-in `inf_frame: 25`, one frame in 25
  reaches the pipeline.
- **Statistics are accumulated across all three branches** (`source.rs:240-241`, `:263`)
  into one `FrameProcessStats`, so the per-source log line reports the pipeline as a
  whole rather than per model.

---

## 5. FFI contract with `libclient_video.so`

The full contract — symbol table, callback table, `Ownership` discipline, the
`SetCallbacks` → `SetSettings` → `InitSources` ordering, and the `PostResults` JSON body
— is documented in `docs/client-object-detection.md` §5 and applies verbatim; the file is
byte-identical.

What matters here is which half of it is exercised. This crate uses the library as a
**frame source only**:

| Direction | Used | Notes |
| --- | --- | --- |
| `SetCallbacks`, `SetSettings`, `InitSources` | yes | `Services::start` (`services.rs:99-108`) boots the library exactly as the detection crate does. |
| `_source_frames_callback`, `_source_metadata_callback`, `_source_status_callback` | yes | Frames, stream metadata and status transitions arrive as before. |
| `PostResults` / `populate_bboxes` / `_post_results_callback` | no | The callback is registered at `client_video.rs:166` so the library's callback table is complete, but the crate never calls `PostResults`. Detections leave via Elasticsearch, not via the library. |

`ClientVideo` is still constructed unconditionally in `Services::new`
(`services.rs:72`), and `ClientVideo::new` (`client_video.rs:110`) `dlopen`s
`secrets/libclient_video.so` relative to the process working directory — so the library
must be present for the crate to start, and the process must run from the crate root.
See §12.

---

## 6. Inference against Triton

The mechanism is unchanged from `docs/client-object-detection.md` §6 and is worth reading there
in full: there is no `config.pbtxt` in the model repository, the client builds the
equivalent JSON at runtime in `build_model_config` (`inference.rs:193-270`) and uploads
it as the `config` parameter of `RepositoryModelLoadRequest` (`load_model`, `:273-297`),
which is why `--model-control-mode=explicit` in `services/docker-compose-triton.yml` is
load-bearing. `DeviceType` still selects `tensorrt_plan`/`model.plan`/`KIND_GPU` versus
`onnxruntime_onnx`/`model.onnx`/`KIND_CPU`, and the GPU-only `optimization` block.

What this crate changes is the arity. There are now three `InferenceModel`s, one per
`ModelPurpose`, each with its own client, its own generated Triton config and its own
precision. Notably `device_type` is a single global setting
(`inference_config.device_type`) applied to all three: all models load on the same
platform.

### Which `infer()` path each model takes

`infer(Vec<Vec<u8>>) -> Vec<Vec<u8>>` (`inference.rs:301`) computes the per-sample output
size as `product(output_shape) × precision_size` (`:305-314`) and then chooses between a
fast path and a chunked path at `:321`.

| Caller | Inputs per call | Path |
| --- | --- | --- |
| `yolo::process_frame` (`yolo.rs:454`) | always 1 | fast path — one `model_infer`, output sliced inline |
| `dino::process_frame` (`dino.rs:198`) | always 1 | fast path |
| `dino::process_bboxes` (`dino.rs:271`) | one per detected box | fast path while ≤ `batch_max_size` (16); chunked above it |

`process_bboxes` is the first caller in either crate that can exercise the **chunked**
path: it hands `infer()` every crop of the frame in a single call and lets `infer()`
decide how to split it. Above 16 boxes it chunks by `batch_max_size`, spawns one task per
chunk, splits each output on `spawn_blocking`, and places results into pre-allocated
slots by index so no sort is needed (`:358-431`).

With `output_shape: [768]` and FP16, `output_size_per_sample` is 1,536 bytes for both
DINO models. For `yolo26x` — `[300, 6]`, FP32 — it is 7,200.

Requests go to `triton_config.url`, `http://localhost:8001` in the checked-in config —
Triton's **gRPC** port (8000 is HTTP, 8002 metrics).

---

## 7. Preprocessing

`src/processing.rs` now carries two fused preprocessors that share their geometry and
differ only in normalization:

| Function | Normalization | Used by |
| --- | --- | --- |
| `resize_letterbox_and_normalize` (`:323`) | `x / 255` | `yolo::preprocess_frame` (`yolo.rs:38`) |
| `resize_letterbox_and_normalize_imagenet` (`:442`) | `(x/255 − mean) / std`, per channel | `dino::preprocess_frame` (`dino.rs:33`), `dino::preprocess_bbox` (`dino.rs:158`) |

Both call the same `calculate_letterbox` (`:295`), which scales by the **longer** side —
`scale = target / max(in_h, in_w)` — and centres the result so padding lands symmetrically
on the two shorter edges. Both resample with **nearest neighbour** off a precomputed
`x_offsets` table (`:461-464`), both write **planar** `[R… | G… | B…]` output through
three `from_raw_parts_mut` views of one buffer allocated exactly once at the final
precision (`:455-458`, `:481-487`), and both pre-fill the whole buffer with the
*already-normalized* pad colour before writing real pixels, so padded regions are never
touched twice.

Three things are specific to the ImageNet variant:

1. **The padding value is per channel.** `PAD_GRAY_COLOR = 114` (`:16`) is looked up in
   each channel's LUT separately (`:476-478` FP16, `:592-594` FP32), so the three planes
   are filled with three different constants rather than one shared grey.
2. **Normalization is a per-channel LUT lookup**, not arithmetic. `IMAGENET_MEAN` and
   `IMAGENET_STD` (`:14-15`) are the standard ImageNet statistics, and six 256-entry
   tables — R/G/B × FP32/FP16 (`:30-37`, built at `:194-257`) — collapse the whole
   `(x/255 − mean) / std` chain into one indexed load per component.
3. **The inner loop is unrolled four pixels at a time** (`:516-566` FP16, `:632-682`
   FP32) with `get_unchecked` writes and hoisted `pad_x`/`pad_y`/`inv_scale`, with a
   scalar tail for the remainder.

### The lookup tables

| Table | Size | Direction | Built by | Used by |
| --- | --- | --- | --- | --- |
| `F16_TO_F32_LUT` (`:19`) | 65,536 × `f32` | FP16 bits → `f32` | `:58` | YOLO decoders, `dino::postprocess` |
| `F16_LUT` (`:25`) | 256 × `u16` | `u8` → `x/255` as FP16 bits | `:151` | plain preprocessing |
| `F32_LUT` (`:27`) | 256 × `f32` | `u8` → `x/255` | `:181` | plain preprocessing |
| `IMAGENET_{R,G,B}_F32_LUT` (`:30-32`) | 256 × `f32` each | `u8` → ImageNet-normalized `f32` | `:194-224` | ImageNet preprocessing, FP32 |
| `IMAGENET_{R,G,B}_F16_LUT` (`:35-37`) | 256 × `u16` each | `u8` → ImageNet-normalized FP16 bits | `:227-257` | ImageNet preprocessing, FP16 |
| `F32_TO_F16_LUT` (`:22`) | 32,768 × `u16` | `f32` in [−4, 4] → FP16 bits | `:105` | building the three f16 ImageNet tables |

All are `OnceLock`s, built lazily on first use. `F32_TO_F16_LUT` is the one that is not a
direct-indexed exact table: `get_f32_to_f16_lut` (`:138-148`) clamps its argument to
[−4, 4] and quantizes it into 32,768 buckets. It has exactly one consumer — the three
`create_imagenet_*_f16_lut` functions, each of which calls it 256 times at
initialization — so it converts 768 values once at startup and is never touched on the
frame path. ImageNet-normalized `u8` inputs land in roughly [−2.2, 2.7], inside the
table's range.

### Cropping for the box embedder

`dino::preprocess_bbox` (`dino.rs:89`) turns one `ResultBBOX` into one model input:

1. **Expand the box 1.5× about its centre** (`:101-113`). Computing the centre and
   re-deriving the corners from `1.5 × width` / `1.5 × height` keeps the expansion
   symmetric. The extra context around the object is what the embedder sees.
2. **Clamp to the frame** (`:115-118`): `x1`/`y1` floor at 0, `x2`/`y2` cap at the frame
   dimensions, then cast to `u32`.
3. **Row-wise copy** (`:135-144`): for each row in `[y1, y2)`, one `extend_from_slice` of
   the `[x1*3, x2*3)` byte range out of the frame buffer, into a `Vec` pre-sized to
   `bbox_width * bbox_height * 3`. The crop is contiguous RGB24 with stride
   `bbox_width * 3`, which is exactly what the preprocessor expects.
4. **Letterbox + ImageNet-normalize** the crop to the model's square input (`:158-166`).
   Aspect ratio is preserved and the remainder padded, so a tall thin box is not
   stretched.

Both DINO entry points derive `target_size` from the **last** element of the configured
`input_shape` (`dino.rs:177-182`, `:234-239`): 512 for `dinov3-512`, 224 for
`dinov3-224`.

---

## 8. From tensor to embedding

`dino::postprocess` (`dino.rs:47`) converts a batch of raw output buffers into
`ResultEmbedding`s. There is no decoding to do — a DINOv3 output *is* the feature vector
— so the whole job is a width conversion:

- **FP16** (`:60-69`): element count is `len / 2`, and each `u16` goes through
  `F16_TO_F32_LUT` into a `Vec<f32>` sized up front.
- **FP32** (`:70-76`): element count is `len / 4`, and the buffer is reinterpreted in
  place as `f32` rather than converted.

`ResultEmbedding` (`processing.rs:41`) is `{ data: Vec<f32> }`, `Clone` and `Serialize`.
That `Serialize` is what lets the Elastic layer drop the vector straight into a
`serde_json` document (`elastic.rs:69`); the embedding is stored as a JSON array of
numbers.

The vectors are **not L2-normalized** anywhere in this crate. That is consistent with the
index mapping in `elasticsearch-setup.md`, which declares `"similarity": "cosine"` —
Elasticsearch normalizes at index and query time for cosine similarity, so raw model
magnitudes are what should be sent.

`dino::process_frame` (`:172`) wraps preprocess → infer → postprocess for a single frame
and takes the first (only) embedding out of the batch (`:210-213`).
`dino::process_bboxes` (`:228`) does the same for `n` crops, with one
`spawn_blocking` per crop for the preprocessing stage joined by `try_join_all`
(`:245-259`), a single `infer()` call for all of them (`:271`), and one `spawn_blocking`
for the whole postprocess batch (`:278`). Both return a `FrameProcessStats` alongside
their result.

---

## 9. The Elasticsearch write path

`src/utils/elastic.rs` is a queue with a timer on it. `Elastic` holds one field — the
document queue (`:17-19`) — because the `Elasticsearch` client itself is moved into the
flusher task at construction.

```
process_frame                       elastic.rs
   │
   ├─ populate_embeddings(source_id, ts, &[ResultEmbedding])          :58
   │     └─ one json!{timestamp, channel_id, embedding} per vector    :66-70
   │        pushed into FixedSizeQueue<Value>, capacity 1000          :73, :14
   │
   └─ (returns immediately — the frame path never waits on Elastic)

background flusher task, spawned in Elastic::new                      :37-52
   │  every 2 s (FLUSH_INTERVAL, :15)
   ├─ queue.receiver.drain()  → takes everything currently queued      :44, queue.rs:92
   └─ if non-empty: send_bulk(...)                                     :47
         ├─ interleave  {"index": {"_index": <index_name>}}  +  doc    :90-92
         ├─ POST via BulkParts::None (index named per action line)      :96
         └─ inspect response["errors"], log outcome                     :103-112
```

Design points:

- **Flushing is purely time-based.** There is no size trigger; the flusher drains
  whatever accumulated in the last 2 s and sends it as one `_bulk` request. Frame rate
  and `inf_frame` therefore set the batch size, not the other way round.
- **The write is fully decoupled from the frame path.** `populate_embeddings` only
  enqueues, so `process_frame` never blocks on Elasticsearch. The time it measures as
  `results` (`source.rs:268-273`) is enqueue time, not round-trip time.
- **`drain()` exists for this.** The frame queue uses `recv()` (one item, waiting on a
  `Notify`); the document queue uses `drain()` (everything at once, no waiting), which is
  the one method `queue.rs` adds over the detection crate's copy.
- **`BulkParts::None`** means no index is in the URL path; each action line names the
  index explicitly from `config.index_name`, so retargeting is pure configuration.
- **The transport is lazy.** `Transport::single_node` (`:24-25`) constructs a client
  without contacting the cluster, so nothing in `Services::new` proves Elasticsearch is
  reachable — the first `_bulk` request is the first connection attempt. This is why the
  moon task wiring gates the client behind the Elastic compose healthcheck (§12).

### The document

```json
{
  "timestamp": 1765432198765,
  "channel_id": 1,
  "embedding": [0.0123, -0.4567, ...]
}
```

| Field | Source | Notes |
| --- | --- | --- |
| `timestamp` | `Utc::now().timestamp_millis()` at `source.rs:292` | Epoch milliseconds, stamped when the completed frame is handed to the Elastic layer. |
| `channel_id` | `frame.source_id`, i.e. the configured source id | Mapped as `keyword` in the index template. |
| `embedding` | `ResultEmbedding::data` | 768 `f32` values. |

Every embedding from a frame — the whole-frame vector and each crop vector — becomes one
document, all carrying the same `timestamp` and `channel_id`.

### Relationship to `elasticsearch-setup.md`

**The client never creates the index or its mapping.** It only ever issues `_bulk` index
actions against the name in `elastic_config.index_name`, `embeddings-live` in the
checked-in config. Everything that gives that name meaning is provisioned by hand, once,
via the Kibana Dev Tools commands in `elasticsearch-setup.md`:

| Provisioned by `elasticsearch-setup.md` | What it does for this crate |
| --- | --- |
| ILM policy `embeddings-phasing-policy` | Rolls the write index over at 7 days / 50 GB primary shard, then moves it to `warm`, shrinks it to one shard and marks it read-only. |
| Component template `embedding-live-object-template` | The mapping: `timestamp` as `date`/`epoch_millis`, `channel_id` as `keyword`, `embedding` as `dense_vector` with `dims: 768`, `element_type: bfloat16`, `index_options.type: bbq_disk`, `similarity: cosine`. |
| Index template `embedding-live-template` | Binds the component template and the ILM policy to the `embeddings-live-*` pattern and adds the `embeddings_search` read alias. |
| `PUT embeddings-live-000001` with `"embeddings-live": {"is_write_index": true}` | Creates the first concrete index and makes `embeddings-live` the **rollover write alias**. |

So `index_name: embeddings-live` is an *alias*, not an index. Bulk actions against it are
routed to whichever concrete `embeddings-live-00000N` index currently holds
`is_write_index: true`, and ILM moves that pointer forward over time without the client
knowing. The three parameters the client depends on for its writes to be accepted and
searchable — `dims: 768`, the `epoch_millis` date format, and `channel_id` as `keyword` —
all live in the component template, matching the 768-element FP16 DINOv3 outputs, the
`timestamp_millis()` stamp and the numeric source id respectively.

Run the four commands in `elasticsearch-setup.md` against a fresh cluster before starting
the client (§12).

---

## 10. Statistics

`src/statistics.rs` is byte-identical to the detection crate's; see
`docs/client-object-detection.md` §9 for the reporter mechanics (a 1 s tick that logs one
`inference statistics` event per source and then resets the counters, so every number is
a one-second window, and averages divide accumulated microseconds by `frames_success`).

What differs is where the numbers come from, because there are three model branches
feeding one `FrameProcessStats`:

| Counter | This crate |
| --- | --- |
| `frames_total` | Incremented once per skipped frame (`source.rs:197-199`) and once per processed frame (`:126-128`). |
| `frames_expected` | Incremented only at the processed-frame site (`:129-131`). |
| `frames_success` | `process_frame` returned `Ok` (`:132-140`). |
| `frames_failed` | `process_frame` returned `Err` (`:141-145`), or the frame was evicted from the one-slot queue by the drop callback (`:98-100`). |

| Timer (µs, averaged) | Measures |
| --- | --- |
| `avg_queue` | From `added` (stamped on enqueue, `source.rs:190`) to the start of `process_frame` (`:209`). |
| `avg_pre_proc` | Sum of the preprocessing stages of every branch that ran: YOLO's (`yolo.rs:445`), the frame embedder's (`dino.rs:188`), and — when boxes were found — the whole crop-preprocessing fan-out (`dino.rs:242-266`). |
| `avg_inference` | Sum of the `infer()` calls of every branch that ran. |
| `avg_post_proc` | Sum of the decode / conversion stages of every branch that ran. |
| `avg_results` | Time to enqueue the documents into the Elastic queue (`source.rs:268-273`). |
| `avg_processing` | Sum of each branch's own wall time (`yolo.rs:512`, `dino.rs:222`, `:290`) plus the queue time added at `source.rs:281`. |

Because `FrameProcessStats::accumulate` (`statistics.rs:50-57`) adds field by field
across branches, the per-stage timers are **totals of model work**, not wall-clock
durations: the detector and the frame embedder run concurrently (`source.rs:229`), so
their contributions overlap in time. Read them as "how much inference this pipeline
performed per frame", and use `avg_processing` minus `avg_queue` for a per-branch wall
time only when a single branch ran.

GPU telemetry via NVML is unchanged and still `device_type == GPU` only.

---

## 11. Configuration reference — `secrets/config.yaml`

Deserialized into `AppConfig` (`src/utils/config.rs:191`) by `serde_yaml`. Unknown keys
are ignored; missing keys without a `#[serde(default)]` are a hard parse error.

### Top level

| Key | Type | Effect |
| --- | --- | --- |
| `local` | `bool` | When true, adds the `logs/app.log` file layer. Stdout logging is unconditional. |
| `sources_config` | map | Sources, frame skipping and thresholds. |
| `elastic_config` | map | **New.** See below. |
| `triton_config.url` | string | Triton gRPC endpoint. |
| `inference_config` | map | Device, instance counts, and the three models. |

`hardware_name` is a field of `AppConfig` but is `#[serde(skip)]` (`:199`) — resolved at
startup, never read from the file.

### `elastic_config`

`ElasticConfig` (`config.rs:63-67`), reached through `AppConfig::elastic_config()`
(`:300`) and cloned into `Elastic::new` at `services.rs:64`.

| Key | Type | Effect |
| --- | --- | --- |
| `url` | string | Elasticsearch HTTP endpoint handed to `Transport::single_node`. Plain HTTP; no credentials, TLS settings or timeouts are configurable here. |
| `index_name` | string | Written into every `_bulk` action line (`elastic.rs:90`). Points at the ILM rollover write alias, `embeddings-live` (§9). |

Both fields are mandatory.

### `sources_config`

Unchanged from the detection crate — see `docs/client-object-detection.md` §10 for the full
override-and-clamp table. In short: `ids` lists the sources to start, `default` supplies
the baseline `inf_frame` / `conf_threshold` / `nms_iou_threshold`, and an optional
`custom` map overrides them per source through a range filter (`inf_frame` 1..=30,
thresholds 0.0..=1.0).

Which of those the pipeline consults here:

| Field | Consumed by |
| --- | --- |
| `inf_frame` | `add_to_queue` (`source.rs:186`) — one frame in N enters the pipeline. |
| `conf_threshold` | The YOLO26 decoder, i.e. which detections become crops. |
| `nms_iou_threshold` | The YOLOv9 decoder only; parsed and unused under `model_type: YOLO26`. |

The DINO stages take no per-source parameters — every frame that reaches them is embedded
whole, and every surviving box is cropped and embedded.

### `inference_config`

| Key | Type | Effect |
| --- | --- | --- |
| `device_type` | `GPU` \| `CPU` | Applies to **all three** models: platform, model filename, instance kind, the `optimization` block, and whether NVML runs. |
| `instances.default` | `u32` | Instance count when the hardware, or the purpose under that hardware, is not listed. |
| `instances.custom` | `map<String, map<ModelPurpose, u32>>` | Keyed first by `hardware_name` — the literal `"CPU"`, or the NVML device name — then by `ModelPurpose`, so the detector and the two embedders can each get their own count on the same hardware. Resolved by `InstancesConfig::resolve` (`:84`), which falls back to `default` when either level is absent. |
| `models` | `map<ModelPurpose, ModelConfig>` | Three purposes now, see below. |

`ModelPurpose` (`config.rs:113-118`) is the dispatch key for the whole pipeline:

| Variant | Resolved at | Pipeline role |
| --- | --- | --- |
| `FrameDetection` | `source.rs:217-219` | Detector. Its boxes are the crop list. |
| `FrameEmbedding` | `source.rs:220-222` | Embeds the whole frame. |
| `BBOXEmbedding` | `source.rs:249-251` | Embeds each expanded crop. |

Which *code path* runs is chosen by `ModelPurpose`, not by `model_type`:
`FrameDetection` goes to `processing::yolo`, both embedding purposes go to
`processing::dino`. `ModelType` (`config.rs:94-99`, now `YOLOV9 | YOLO26 | DINOV3`)
selects only the output decoder *within* the YOLO path (`yolo.rs:490-500`).

`ModelConfig` (`config.rs:14-25`) — all fields mandatory:

| Field | Notes |
| --- | --- |
| `name` | Triton model name; must match the directory in `triton_models/`. |
| `model_type` | `YOLOV9`, `YOLO26` or `DINOV3`. |
| `precision` | `FP32` or `FP16`. Becomes `TYPE_FP32`/`TYPE_FP16` in the generated Triton config, the request datatype, the preprocessing output format, and the postprocessing element width. |
| `input_name` / `output_name` | Tensor names as compiled into the model. |
| `input_shape` | Without the batch dimension. The **last** element is the letterbox target size (`yolo.rs:434-438`, `dino.rs:177-182`, `:234-239`). |
| `output_shape` | Product × precision size gives the per-sample output byte count (`inference.rs:305-314`). Must be 2-dimensional for the YOLO decoders; the DINO path reads it only through that product. |
| `batch_max_size` | `max_batch_size` in the generated config, and the fast-path threshold in `infer()`. |
| `batch_max_queue_delay` | `dynamic_batching.max_queue_delay_microseconds`. |
| `batch_preferred_sizes` | `dynamic_batching.preferred_batch_size`. |

### The checked-in configuration

`local: true`, one source (id 1), `inf_frame: 25`, `conf_threshold: 0.25`,
`nms_iou_threshold: 0.5` (unused under YOLO26), no `custom` block. Elastic at
`http://localhost:9200`, index `embeddings-live`. Triton at `http://localhost:8001`,
`device_type: CPU`, one instance.

| | `FrameDetection` | `FrameEmbedding` | `BBOXEmbedding` |
| --- | --- | --- | --- |
| `name` | `yolo26x` | `dinov3-512` | `dinov3-224` |
| `precision` | FP32 | FP16 | FP16 |
| `input_name` | `images` | `images` | `images` |
| `input_shape` | `[3, 640, 640]` | `[3, 512, 512]` | `[3, 224, 224]` |
| `output_name` | `output` | `output` | `output` |
| `output_shape` | `[300, 6]` | `[768]` | `[768]` |
| `batch_max_size` | 16 | 16 | 16 |
| `batch_max_queue_delay` | 2500 µs | 1000 µs | 1000 µs |
| `batch_preferred_sizes` | `[2, 4, 6, 8]` | `[4, 8, 16]` | `[4, 8, 16]` |

Reading across that table: the detector runs at **FP32** on 640 px frames and emits at
most 300 rows of `[x1, y1, x2, y2, score, class_id]`; both embedders run at **FP16** and
emit a 768-dimensional vector, which is what the `dense_vector` mapping in
`elasticsearch-setup.md` declares. The whole-frame embedder sees 512 px, the crop
embedder 224 px — crops are small regions upsampled to a small input, whole frames get
the larger one. The DINO models also carry a shorter dynamic-batching delay (1 ms versus
2.5 ms) and preferred batch sizes reaching 16, which suits the crop path, where a single
frame can present many inputs at once.

---

## 12. Building and running

### Prerequisites

1. **`libclient_video.so` in `client-image-retrieval/secrets/`.** `.gitignore` ends with
   `*.so`, so the compiled video library is never committed. `ClientVideo::new`
   (`client_video.rs:110`) `dlopen`s `secrets/libclient_video.so` relative to the process
   working directory, and `Services::new` constructs it unconditionally
   (`services.rs:72`). A built copy exists in the sibling project at
   `client-real-time/client/secrets/libclient_video.so`; copying it in is a deliberate
   manual step.
2. **The three models in the Triton model repository.** With `device_type: CPU` the
   client uploads a config declaring `onnxruntime_onnx` and
   `default_model_filename: model.onnx`, so it needs
   `triton_models/{yolo26x,dinov3-512,dinov3-224}/1/model.onnx`. On GPU it needs
   `model.plan` in the same places. `triton_models/` currently holds
   `dinov3-224/1/model.plan`, `dinov3-512/1/model.plan` and `yolov9-e/1/model.plan`, so a
   GPU run needs `yolo26x` added and a CPU run needs ONNX exports of all three.
   `model-optimization/` is where those exports are produced. No `config.pbtxt` is
   required — the client generates and uploads the model configuration itself (§6).
3. **The Elasticsearch index templates applied.** Run the four Kibana Dev Tools commands
   in `elasticsearch-setup.md` — ILM policy, component template, index template, and the
   initial `embeddings-live-000001` index with the `embeddings-live` write alias — before
   the first write (§9).
4. **Working directory is the crate root.** Both `secrets/config.yaml`
   (`config.rs:252`) and `secrets/libclient_video.so` (`client_video.rs:23`) resolve
   relative to the process CWD.
5. **`device_type` matches the Triton profile you boot.** The client is what tells Triton
   which platform to load, so booting the CPU compose profile with `device_type: GPU`
   asks an unaccelerated Triton for a `tensorrt_plan`. There is no environment-variable
   override; `secrets/config.yaml` is the only place to switch it.

### Build

```bash
cd client-image-retrieval
cargo build --release
```

`.cargo/config.toml` sets `CMAKE_POLICY_VERSION_MINIMUM=3.5` so the build works out of the
box: `triton-client 0.2.0` pulls in `prost-build 0.10.4`, which vendors protobuf and
builds it with CMake, and CMake ≥ 4.0 dropped compatibility with the
`cmake_minimum_required(VERSION < 3.5)` that vendored protobuf declares. The faster
alternative, noted in the file itself, is to install `protoc`
(`dnf install protobuf-compiler`) and set `PROTOC_NO_VENDOR=1`.

Beyond the detection crate's dependency set, `Cargo.toml` adds
`elasticsearch = "9.1.0-alpha.1"` (an alpha release of the official client, matching the
9.x cluster in `services/docker-compose-elastic.yml`) and `chrono = "0.4.42"` for the
document timestamp.

### Run

```bash
moon run client-retrieval:cpu     # or
moon run client-retrieval:gpu
```

Each task depends on **both** the matching `services:triton-*` task and
`services:elastic`, so the whole backing stack is started for you. Both service tasks run
`docker compose up --detach --wait`, which blocks on the compose healthchecks and exits
only once the services report ready — Triton serving, and the Elasticsearch cluster at
`wait_for_status=yellow` (`services/docker-compose-elastic.yml`). By the time `cargo run`
starts, `InferenceModel::new`'s `server_ready()` check is guaranteed to pass, and the
cluster is answering queries. The Elastic gate matters because the client's transport is
lazy (§9): nothing in startup would otherwise notice a cluster that is not there.

The gating depends on those service tasks being **non-persistent**: moon starts
`persistent` tasks concurrently and never waits for one to finish, so a persistent task
would not gate anything. Confirm the shape with
`moon action-graph client-retrieval:cpu --dot` — it must show `RunTask(services:triton-cpu)`
and `RunTask(services:elastic)` feeding `RunPersistentTask(client-retrieval:cpu)`.

Because both stacks run detached, nothing streams their output and they outlive the
client:

```bash
moon run services:triton-logs     # follow Triton's output
moon run services:elastic-logs    # follow elasticsearch / kibana / logstash
moon run services:triton-down
moon run services:elastic-down
```

Kibana comes up with the Elastic stack on port 5601 — that is where the
`elasticsearch-setup.md` Dev Tools commands are run, and where the written vectors can be
inspected.

Both client tasks set `RUST_LOG=INFO`; a `RUST_LOG` exported in your shell takes
precedence.

The manual equivalent, if you are not going through moon:

1. `cd services && docker compose -f docker-compose-elastic.yml up --detach --wait`
2. `docker compose -f docker-compose-triton.yml --profile cpu up --detach --wait`
   (run from `services/`, since the compose file mounts `../triton_models` relative to
   its own directory)
3. Apply the `elasticsearch-setup.md` commands in Kibana at `http://localhost:5601`.
4. Place `libclient_video.so` in `client-image-retrieval/secrets/` and the models in
   `triton_models/<name>/1/model.{onnx,plan}`.
5. `cd client-image-retrieval && RUST_LOG=INFO ./target/release/client`

### What success looks like

On a healthy start you should see, in order: three `Initiated model instances` /
`Successfully initiated model` pairs from `InferenceModels::start`, then per-source
`inference statistics` lines once a second from the statistics reporter, and
`Successfully sent bulk request to Elastic, Total N` from the flusher every 2 s once
frames are flowing (`elastic.rs:108-111`). With `inf_frame: 25` at 25 fps, that is
roughly one frame per second reaching the pipeline, and `N` per flush is
`2 × (1 + boxes_per_frame)`.
