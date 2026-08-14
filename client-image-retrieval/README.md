# Client Image Retrieval
The following project allows consuming a large scale of video sources, ingesting the raw frames into `raw vectors` and `metadata` (Using an embedding model - currently supporting **DINOv3**), indexing them eventually in `ElasticSearch`, in order to later search upon the same indexed frames.

## Architecture overview
The following diagram emphesizes the architecture of the whole Image Retrieval chain.<br>
This project is the `FEED` part of this chain:<br>
![Architecture](../assets/image-retrieval.png)

## Technical details
* This project leverages **NVIDIA's TritonServer** tool, leveraging high throughput when doing inference on GPUs with dynamic batching, alongside **Rust's** native multithreading nature to squeeze performance
* ElasticSearch indexes are configured with the **disk_bbq** setting, to allow searching upon large quantities of vectors and not bloating the RAM during the process (vectors are stored on disk, KNN search stores only clusters in RAM and references vectors on disk for search).<br>Refer to [ElasticSearch Setup overview](../elasticsearch-setup.md) for more details
