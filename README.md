# Real Time AI Analytics
The following repository is a set of projects revolving around the same principal - performing AI analytics under strict real time constraints.<br>
The projects are the following:
* **[Object Detection](client-object-detection)** - Performing analytics in real time(`sub-33ms` constraints) over live video feed and displaying a BBOX square to the user on screen
* **[Image Retrieval Indexing](client-image-retrieval)** - Consuming video feed in real time, and indexing the individual frames into ElasticSearch database. 
* **[Image Retrieval Search](image-retrieval-search)** - Allowing users to upload an image and find similar images from the indexed ElasticSearch frames, using a highly configurable vector search setting.
* **[Model Optimization](model-optimization)** - Set of scripts dedicated to converting a machine learning model into `ONNX`/`TensorRT` at specific settings, to allow highter throughput when used with other applications

## Prerequisites
- **Rust** toolchain installed for building the application.
To get started with running the application, you must have the following prerequisites:
- A machine with a **compatible NVIDIA GPU** and the necessary drivers installed to gain the most out of your hardware. Alternatively you can use **CPU**.

**NOTE** - In order to allow video consuming for `client-object-detection` and `client-image-retrieval`, you will need to compile an FFI library that is responsible for it.<br>
For more information about compiling the library please refer to [Video Player](https://github.com/yveskrei/video-player) Repository

After retrieving the `libclient_video.so` file, plant it in each of the projects `secrets` folder.

## Getting started
Before running, make sure you place your models in `triton_models` directory at the root of the repository. This will be used for all the three projects included.<br>
```
triton_models/
└── MODEL_NAME/
    └── 1/
        ├── model.onnx
        └── model.plan
```
**NOTE** - The projects load the models dynamically with their own config, so `config.pbtxt` is not required here

Run the following commands, depending on what project you want to use. Each project has a GPU and CPU variant(decides how to run NVIDIA TritonServer):
```bash
moon run client-detection:(gpu/cpu)
moon run client-retrieval:(gpu/cpu)
moon run retrieval-search:(gpu/cpu)
```
