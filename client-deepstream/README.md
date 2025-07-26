# Nvidia DeepStream Client
Implementation of [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository on YOLOV9 models.<br>
The implementation is intended mostly for edge devices getting raw stream input, and systems that perform end to end video analytics, compatible with GStreamer formats.<br>
Using **DeepStream 7.1**

**Note** - This is a POC only, and not production ready material

## Setup
First, we need to configure the application before running.<br>
Move the `export_yolo_deepstream.py` file to the base of your [YOLOV9](https://github.com/WongKinYiu/yolov9) repository, together with the chosen model. Now run the following command to convert the models to be compatible with the deepstream setup:

```
pip install -r requirements.txt
python export_yolo_deepstream.py --model_path <PATH_TO_MODEL_PT>
```
This will create a compatible onnx model that will be used for deepstream. We need to use this export method over regular `export.py` module found in yolov9's repository, because we perform postprocessing(eliminating class probabilities) at the GPU level, to boost performance.

Next, we would be building docker image for the client, containing extra dependencies and necessary setup.<br>
Run the following command:

```
docker build -t deepstream:7.1-gc-triton-devel-deps-fix
```

And then to run and get inside of the image for further actions:

```
docker compose up -d

docker exec -it deepstream bash
```

Next, we need to edit the `config_infer_yolov9.txt` configuration file to read our onnx file. Edit the following properties in your file:

```
[properties]
...
onnx-file=yolov9-e.onnx # Path to your ONNX File
model-engine-file=model_b1_gpu0_fp16.engine # Path to tensorrt file(Will compile onnx to tensorrt if not found)
network-mode=2 # Type of model(2=Float16)
```

For the `deepstream_app_config.txt` file, make sure you have this field setup correctly:

```
[primary-gie]
...
config-file=config_infer_yolov9.txt # Path to the other configuration
```

Now we should be set to run the application:
```
deepstream-app -c deepstream_app_config.txt
```