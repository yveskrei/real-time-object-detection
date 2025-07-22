import time
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import threading
import GPUtil
import traceback
import argparse
import cv2
import os

MODEL_FP16 = 'FP16'
MODEL_FP32 = 'FP32'

def generate_dummy_input(model_type):
    """Generate a single dummy input (batch size = 1)."""
    rand_input = np.random.rand(1, 3, 640, 640)

    if model_type == MODEL_FP16:
        return rand_input.astype(np.float16)
    elif model_type == MODEL_FP32:
        return rand_input.astype(np.float32)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'FP16' or 'FP32'.")

def load_image_input(image_path, model_type):
    """
    Load image from disk, preprocess and return numpy array
    with shape (1, 3, 640, 640) in model_type float32 or float16.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load with OpenCV in BGR
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 640x640 (model input size)
    img_resized = cv2.resize(img, (640, 640))

    # Normalize to 0-1 float
    img_norm = img_resized.astype(np.float32) / 255.0

    # Change to CHW format
    img_chw = np.transpose(img_norm, (2, 0, 1))

    # Add batch dim
    img_batch = np.expand_dims(img_chw, axis=0)

    if model_type == MODEL_FP16:
        return img_batch.astype(np.float16)
    else:
        return img_batch.astype(np.float32)

def get_gpu_util():
    """Get current GPU utilization percentage."""
    gpus = GPUtil.getGPUs()
    return max((gpu.load * 100 for gpu in gpus), default=0.0)

def infer(client, model_name, model_version, input_name, input_data):
    """Perform inference on the model."""
    inputs = [
        grpcclient.InferInput(input_name, input_data.shape, np_to_triton_dtype(input_data.dtype))
    ]
    inputs[0].set_data_from_numpy(input_data)
    return client.infer(model_name=model_name, model_version=model_version, inputs=inputs)

class VideoStreamSimulator:
    """Simulates a video stream sending frames at specified FPS."""
    
    def __init__(self, stream_id, fps, client, model_name, model_version, input_name, input_data, infer_every_n_frames):
        self.stream_id = stream_id
        self.fps = fps
        self.frame_interval = 1.0 / fps  # seconds between frames
        self.client = client
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_data = input_data
        self.infer_every_n_frames = infer_every_n_frames

        self.latencies = []
        self.frames_got = 0
        self.frames_processed = 0
        self.frames_completed = 0
        self.frames_missed = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the video stream simulation."""
        self.running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the video stream simulation."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run_stream(self):
        """Run the video stream loop."""
        last_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_frame_time >= self.frame_interval:
                self._send_frame()
                last_frame_time = current_time
            else:
                time.sleep(0.001)
    
    def _send_frame(self):
        """Send a single frame for inference."""
        self.frames_got += 1

        if self.frames_got % self.infer_every_n_frames != 0:
            return  # Skip inference for this frame

        self.frames_processed += 1
        start_time = time.time()

        try:
            infer(self.client, self.model_name, self.model_version, self.input_name, self.input_data)

            with self.lock:
                latency_ms = (time.time() - start_time) * 1000
                self.latencies.append(latency_ms)

                if latency_ms <= self.frame_interval * 1000:
                    self.frames_completed += 1
                else:
                    self.frames_missed += 1

        except Exception as e:
            print(f"Stream {self.stream_id} error: {e}")

    def get_stats(self, test_duration_sec):
        """Get statistics for this stream."""
        with self.lock:
            latencies = self.latencies.copy()

        if not latencies:
            return {
                "stream_id": self.stream_id,
                "frames_got": self.frames_got,
                "frames_processed": self.frames_processed,
                "frames_completed": self.frames_completed,
                "frames_missed": self.frames_missed,
                "frame_miss_rate": 0.0,
                "avg_latency_ms": float('inf'),
                "p95_latency_ms": float('inf'),
                "p99_latency_ms": float('inf'),
                "min_latency_ms": float('inf'),
                "max_latency_ms": float('inf'),
                "actual_fps": 0.0
            }
        else:
            return {
                "stream_id": self.stream_id,
                "frames_got": self.frames_got,
                "frames_processed": self.frames_processed,
                "frames_completed": self.frames_completed,
                "frames_missed": self.frames_missed,
                "frame_miss_rate": (self.frames_missed / self.frames_got) * 100 if self.frames_got > 0 else 0,
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "actual_fps": self.frames_completed / test_duration_sec
            }

def test_video_streams(num_streams, client, model_name, model_version, input_name, input_data, target_fps,
                       test_duration_sec, infer_every_n_frames):
    """Test with specified number of video streams."""
    print(f"\n=== Testing {num_streams} video streams at {target_fps} FPS (Inference every {infer_every_n_frames} frames) ===")
    
    streams = []
    for i in range(num_streams):
        stream = VideoStreamSimulator(i, target_fps, client, model_name, model_version, input_name, input_data, infer_every_n_frames)
        streams.append(stream)
        stream.start()
    
    start_time = time.time()
    time.sleep(test_duration_sec)

    for stream in streams:
        stream.stop()
    
    total_duration = time.time() - start_time

    all_latencies = []
    total_frames_got = 0
    total_frames_processed = 0
    total_frames_completed = 0
    total_frames_missed = 0
    stream_stats = []

    for stream in streams:
        stats = stream.get_stats(test_duration_sec)
        stream_stats.append(stats)

        with stream.lock:
            all_latencies.extend(stream.latencies)
        
        total_frames_got += stats["frames_got"]
        total_frames_processed += stats["frames_processed"]
        total_frames_completed += stats["frames_completed"]
        total_frames_missed += stats["frames_missed"]

    if all_latencies:
        avg_latency = np.mean(all_latencies)
        p95_latency = np.percentile(all_latencies, 95)
        p99_latency = np.percentile(all_latencies, 99)
        min_latency = np.min(all_latencies)
        max_latency = np.max(all_latencies)
    else:
        avg_latency = p95_latency = p99_latency = min_latency = max_latency = float('inf')

    gpu_util = get_gpu_util()
    
    expected_total_fps = num_streams * target_fps / infer_every_n_frames
    actual_total_fps = total_frames_completed / total_duration
    fps_completion_rate = (actual_total_fps / expected_total_fps) * 100 if expected_total_fps > 0 else 0

    frame_process_rate = (total_frames_processed / total_frames_got) * 100 if total_frames_got > 0 else 0
    frame_completion_rate = (total_frames_completed / total_frames_processed) * 100 if total_frames_processed > 0 else 0
    missed_frame_rate = (total_frames_missed / total_frames_processed) * 100 if total_frames_processed > 0 else 0

    results = {
        "num_streams": num_streams,
        "overall_stats": {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "expected_total_fps": expected_total_fps,
            "actual_total_fps": actual_total_fps,
            "fps_completion_rate": fps_completion_rate,
            "gpu_util_percent": gpu_util,
            
            "total_frames_got": total_frames_got,
            "total_frames_processed": total_frames_processed,
            "frame_process_rate": frame_process_rate,
            "total_frames_completed": total_frames_completed,
            "frame_completion_rate": frame_completion_rate,
            "total_frames_missed": total_frames_missed,
            "missed_frame_rate": missed_frame_rate,
        },
        "per_stream_stats": stream_stats
    }

    print("---- General Statistics ----")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  P95 Latency: {p95_latency:.2f} ms")
    print(f"  P99 Latency: {p99_latency:.2f} ms")
    print(f"  Min Latency: {min_latency:.2f} ms")
    print(f"  Max Latency: {max_latency:.2f} ms")
    print(f"  Expected FPS: {expected_total_fps:.2f} FPS")
    print(f"  Actual FPS: {actual_total_fps:.2f} FPS")
    print(f"  FPS Completion Rate: {fps_completion_rate:.2f}%")
    print(f"  GPU Utilization: {gpu_util:.2f}%")
    print("\n---- Frame Statistics ----")
    print(f"  Total Frames Got: {total_frames_got}")
    print(f"  Total Frames Processed: {total_frames_processed}")
    print(f"  Frame Process Rate: {frame_process_rate:.2f}%")
    print(f"  Total Frames Completed: {total_frames_completed}")
    print(f"  Frame Completion Rate: {frame_completion_rate:.2f}%")
    print(f"  Total Frames Missed: {total_frames_missed}")
    print(f"  Missed Frame Rate: {missed_frame_rate:.2f}%")

    return results

def run_full_test(args):
    print(f"Starting video stream throughput test")
    print(f"Model: {args.model_name}")
    print(f"Model Version: {args.model_version}")
    print(f"Target FPS per stream: {args.target_fps}")
    print(f"Test duration: {args.test_duration_sec} seconds")
    print(f"Inference every N frames: {args.infer_every_n_frames}")
    print(f"Max latency threshold: {args.max_latency} ms")
    print(f"Max GPU utilization threshold: {args.max_gpu_util} %")
    print(f"Triton URL: {args.triton_url}")
    print(f"Input tensor name: {args.input_name}")
    print(f"Using image input: {'Yes' if args.input_image else 'No (dummy input)'}")
    print(f"Frame processing interval: {1 / args.target_fps * 1000:.2f} ms")

    # Create client
    client = grpcclient.InferenceServerClient(url=args.triton_url)

    # Prepare input data
    if args.input_image:
        input_data = load_image_input(args.input_image, args.model_type)
    else:
        input_data = generate_dummy_input(args.model_type)

    all_results = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "fps_per_stream": args.target_fps,
        "test_duration_sec": args.test_duration_sec,
        "infer_every_n_frames": args.infer_every_n_frames,
        "streams_stats": []
    }
    
    num_streams = args.min_streams
    while num_streams <= args.max_streams:
        try:
            results = test_video_streams(
                num_streams,
                client,
                args.model_name,
                args.model_version,
                args.input_name,
                input_data,
                args.target_fps,
                args.test_duration_sec,
                args.infer_every_n_frames,
            )
            all_results["streams_stats"].append(results)

            overall_stats = results["overall_stats"]
            if (overall_stats["avg_latency_ms"] > args.max_latency or 
                overall_stats["gpu_util_percent"] > args.max_gpu_util):
                
                print(f"⚠️ Performance threshold reached at {num_streams} streams.")
                print(f"   Average latency: {overall_stats['avg_latency_ms']:.2f} ms")
                print(f"   GPU utilization: {overall_stats['gpu_util_percent']:.1f}%")
                print(f"   FPS efficiency: {overall_stats['fps_completion_rate']:.1f}%")
                break

            num_streams += 1

        except Exception as e:
            print(f"Error testing {num_streams} streams: {e}")
            traceback.print_exc()
            break

    out_file = f"{args.model_name}_video_stream_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✅ Results saved to {out_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Triton video stream throughput testing")

    parser.add_argument("--model_name", type=str, required=True, help="Triton model name")
    parser.add_argument("--model_version", type=str, default="1", help="Triton model version (default: 1)")
    parser.add_argument("--input_name", type=str, default="images", help="Model input tensor name (default: images)")
    parser.add_argument("--max_latency", type=int, default=1000, help="Max acceptable average latency in ms (default: 1000)")
    parser.add_argument("--max_gpu_util", type=int, default=100, help="Max GPU utilization percent (default: 100)")
    parser.add_argument("--triton_url", type=str, default="localhost:8001", help="Triton server URL (default: localhost:8001)")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS for each video stream (default: 30)")
    parser.add_argument("--test_duration_sec", type=int, default=30, help="Duration of test per stream count in seconds (default: 30)")
    parser.add_argument("--infer_every_n_frames", type=int, default=1, help="Perform inference every N frames (default: 1)")
    parser.add_argument("--input_image", type=str, default=None, help="Optional path to input image to replace dummy input")
    parser.add_argument("--model_type", type=str, choices=[MODEL_FP16, MODEL_FP32], default=MODEL_FP16, help="Model precision type (FP16 or FP32)")

    parser.add_argument("--min_streams", type=int, default=1, help="Minimum number of streams to test (default: 1)")
    parser.add_argument("--max_streams", type=int, default=32, help="Maximum number of streams to test (default: 32)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_full_test(args)
