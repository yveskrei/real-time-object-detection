import time
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import threading
import GPUtil
import traceback

MODEL_FP16 = 'FP16'
MODEL_FP32 = 'FP32'

# === CONFIG ===
MODEL_NAME = "yolov9-e-fp16"  # Change as needed
MODEL_TYPE = MODEL_FP16
MODEL_VERSION = "1"
INPUT_NAME = "images"
MAX_LATENCY_MS = 1000         # Max acceptable average latency
MAX_GPU_UTIL = 100            # Max GPU usage (%)
TRITON_URL = "localhost:8001"  # Triton server address
TARGET_FPS = 30               # FPS for each video stream
TEST_DURATION_SEC = 60        # How long to run each test


def generate_input():
    """Generate a single dummy input (batch size = 1)."""
    rand_input = np.random.rand(1, 3, 640, 640)

    if MODEL_TYPE == MODEL_FP16:
        return rand_input.astype(np.float16)
    elif MODEL_TYPE == MODEL_FP32:
        return rand_input.astype(np.float32)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}. Use 'FP16' or 'FP32'.")


def get_gpu_util():
    """Get current GPU utilization percentage."""
    gpus = GPUtil.getGPUs()
    return max((gpu.load * 100 for gpu in gpus), default=0.0)


def infer(client, input_data):
    """Perform inference on the model."""
    inputs = [
        grpcclient.InferInput(INPUT_NAME, input_data.shape, np_to_triton_dtype(input_data.dtype))
    ]
    inputs[0].set_data_from_numpy(input_data)
    return client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION, inputs=inputs)


class VideoStreamSimulator:
    """Simulates a video stream sending frames at specified FPS."""
    
    def __init__(self, stream_id, fps, client, input_data):
        self.stream_id = stream_id
        self.fps = fps
        self.frame_interval = 1.0 / fps  # seconds between frames
        self.client = client
        self.input_data = input_data
        self.latencies = []
        self.frames_sent = 0
        self.frames_completed = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the video stream simulation."""
        self.running = True
        self.thread = threading.Thread(target=self._run_stream)
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
            
            # Check if it's time to send the next frame
            if current_time - last_frame_time >= self.frame_interval:
                self._send_frame()
                last_frame_time = current_time
            else:
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.001)
    
    def _send_frame(self):
        """Send a single frame for inference."""
        self.frames_sent += 1
        start_time = time.time()
        
        try:
            infer(self.client, self.input_data)
            latency_ms = (time.time() - start_time) * 1000
            
            with self.lock:
                self.latencies.append(latency_ms)
                self.frames_completed += 1
                
        except Exception as e:
            print(f"Stream {self.stream_id} error: {e}")
    
    def get_stats(self):
        """Get statistics for this stream."""
        with self.lock:
            latencies = self.latencies.copy()
        
        if not latencies:
            return {
                "stream_id": self.stream_id,
                "frames_sent": self.frames_sent,
                "frames_completed": self.frames_completed,
                "avg_latency_ms": float('inf'),
                "p95_latency_ms": float('inf'),
                "p99_latency_ms": float('inf'),
                "min_latency_ms": float('inf'),
                "max_latency_ms": float('inf'),
                "actual_fps": 0.0
            }
        
        return {
            "stream_id": self.stream_id,
            "frames_sent": self.frames_sent,
            "frames_completed": self.frames_completed,
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "actual_fps": self.frames_completed / TEST_DURATION_SEC
        }


def test_video_streams(num_streams):
    """Test with specified number of video streams."""
    print(f"\n=== Testing {num_streams} video streams at {TARGET_FPS} FPS ===")
    
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    input_data = generate_input()
    
    # Create and start video streams
    streams = []
    for i in range(num_streams):
        stream = VideoStreamSimulator(i, TARGET_FPS, client, input_data)
        streams.append(stream)
        stream.start()
    
    # Let streams run for the test duration
    start_time = time.time()
    time.sleep(TEST_DURATION_SEC)
    
    # Stop all streams
    for stream in streams:
        stream.stop()
    
    total_duration = time.time() - start_time
    
    # Collect statistics
    all_latencies = []
    total_frames_sent = 0
    total_frames_completed = 0
    stream_stats = []
    
    for stream in streams:
        stats = stream.get_stats()
        stream_stats.append(stats)
        
        with stream.lock:
            all_latencies.extend(stream.latencies)
        
        total_frames_sent += stats["frames_sent"]
        total_frames_completed += stats["frames_completed"]
    
    # Calculate overall statistics
    if all_latencies:
        avg_latency = np.mean(all_latencies)
        p95_latency = np.percentile(all_latencies, 95)
        p99_latency = np.percentile(all_latencies, 99)
        min_latency = np.min(all_latencies)
        max_latency = np.max(all_latencies)
    else:
        avg_latency = p95_latency = p99_latency = min_latency = max_latency = float('inf')
    
    overall_throughput = total_frames_completed / total_duration
    gpu_util = get_gpu_util()
    
    # Calculate expected vs actual FPS
    expected_total_fps = num_streams * TARGET_FPS
    actual_total_fps = overall_throughput
    
    results = {
        "num_streams": num_streams,
        "target_fps_per_stream": TARGET_FPS,
        "test_duration_sec": TEST_DURATION_SEC,
        "overall_stats": {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "total_frames_sent": total_frames_sent,
            "total_frames_completed": total_frames_completed,
            "overall_throughput_fps": overall_throughput,
            "expected_total_fps": expected_total_fps,
            "actual_total_fps": actual_total_fps,
            "fps_efficiency": (actual_total_fps / expected_total_fps) * 100 if expected_total_fps > 0 else 0,
            "gpu_util_percent": gpu_util,
            "frame_completion_rate": (total_frames_completed / total_frames_sent) * 100 if total_frames_sent > 0 else 0
        },
        "per_stream_stats": stream_stats
    }
    
    # Print results
    print(f"Overall Results:")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  P95 Latency: {p95_latency:.2f} ms")
    print(f"  P99 Latency: {p99_latency:.2f} ms")
    print(f"  Min Latency: {min_latency:.2f} ms")
    print(f"  Max Latency: {max_latency:.2f} ms")
    print(f"  Total Throughput: {overall_throughput:.2f} FPS")
    print(f"  Expected FPS: {expected_total_fps:.2f}")
    print(f"  Actual FPS: {actual_total_fps:.2f}")
    print(f"  FPS Efficiency: {(actual_total_fps / expected_total_fps) * 100:.1f}%")
    print(f"  GPU Utilization: {gpu_util:.1f}%")
    print(f"  Frame Completion Rate: {(total_frames_completed / total_frames_sent) * 100:.1f}%")
    
    return results


def run_full_test(min_streams: int = 1, max_streams:int = 32):
    """Run the complete video stream throughput test."""
    print(f"Starting video stream throughput test")
    print(f"Model: {MODEL_NAME}")
    print(f"Target FPS per stream: {TARGET_FPS}")
    print(f"Test duration: {TEST_DURATION_SEC} seconds")
    print(f"Min streams to test: {min_streams}")
    print(f"Max streams to test: {max_streams}")
    
    all_results = {}
    
    num_streams = min_streams
    while num_streams <= max_streams:
        try:
            results = test_video_streams(num_streams)
            all_results[f"streams_{num_streams}"] = results
            
            # Check if we've hit our limits
            overall_stats = results["overall_stats"]
            if (overall_stats["avg_latency_ms"] > MAX_LATENCY_MS or 
                overall_stats["gpu_util_percent"] > MAX_GPU_UTIL):
                
                print(f"⚠️ Performance threshold reached at {num_streams} streams.")
                print(f"   Average latency: {overall_stats['avg_latency_ms']:.2f} ms")
                print(f"   GPU utilization: {overall_stats['gpu_util_percent']:.1f}%")
                print(f"   FPS efficiency: {overall_stats['fps_efficiency']:.1f}%")
                break
            
            num_streams += 1
            
        except Exception as e:
            print(f"Error testing {num_streams} streams: {e}")
            traceback.print_exc()
            break
    
    # Save results to JSON
    out_file = f"{MODEL_NAME}_video_stream_results_{TARGET_FPS}fps.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {out_file}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    for stream_count, data in all_results.items():
        stats = data["overall_stats"]
        print(f"{data['num_streams']} streams: "
              f"Avg {stats['avg_latency_ms']:.1f}ms, "
              f"P99 {stats['p99_latency_ms']:.1f}ms, "
              f"Efficiency {stats['fps_efficiency']:.1f}%, "
              f"GPU {stats['gpu_util_percent']:.1f}%")


if __name__ == "__main__":
    # You can also test a specific number of streams
    # test_video_streams(4)
    
    # Or run the full test
    run_full_test()