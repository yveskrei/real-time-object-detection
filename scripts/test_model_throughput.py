import time
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_CONCURRENCY = 128         # Max concurrent requests to test
TRITON_URL = "localhost:8001"  # Triton server address


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
    gpus = GPUtil.getGPUs()
    return max((gpu.load * 100 for gpu in gpus), default=0.0)


def infer(client, input_data):
    inputs = [
        grpcclient.InferInput(INPUT_NAME, input_data.shape, np_to_triton_dtype(input_data.dtype))
    ]
    inputs[0].set_data_from_numpy(input_data)
    return client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION, inputs=inputs)


def run_concurrent_infer(client, concurrency):
    input_data = generate_input()
    latencies = []
    requests_sent = 0
    requests_completed = 0
    start_time = time.time()

    def task():
        nonlocal requests_sent, requests_completed
        requests_sent += 1
        start = time.time()
        try:
            infer(client, input_data)
            latency_ms = (time.time() - start) * 1000
            requests_completed += 1
            return latency_ms
        except Exception:
            print(traceback.format_exc())
            return None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task) for _ in range(concurrency)]
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                latencies.append(result)

    total_duration = (time.time() - start_time)
    avg_latency = np.mean(latencies) if latencies else float('inf')
    p95_latency = np.percentile(latencies, 95) if latencies else float('inf')
    p99_latency = np.percentile(latencies, 99) if latencies else float('inf')
    min_latency = np.min(latencies) if latencies else float('inf')
    max_latency = np.max(latencies) if latencies else float('inf')
    throughput = len(latencies) / total_duration if total_duration > 0 else 0.0
    gpu_util = get_gpu_util()

    return {
        "average_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_fps": throughput,
        "total_duration_sec": total_duration,
        "gpu_util_percent": gpu_util,
        "requests_sent": requests_sent,
        "requests_completed": requests_completed,
    }


def test_throughput():
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    results = {}

    concurrency = 1
    while concurrency <= MAX_CONCURRENCY:
        metrics = run_concurrent_infer(client, concurrency)

        print(f"\n=== Concurrency: {concurrency} ===")
        for key, val in metrics.items():
            print(f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}")

        results[f"concurrency_{concurrency}"] = metrics

        if metrics["average_latency_ms"] > MAX_LATENCY_MS or metrics["gpu_util_percent"] > MAX_GPU_UTIL:
            print("⚠️ Threshold reached. Stopping test.")
            break

        concurrency += 1

    # Save results to JSON
    out_file = f"{MODEL_NAME}_throughput_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_file}")


if __name__ == "__main__":
    test_throughput()