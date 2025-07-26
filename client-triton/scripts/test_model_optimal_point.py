import argparse
import csv
import subprocess
import os
import threading
import time
from tempfile import NamedTemporaryFile
import GPUtil
import re
import traceback


def get_gpu_util():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0.0
    return max([gpu.load * 100 for gpu in gpus])


def monitor_gpu_util(threshold, process):
    while process.poll() is None:
        util = get_gpu_util()
        if util > threshold:
            print(f"GPU utilization too high ({util:.1f}%), terminating perf_analyzer.")
            process.terminate()
            break
        time.sleep(0.5)


def update_model_config(config_path, instance_count, max_batch_size):
    with open(config_path, 'r') as f:
        config = f.read()

    config = re.sub(r'count:\s*\d+', f'count: {instance_count}', config)
    config = re.sub(r'max_batch_size:\s*\d+', f'max_batch_size: {max_batch_size}', config)

    with open(config_path, 'w') as f:
        f.write(config)


def load_model_in_triton(model_name):
    curl_cmd = [
        "curl", "-X", "POST",
        f"http://localhost:8000/v2/repository/models/{model_name}/load"
    ]
    result = subprocess.run(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    message = result.stdout.decode()

    if message:
        print("[Load]", message)

def unload_model_in_triton(model_name):
    curl_cmd = [
        "curl", "-X", "POST",
        f"http://localhost:8000/v2/repository/models/{model_name}/unload"
    ]
    result = subprocess.run(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    message = result.stdout.decode()

    if message:
        print("[Unload]", message)


def run_perf_analyzer(perf_analyzer_path, model_name, batch_size, instance_count, gpu_util_threshold):
    with NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_csv:
        output_file = temp_csv.name

    cmd = [
        perf_analyzer_path,
        '-m', model_name,
        '-b', str(batch_size),
        '--concurrency-range', f'{instance_count}:{instance_count}',
        '--measurement-mode', 'time_windows',
        '-f', output_file,
        '--measurement-interval', '5000'
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    monitor_thread = threading.Thread(target=monitor_gpu_util, args=(gpu_util_threshold, process))
    monitor_thread.start()
    monitor_thread.join()
    process.wait()

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f.readlines())
            for row in reader:
                try:
                    concurrency = int(row['Concurrency'])
                    latency = float(row['p95 latency']) / 1e6
                    throughput = float(row['Inferences/Second'])
                    results.append({
                        'concurrency': concurrency,
                        'latency_ms': latency,
                        'throughput': throughput
                    })
                except Exception:
                    traceback.print_exc()
                    continue
        os.remove(output_file)

    return results if results else None


def find_best_config(perf_analyzer_path, model_name, config_path, latency_threshold_ms, gpu_util_threshold):
    best_config = None
    print(f"=== Testing model for best batch/instances combination ===")

    for instance_count in range(1,6):
        for batch_size in range(1,16,2):
            unload_model_in_triton(model_name)
            update_model_config(config_path, instance_count, batch_size)
            load_model_in_triton(model_name)


            print(f"Testing model instance count: {instance_count}, batch size: {batch_size}")

            time.sleep(2)  # Wait for model to load

            metrics = run_perf_analyzer(perf_analyzer_path, model_name, batch_size, instance_count, gpu_util_threshold)

            if metrics is None:
                print("GPU threshold reached or no results returned. Cooling down...")
                time.sleep(5)
                continue

            for entry in metrics:
                if entry['latency_ms'] <= latency_threshold_ms:
                    if best_config is None or entry['throughput'] > best_config['throughput']:
                        best_config = {
                            'batch_size': batch_size,
                            'instances': instance_count,
                            'latency_ms': entry['latency_ms'],
                            'throughput': entry['throughput']
                        }

    return best_config


def main():
    parser = argparse.ArgumentParser(description="Find optimal Triton batch size and instance count under latency constraint")
    parser.add_argument('--perf_analyzer', required=True, help='Path to perf_analyzer binary')
    parser.add_argument('--model', required=True, help='Model name deployed in Triton')
    parser.add_argument('--config_path', required=True, help='Path to model config.pbtxt')
    parser.add_argument('--latency_ms', type=float, required=True, help='Max acceptable p95 latency in milliseconds')
    parser.add_argument('--gpu_util_threshold', type=float, default=90.0, help='Maximum allowed GPU utilization percentage (default: 90)')

    args = parser.parse_args()

    best = find_best_config(args.perf_analyzer, args.model, args.config_path, args.latency_ms, args.gpu_util_threshold)

    if best:
        print("\nOptimal Configuration Found:")
        print(f"Batch Size     : {best['batch_size']}")
        print(f"Model Instances: {best['instances']}")
        print(f"Latency (p95)  : {best['latency_ms']:.2f} ms")
        print(f"Throughput     : {best['throughput']:.2f} infer/sec")
    else:
        print("No configuration found within the given latency or GPU constraint.")


if __name__ == '__main__':
    main()
