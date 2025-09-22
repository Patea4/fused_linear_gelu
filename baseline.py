import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


def benchmark_operation(func, inputs, warmup=10, runs=100):
    device = inputs[0].device

    # Warmup
    for _ in range(warmup):
        _ = func(*inputs)

    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(runs):
        result = func(*inputs)

    torch.cuda.synchronize()

    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / runs * 1000
    return result, avg_time


def separate_linear_gelu(x, weight, bias):
    """Baseline PyTorch operation: Linear + Gelu as two separate operations"""
    linear = F.linear(x, weight, bias)
    return F.gelu(linear)


def test_configuration(batch_size, input_dim, output_dim, device):
    "Test the specific configuration given by batch_size, input_dim and output_dim"

    print("\n" + ("=" * 60))
    print(f"Testing: Batch={batch_size}, Input={input_dim}, Output={output_dim}")
    print(f"Device: {device}")
    print("=" * 60)

    x = torch.randn(batch_size, input_dim, device=device)
    weight = torch.randn(output_dim, input_dim, device=device)
    bias = torch.randn(output_dim, device=device)

    print(f"Input tensor shape: {x.shape[0]} x {x.shape[1]}")
    print(f"Weight tensor shape: {weight.shape[0]} x {weight.shape[1]}")
    print(f"Bias tensor shape: {bias.shape[0]}")

    result, avg_time = benchmark_operation(separate_linear_gelu, [x, weight, bias], warmup=5, runs=50)

    print(f"Output shape: {result.shape[0]} x {result.shape[1]}")
    print(f"Average time: {avg_time:.3f} ms")

    # y_ij = 2 I operations. so 2I * how many y = 2I * O * B
    total_ops = batch_size * input_dim * output_dim * 2

    # operations / seconds / billion. So Billions of FLOPs per second = GFLOPS
    print(f"Throughput: {total_ops / (avg_time * 1e-3) / 1e9:.2f} GFLOPS")

    memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"GPU Memory used: {memory_used:.1f} MB")

    return result, avg_time


def main():
    print("=" * 60)
    print("PyTorch Linear + Gelu Baseline Benchmark")

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        exit()

    print(f"Using GPU: {torch.cuda.get_device_name()}")

    print("=" * 60)
    device = "cuda"

    test_configs = [(128, 512, 512), (256, 1024, 1024), (512, 2048, 2048), (1024, 4096, 4096)]

    for batch_size, input_dim, output_dim in test_configs:
        try:
            test_configuration(batch_size, input_dim, output_dim, device)
        except RuntimeError as e:
            print(f"Failed for config ({batch_size}, {input_dim}, {output_dim}): {e}")


if __name__ == "__main__":
    main()
