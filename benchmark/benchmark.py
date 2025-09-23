import torch
import torch.nn.functional as F
import time

def test_shapes():
    print("Testing different tensor shapes...")

    test_cases = [
        ([4, 8], 8, 6, True,  "2D with bias"),
        ([4, 8], 8, 6, False, "2D without bias"),
        ([2, 4, 8], 8, 6, True,  "3D (batched) with bias"),
        ([3, 2, 4, 8], 8, 6, True,  "4D (double batched) with bias"),
        ([5, 3, 2, 4, 8], 8, 6, False, "5D without bias"),
    ]

    try:
        import fused_linear_gelu

        for input_shape, input_dim, output_dim, has_bias, desc in test_cases:
            print(f"\nTesting {desc}: {input_shape} -> [..., {output_dim}]")

            torch.manual_seed(42)
            x = torch.randn(input_shape, device="cuda", dtype=torch.float32)
            weight = torch.randn(output_dim, input_dim, device="cuda", dtype=torch.float32)
            bias = torch.randn(output_dim, device="cuda", dtype=torch.float32) if has_bias else None

            ref = F.gelu(F.linear(x, weight, bias))

            fused = fused_linear_gelu.linear_gelu(x, weight, bias)
            cublas = fused_linear_gelu.linear_gelu_cublas(x, weight, bias)

            ok_shape = (ref.shape == fused.shape == cublas.shape)
            print(f"Shape: {fused.shape}" if ok_shape else f"ERROR: Shape mismatch: ref {ref.shape}, fused {fused.shape}, cublas {cublas.shape}")
            if not ok_shape:
                continue

            atol, rtol = 1e-5, 1e-4

            def report(name, y):
                ok = torch.allclose(ref, y, rtol=rtol, atol=atol)
                maxdiff = (ref - y).abs().max().item()
                if ok:
                    print(f"  {name:<8} matches (max diff: {maxdiff:.2e})")
                else:
                    print(f"  {name:<8} ERROR (max diff: {maxdiff:.2e})")
                    print(f"    Ref sample:   {ref.flatten()[:3]}")
                    print(f"    {name} sample: {y.flatten()[:3]}")

            report("Fused", fused)
            report("cuBLAS", cublas)

        print("\nShape testing complete!")
        return True

    except Exception as e:
        print(f"ERROR: Shape testing failed: {e}")
        return False


def benchmark_vs_pytorch():
    print("\nPerformance comparison across shapes...")

    test_configs = [
        ([128, 512], "2D - Small"),
        ([256, 1024], "2D - Medium"),
        ([32, 128, 512], "3D - Transformer-like"),
        ([16, 32, 128, 512], "4D - Batch of sequences"),
    ]

    try:
        import fused_linear_gelu

        results = []

        for shape, desc in test_configs:
            input_dim = shape[-1]
            output_dim = input_dim  # square-ish for parity

            print(f"\n{desc}: {shape} -> {shape[:-1] + [output_dim]}")

            x = torch.randn(shape, device="cuda", dtype=torch.float32)
            w = torch.randn(output_dim, input_dim, device="cuda", dtype=torch.float32)
            b = torch.randn(output_dim, device="cuda", dtype=torch.float32)

            # Warmup
            for _ in range(10):
                _ = F.gelu(F.linear(x, w, b))
                _ = fused_linear_gelu.linear_gelu(x, w, b)
                _ = fused_linear_gelu.linear_gelu_cublas(x, w, b)
            torch.cuda.synchronize()

            iters = 100

            # PyTorch
            t0 = time.perf_counter()
            for _ in range(iters):
                y_ref = F.gelu(F.linear(x, w, b))
            torch.cuda.synchronize()
            t_pt = (time.perf_counter() - t0) / iters * 1000

            # Fused (single-kernel)
            t0 = time.perf_counter()
            for _ in range(iters):
                y_fused = fused_linear_gelu.linear_gelu(x, w, b)
            torch.cuda.synchronize()
            t_fused = (time.perf_counter() - t0) / iters * 1000

            # cuBLAS path
            t0 = time.perf_counter()
            for _ in range(iters):
                y_cublas = fused_linear_gelu.linear_gelu_cublas(x, w, b)
            torch.cuda.synchronize()
            t_cublas = (time.perf_counter() - t0) / iters * 1000

            sp_fused = t_pt / t_fused if t_fused > 0 else float("inf")
            sp_cublas = t_pt / t_cublas if t_cublas > 0 else float("inf")

            print(f"  PyTorch: {t_pt:.3f} ms")
            print(f"  Fused:   {t_fused:.3f} ms  (speedup vs PT: {sp_fused:.2f}x)")
            print(f"  cuBLAS:  {t_cublas:.3f} ms (speedup vs PT: {sp_cublas:.2f}x)")

            results.append((desc, sp_fused, sp_cublas, t_pt, t_fused, t_cublas))

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Test Case':<28} {'Speedup Fused':<15} {'Speedup cuBLAS':<16} {'PyTorch (ms)':<12} {'Fused (ms)':<11} {'cuBLAS (ms)':<11}")
        print("-" * 60)
        for desc, spf, spc, tpt, tfu, tcb in results:
            print(f"{desc:<28} {spf:<15.2f} {spc:<16.2f} {tpt:<12.3f} {tfu:<11.3f} {tcb:<11.3f}")

        avg_sp_fused = sum(r[1] for r in results) / len(results)
        avg_sp_cublas = sum(r[2] for r in results) / len(results)
        print(f"\nAverage speedup (Fused):  {avg_sp_fused:.2f}x")
        print(f"Average speedup (cuBLAS): {avg_sp_cublas:.2f}x")

        return True

    except Exception as e:
        print(f"ERROR: Benchmarking failed: {e}")
        return False


def main():
    print("Comprehensive Fused Linear+GELU Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    if test_shapes():
        benchmark_vs_pytorch()
    else:
        print("ERROR: Fix shape issues before benchmarking!")


if __name__ == "__main__":
    main()
