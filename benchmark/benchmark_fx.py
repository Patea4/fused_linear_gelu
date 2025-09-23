import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fx_pass import fuse_linear_gelu_fx

# --------------------------
# Test models
# --------------------------
class TinyMLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024, depth=4):
        super().__init__()
        layers = []
        din = d_in
        for _ in range(depth):
            layers += [nn.Linear(din, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out), nn.GELU()]
            din = d_out
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: [B, d_in]
        return self.net(x)

class ToyTransformerFFN(nn.Module):
    def __init__(self, d_model=1024, d_ff=4096, depth=6):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks += [nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.GELU()]
        self.ffn = nn.Sequential(*blocks)
    def forward(self, x):  # x: [B, T, d_model]
        b, t, d = x.shape
        y = self.ffn(x.view(b*t, d)).view(b, t, d)
        return y

# --------------------------
# Benchmark utils
# --------------------------
def cuda_time_ms(fn, iters=100, warmup=20):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    # measure
    starter.record()
    for _ in range(iters):
        _ = fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters

@dataclass
class BenchConfig:
    shapes: List[Tuple[int, ...]]
    iters: int = 100
    warmup: int = 20

def make_inputs(shapes: List[Tuple[int, ...]], dtype=torch.float32):
    xs = []
    for s in shapes:
        xs.append(torch.randn(*s, device="cuda", dtype=dtype))
    return xs

def validate(ref, out, name, rtol=1e-4, atol=1e-5):
    ok = torch.allclose(ref, out, rtol=rtol, atol=atol)
    md = (ref - out).abs().max().item()
    return ok, md

# --------------------------
# End-to-end benchmark
# --------------------------
def run_benchmark(
    model_name: str = "mlp",
    prefer_cublas: bool = True,
    shapes: List[Tuple[int, ...]] = None,
    iters: int = 100,
    warmup: int = 20,
    d_in: int = 1024,
    d_hidden: int = 4096,
    d_out: int = 1024,
    depth: int = 4,
    d_model: int = 1024,
    d_ff: int = 4096,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if shapes is None:
        if model_name == "mlp":
            # batch sizes to sweep
            shapes = [(32, d_in), (128, d_in), (512, d_in), (2048, d_in)]
        else:
            # (B, T, d_model)
            shapes = [(8, 128, d_model), (16, 256, d_model), (32, 128, d_model)]

    if model_name == "mlp":
        base = TinyMLP(d_in=d_in, d_hidden=d_hidden, d_out=d_out, depth=depth).eval().cuda()
        ex_shapes = shapes
    elif model_name == "ffn":
        base = ToyTransformerFFN(d_model=d_model, d_ff=d_ff, depth=depth).eval().cuda()
        ex_shapes = shapes
    else:
        raise ValueError("model_name must be 'mlp' or 'ffn'")

    fused_cublas = fuse_linear_gelu_fx(base, prefer_cublas=True)
    fused_single = fuse_linear_gelu_fx(base, prefer_cublas=False)

    rows = []
    for shape in ex_shapes:
        xs = make_inputs([shape])
        x = xs[0]

        with torch.no_grad():
            y_ref = base(x)

            y_cublas = fused_cublas(x)
            ok_cu, md_cu = validate(y_ref, y_cublas, "cuBLAS")

            y_fused = fused_single(x)
            ok_fu, md_fu = validate(y_ref, y_fused, "Fused")

        def run_ref():   # noqa: E306
            with torch.no_grad():
                return base(x)
        def run_cublas():
            with torch.no_grad():
                return fused_cublas(x)
        def run_fused():
            with torch.no_grad():
                return fused_single(x)

        t_ref   = cuda_time_ms(run_ref, iters=iters, warmup=warmup)
        t_cu    = cuda_time_ms(run_cublas, iters=iters, warmup=warmup)
        t_fused = cuda_time_ms(run_fused, iters=iters, warmup=warmup)

        rows.append({
            "shape": shape,
            "t_ref": t_ref,
            "t_cu": t_cu,
            "t_fused": t_fused,
            "sp_cu": t_ref / t_cu if t_cu > 0 else float("inf"),
            "sp_fused": t_ref / t_fused if t_fused > 0 else float("inf"),
            "ok_cu": ok_cu, "md_cu": md_cu,
            "ok_fused": ok_fu, "md_fused": md_fu,
        })

    print("\n============================================================")
    print(f"FX BENCH — {model_name.upper()}")
    print("============================================================")
    print(f"{'Shape':<18} {'PT (ms)':>9} {'cuBLAS (ms)':>12} {'Fused (ms)':>12} "
          f"{'Spd cuBLAS':>11} {'Spd Fused':>10} {'OK cuBLAS':>10} {'MaxDiff cu':>12} {'OK fused':>10} {'MaxDiff fu':>12}")
    print("-" * 120)
    sp_cu_avg = sp_fu_avg = 0.0
    for r in rows:
        print(f"{str(r['shape']):<18} {r['t_ref']:>9.3f} {r['t_cu']:>12.3f} {r['t_fused']:>12.3f} "
              f"{r['sp_cu']:>11.2f} {r['sp_fused']:>10.2f} {str(r['ok_cu']):>10} {r['md_cu']:>12.2e} {str(r['ok_fused']):>10} {r['md_fused']:>12.2e}")
        sp_cu_avg += r['sp_cu']; sp_fu_avg += r['sp_fused']
    sp_cu_avg /= len(rows); sp_fu_avg /= len(rows)
    print("-" * 120)
    print(f"{'AVERAGE':<18} {'':>9} {'':>12} {'':>12} {sp_cu_avg:>11.2f} {sp_fu_avg:>10.2f}")
    print("============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "ffn"], default="mlp")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--d_in", type=int, default=1024)
    parser.add_argument("--d_hidden", type=int, default=4096)
    parser.add_argument("--d_out", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    run_benchmark(
        model_name=args.model,
        iters=args.iters,
        warmup=args.warmup,
        d_in=args.d_in,
        d_hidden=args.d_hidden,
        d_out=args.d_out,
        depth=args.depth,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )
