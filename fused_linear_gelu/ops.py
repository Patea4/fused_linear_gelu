import torch
from torch import Tensor

__all__ = ["mymuladd"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """a * b + c in a fused kernel"""
    return torch.ops.fused_linear_gelu.mymuladd(a,b,c)

@torch.library.register_fake("fused_linear_gelu::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

