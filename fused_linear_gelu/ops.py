import torch
from torch import Tensor
from typing import Optional

__all__ = ["linear_gelu", "linear_gelu_cublas"]

def linear_gelu(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Fused Linear + GELU: GELU(input @ weight.T + bias)"""
    return torch.ops.fused_linear_gelu.linear_gelu(input, weight, bias)

def linear_gelu_cublas(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return torch.ops.fused_linear_gelu.linear_gelu_cublas(input, weight, bias)

@torch.library.register_fake("fused_linear_gelu::linear_gelu")
def _fake_linear_gelu(input, weight, bias):
    torch._check(input.dim() >= 1)
    torch._check(weight.dim() == 1 or weight.dim() == 2)
    torch._check(input.dtype == torch.float32)
    torch._check(weight.dtype == torch.float32)
    torch._check(input.device == weight.device)
    
    in_features = input.size(-1)
    if weight.dim() == 2:
        torch._check(in_features == weight.size(1))
        out_features = weight.size(0)
        output_shape = list(input.shape)
        output_shape[-1] = out_features
    else:
        torch._check(in_features == weight.size(0))
        output_shape = list(input.shape[:-1])  # Remove last dimension
    
    if bias is not None:
        torch._check(bias.dtype == torch.float32)
        torch._check(bias.device == input.device)
        if weight.dim() == 2:
            torch._check(bias.dim() == 1 and bias.size(0) == weight.size(0))
        else:
            torch._check(bias.dim() == 0)
    
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)

@torch.library.register_fake("fused_linear_gelu::linear_gelu_cublas")
def _fake_linear_gelu_cublas(input, weight, bias):
    # Same shape/dtype/device checks and output-shape logic as the fused version
    return _fake_linear_gelu(input, weight, bias)
