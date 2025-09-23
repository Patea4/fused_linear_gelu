import torch
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule, replace_pattern

import fused_linear_gelu

# --- patterns ---
def _pat_linear_gelu(x, w, b):
    return F.gelu(F.linear(x, w, b))

def _rep_linear_gelu_cublas(x, w, b):
    return torch.ops.fused_linear_gelu.linear_gelu_cublas(x, w, b)

def _rep_linear_gelu_fused(x, w, b):
    return torch.ops.fused_linear_gelu.linear_gelu(x, w, b)

# Some models lower nn.Linear to addmm. This pattern hits that too.
def _pat_addmm_gelu(x, w, b):
    return F.gelu(torch.addmm(b, x, w.t()))

def fuse_linear_gelu_fx(module: torch.nn.Module, prefer_cublas: bool = True) -> GraphModule:
    gm = symbolic_trace(module)
    rep = _rep_linear_gelu_cublas if prefer_cublas else _rep_linear_gelu_fused

    # Replace F.linear(...)->gelu(...)
    replace_pattern(gm, _pat_linear_gelu, rep)
    # Replace addmm(...)->gelu(...)
    replace_pattern(gm, _pat_addmm_gelu, rep)

    gm.recompile()
    return gm
