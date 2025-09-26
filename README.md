# Fused Linear + GELU PyTorch Extension

This project explores **operator fusion in PyTorch** by combining a `Linear` (matrix multiply + bias) and `GELU` activation into a **single CUDA kernel**.

The motivation: reduce kernel launch overhead, improve memory locality, and benchmark against PyTorch’s native implementation.

---

## Implementations

There are currently **two backends** in this repository:

1. **cuBLAS + single-pass epilogue**  
   - Uses cuBLAS for the matrix multiply.  
   - Adds bias + GELU activation in a single pass afterward.  

2. **Custom fused CUDA kernel (work in progress)**  
   - A hand-written CUDA kernel that performs matmul, bias, and GELU **with one kernel launch**.  
   - Currently optimizing performance for the matmul operation.
   - The goal is to compare performance against both the cuBLAS and PyTorch implementations.

---

## Installation

### Requirements
- CUDA 12+  
- PyTorch (built with CUDA support)  
- Python 3.9+  
- A GPU with compute capability 8.0+ (tested on RTX 4080-S)

### 1. Install Python requirements

The project provides both `pyproject.toml` and `uv.lock`, so you can use either **uv** or your preferred Python environment manager.

```
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```
### 2. Build the extension

From the project root:
```
python setup.py install
```

### 3. Use the fused operator

After installation, you can apply FX-based graph rewriting to substitute the default `Linear → GELU` pattern with the fused operator:

```
from fused_linear_gelu import fuse_linear_gelu_fx

fused_cublas = fuse_linear_gelu_fx(model, prefer_cublas=True)   # cuBLAS + epilogue
fused_single = fuse_linear_gelu_fx(model, prefer_cublas=False)  # custom fused kernel (WIP)
```

## References

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)  
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
- [Optimizing CUDA Matrix Multiplication (blog)](https://siboehm.com/articles/22/CUDA-MMM)  

---
