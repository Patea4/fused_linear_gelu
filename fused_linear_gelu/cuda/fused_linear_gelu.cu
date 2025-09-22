#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <Python.h>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
  The import from Python will load the .so consisting of this file
  in this extension, so that the TORCH_LIBRARY static initializers
  below are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace fused_linear_gelu {

// __global__ void linear_gelu(const float *X, const float *W, const float
// *bias,
//                             float *Y, int B, int In, int Out) {
//
// };
//
// at::Tensor linear_gelu_cuda(const at::Tensor &input, const at::Tensor
// &weight,
//                             c10::optional<at::Tensor> bias) {}

__global__ void muladd_kernel(int n, const float *a, const float *b, float c,
                              float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = a[idx] * b[idx] + c;
  }
}

at::Tensor mymuladd_cuda(const at::Tensor &a, const at::Tensor &b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float *a_ptr = a_contig.data_ptr<float>();
  const float *b_ptr = b_contig.data_ptr<float>();
  float *result_ptr = result.data_ptr<float>();

  int n = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  muladd_kernel<<<(n + 255) / 256, 256, 0>>>(n, a_ptr, b_ptr, c, result_ptr);
  return result;
}

TORCH_LIBRARY(fused_linear_gelu, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(fused_linear_gelu, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
}

} // namespace fused_linear_gelu
