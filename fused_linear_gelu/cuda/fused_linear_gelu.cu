#include "c10/core/DeviceType.h"
#include "c10/cuda/CUDAException.h"
#include "c10/util/Exception.h"
#include "c10/util/MaybeOwned.h"
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

__device__ float gelu_activation_approx(float x) {
    const float sqrt2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float a = 0.044715f;

    float x_cubed = x * x * x;
    float inner = sqrt2_over_pi * (x + a * x_cubed);
    float tanh_inner = tanhf(inner);

    return x * 0.5f * (1.0 + tanh_inner);
}

__device__ inline float gelu_exact(float x) {
    const float inv_sqrt2 = 0.70710678118f; // 1/sqrt(2)
    return 0.5 * x * (1.f + erff(x * inv_sqrt2));
}

__device__ float gemm_core_naive(const float *__restrict__ A,
                                 const float *__restrict__ B,
                                 int M,
                                 int K,
                                 int N,
                                 int x,
                                 int y) {
    float acc = 0.f;
    for (int i = 0; i < K; ++i) {
        acc = fmaf(A[x * K + i], B[i * N + y], acc);
    }
    return acc;
}

__global__ void linear_gelu(const float *__restrict__ X,
                            const float *__restrict__ W,
                            const float *__restrict__ bias,
                            float *__restrict__ Y,
                            int total_batch_size,
                            int in_features,
                            int out_features) {

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= total_batch_size || output_idx >= out_features)
        return;

    float acc = gemm_core_naive(X, W, total_batch_size, in_features,
                                out_features, batch_idx, output_idx);

    if (bias) {
        acc += bias[output_idx];
    }
    Y[batch_idx * out_features + output_idx] = gelu_exact(acc);
};

int compute_batch_size(const at::Tensor &tensor) {
    int batch_size = 1;
    for (int i = 0; i < tensor.dim() - 1; i++) {
        batch_size *= tensor.size(i);
    }
    return batch_size;
}

at::Tensor linear_gelu_cuda(const at::Tensor &input,
                            const at::Tensor &weight,
                            c10::optional<at::Tensor> bias_opt) {

    auto bias = bias_opt.has_value()
                    ? c10::MaybeOwned<at::Tensor>::borrowed(*bias_opt)
                    : c10::MaybeOwned<at::Tensor>::owned(at::Tensor{});

    //  Basic Dimension Checks
    TORCH_CHECK(input.dim() >= 1);
    TORCH_CHECK(weight.dim() == 1 || weight.dim() == 2);

    // Device Checks
    TORCH_CHECK(input.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(weight.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(input.device() == weight.device());

    if (bias->defined()) {
        TORCH_CHECK(bias->device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(input.device() == bias->device());
    }

    // Type Checks
    TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "Weight must be float32");

    if (bias->defined()) {
        TORCH_CHECK(bias->dtype() == torch::kFloat, "Bias must be float32");
    }

    // Compute Dimensions
    int in_features = input.size(-1);
    bool weight_is_matrix = (weight.dim() == 2);
    int out_features = weight_is_matrix ? weight.size(0) : 1;
    int weight_in_features = weight_is_matrix ? weight.size(1) : weight.size(0);
    int total_batch_size = compute_batch_size(input);

    // Shape Compatibility Checks
    TORCH_CHECK(in_features == weight_in_features);

    if (bias->defined()) {
        if (weight_is_matrix) {
            TORCH_CHECK(bias->dim() == 1);
            TORCH_CHECK(bias->size(0) == out_features);
        } else {
            TORCH_CHECK(bias->dim() == 0);
        }
    }

    TORCH_CHECK(in_features > 0 && out_features > 0);

    auto output_shape = input.sizes().vec();
    if (weight_is_matrix) {
        output_shape.back() = out_features;
    } else {
        output_shape.pop_back();
    }

    at::Tensor output = at::empty(output_shape, input.options());

    at::Tensor input_contig = input.contiguous();
    at::Tensor weight_contig = weight.contiguous();

    at::Tensor bias_contig;
    if (bias->defined()) {
        bias_contig = bias->contiguous();
    }

    at::Tensor weight_transpose;
    if (weight_is_matrix) {
        weight_transpose =
            weight_contig.transpose(0, 1).contiguous(); // [K, N] row-major
    } else {
        weight_transpose = weight_contig;
    }

    at::Tensor input_2d = input_contig.view({total_batch_size, in_features});
    at::Tensor output_2d;
    if (weight_is_matrix) {
        output_2d = output.view({total_batch_size, out_features});
    } else {
        output_2d = output.view({total_batch_size, 1});
    }

    const int block_size_x = 32;
    const int block_size_y = weight_is_matrix ? 8 : 1;

    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((total_batch_size + block_size_x - 1) / block_size_x,
                   (out_features + block_size_y - 1) / block_size_y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    linear_gelu<<<grid_size, block_size, 0, stream>>>(
        input_2d.data_ptr<float>(), weight_transpose.data_ptr<float>(),
        bias->defined() ? bias_contig.data_ptr<float>() : nullptr,
        output_2d.data_ptr<float>(), total_batch_size, in_features,
        out_features);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

__global__ void epilogue_bias_gelu(const float *__restrict__ Z,
                                   const float *__restrict__ bias,
                                   float *__restrict__ Y,
                                   int M,
                                   int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
        return;

    float v = Z[m * N + n];
    if (bias)
        v += bias[n];
    const float inv_sqrt2 = 0.7071067811865475f;
    float out = 0.5f * v * (1.f + erff(v * inv_sqrt2));
    Y[m * N + n] = out;
}

at::Tensor linear_gelu_cublas_cuda(const at::Tensor &input,
                                   const at::Tensor &weight,
                                   c10::optional<at::Tensor> bias_opt) {
    auto bias = bias_opt.has_value()
                    ? c10::MaybeOwned<at::Tensor>::borrowed(*bias_opt)
                    : c10::MaybeOwned<at::Tensor>::owned(at::Tensor{});

    TORCH_CHECK(input.dim() >= 1);
    TORCH_CHECK(weight.dim() == 2);
    TORCH_CHECK(input.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(weight.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(input.device() == weight.device());
    if (bias->defined()) {
        TORCH_CHECK(bias->device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(input.device() == bias->device());
    }
    TORCH_CHECK(input.dtype() == torch::kFloat);
    TORCH_CHECK(weight.dtype() == torch::kFloat);
    if (bias->defined()) {
        TORCH_CHECK(bias->dtype() == torch::kFloat);
    }

    int64_t in_features = input.size(-1);
    bool weight_is_matrix = (weight.dim() == 2);
    int64_t out_features = weight_is_matrix ? weight.size(0) : 1;
    int64_t weight_in_features =
        weight_is_matrix ? weight.size(1) : weight.size(0);
    TORCH_CHECK(in_features == weight_in_features);
    if (bias->defined()) {
        TORCH_CHECK(bias->dim() == 1);
        TORCH_CHECK(bias->size(0) == out_features);
    }
    TORCH_CHECK(in_features > 0 && out_features > 0);

    auto compute_batch_size = [](const at::Tensor &t) -> int64_t {
        int64_t b = 1;
        for (int i = 0; i < t.dim() - 1; ++i)
            b *= t.size(i);
        return b;
    };
    int64_t total_batch_size = compute_batch_size(input);
    int64_t M = total_batch_size;
    int64_t K = in_features;
    int64_t N = out_features;

    auto output_shape = input.sizes().vec();
    if (weight_is_matrix) {
        output_shape.back() = out_features;
    } else {
        output_shape.pop_back();
    }
    at::Tensor output = at::empty(output_shape, input.options());

    at::Tensor input_contig = input.contiguous();
    at::Tensor weight_contig = weight.contiguous();

    at::Tensor bias_contig;
    if (bias->defined()) {
        bias_contig = bias->contiguous();
    }

    at::Tensor input_2d = input_contig.view({M, K});
    at::Tensor output_2d = output.view({M, N});

    at::mm_out(output_2d, input_2d, weight_contig.transpose(0, 1));

    const float *bias_ptr =
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 block(16, 16), grid((int((N + 15) / 16)), int((M + 15) / 16));
    epilogue_bias_gelu<<<grid, block, 0, stream>>>(
        output_2d.data_ptr<float>(), bias_ptr, output_2d.data_ptr<float>(),
        (int)M, (int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

TORCH_LIBRARY(fused_linear_gelu, m) {
    m.def("linear_gelu(Tensor input, Tensor weight, Tensor? bias=None) -> "
          "Tensor");
    m.def("linear_gelu_cublas(Tensor input, Tensor weight, Tensor? bias=None) "
          "-> Tensor");
}

TORCH_LIBRARY_IMPL(fused_linear_gelu, CUDA, m) {
    m.impl("linear_gelu", &linear_gelu_cuda);
    m.impl("linear_gelu_cublas", &linear_gelu_cublas_cuda);
}

} // namespace fused_linear_gelu
