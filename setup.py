from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob



def get_extensions():
    debug_mode = os.getenv("DEBUG", 0) == "1"

    if debug_mode:
        print("Compiling in DEBUG mode")

    sources = list(glob.glob("fused_linear_gelu/cuda/*.cu"))
    if not sources:
        raise RuntimeError("No CUDA sources files found in fused_linear_gelu/cuda/")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": ["-O3" if not debug_mode else "-O0"],
    }

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    ext_modules = [
        CUDAExtension(
            name="fused_linear_gelu._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name="fused_linear_gelu",
    description="Fused linear + GELU Cuda kernel for PyTorch",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
