from torch.utils.cpp_extension import load
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "cuda_build")
os.makedirs(build_dir, exist_ok=True)

cpp_dir = os.path.join(current_dir, "flash_attention.cpp")
cu_dir = os.path.join(current_dir, "kernel.cu")

try:
    cuda_ext = load(
        name='flash_attention_cuda',
        sources=[cpp_dir, cu_dir],
        extra_cflags=['/std:c++17'],
        extra_cuda_cflags=[
            '-arch=sm_75',
            '--use_fast_math',
            '-std=c++14',
            '--ptxas-options=-v',
        ],
        verbose=True,
        build_directory=build_dir
    )
    print(f"CUDA extension built successfully to {build_dir}")

except Exception as e:
    print(f"\n ERROR: Failed to build CUDA extension")
    print(f"Build directory: {build_dir}")
    print(f"Source files: {os.path.basename(cpp_dir)}, {os.path.basename(cu_dir)}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    sys.exit(1)