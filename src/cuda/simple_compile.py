from torch.utils.cpp_extension import load
import os

build_dir = "./cuda_build"
os.makedirs(build_dir, exist_ok=True)

cuda_ext = load(
    name='flash_attention_cuda',
    sources=['flash_attention.cpp', 'kernel.cu'],
    extra_cflags=['/std:c++17'],
    extra_cuda_cflags=[
        '-arch=sm_75',
        '--use_fast_math',
        '-std=c++14',
    ],
    verbose=False,
    build_directory=build_dir
)