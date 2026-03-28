import sys
import os
import math

import torch
from attention import MultiHeadAttentionBlock
from flash_attn.flash_attn_interface import flash_attn_func

import random
import numpy as np

def set_seed(seed=42):
    """Фиксирует seed для воспроизводимости результатов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

IS_BENCH = "-bench" in sys.argv
IS_DEBUG = "-debug" in sys.argv

# cuda_build folder path
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_build_dir = os.path.join(current_dir, 'cuda', 'cuda_build')

# adding cuda_build in sys.path
if cuda_build_dir not in sys.path:
    sys.path.insert(0, cuda_build_dir)
    print(f"Added to path: {cuda_build_dir}")

from flash_attention_cuda import forward

batch_size = 1
d_model = 64
num_heads = 1
head_dim = d_model // num_heads
seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192,] if not IS_DEBUG else [8192]
'''247,   8192*2'''
mask = None

attentionBlock = MultiHeadAttentionBlock(d_model, num_heads, dropout=0.0).cuda().float()

results = []
num_iters = 1000 if IS_BENCH else 1 if IS_DEBUG else 50
num_warms = 100 if IS_BENCH else 0 if IS_DEBUG else 5
Br = 64 if IS_BENCH else 64
Bc = 64 if IS_BENCH else 64
# Br sweep (powers of two, starting from 16)
Br_list = [16, 32, 64]
Bc_list = [16, 32, 64]
our_kernel_sweep_results = []  # (seq_len, Br, Bc, time_ms)

for seq_len in seq_lens:
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float32)

    # Create qkv tensor for flash_attn_func
    q = query.view(batch_size * seq_len, num_heads, head_dim).half()
    k = key.view(batch_size * seq_len, num_heads, head_dim).half()
    v = value.view(batch_size * seq_len, num_heads, head_dim).half()
    qkv = torch.stack([q, k, v], dim=1)  # (total_q, 3, num_heads, head_dim)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len,
                          dtype=torch.int32, device='cuda')

    # our kernel: [batch, heads, seq_len, d_k]
    q_cuda = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous().half()
    k_cuda = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous().half()
    v_cuda = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous().half()
    
    # size check
    print(f"q_cuda shape: {q_cuda.shape}")  # need to be (batch, num_heads, seq_len, head_dim)
    
    # our kernel check
    if (not IS_DEBUG):
        try:
            output = forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=Br, Bc=Bc)
            print(f"Output contains nan: {torch.isnan(output).any()}")
            print(f"Output contains inf: {torch.isinf(output).any()}")
        except Exception as e:
            print(f"Error during forward call: {e}")

    def torch_reference_attention(q, k, v, scale=1.0):
        qf = q.float()
        kf = k.float()
        vf = v.float()

        batch, heads, seq_len, d_k = qf.shape
        scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, vf)
        return output.half()

    output_cuda = forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=Br, Bc=Bc)
    output_ref = torch_reference_attention(q_cuda, k_cuda, v_cuda)

    # --- diff нашего ядра ---
    diff_cuda = (output_cuda.float() - output_ref.float()).abs()
    print(f"Our kernel max diff vs ref: {diff_cuda.max().item():.9f}")

    # --- FLASH ATTN OUTPUT ---
    output_flash = flash_attn_func(
        qkv,
        cu_seqlens,
        0.0,
        seq_len,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=False
    )

    # flash_attn_func возвращает (total_q, num_heads, head_dim)
    # приводим к (batch, heads, seq_len, head_dim)
    output_flash = output_flash.view(batch_size, seq_len, num_heads, head_dim) \
                            .transpose(1, 2) \
                            .contiguous()

    # --- diff flash attn ---
    diff_flash = (output_flash.float() - output_ref.float()).abs()
    print(f"Torch max diff vs ref: {diff_flash.max().item():.9f}")

    rel_diff_cuda = (
        (output_cuda.float() - output_ref.float()).abs() /
        (output_ref.float().abs() + 1e-6)
    )
    rel_diff_flash = (
        (output_flash.float() - output_ref.float()).abs() /
        (output_ref.float().abs() + 1e-6)
    )
    print(f"Our mean relative diff: {rel_diff_cuda.mean().item():.9f}")
    print(f"Torch mean relative diff: {rel_diff_flash.mean().item():.9f}")
    
    startCore = torch.cuda.Event(enable_timing=True)
    endCore = torch.cuda.Event(enable_timing=True)
    startFull = torch.cuda.Event(enable_timing=True)
    endFull = torch.cuda.Event(enable_timing=True)
    startFlash = torch.cuda.Event(enable_timing=True)
    endFlash = torch.cuda.Event(enable_timing=True)
    startOurCuda = torch.cuda.Event(enable_timing=True)
    endOurCuda = torch.cuda.Event(enable_timing=True)
    if (IS_DEBUG):
        sys.exit(0)
    # GPU warm up
    for _ in range(num_warms):
        with torch.no_grad():
            attentionBlock.forward(query, key, value, mask)
    torch.cuda.synchronize()

    # attention core perf measure
    startCore.record()
    for _ in range(num_iters):
        attentionBlock.attention(query, key, value, mask, None)
    endCore.record()
    torch.cuda.synchronize()

    # full block perf measure
    startFull.record()
    for _ in range(num_iters):
        attentionBlock.forward(query, key, value, mask)
    endFull.record()
    torch.cuda.synchronize()

    # built-in flash attention measure (torch)
    startFlash.record()
    for _ in range(num_iters):
        flash_attn_func(qkv, cu_seqlens, 0.0, seq_len, softmax_scale=1.0/math.sqrt(head_dim), causal=False)
    endFlash.record()
    torch.cuda.synchronize()

    # our CUDA kernel
    # in C++ wrapper: actual_scale = scale / sqrtf(d_k)
    
    # warm-up
    for _ in range(num_warms):
        with torch.no_grad():
            forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    
    startOurCuda.record()
    for _ in range(num_iters):
        forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=Br, Bc=Bc)
    endOurCuda.record()
    torch.cuda.synchronize()

    coreAvgTimeMs =  startCore.elapsed_time(endCore) / num_iters
    fullAvgTimeMs =  startFull.elapsed_time(endFull) / num_iters
    flashAvgTimeMs = startFlash.elapsed_time(endFlash) / num_iters
    our_cuda_time =  startOurCuda.elapsed_time(endOurCuda) / num_iters
    
    print(f"seq len: {seq_len} tokens")
    print(f"attention core: {coreAvgTimeMs:.3f} ms")
    print(f"full block: {fullAvgTimeMs:.3f} ms")
    print(f"flash attention (built-in): {flashAvgTimeMs:.3f} ms")
    print(f"our CUDA kernel: {our_cuda_time:.3f} ms")
    print(f"Speedup vs built-in flash: {flashAvgTimeMs/our_cuda_time:.4f}x")
    print()

    results.append((seq_len, coreAvgTimeMs, fullAvgTimeMs, flashAvgTimeMs, our_cuda_time))

    for br_now in Br_list:
        for bc_now in Bc_list:
            # warm-up
            for _ in range(num_warms):
                with torch.no_grad():
                    forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=br_now, Bc=bc_now)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(num_iters):
                forward(q_cuda, k_cuda, v_cuda, scale=1.0, Br=br_now, Bc=bc_now)
            end.record()
            torch.cuda.synchronize()

            avg_ms = start.elapsed_time(end) / num_iters

            our_kernel_sweep_results.append((seq_len, br_now, bc_now, avg_ms))

# saving results
import csv

benchmarks_dir = os.path.join(current_dir, 'benchmarks')
os.makedirs(benchmarks_dir, exist_ok=True)

csv_path = os.path.join(benchmarks_dir, 'attention_benchmark.csv')
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq_len", "core_ms", "full_ms", "flash_ms", "our_cuda_ms"])
    writer.writerows(results)
print(f"results saved to {csv_path}")

csv_path_sweep = os.path.join(benchmarks_dir, 'our_kernel_sweep.csv')
with open(csv_path_sweep, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq_len", "Br", "Bc", "time_ms"])
    writer.writerows(our_kernel_sweep_results)

print(f"Sweep results saved to {csv_path_sweep}")