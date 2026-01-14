import sys
import os
import math

import torch
from attention import MultiHeadAttentionBlock
from flash_attn.flash_attn_interface import flash_attn_func

# cuda_build folder path
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_build_dir = os.path.join(current_dir, 'cuda', 'cuda_build')

# adding cuda_build in sys.path
if cuda_build_dir not in sys.path:
    sys.path.insert(0, cuda_build_dir)
    print(f"Added to path: {cuda_build_dir}")

from flash_attention_cuda import qk_mul

batch_size = 1
d_model = 512
num_heads = 16
head_dim = d_model // num_heads
seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
mask = None

attentionBlock = MultiHeadAttentionBlock(d_model, num_heads, dropout=0.0).cuda().half()

results = []
num_iters = 100

for seq_len in seq_lens:
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)
    key = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)
    value = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)

    # Create qkv tensor for flash_attn_func
    q = query.view(batch_size * seq_len, num_heads, head_dim)
    k = key.view(batch_size * seq_len, num_heads, head_dim)
    v = value.view(batch_size * seq_len, num_heads, head_dim)
    qkv = torch.stack([q, k, v], dim=1)  # (total_q, 3, num_heads, head_dim)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len,
                          dtype=torch.int32, device='cuda')

    # Create qkv tensor for our CUDA kernel (batch, heads, seq_len, d_k)
    q_cuda = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    k_cuda = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    
    startCore = torch.cuda.Event(enable_timing=True)
    endCore = torch.cuda.Event(enable_timing=True)
    startFull = torch.cuda.Event(enable_timing=True)
    endFull = torch.cuda.Event(enable_timing=True)
    startFlash = torch.cuda.Event(enable_timing=True)
    endFlash = torch.cuda.Event(enable_timing=True)
    startOurCuda = torch.cuda.Event(enable_timing=True)
    endOurCuda = torch.cuda.Event(enable_timing=True)

    # GPU warm up
    for _ in range(5):
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
        flash_attn_func(qkv, cu_seqlens, 0.0, seq_len, softmax_scale=None, causal=False)
    endFlash.record()
    torch.cuda.synchronize()

    # our CUDA kernel
    # convert to float32
    q_cuda_f32 = q_cuda.float()
    k_cuda_f32 = k_cuda.float()
    scale = 1.0 / math.sqrt(head_dim)
    
    # warm-up
    for _ in range(5):
        with torch.no_grad():
            qk_mul(q_cuda_f32, k_cuda_f32, scale)
    torch.cuda.synchronize()
    
    startOurCuda.record()
    for _ in range(num_iters):
        qk_mul(q_cuda_f32, k_cuda_f32, scale)
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
    print(f"our CUDA kernel (Q*K only): {our_cuda_time:.3f} ms")
    print()

    results.append((seq_len, coreAvgTimeMs, fullAvgTimeMs, flashAvgTimeMs, our_cuda_time))

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