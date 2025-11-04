import torch
from attention import MultiHeadAttentionBlock

batch_size = 1
d_model = 512
num_heads = 16
seq_lens = [64, 128, 256, 512, 1024, 2048, 4096]
mask = None

attentionBlock = MultiHeadAttentionBlock(d_model, num_heads, dropout=0.0).cuda()

results = []
num_iters = 100

for seq_len in seq_lens:

    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model, device='cuda')
    key = torch.randn(batch_size, seq_len, d_model, device='cuda')
    value = torch.randn(batch_size, seq_len, d_model, device='cuda')

    startCore = torch.cuda.Event(enable_timing=True)
    endCore = torch.cuda.Event(enable_timing=True)
    startFull = torch.cuda.Event(enable_timing=True)
    endFull = torch.cuda.Event(enable_timing=True)

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

    coreAvgTimeMs = startCore.elapsed_time(endCore) / num_iters
    fullAvgTimeMs = startFull.elapsed_time(endFull) / num_iters
    print(f"seq len: {seq_len} tokens")
    print(f"attention core: {coreAvgTimeMs:.3f} ms")
    print(f"full block: {fullAvgTimeMs:.3f} ms\n")

    results.append((seq_len, coreAvgTimeMs, fullAvgTimeMs))

#saving results
import csv

with open("src/benchmarks/attention_benchmark.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq_len", "core_ms", "full_ms"])
    writer.writerows(results)
print("results saved to src/benchmarks/attention_benchmark.csv")