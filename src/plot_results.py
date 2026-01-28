import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("src/benchmarks/attention_benchmark.csv")

plt.figure(figsize=(8,5))
plt.plot(df["seq_len"], df["core_ms"], marker="o", alpha = 0.7, label="Attention core")
plt.plot(df["seq_len"], df["full_ms"], marker="o", label="Full block")
plt.plot(df["seq_len"], df["flash_ms"], marker="x", ls = '--', alpha = 0.7, label="Flash attention")
plt.plot(df["seq_len"], df["our_cuda_ms"], marker="s", alpha=0.8, label="Our CUDA (32x32)")

plt.xticks(df["seq_len"])
seq_ticks = sorted(df["seq_len"].unique().tolist())
plt.xticks(seq_ticks, [str(x) for x in seq_ticks])
plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Sequence length")
plt.ylabel("Time (ms)")
plt.title("Classic Multi-Head vs Flash Attention (RTX 2070S)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("src/benchmarks/perf_plot.png")
print("plot saved to src/benchmarks/perf_plot.png")
plt.show()

df2 = pd.read_csv("src/benchmarks/our_kernel_numwarps_sweep.csv")

plt.figure(figsize=(8, 5))
for nw in sorted(df2["num_warps"].unique()):
    sub = df2[df2["num_warps"] == nw]
    plt.plot(
        sub["seq_len"],
        sub["time_ms"],
        marker="o",
        label=f"numWarps={nw}"
    )

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Sequence length")
plt.ylabel("Time (ms)")
plt.title("Our FlashAttention kernel: numWarps sweep")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("src/benchmarks/our_kernel_numwarps_sweep.png")
plt.show()