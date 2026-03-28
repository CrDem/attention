import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("src/benchmarks/attention_benchmark.csv")

plt.figure(figsize=(8,5))
plt.plot(df["seq_len"], df["core_ms"], marker="o", alpha = 0.7, label="Attention core")
plt.plot(df["seq_len"], df["full_ms"], marker="o", label="Full block")
plt.plot(df["seq_len"], df["flash_ms"], marker="x", ls = '--', alpha = 0.7, label="Flash attention")
plt.plot(df["seq_len"], df["our_cuda_ms"], marker="s", alpha=0.8, label="Our CUDA kernel")

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

import seaborn as sns

df2 = pd.read_csv("src/benchmarks/our_kernel_sweep.csv")

seq_lens = sorted(df2["seq_len"].unique())

for seq in seq_lens:
    if seq == 64 or seq == 247 or seq == 256 or seq == 512: continue
    sub = df2[df2["seq_len"] == seq]

    pivot = sub.pivot(index="Br", columns="Bc", values="time_ms")

    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f")

    plt.title(f"Our kernel heatmap (seq_len={seq})")
    plt.xlabel("Bc")
    plt.ylabel("Br")

    plt.tight_layout()
    plt.savefig(f"src/benchmarks/heatmap_seq_{seq}.png")

plt.show()

plt.figure(figsize=(8, 5))

Br_values = sorted(df2["Br"].unique())

for br in Br_values:
    sub_br = df2[df2["Br"] == br]

    # усреднять не надо — у тебя уникальные точки
    for bc in sorted(sub_br["Bc"].unique()):
        sub = sub_br[sub_br["Bc"] == bc]

        plt.plot(
            sub["seq_len"],
            sub["time_ms"],
            marker="o",
            label=f"Br={br}, Bc={bc}",
            alpha=0.7
        )

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Sequence length")
plt.ylabel("Time (ms)")
plt.title("Our kernel: Br/Bc sweep (line plot)")

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig("src/benchmarks/our_kernel_Br_Bc_lines.png")
plt.show()