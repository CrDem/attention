import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


df = pd.read_csv("src/benchmarks/attention_benchmark.csv")

plt.figure(figsize=(8,5))
plt.plot(df["seq_len"], df["core_ms"], marker="o", label="Attention core")
plt.plot(df["seq_len"], df["full_ms"], marker="o", label="Full block")

plt.xticks(df["seq_len"])
seq_ticks = sorted(df["seq_len"].unique().tolist())
plt.xticks(seq_ticks, [str(x) for x in seq_ticks])
plt.xscale("log", base=2)

plt.xlabel("Sequence length")
plt.ylabel("Time (ms)")
plt.title("Multi-Head Attention Performance (RTX 2070S)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("src/benchmarks/perf_plot.png")
print("plot saved to src/benchmarks/perf_plot.png")
plt.show()