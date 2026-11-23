import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/comparison/metrics_summary.json", encoding="utf-8-sig") as f:
    results = json.load(f)

metrics = ["accuracy", "precision_macro", "recall_macro", "macro_f1", "avg_latency_ms"]

labels = ["Accuracy", "Precision", "Recall", "Macro F1", "Latency (ms)"]

a = [results["Model A"][m] for m in metrics]
b = [results["Model B"][m] for m in metrics]

# Separate scale for latency so the other metrics remain readable
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(4)
width = 0.35

axes[0].bar(x - width/2, a[:4], width, label="Model A")
axes[0].bar(x + width/2, b[:4], width, label="Model B")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels[:4])
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Classification Metrics")
axes[0].legend()

axes[1].bar(["Model A", "Model B"], [a[4], b[4]])
axes[1].set_ylabel("Milliseconds")
axes[1].set_title("Average Inference Latency")

plt.suptitle("Champion-Challenger Model Comparison")
plt.tight_layout()

plt.savefig("outputs/comparison/metric_comparison.png", dpi=300)
plt.close()

print("Comparison chart regenerated successfully!")
