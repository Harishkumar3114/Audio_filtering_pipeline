import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

out = Path("./Phase1_filter")
all_df   = pd.read_csv(out / "all_metrics.csv")
hard_rej = pd.read_csv(out / "hard_rejected.csv")

all_df["status"] = "passed"
soft_rej_mask = all_df["soft_score"] < 0.40
all_df.loc[soft_rej_mask, "status"] = "soft_rejected"

plot_dir = out / "Validation_plot"
plot_dir.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.40

fig, ax = plt.subplots(figsize=(10, 5))
for status, color in [("passed", "#55A868"), ("soft_rejected", "#C44E52")]:
    subset = all_df[all_df["status"] == status]["soft_score"]
    ax.hist(subset, bins=40, alpha=0.6, color=color, label=status, density=True)
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"threshold={THRESHOLD}")
ax.set_xlabel("Soft Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Score Distribution — Passed vs Soft Rejected", fontsize=12, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(plot_dir  / "plot1_score_distribution.png", dpi=150)
plt.close(fig)
print("Saved plot1")

SCORE_METRICS = ["snr_db", "vad_ratio", "c50_db", "spectral_flatness", "zcr", "kurtosis"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, metric in enumerate(SCORE_METRICS):
    ax = axes[i]
    data = [
        all_df[all_df["status"] == "passed"][metric].dropna().values,
        all_df[all_df["status"] == "soft_rejected"][metric].dropna().values,
    ]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#55A868"); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor("#C44E52"); bp["boxes"][1].set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Passed", "Rejected"], fontsize=9)
    ax.set_title(metric, fontsize=10, fontweight="bold")

fig.suptitle("Per-Metric Distribution — Passed vs Rejected",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(plot_dir / "plot2_metric_boxplot_pass_vs_reject.png", dpi=150)
plt.close(fig)
print("Saved plot2")

import numpy as np

langs = sorted(all_df["language"].unique())
passed_counts   = [len(all_df[(all_df["language"]==l) & (all_df["status"]=="passed")]) for l in langs]
soft_rej_counts = [len(all_df[(all_df["language"]==l) & (all_df["status"]=="soft_rejected")]) for l in langs]
hard_rej_counts = []
for l in langs:
    if "language" in hard_rej.columns:
        hard_rej_counts.append(len(hard_rej[hard_rej["language"]==l]))
    else:
        hard_rej_counts.append(0)

x = np.arange(len(langs))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, passed_counts,   label="Passed",        color="#55A868", alpha=0.85)
ax.bar(x, soft_rej_counts, bottom=passed_counts,   label="Soft rejected", color="#C44E52", alpha=0.85)
ax.bar(x, hard_rej_counts,
       bottom=[p+s for p,s in zip(passed_counts, soft_rej_counts)],
       label="Hard rejected", color="#937860", alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(langs, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("File count", fontsize=11)
ax.set_title("Per-Language Rejection Breakdown", fontsize=12, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(plot_dir  / "plot3_per_language_rejection.png", dpi=150)
plt.close(fig)
print("Saved plot3")


    
scores = np.sort(all_df["soft_score"].dropna().values)
cdf    = np.arange(1, len(scores)+1) / len(scores)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(scores, cdf, color="#4C72B0", linewidth=2)
ax.axvline(THRESHOLD, color="red", linestyle="--", linewidth=1.5,
           label=f"current threshold={THRESHOLD} → removes {(scores < THRESHOLD).mean():.1%}")
ax.fill_betweenx([0, 1], 0, THRESHOLD, alpha=0.08, color="red")
ax.set_xlabel("Soft Score", fontsize=11)
ax.set_ylabel("Cumulative fraction of files", fontsize=11)
ax.set_title("Soft Score CDF — Threshold Tuning Guide", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(plot_dir  / "plot4_score_cdf.png", dpi=150)
plt.close(fig)
print("Saved plot4")

import pandas as pd
df = pd.read_csv("./Phase1_filter/all_metrics.csv")

print(f"Rejected: {(df['soft_score'] < 0.50).mean():.1%}")

print(df.groupby("language").apply(
    lambda g: (g["soft_score"] < 0.50).mean()
).round(3).sort_values(ascending=False))
