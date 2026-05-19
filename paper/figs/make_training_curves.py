"""Training-curves and cardinality-regression figure for §5.
Uses the eval metrics from epoch_001 of the in-progress distillation run."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CKPT = Path("/Volumes/KINGSTON/vast_stage1_checkpoints_NEWRUN")
metrics = json.loads((CKPT / "best/metrics.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

# Left: count-error breakdown by gt cardinality
buckets = ["1–2 items", "3+ items"]
exact = [metrics["count_0_1/exact_count_accuracy"] * 100, metrics["count_2_3/exact_count_accuracy"] * 100]
mae = [metrics["count_0_1/count_mae"], metrics["count_2_3/count_mae"]]
x = range(len(buckets))
b1 = ax1.bar([i - 0.18 for i in x], exact, width=0.36, color="#2c5985", label="exact-count acc (%)")
ax1.set_ylim(0, 100)
ax1.set_ylabel("exact count accuracy (%)", color="#2c5985")
ax1.set_xticks(list(x))
ax1.set_xticklabels(buckets)
ax1.set_title("(a) Cardinality regression by GT count")
ax2_twin = ax1.twinx()
b2 = ax2_twin.bar([i + 0.18 for i in x], mae, width=0.36, color="#c2522e", label="count MAE")
ax2_twin.set_ylabel("count MAE (items)", color="#c2522e")
ax2_twin.set_ylim(0, max(mae) * 1.4)
ax2_twin.spines["top"].set_visible(False)
for ax in [ax1, ax2_twin]:
    ax.spines["right"].set_visible(False) if ax is ax1 else None

# Right: over- vs under-detection rate
labels = ["overcount", "exact", "undercount"]
oc = metrics["overcount_rate"] * 100
uc = metrics["undercount_rate"] * 100
ec = (1 - oc / 100 - uc / 100) * 100
vals = [oc, ec, uc]
colors = ["#c2522e", "#2c5985", "#7d8a99"]
ax2.bar(labels, vals, color=colors)
ax2.set_ylabel("share of dev images (%)")
ax2.set_ylim(0, 80)
for i, v in enumerate(vals):
    ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=7)
ax2.set_title(f"(b) Direction of count error  (Δn = +{metrics['count_bias']:.2f})")

plt.tight_layout()
out = Path(__file__).parent / "training_curves.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
print(f"Numbers: exact={exact}, MAE={mae}, overcount={oc:.1f}%, undercount={uc:.1f}%, bias=+{metrics['count_bias']:.2f}")
