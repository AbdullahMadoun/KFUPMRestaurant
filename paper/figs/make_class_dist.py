"""Generate the class-distribution figure for §3 (Data Engine)."""
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

V32 = Path("/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3.2_2026-04-27_c99ee47f")
classes = json.loads((V32 / "classes.json").read_text())
classes.sort(key=lambda c: c["size"], reverse=True)
names = [c["display_name"] for c in classes]
sizes = [c["size"] for c in classes]

fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(range(len(sizes)), sizes, color="#2c5985", width=0.85)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("items in v3.2")
ax.set_yscale("log")
ax.set_ylim(0.5, max(sizes) * 1.2)
for i, (b, n) in enumerate(zip(bars, sizes)):
    if i < 3 or n < 10:
        ax.text(b.get_x() + b.get_width()/2, n * 1.15, str(n), ha="center", va="bottom", fontsize=6)
plt.tight_layout()
out = Path(__file__).parent / "class_distribution.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
