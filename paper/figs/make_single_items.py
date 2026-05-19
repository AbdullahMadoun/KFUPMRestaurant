"""Build a small grid of single-item masked crops to illustrate why
segmentation matters: clean per-item crops feed downstream apps like
calorie / portion-size estimation. Pulls real crops from v3.2."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams.update({
    "font.size": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

V32 = Path("/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3.2_2026-04-27_c99ee47f")
ROOT = Path(__file__).resolve().parents[2]

# Hand-picked diverse classes — one good representative each
WANTED = [
    "rice", "roast chicken", "hummus (cling-wrapped)", "tabbouleh salad",
    "soup", "cake", "grilled fish", "lemon",
]


def main():
    # Find one is_reference=True crop per wanted class
    by_class: dict = {}
    with (V32 / "items.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            cls = r.get("class_display_name") or r.get("class_slug")
            if cls in WANTED and cls not in by_class and r.get("is_reference"):
                by_class[cls] = r
            if len(by_class) == len(WANTED):
                break

    chosen = [(c, by_class[c]) for c in WANTED if c in by_class]
    print(f"chose {len(chosen)} classes: {[c for c,_ in chosen]}")

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    axes = axes.flatten()
    for ax, (cls, r) in zip(axes, chosen):
        crop = Image.open(V32 / r["crop_path"]).convert("RGB")
        ax.imshow(crop)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(3); spine.set_edgecolor("#000")
        ax.set_title(cls, fontsize=10, fontweight="bold", pad=6)
    for ax in axes[len(chosen):]:
        ax.axis("off")

    pdf_out = Path(__file__).parent / "single_items.pdf"
    fig.savefig(pdf_out, bbox_inches="tight", dpi=180)
    print(f"wrote {pdf_out}")

    png_out = ROOT / "presentation" / "assets" / "single_items.png"
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, bbox_inches="tight", dpi=160, facecolor="white")
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
