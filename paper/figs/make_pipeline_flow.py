"""Build a single pipeline-flow figure showing one tray going through the
data engine: ORIGINAL  →  +BBOXES (Qwen-72B)  →  +MASKS (SAM3)  →  CROPS (DINOv2 → 32-class)

Outputs:
  - paper/figs/pipeline_flow.pdf      (for LaTeX inclusion)
  - presentation/assets/flow.png      (for the deck)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update({
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

V32 = Path("/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3.2_2026-04-27_c99ee47f")
SRC_ID = "frame_frame_039825_00"  # 4 items: rice, roast chicken, rice, lemon

PALETTE = {
    "rice": "#FF6B6B",
    "roast chicken": "#FFD93D",
    "lemon": "#A7F3D0",
    "chicken/meat koftas": "#C4B5FD",
    "mixed salad": "#93C5FD",
    "soup": "#F9A8D4",
}


def load_items(src_id: str):
    rows = []
    with (V32 / "items.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["src_image_id"] == src_id:
                rows.append(r)
    return rows


def get_scale(item) -> tuple[float, float]:
    """sx, sy from mask-native space → image-on-disk space."""
    img = Image.open(V32 / item["image_path"])
    msk = Image.open(V32 / item["mask_path"])
    return img.size[0] / msk.size[0], img.size[1] / msk.size[1]


def main():
    items = load_items(SRC_ID)
    if not items:
        print(f"no items for {SRC_ID}")
        sys.exit(1)
    print(f"loaded {len(items)} items: {[i['class_display_name'] for i in items]}")

    img = Image.open(V32 / items[0]["image_path"]).convert("RGB")
    sx, sy = get_scale(items[0])

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), gridspec_kw={"wspace": 0.04})

    # ── PANEL 1: original ────────────────────────────────────────────────
    axes[0].imshow(img)
    axes[0].set_title("① Tray photo", fontsize=11, fontweight="bold", loc="left", pad=10)
    axes[0].axis("off")

    # ── PANEL 2: +bboxes (Qwen) ─────────────────────────────────────────
    axes[1].imshow(img)
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        color = PALETTE.get(it["class_display_name"], "#FF6B6B")
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=4, edgecolor="black", facecolor="none")
        axes[1].add_patch(rect)
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2.5, edgecolor=color, facecolor="none")
        axes[1].add_patch(rect)
    axes[1].set_title("② Qwen-VL bboxes", fontsize=11, fontweight="bold", loc="left", pad=10)
    axes[1].axis("off")

    # ── PANEL 3: +masks (SAM3) ──────────────────────────────────────────
    overlay = np.array(img).copy()
    composite = np.array(img).astype(np.float32)
    for it in items:
        mask_path = V32 / it["mask_path"]
        mask_img = Image.open(mask_path).convert("L")
        mask_resized = mask_img.resize(img.size, Image.NEAREST)
        m = np.array(mask_resized) > 127
        color_hex = PALETTE.get(it["class_display_name"], "#FF6B6B").lstrip("#")
        rgb = np.array([int(color_hex[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
        for c in range(3):
            composite[..., c] = np.where(m, composite[..., c] * 0.45 + rgb[c] * 0.55, composite[..., c])
    axes[2].imshow(composite.astype(np.uint8))
    axes[2].set_title("③ SAM3 masks", fontsize=11, fontweight="bold", loc="left", pad=10)
    axes[2].axis("off")

    # ── PANEL 4: classified crops ──────────────────────────────────────
    axes[3].axis("off")
    axes[3].set_title("④ DINOv2 → 32-class", fontsize=11, fontweight="bold", loc="left", pad=10)
    n = len(items)
    cell_w, cell_h = 1 / max(n, 1), 1.0
    for i, it in enumerate(items):
        crop = Image.open(V32 / it["crop_path"]).convert("RGB")
        crop = crop.resize((180, 180), Image.LANCZOS)
        ax_cell = fig.add_axes([
            axes[3].get_position().x0 + i * (axes[3].get_position().width / n),
            axes[3].get_position().y0 + 0.1,
            axes[3].get_position().width / n - 0.005,
            axes[3].get_position().height - 0.2,
        ])
        ax_cell.imshow(crop)
        color = PALETTE.get(it["class_display_name"], "#FF6B6B")
        for spine in ax_cell.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax_cell.set_xticks([]); ax_cell.set_yticks([])
        ax_cell.set_xlabel(it["class_display_name"], fontsize=8, fontweight="bold")

    pdf_out = Path(__file__).parent / "pipeline_flow.pdf"
    fig.savefig(pdf_out, bbox_inches="tight", dpi=200)
    print(f"wrote {pdf_out}")

    png_out = Path(__file__).resolve().parents[2] / "presentation" / "assets" / "flow.png"
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, bbox_inches="tight", dpi=160, facecolor="white")
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
