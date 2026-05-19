"""Build a 6-image qualitative grid: real tray photos with predicted vs GT
class lists from the actual base / distilled runs (596-image full eval).

For each chosen image we render:
  ┌──────────────┐  GT:        rice, chicken
  │   tray.jpg   │  BASE:      —  ← what the off-the-shelf 7B gave
  │              │  DISTILLED: rice ✓ chicken ✓
  └──────────────┘

Outputs:
  - paper/figs/qualitative_grid.pdf
  - presentation/assets/qual_grid.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams.update({
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
V32 = Path("/Users/abdulrazzak/Resturant_Pipeline_Feb/results_v2/exports/v3.2_2026-04-27_c99ee47f")
EVAL = ROOT / "test_set.jsonl"
PRED_BASE = ROOT / "predictions_base.jsonl"
PRED_DIST = ROOT / "predictions_distilled.jsonl"


def load(p): return [json.loads(l) for l in p.open() if l.strip()]


def by_id(rows): return {r["src_image_id"]: r for r in rows}


def multiset_match(p, g):
    return Counter(c.strip().lower() for c in p) == Counter(c.strip().lower() for c in g)


def select_examples(eval_set, base, dist, n=6):
    """Pick a diverse set of trays:
       - 2 where distilled is correct AND base failed (clear win)
       - 1 where both correct (sanity)
       - 1 where distilled was wrong but base also wrong (hard case)
       - 1 multi-item where distilled correct
       - 1 with rare class
    """
    base_by, dist_by = by_id(base), by_id(dist)
    win = []
    both_correct = []
    both_wrong = []
    multi_dist_win = []
    for gt in eval_set:
        iid = gt["src_image_id"]
        b, d = base_by.get(iid), dist_by.get(iid)
        if not b or not d:
            continue
        b_ok = multiset_match(b["pred_classes"], gt["gt_classes"])
        d_ok = multiset_match(d["pred_classes"], gt["gt_classes"])
        n_gt = len(gt["gt_classes"])
        if d_ok and not b_ok and n_gt == 1:
            win.append((gt, b, d))
        if d_ok and not b_ok and n_gt >= 2:
            multi_dist_win.append((gt, b, d))
        if b_ok and d_ok:
            both_correct.append((gt, b, d))
        if not b_ok and not d_ok and n_gt >= 2:
            both_wrong.append((gt, b, d))
    chosen = []
    chosen += win[:2]
    chosen += multi_dist_win[:2]
    chosen += both_correct[:1]
    chosen += both_wrong[:1]
    return chosen[:n]


def fmt_classes(lst, mark_correct_against=None):
    if not lst:
        return "—"
    if mark_correct_against is None:
        return ", ".join(lst)
    gt = Counter(c.strip().lower() for c in mark_correct_against)
    out_parts = []
    for c in lst:
        ck = c.strip().lower()
        if gt.get(ck, 0) > 0:
            out_parts.append(f"{c} ✓")
            gt[ck] -= 1
        else:
            out_parts.append(f"{c} ✗")
    return ", ".join(out_parts)


def main():
    eval_set = load(EVAL)
    base = load(PRED_BASE)
    dist = load(PRED_DIST)
    chosen = select_examples(eval_set, base, dist, n=6)
    print(f"chose {len(chosen)} examples")

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.10})
    axes = axes.flatten()

    for ax, (gt, b, d) in zip(axes, chosen):
        img = Image.open(V32 / "images" / f"{gt['src_image_id']}.jpg").convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5); spine.set_edgecolor("#000000")
        gt_txt = ", ".join(gt["gt_classes"])
        b_txt = fmt_classes(b["pred_classes"], gt["gt_classes"])
        d_txt = fmt_classes(d["pred_classes"], gt["gt_classes"])
        title = (
            f"$\\bf{{GT:}}$ {gt_txt}\n"
            f"$\\bf{{BASE:}}$ {b_txt}\n"
            f"$\\bf{{DISTILLED:}}$ {d_txt}"
        )
        ax.set_title(title, fontsize=9, loc="left", pad=8, linespacing=1.4)

    # Hide unused subplots
    for ax in axes[len(chosen):]:
        ax.axis("off")

    pdf_out = Path(__file__).parent / "qualitative_grid.pdf"
    fig.savefig(pdf_out, bbox_inches="tight", dpi=200)
    print(f"wrote {pdf_out}")

    png_out = ROOT / "presentation" / "assets" / "qual_grid.png"
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, bbox_inches="tight", dpi=160, facecolor="white")
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
