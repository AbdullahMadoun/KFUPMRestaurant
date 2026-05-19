"""Generate result figures from predictions_base.jsonl + predictions_distilled.jsonl
+ test_set.jsonl. Falls back to placeholder figures if a file is missing."""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).parent
PRED_DIR = ROOT / "results"
EVAL = ROOT / "test_set.jsonl"


def load(p: Path):
    if not p.exists():
        return None
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def placeholder(out: Path, title: str):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.text(0.5, 0.5, f"[awaiting data]\n{title}", ha="center", va="center", fontsize=9, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  placeholder: {out.name}")


def metrics_bundle(preds, eval_set):
    by_id = {p["src_image_id"]: p for p in preds}
    rows = []
    for gt in eval_set:
        p = by_id.get(gt["src_image_id"])
        if not p:
            continue
        rows.append({
            "n_gt": len(gt["gt_classes"]),
            "n_pred": p["n_pred"],
            "count_err": p["n_pred"] - len(gt["gt_classes"]),
            "count_exact": p["n_pred"] == len(gt["gt_classes"]),
            "multiset_match": Counter(c.lower() for c in p["pred_classes"]) == Counter(c.lower() for c in gt["gt_classes"]),
            "dish_correct": (p["n_pred"] == len(gt["gt_classes"])) and
                            (Counter(c.lower() for c in p["pred_classes"]) == Counter(c.lower() for c in gt["gt_classes"])),
        })
    return rows


def fig_dish_correct_bar(base_rows, dist_rows, out: Path):
    """dish-correct rate broken down by gt cardinality"""
    def by_card(rows):
        out = {}
        for r in rows:
            k = "1" if r["n_gt"] == 1 else "2" if r["n_gt"] == 2 else "3+"
            out.setdefault(k, []).append(r["dish_correct"])
        return {k: 100 * sum(v) / len(v) if v else 0 for k, v in out.items()}

    base = by_card(base_rows)
    dist = by_card(dist_rows)
    cards = ["1", "2", "3+"]
    base_v = [base.get(c, 0) for c in cards]
    dist_v = [dist.get(c, 0) for c in cards]

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    x = range(len(cards))
    w = 0.36
    ax.bar([i - w/2 for i in x], base_v, width=w, color="#7d8a99", label="Base 7B")
    ax.bar([i + w/2 for i in x], dist_v, width=w, color="#2c5985", label="Distilled 7B")
    for i, (a, b) in enumerate(zip(base_v, dist_v)):
        ax.text(i - w/2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=7)
        ax.text(i + w/2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c} item{'s' if c != '1' else ''}" for c in cards])
    ax.set_ylabel("dish-correct (%)")
    ax.set_ylim(0, max(max(base_v), max(dist_v), 1) * 1.25 + 5)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {out.name}")


def fig_count_error_hist(base_rows, dist_rows, out: Path):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    bins = list(range(-4, 6))
    base_e = [r["count_err"] for r in base_rows]
    dist_e = [r["count_err"] for r in dist_rows]
    ax.hist(base_e, bins=bins, alpha=0.55, color="#7d8a99", label=f"Base 7B (μ={sum(base_e)/max(len(base_e),1):.2f})", align="left")
    ax.hist(dist_e, bins=bins, alpha=0.65, color="#2c5985", label=f"Distilled 7B (μ={sum(dist_e)/max(len(dist_e),1):.2f})", align="left")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"count error  $|\mathcal{P}| - |\mathcal{G}|$")
    ax.set_ylabel("count of held-out images")
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote: {out.name}")


def fig_qualitative_placeholder(out: Path):
    placeholder(out, "qualitative grid (build after both arms complete)")


def main():
    eval_set = load(EVAL)
    if eval_set is None:
        print("No test_set.jsonl found.")
        sys.exit(0)

    base = load(PRED_DIR / "predictions_base.jsonl")
    dist = load(PRED_DIR / "predictions_distilled.jsonl")

    if base is None or dist is None:
        if base is None:
            print("predictions_base.jsonl not found")
        if dist is None:
            print("predictions_distilled.jsonl not found")
        placeholder(FIGS / "dish_correct_bar.pdf", "dish_correct_bar")
        placeholder(FIGS / "count_error_hist.pdf", "count_error_hist")
        placeholder(FIGS / "qualitative_grid.pdf", "qualitative_grid")
        return

    base_rows = metrics_bundle(base, eval_set)
    dist_rows = metrics_bundle(dist, eval_set)
    print(f"Loaded {len(base_rows)} base / {len(dist_rows)} distilled rows")
    fig_dish_correct_bar(base_rows, dist_rows, FIGS / "dish_correct_bar.pdf")
    fig_count_error_hist(base_rows, dist_rows, FIGS / "count_error_hist.pdf")
    fig_qualitative_placeholder(FIGS / "qualitative_grid.pdf")


if __name__ == "__main__":
    main()
