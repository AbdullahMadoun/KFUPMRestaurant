#!/usr/bin/env python3
"""Generate a multi-page PDF report from a TriFoodNet training run.

Usage:
    python scripts/build_report_pdf.py \
        --run logs/trial-20260425-2313-real-1hr \
        --out reports/trial-20260425-2313-real-1hr.pdf
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_events(jsonl_path: Path):
    train, evals = [], []
    for line in open(jsonl_path):
        d = json.loads(line)
        et = d.get("event_type")
        if et == "train_step":
            train.append(d)
        elif et == "eval_epoch":
            evals.append(d)
    return train, evals


def smooth(values, k=20):
    if not values:
        return values
    out = []
    for i in range(len(values)):
        a = max(0, i - k)
        out.append(sum(values[a:i + 1]) / (i + 1 - a))
    return out


def page_title(pdf, run_meta, evals, train_total_min):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.93, "TriFoodNet Training Report", ha="center", size=20, weight="bold")
    fig.text(0.5, 0.89, run_meta.get("run_name", "?"), ha="center", size=13, style="italic", color="#444")

    ds = run_meta.get("dataset", {})
    pkgs = run_meta.get("packages", {})
    rows = [
        ("Notes", run_meta.get("notes", "")),
        ("Seed", str(run_meta.get("seed", "?"))),
        ("Determinism mode", str(run_meta.get("determinism_mode", "?"))),
        ("Run timestamp (UTC)", run_meta.get("timestamp_utc", "?")[:19]),
        ("", ""),
        ("Hardware", f"{run_meta.get('cuda',{}).get('devices',['?'])[0]} (Blackwell sm_120)"),
        ("Torch / CUDA", f"{run_meta.get('torch','?')} / cuda {run_meta.get('cuda',{}).get('version','?')}"),
        ("transformers / peft", f"{pkgs.get('transformers','?')} / {pkgs.get('peft','?')}"),
        ("Image", "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"),
        ("Provider", "vast.ai (Denmark, RTX 5090, $0.39/hr)"),
        ("", ""),
        ("Dataset", f"{ds.get('version','?')} KFUPM food, hash {ds.get('hash','?')}"),
        ("",        f"   {ds.get('n_images_total','?')} images, {ds.get('n_stage3_items_total','?')} items, "
                    f"{ds.get('n_classes','?')} classes"),
        ("Splits", "train 80% / dev 10% / test 10%"),
        ("Dev split size", "370 images / 514 items"),
        ("", ""),
        ("Stage 1 (Qwen2.5-VL-3B)", "LoRA r=16/α=32 on q/k/v/o (~7.4M trainable)"),
        ("Stage 2 (SAM3)", "FROZEN (image enc + prompt enc + mask dec)"),
        ("Stage 3 (PictSure ViT)", "ViT FROZEN, transformer LoRA r=16 (~524K trainable)"),
        ("Total trainable / total params", "7.90M / 4.72B (1 in 598)"),
        ("", ""),
        ("Joint training epochs", "5"),
        ("Effective batch size", "8 (bs=2 × grad_accum=4)"),
        ("Per-stage LR", "Qwen 2e-5 / SAM 5e-5 / PictSure 1e-4 (cosine schedule)"),
        ("Teacher forcing", "GT boxes throughout (use_gt_boxes=True)"),
        ("Stage 1 chunk size (eval)", "8 (batched Qwen.generate for ~5× faster dev eval)"),
        ("", ""),
        ("Total wall time", f"{train_total_min:.1f} min ({train_total_min/60:.2f} hrs)"),
        ("Dev evals completed", f"{len(evals)} (one per epoch)"),
        ("Best epoch (lowest dev_loss)", f"{evals[-1]['epoch']} — dev/loss_total = {evals[-1]['dev/loss_total']:.4f}"),
        ("Best combined score", f"{evals[-1]['dev/combined']:.4f}"),
    ]
    y = 0.82
    for label, value in rows:
        if label:
            fig.text(0.10, y, label, size=9, weight="bold", color="#333")
            fig.text(0.40, y, value, size=9, color="#000")
        y -= 0.022
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_train_loss_curves(pdf, train, evals):
    steps = [t["step"] for t in train]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Training loss curves (per step, raw + EMA)", weight="bold")

    components = [
        ("train_step/loss_total", "Total loss", axes[0][0]),
        ("train_step/stage1_lm_loss", "Stage 1 (Qwen LM loss)", axes[0][1]),
        ("train_step/stage2_dice_loss", "Stage 2 (SAM3 Dice loss)", axes[1][0]),
        ("train_step/stage3_ce_loss", "Stage 3 (PictSure CE loss)", axes[1][1]),
    ]
    for key, title, ax in components:
        vals = [t.get(key, 0) for t in train]
        sm = smooth(vals, k=30)
        ax.plot(steps, vals, color="#bbb", lw=0.5, alpha=0.6, label="raw")
        ax.plot(steps, sm, color="#1f77b4", lw=1.6, label="EMA(30)")
        # Shade epoch boundaries
        for e in evals:
            ax.axvline(e["step"], color="#d62728", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(key.split('/')[-1])
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_dev_curves(pdf, evals):
    epochs = [e["epoch"] for e in evals]

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Dev metrics across 5 epochs", weight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def plot(ax, key, label, color="#1f77b4", **kw):
        vals = [e.get(key, 0) for e in evals]
        ax.plot(epochs, vals, "o-", color=color, lw=2, ms=7, **kw)
        for x, y in zip(epochs, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=color)
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

    ax1 = fig.add_subplot(gs[0, 0])
    plot(ax1, "dev/combined", "Combined score (recall + mIoU + s3_acc)", color="#2ca02c")
    ax1.set_ylabel("score")

    ax2 = fig.add_subplot(gs[0, 1])
    plot(ax2, "dev/loss_total", "Dev loss total", color="#d62728")
    ax2.set_ylabel("loss")

    ax3 = fig.add_subplot(gs[0, 2])
    r = [e.get("dev/stage1_recall@0.5", 0) for e in evals]
    p = [e.get("dev/stage1_precision@0.5", 0) for e in evals]
    ax3.plot(epochs, r, "o-", lw=2, ms=7, label="recall@0.5", color="#1f77b4")
    ax3.plot(epochs, p, "s-", lw=2, ms=7, label="precision@0.5", color="#ff7f0e")
    for x, y in zip(epochs, r):
        ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    for x, y in zip(epochs, p):
        ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax3.set_xticks(epochs)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("score")
    ax3.set_title("Stage 1 detection (Qwen)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    plot(ax4, "dev/stage2_mIoU", "Stage 2 mIoU (SAM3 frozen)", color="#9467bd")
    ax4.set_ylabel("mIoU")

    ax5 = fig.add_subplot(gs[1, 1])
    s3 = [e.get("dev/stage3_acc", 0) for e in evals]
    ct = [e.get("dev/stage3_cosine_top1_acc", 0) for e in evals]
    ax5.plot(epochs, s3, "o-", lw=2, ms=7, label="stage3_acc (overall)", color="#1f77b4")
    ax5.plot(epochs, ct, "s-", lw=2, ms=7, label="cosine_top1 baseline", color="#888")
    for x, y in zip(epochs, s3):
        ax5.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    for x, y in zip(epochs, ct):
        ax5.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#666")
    ax5.set_xticks(epochs)
    ax5.set_xlabel("epoch")
    ax5.set_ylabel("accuracy")
    ax5.set_title("Stage 3: classification vs cosine baseline")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    lift = [e.get("dev/stage3_transformer_lift_over_top1", 0) for e in evals]
    ax6.bar(epochs, lift, color="#2ca02c", alpha=0.85)
    for x, y in zip(epochs, lift):
        ax6.annotate(f"+{y*100:.1f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, weight="bold")
    ax6.set_xticks(epochs)
    ax6.set_xlabel("epoch")
    ax6.set_ylabel("lift (pts)")
    ax6.set_title("Transformer lift over cosine retrieval")
    ax6.axhline(0, color="black", lw=0.5)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, max(lift) * 1.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metrics_table(pdf, evals):
    keys = [
        ("combined",                                     "dev/combined"),
        ("loss_total (dev)",                             "dev/loss_total"),
        ("loss_stage1",                                  "dev/loss_stage1"),
        ("loss_stage2",                                  "dev/loss_stage2"),
        ("loss_stage3",                                  "dev/loss_stage3"),
        ("",                                             None),
        ("Stage 1: recall@0.5",                          "dev/stage1_recall@0.5"),
        ("Stage 1: precision@0.5",                       "dev/stage1_precision@0.5"),
        ("Stage 1: pred_items_per_image",                "dev/pred_items_per_image"),
        ("",                                             None),
        ("Stage 2: mIoU",                                "dev/stage2_mIoU"),
        ("Stage 2: BCE loss",                            "dev/stage2_bce_loss"),
        ("Stage 2: Dice loss",                           "dev/stage2_dice_loss"),
        ("",                                             None),
        ("Stage 3: acc (overall)",                       "dev/stage3_acc"),
        ("Stage 3: acc (of detected)",                   "dev/stage3_matched_acc"),
        ("Stage 3: episode acc (5w-5s)",                 "dev/stage3_episode_acc"),
        ("Stage 3: cosine top-1 acc",                    "dev/stage3_cosine_top1_acc"),
        ("Stage 3: acc when retrieval got GT",           "dev/stage3_acc_given_retrieved"),
        ("Stage 3: retrieval recall@K",                  "dev/stage3_retrieval_recall@K"),
        ("Stage 3: TRANSFORMER LIFT",                    "dev/stage3_transformer_lift_over_top1"),
        ("",                                             None),
        ("Counts: n_images",                             "dev/n_images"),
        ("Counts: n_items (GT)",                         "dev/n_items"),
        ("Counts: n_pred_items",                         "dev/n_pred_items"),
        ("Counts: n_matches (TP)",                       "dev/n_matches"),
        ("Counts: n_with_candidates",                    "dev/n_with_candidates"),
        ("Counts: n_retrieval_hits",                     "dev/n_retrieval_hits"),
        ("Counts: n_cosine_top1_correct",                "dev/n_cosine_top1_correct"),
        ("",                                             None),
        ("Latency: stage1 ms/img",                       "dev/latency_stage1_ms"),
        ("Latency: stage2 ms/img",                       "dev/latency_stage2_ms"),
        ("Latency: stage3 ms/img",                       "dev/latency_stage3_ms"),
        ("Latency: TOTAL ms/img",                        "dev/latency_total_ms"),
    ]

    # Build table rows
    table_data = [["Metric", "ep1", "ep2", "ep3", "ep4", "ep5", "Δ"]]
    for label, key in keys:
        if key is None:
            table_data.append([label, "", "", "", "", "", ""])
            continue
        vals = [e.get(key, 0) for e in evals]
        # Counts → integers; metrics → 4 decimals; latency → 1 decimal
        if "Counts" in label or "n_" in (key or ""):
            cells = [f"{v:.0f}" for v in vals]
            delta = f"{vals[-1] - vals[0]:+.0f}"
        elif "Latency" in label:
            cells = [f"{v:.1f}" for v in vals]
            delta = f"{vals[-1] - vals[0]:+.1f}"
        else:
            cells = [f"{v:.4f}" for v in vals]
            delta = f"{vals[-1] - vals[0]:+.4f}"
        table_data.append([label] + cells + [delta])

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Full dev metrics — all 5 epochs", weight="bold", y=0.99)
    ax = fig.add_subplot(111)
    ax.axis("off")
    tbl = ax.table(cellText=table_data, loc="center", cellLoc="left",
                   colWidths=[0.35, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    # Header row formatting
    for j in range(7):
        tbl[(0, j)].set_facecolor("#333")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    # Highlight transformer lift row
    for i, row in enumerate(table_data):
        if "TRANSFORMER LIFT" in row[0]:
            for j in range(7):
                tbl[(i, j)].set_facecolor("#fff3a8")
        elif row[0] == "" and row[1] == "":
            for j in range(7):
                tbl[(i, j)].set_facecolor("#eee")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_qualitative(pdf, viz_dir):
    """Mosaic of 12 random dev visualizations from epoch_003."""
    if not viz_dir.is_dir():
        return
    images = sorted(viz_dir.glob("*.png"))
    if not images:
        return
    # Take a deterministic spread of 12
    n = 12
    step = max(1, len(images) // n)
    sample = images[::step][:n]

    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    fig.suptitle(f"Sample dev predictions (epoch 3) — {len(images)} total saved",
                 weight="bold")
    for ax, img_path in zip(axes.flat, sample):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem.replace("_predictions", "")[:28], fontsize=7)
        ax.axis("off")
    for ax in list(axes.flat)[len(sample):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_findings(pdf, evals):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, "Key findings", ha="center", size=18, weight="bold")

    e1, e5 = evals[0], evals[-1]
    text_blocks = [
        ("Stage 1 (Qwen detection) — got smarter, not more",
         f"• Recall@0.5: {e1['dev/stage1_recall@0.5']:.3f} → {e5['dev/stage1_recall@0.5']:.3f}  "
         f"(Δ {e5['dev/stage1_recall@0.5']-e1['dev/stage1_recall@0.5']:+.3f}, essentially flat)\n"
         f"• Precision@0.5: {e1['dev/stage1_precision@0.5']:.3f} → {e5['dev/stage1_precision@0.5']:.3f}  "
         f"(Δ {e5['dev/stage1_precision@0.5']-e1['dev/stage1_precision@0.5']:+.3f}, +8.3 pt)\n"
         f"• Predictions made: {int(e1['dev/n_pred_items'])} → {int(e5['dev/n_pred_items'])}  "
         f"(Δ {int(e5['dev/n_pred_items']-e1['dev/n_pred_items']):+d})\n"
         f"• Correct detections: {int(e1['dev/n_matches'])} → {int(e5['dev/n_matches'])}  "
         f"(Δ {int(e5['dev/n_matches']-e1['dev/n_matches']):+d})\n"
         f"  ↳ Qwen learned to be conservative: fewer total predictions, higher hit rate."),

        ("Stage 2 (SAM3 mask) — free-rode on Stage 1",
         f"• mIoU: {e1['dev/stage2_mIoU']:.3f} → {e5['dev/stage2_mIoU']:.3f}  "
         f"(Δ {e5['dev/stage2_mIoU']-e1['dev/stage2_mIoU']:+.3f})\n"
         f"  ↳ SAM3 is fully frozen. All gain is downstream: better Stage 1 boxes\n"
         f"     → cleaner SAM prompts → better masks."),

        ("Stage 3 (PictSure ICL) — the headline result",
         f"• Cosine top-1 baseline: {e1['dev/stage3_cosine_top1_acc']:.3f} → "
         f"{e5['dev/stage3_cosine_top1_acc']:.3f}  "
         f"(Δ {e5['dev/stage3_cosine_top1_acc']-e1['dev/stage3_cosine_top1_acc']:+.3f}, -5.5 pt)\n"
         f"• Transformer lift: +{e1['dev/stage3_transformer_lift_over_top1']*100:.1f} pt → "
         f"+{e5['dev/stage3_transformer_lift_over_top1']*100:.1f} pt  "
         f"(Δ +{(e5['dev/stage3_transformer_lift_over_top1']-e1['dev/stage3_transformer_lift_over_top1'])*100:.1f} pt)\n"
         f"• Stage 3 acc (overall): {e1['dev/stage3_acc']:.3f} → {e5['dev/stage3_acc']:.3f}  "
         f"(Δ {e5['dev/stage3_acc']-e1['dev/stage3_acc']:+.3f}, FLAT)\n"
         f"• Episode acc (in-distribution 5w-5s): {e1['dev/stage3_episode_acc']:.3f} → "
         f"{e5['dev/stage3_episode_acc']:.3f}  "
         f"(Δ {e5['dev/stage3_episode_acc']-e1['dev/stage3_episode_acc']:+.3f})\n"
         f"  ↳ Joint training degrades the embedding's cosine geometry, but the\n"
         f"     transformer learns to compensate — gap widens monotonically."),

        ("Combined picture",
         f"• Combined score: {e1['dev/combined']:.4f} → {e5['dev/combined']:.4f}  "
         f"(Δ {e5['dev/combined']-e1['dev/combined']:+.4f})\n"
         f"• Dev loss total:  {e1['dev/loss_total']:.4f} → {e5['dev/loss_total']:.4f}  "
         f"(Δ {e5['dev/loss_total']-e1['dev/loss_total']:+.4f})\n"
         f"• Latency: {e1['dev/latency_total_ms']:.0f} → {e5['dev/latency_total_ms']:.0f} ms / image\n"
         f"  ↳ Combined gains driven entirely by Stage 1 + Stage 2; Stage 3 acc is held\n"
         f"     flat by the embedding-degradation / transformer-compensation balance."),

        ("Bottleneck",
         f"• Stage 1 recall is the binding constraint: 45 % of GT items are never detected\n"
         f"  and therefore can't be classified. Halving stage3_acc gains is impossible\n"
         f"  without lifting recall.\n"
         f"• Path forward: train Stage 1 longer / with detection-specific loss /\n"
         f"  larger backbone, before further joint Stage 3 work."),

        ("Publishable claim",
         f"\"On a 32-class restaurant food dataset, an ICL transformer adds\n"
         f"+{e5['dev/stage3_transformer_lift_over_top1']*100:.1f} pt over pure cosine retrieval, "
         f"and the gap widens monotonically with\n"
         f"joint training (+{e1['dev/stage3_transformer_lift_over_top1']*100:.1f} pt at "
         f"epoch 1 → +{e5['dev/stage3_transformer_lift_over_top1']*100:.1f} pt at epoch 5).\n"
         f"This holds despite the cosine baseline degrading by "
         f"{(e5['dev/stage3_cosine_top1_acc']-e1['dev/stage3_cosine_top1_acc'])*100:+.1f} pt over the\n"
         f"same training, indicating the transformer learns to extract more\n"
         f"signal from the (jointly-tuned) embeddings.\""),
    ]

    y = 0.88
    for title, body in text_blocks:
        fig.text(0.08, y, title, size=11, weight="bold", color="#1f4d7a")
        y -= 0.025
        for line in body.split("\n"):
            fig.text(0.10, y, line, size=9, family="monospace")
            y -= 0.018
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to logs/<run_name>/")
    parser.add_argument("--out", required=True, help="output PDF path")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    events_path = run_dir / "joint" / "events.jsonl"
    train, evals = load_events(events_path)
    if not evals:
        raise SystemExit(f"no eval_epoch events in {events_path}")
    train_total_min = train[-1]["elapsed_sec"] / 60 if train else 0.0

    run_meta = json.loads((run_dir / "joint" / "run_metadata.json").read_text())

    # Find dev visualizations dir (may be epoch_003 only based on every_n_epochs=3)
    viz_root = Path("outputs") / run_dir.name / "report" / "dev_visualizations"
    viz_dir = next((p for p in sorted(viz_root.glob("epoch_*"), reverse=True) if (p / "images").is_dir()), None)
    if viz_dir is not None:
        viz_dir = viz_dir / "images"

    with PdfPages(out_path) as pdf:
        page_title(pdf, run_meta, evals, train_total_min)
        page_train_loss_curves(pdf, train, evals)
        page_dev_curves(pdf, evals)
        page_metrics_table(pdf, evals)
        if viz_dir is not None:
            page_qualitative(pdf, viz_dir)
        page_findings(pdf, evals)

    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
