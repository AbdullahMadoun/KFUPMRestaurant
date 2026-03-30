# =============================================================================
# FILE: experiment_report.py
# CATEGORY: EVAL
# PURPOSE: Markdown, CSV, JSON, and SVG report generator over experiment logs.
# DEPENDENCIES: None
# USED BY: post_training_artifacts.py
# KEY CLASSES/FUNCTIONS: MetricSpec, RunReport, parse_args, main, generate_report, resolve_run_dirs, normalize_run_dir, discover_run_dirs, choose_baseline_run, load_run_report, read_json, read_jsonl
# LAST MODIFIED: 2026-03-21T12:48:35.373924+00:00
# SNAPSHOT NOTES: no major inline issues detected during snapshot generation
# =============================================================================
"""
Generate experiment reports from TriFoodNet JSONL logs.

The report generator stays lightweight on purpose:

- input: one or more `logs/<run>/joint` directories
- output: markdown summaries, CSV/JSON summaries, and SVG charts
- no heavy data stack required
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SVG_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#ea580c",
    "#7c3aed",
    "#0f766e",
    "#c2410c",
    "#4f46e5",
]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    slug: str
    higher_is_better: bool
    description: str


DEFAULT_METRICS: List[MetricSpec] = [
    MetricSpec("train_step/loss_total", "Train Step Total Loss", "train_step_loss_total", False, "Joint training objective over optimizer steps."),
    MetricSpec("train_step/loss_stage1", "Train Step Stage 1 Loss", "train_step_loss_stage1", False, "Grounding-language loss during training."),
    MetricSpec("train_step/loss_stage2", "Train Step Stage 2 Loss", "train_step_loss_stage2", False, "Segmentation loss during training."),
    MetricSpec("train_step/loss_stage3", "Train Step Stage 3 Loss", "train_step_loss_stage3", False, "Few-shot classification loss during training."),
    MetricSpec("train_step/stage3_acc", "Train Step Stage 3 Accuracy", "train_step_stage3_acc", True, "Episode-level Stage 3 training accuracy."),
    MetricSpec("train/loss_total", "Train Eval Total Loss", "train_eval_loss_total", False, "Teacher-forced train-split objective loss for overfitting tracking."),
    MetricSpec("train/stage1_recall@0.5", "Train Eval Stage 1 Recall@0.5", "train_stage1_recall_at_0_5", True, "Grounding recall against the train split in inference mode."),
    MetricSpec("train/stage2_mIoU", "Train Eval Stage 2 mIoU", "train_stage2_miou", True, "End-to-end segmentation quality using Qwen-prompted SAM3 masks on the train split."),
    MetricSpec("train/stage3_acc", "Train Eval Stage 3 Accuracy", "train_stage3_acc", True, "End-to-end item classification accuracy against the train split."),
    MetricSpec("train/stage3_matched_acc", "Train Eval Stage 3 Matched Accuracy", "train_stage3_matched_acc", True, "Classification accuracy on train items whose predicted boxes matched ground truth."),
    MetricSpec("train/stage3_episode_acc", "Train Eval Stage 3 Episode Accuracy", "train_stage3_episode_acc", True, "Teacher-forced PictSure ICL episode accuracy on train masked crops."),
    MetricSpec("dev/loss_total", "Dev Total Loss", "dev_loss_total", False, "Teacher-forced dev objective loss for overfitting tracking."),
    MetricSpec("dev/stage1_recall@0.5", "Dev Stage 1 Recall@0.5", "dev_stage1_recall_at_0_5", True, "Grounding recall against the held-out dev split."),
    MetricSpec("dev/stage2_mIoU", "Dev Stage 2 mIoU", "dev_stage2_miou", True, "End-to-end segmentation quality using Qwen-prompted SAM3 masks on the held-out dev split."),
    MetricSpec("dev/stage3_acc", "Dev Stage 3 Accuracy", "dev_stage3_acc", True, "End-to-end item classification accuracy against the held-out dev split."),
    MetricSpec("dev/stage3_matched_acc", "Dev Stage 3 Matched Accuracy", "dev_stage3_matched_acc", True, "Classification accuracy on dev items whose predicted boxes matched ground truth."),
    MetricSpec("dev/stage3_episode_acc", "Dev Stage 3 Episode Accuracy", "dev_stage3_episode_acc", True, "Teacher-forced PictSure ICL episode accuracy on held-out dev masked crops."),
    MetricSpec("dev/latency_total_ms", "Dev Inference Latency", "dev_latency_total_ms", False, "Average end-to-end dev-image latency in milliseconds."),
    MetricSpec("train_step/lr", "Learning Rate", "train_step_lr", True, "Optimizer learning rate progression."),
    MetricSpec("train_step/samples_per_sec", "Training Throughput", "train_step_samples_per_sec", True, "Measured samples processed per second."),
    MetricSpec("gpu/mem_peak_allocated_gb", "Peak GPU Memory", "gpu_mem_peak_allocated_gb", False, "Peak allocated GPU memory per logged event."),
]


SUMMARY_METRICS: List[MetricSpec] = [
    MetricSpec("joint/combined", "Best Joint Combined", "best_joint_combined", True, "Checkpoint selection score."),
    MetricSpec("dev/stage1_recall@0.5", "Best Dev Stage 1 Recall@0.5", "best_dev_stage1_recall_at_0_5", True, "Best dev grounding recall."),
    MetricSpec("dev/stage2_mIoU", "Best Dev Stage 2 mIoU", "best_dev_stage2_miou", True, "Best dev segmentation quality."),
    MetricSpec("dev/stage3_acc", "Best Dev Stage 3 Accuracy", "best_dev_stage3_acc", True, "Best dev end-to-end classification accuracy."),
    MetricSpec("dev/loss_total", "Min Dev Total Loss", "min_dev_loss_total", False, "Lowest observed dev total loss."),
]


@dataclass
class RunReport:
    name: str
    root_dir: Path
    events: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    config: Dict[str, Any]
    latest: Dict[str, Any]
    best: Dict[str, Any]
    status: Dict[str, Any]

    def scalar_series(self, metric_key: str) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for index, event in enumerate(self.events):
            value = _to_float(event.get(metric_key))
            if value is None:
                continue
            x_value = _to_float(event.get("step"))
            if x_value is None:
                x_value = _to_float(event.get("epoch"))
            if x_value is None:
                x_value = float(index + 1)
            points.append((x_value, value))
        return points

    def latest_value(self, metric_key: str) -> Optional[float]:
        value = _to_float(self.latest.get(metric_key))
        if value is not None:
            return value
        series = self.scalar_series(metric_key)
        if series:
            return series[-1][1]
        return None

    def best_value(self, spec: MetricSpec) -> Optional[float]:
        if spec.key == "joint/combined":
            entry = self.best.get(spec.key)
            if isinstance(entry, dict):
                return _to_float(entry.get("value"))
            return None
        series = self.scalar_series(spec.key)
        if not series:
            return None
        values = [point[1] for point in series]
        return max(values) if spec.higher_is_better else min(values)

    def mean_value(self, metric_key: str) -> Optional[float]:
        series = self.scalar_series(metric_key)
        if not series:
            return None
        return sum(point[1] for point in series) / len(series)

    @property
    def device(self) -> str:
        return str(self.metadata.get("device", "unknown"))

    @property
    def notes(self) -> str:
        return str(self.metadata.get("notes", "") or "")

    @property
    def run_status(self) -> str:
        status = self.status.get("status")
        if status:
            return str(status)
        return "unknown"

    @property
    def elapsed_sec(self) -> Optional[float]:
        return _to_float(self.status.get("elapsed_sec"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default="logs", help="Root directory that contains per-run log folders.")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit run directory like logs/run_name/joint.")
    parser.add_argument("--run-name", action="append", default=[], help="Run name under --logs-root to include.")
    parser.add_argument("--baseline", default="", help="Baseline run name or directory name used for delta tables.")
    parser.add_argument("--output", default="reports/latest", help="Destination directory for markdown, CSV, JSON, and plots.")
    parser.add_argument("--title", default="TriFoodNet Experiment Report", help="Title for the generated report.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs = resolve_run_dirs(
        logs_root=Path(args.logs_root),
        explicit_dirs=[Path(path) for path in args.run_dir],
        run_names=args.run_name,
    )
    if not run_dirs:
        raise FileNotFoundError("No experiment run directories were found. Point --logs-root or --run-dir at logs/<run>/joint paths.")
    generate_report(
        run_dirs=run_dirs,
        output_dir=Path(args.output),
        title=args.title,
        baseline=args.baseline or None,
    )


# --- Snapshot note: Main experiment-report builder that aggregates logs into markdown, CSV, JSON, and SVG. ---
def generate_report(run_dirs: Sequence[Path], output_dir: Path, title: str, baseline: Optional[str] = None):
    runs = [load_run_report(path) for path in run_dirs]
    runs.sort(key=lambda run: run.name)
    baseline_run = choose_baseline_run(runs, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    runs_dir = output_dir / "runs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    generated_plots: Dict[str, str] = {}
    for spec in DEFAULT_METRICS:
        series_map = [(run.name, run.scalar_series(spec.key)) for run in runs]
        series_map = [(name, points) for name, points in series_map if points]
        if not series_map:
            continue
        plot_path = plots_dir / f"{spec.slug}.svg"
        write_line_chart_svg(plot_path, spec.title, series_map, y_label=spec.key)
        generated_plots[spec.slug] = f"plots/{plot_path.name}"

    absolute_metric_values = [
        (run.name, run.best_value(SUMMARY_METRICS[0]))
        for run in runs
        if run.best_value(SUMMARY_METRICS[0]) is not None
    ]
    if absolute_metric_values:
        plot_path = plots_dir / "best_joint_combined.svg"
        write_bar_chart_svg(plot_path, "Best Joint Combined Score", absolute_metric_values)
        generated_plots["best_joint_combined"] = f"plots/{plot_path.name}"

    delta_values = build_delta_values(runs, baseline_run, SUMMARY_METRICS[0])
    if delta_values:
        plot_path = plots_dir / "delta_joint_combined_vs_baseline.svg"
        write_bar_chart_svg(
            plot_path,
            f"Joint Combined Delta vs {baseline_run.name}",
            delta_values,
            signed=True,
        )
        generated_plots["delta_joint_combined_vs_baseline"] = f"plots/{plot_path.name}"

    comparison_rows = build_comparison_rows(runs)
    efficiency_rows = build_efficiency_rows(runs)
    delta_rows = build_delta_rows(runs, baseline_run)

    write_summary_csv(output_dir / "summary.csv", comparison_rows, efficiency_rows)
    write_summary_json(
        output_dir / "summary.json",
        title=title,
        baseline_run=baseline_run.name,
        runs=runs,
        comparison_rows=comparison_rows,
        efficiency_rows=efficiency_rows,
        delta_rows=delta_rows,
        generated_plots=generated_plots,
    )

    run_card_paths = []
    for run in runs:
        run_card_path = runs_dir / f"{slugify(run.name)}.md"
        run_card_path.write_text(render_run_card(run), encoding="utf-8")
        run_card_paths.append((run.name, f"runs/{run_card_path.name}"))

    index_markdown = render_index_markdown(
        title=title,
        baseline_run=baseline_run,
        comparison_rows=comparison_rows,
        efficiency_rows=efficiency_rows,
        delta_rows=delta_rows,
        generated_plots=generated_plots,
        run_card_paths=run_card_paths,
    )
    (output_dir / "index.md").write_text(index_markdown, encoding="utf-8")


def resolve_run_dirs(logs_root: Path, explicit_dirs: Sequence[Path], run_names: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for path in explicit_dirs:
        resolved.append(normalize_run_dir(path))
    for run_name in run_names:
        resolved.append(normalize_run_dir(logs_root / run_name / "joint"))
    if not resolved:
        resolved.extend(discover_run_dirs(logs_root))

    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        if not (path / "events.jsonl").exists():
            raise FileNotFoundError(f"Run directory does not contain events.jsonl: {path}")
        deduped.append(path)
        seen.add(key)
    return deduped


def normalize_run_dir(path: Path) -> Path:
    candidate = path
    if candidate.name != "joint" and (candidate / "joint").is_dir():
        candidate = candidate / "joint"
    return candidate


def discover_run_dirs(logs_root: Path) -> List[Path]:
    if not logs_root.exists():
        return []
    return sorted(path for path in logs_root.glob("*/joint") if (path / "events.jsonl").exists())


def choose_baseline_run(runs: Sequence[RunReport], requested: Optional[str]) -> RunReport:
    if requested:
        for run in runs:
            if run.name == requested or run.root_dir.name == requested or run.root_dir.parent.name == requested:
                return run
        raise ValueError(f"Baseline run not found: {requested}")
    return runs[0]


def load_run_report(run_dir: Path) -> RunReport:
    metadata = canonicalize_payload(read_json(run_dir / "run_metadata.json"))
    config = canonicalize_payload(read_json(run_dir / "config_snapshot.json"))
    latest = canonicalize_payload(read_json(run_dir / "latest_metrics.json"))
    best = canonicalize_payload(read_json(run_dir / "best_metrics.json"))
    status = canonicalize_payload(read_json(run_dir / "run_status.json"))
    events = [canonicalize_payload(event) for event in read_jsonl(run_dir / "events.jsonl")]
    run_name = str(metadata.get("run_name") or run_dir.parent.name)
    return RunReport(
        name=run_name,
        root_dir=run_dir,
        events=events,
        metadata=metadata,
        config=config,
        latest=latest,
        best=best,
        status=status,
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def canonicalize_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [canonicalize_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        normalized[canonicalize_metric_key(str(key))] = canonicalize_payload(value)
    if isinstance(normalized.get("split"), str):
        normalized["split"] = canonicalize_split_name(str(normalized["split"]))
    return normalized


def canonicalize_metric_key(key: str) -> str:
    if key.startswith("metrics/val/"):
        return "metrics/dev/" + key[len("metrics/val/"):]
    if key.startswith("val/"):
        return "dev/" + key[len("val/"):]
    return key


def canonicalize_split_name(split: str) -> str:
    lowered = split.strip().lower()
    if lowered == "val":
        return "dev"
    return lowered


def build_comparison_rows(runs: Sequence[RunReport]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "run": run.name,
                "status": run.run_status,
                "best_joint_combined": run.best_value(SUMMARY_METRICS[0]),
                "best_dev_stage1_recall@0.5": run.best_value(SUMMARY_METRICS[1]),
                "best_dev_stage2_mIoU": run.best_value(SUMMARY_METRICS[2]),
                "best_dev_stage3_acc": run.best_value(SUMMARY_METRICS[3]),
                "min_dev_loss_total": run.best_value(SUMMARY_METRICS[4]),
            }
        )
    return rows


def build_efficiency_rows(runs: Sequence[RunReport]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        batch_size = nested_get(run.config, "joint", "training", "batch_size")
        grad_accum = nested_get(run.config, "joint", "training", "grad_accum_steps")
        effective_batch = None
        if isinstance(batch_size, int) and isinstance(grad_accum, int):
            effective_batch = batch_size * grad_accum
        rows.append(
            {
                "run": run.name,
                "device": run.device,
                "avg_train_samples_per_sec": run.mean_value("train_step/samples_per_sec"),
                "max_gpu_mem_peak_allocated_gb": best_or_latest(run, "gpu/mem_peak_allocated_gb", higher_is_better=True),
                "stage3_loss": nested_get(run.config, "stage3", "loss", "name"),
                "joint_lr": nested_get(run.config, "joint", "training", "learning_rate"),
                "effective_batch": effective_batch,
            }
        )
    return rows


def build_delta_rows(runs: Sequence[RunReport], baseline_run: RunReport) -> List[Dict[str, Any]]:
    delta_rows = []
    for run in runs:
        delta_rows.append(
            {
                "run": run.name,
                "delta_best_joint_combined": delta_against_baseline(run, baseline_run, SUMMARY_METRICS[0]),
                "delta_best_dev_stage1_recall@0.5": delta_against_baseline(run, baseline_run, SUMMARY_METRICS[1]),
                "delta_best_dev_stage2_mIoU": delta_against_baseline(run, baseline_run, SUMMARY_METRICS[2]),
                "delta_best_dev_stage3_acc": delta_against_baseline(run, baseline_run, SUMMARY_METRICS[3]),
            }
        )
    return delta_rows


def build_delta_values(runs: Sequence[RunReport], baseline_run: RunReport, spec: MetricSpec) -> List[Tuple[str, float]]:
    values: List[Tuple[str, float]] = []
    baseline_value = baseline_run.best_value(spec)
    if baseline_value is None:
        return values
    for run in runs:
        candidate = run.best_value(spec)
        if candidate is None:
            continue
        values.append((run.name, candidate - baseline_value))
    return values


def delta_against_baseline(run: RunReport, baseline_run: RunReport, spec: MetricSpec) -> Optional[float]:
    current = run.best_value(spec)
    baseline = baseline_run.best_value(spec)
    if current is None or baseline is None:
        return None
    return current - baseline


def best_or_latest(run: RunReport, metric_key: str, higher_is_better: bool) -> Optional[float]:
    series = run.scalar_series(metric_key)
    if series:
        values = [value for _, value in series]
        return max(values) if higher_is_better else min(values)
    return run.latest_value(metric_key)


def write_summary_csv(path: Path, comparison_rows: Sequence[Dict[str, Any]], efficiency_rows: Sequence[Dict[str, Any]]):
    merged_rows: List[Dict[str, Any]] = []
    efficiency_by_run = {row["run"]: row for row in efficiency_rows}
    for row in comparison_rows:
        merged = dict(row)
        merged.update(efficiency_by_run.get(row["run"], {}))
        merged_rows.append(merged)

    fieldnames = [
        "run",
        "status",
        "best_joint_combined",
        "best_dev_stage1_recall@0.5",
        "best_dev_stage2_mIoU",
        "best_dev_stage3_acc",
        "min_dev_loss_total",
        "avg_train_samples_per_sec",
        "max_gpu_mem_peak_allocated_gb",
        "device",
        "stage3_loss",
        "joint_lr",
        "effective_batch",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: serialize_cell(row.get(field)) for field in fieldnames})


def write_summary_json(
    path: Path,
    *,
    title: str,
    baseline_run: str,
    runs: Sequence[RunReport],
    comparison_rows: Sequence[Dict[str, Any]],
    efficiency_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    generated_plots: Dict[str, str],
):
    payload = {
        "title": title,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": baseline_run,
        "runs": [
            {
                "name": run.name,
                "root_dir": str(run.root_dir),
                "status": run.run_status,
                "device": run.device,
                "notes": run.notes,
            }
            for run in runs
        ],
        "comparison_rows": list(comparison_rows),
        "efficiency_rows": list(efficiency_rows),
        "delta_rows": list(delta_rows),
        "plots": generated_plots,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def render_index_markdown(
    *,
    title: str,
    baseline_run: RunReport,
    comparison_rows: Sequence[Dict[str, Any]],
    efficiency_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    generated_plots: Dict[str, str],
    run_card_paths: Sequence[Tuple[str, str]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- runs_compared: {len(comparison_rows)}",
        f"- baseline_run: {baseline_run.name}",
        "",
        "## Core Metrics",
        "",
        render_markdown_table(
            comparison_rows,
            [
                ("run", "Run"),
                ("status", "Status"),
                ("best_joint_combined", "Best Joint"),
                ("best_dev_stage1_recall@0.5", "Best Dev S1"),
                ("best_dev_stage2_mIoU", "Best Dev S2"),
                ("best_dev_stage3_acc", "Best Dev S3"),
                ("min_dev_loss_total", "Min Dev Loss"),
            ],
        ),
        "",
        "## Efficiency And Setup",
        "",
        render_markdown_table(
            efficiency_rows,
            [
                ("run", "Run"),
                ("device", "Device"),
                ("avg_train_samples_per_sec", "Avg Samples/s"),
                ("max_gpu_mem_peak_allocated_gb", "Peak GPU GB"),
                ("stage3_loss", "Stage 3 Loss"),
                ("joint_lr", "Joint LR"),
                ("effective_batch", "Effective Batch"),
            ],
        ),
        "",
        f"## Improvements vs {baseline_run.name}",
        "",
        render_markdown_table(
            delta_rows,
            [
                ("run", "Run"),
                ("delta_best_joint_combined", "Delta Joint"),
                ("delta_best_dev_stage1_recall@0.5", "Delta S1"),
                ("delta_best_dev_stage2_mIoU", "Delta S2"),
                ("delta_best_dev_stage3_acc", "Delta S3"),
            ],
            signed=True,
        ),
        "",
        "## Trend Charts",
        "",
    ]

    for spec in DEFAULT_METRICS:
        rel_path = generated_plots.get(spec.slug)
        if not rel_path:
            continue
        lines.extend(
            [
                f"### {spec.title}",
                "",
                spec.description,
                "",
                f"![{spec.title}]({rel_path})",
                "",
            ]
        )

    if "best_joint_combined" in generated_plots:
        lines.extend(
            [
                "## Best-Score Comparison",
                "",
                f"![Best Joint Combined]({generated_plots['best_joint_combined']})",
                "",
            ]
        )
    if "delta_joint_combined_vs_baseline" in generated_plots:
        lines.extend(
            [
                "## Baseline Delta Chart",
                "",
                f"![Joint Combined Delta]({generated_plots['delta_joint_combined_vs_baseline']})",
                "",
            ]
        )

    lines.extend(
        [
            "## Run Cards",
            "",
        ]
    )
    for run_name, rel_path in run_card_paths:
        lines.append(f"- [{run_name}]({rel_path})")
    lines.append("")
    return "\n".join(lines)


def render_run_card(run: RunReport) -> str:
    config_rows = [
        {"field": "run", "value": run.name},
        {"field": "status", "value": run.run_status},
        {"field": "device", "value": run.device},
        {"field": "elapsed_sec", "value": run.elapsed_sec},
        {"field": "notes", "value": run.notes or ""},
        {"field": "batch_root", "value": nested_get(run.config, "data", "integration", "batch_root")},
        {"field": "stage3_loss", "value": nested_get(run.config, "stage3", "loss", "name")},
        {"field": "joint_lr", "value": nested_get(run.config, "joint", "training", "learning_rate")},
        {"field": "joint_batch_size", "value": nested_get(run.config, "joint", "training", "batch_size")},
        {"field": "joint_grad_accum", "value": nested_get(run.config, "joint", "training", "grad_accum_steps")},
        {"field": "compile", "value": nested_get(run.config, "hardware", "compile")},
        {"field": "gradient_checkpointing", "value": nested_get(run.config, "hardware", "gradient_checkpointing")},
    ]
    best_rows = [
        {"metric": spec.title, "value": run.best_value(spec)}
        for spec in SUMMARY_METRICS
    ]
    lines = [
        f"# {run.name}",
        "",
        f"- root_dir: `{run.root_dir}`",
        f"- event_count: {len(run.events)}",
        "",
        "## Best Metrics",
        "",
        render_markdown_table(best_rows, [("metric", "Metric"), ("value", "Value")]),
        "",
        "## Config Highlights",
        "",
        render_markdown_table(config_rows, [("field", "Field"), ("value", "Value")]),
        "",
    ]
    return "\n".join(lines)


def render_markdown_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[Tuple[str, str]],
    *,
    signed: bool = False,
) -> str:
    if not rows:
        return "_No data available._"

    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            cells.append(format_cell(value, signed=signed))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_line_chart_svg(
    path: Path,
    title: str,
    series_map: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    *,
    y_label: str,
):
    width, height = 980, 460
    margin_left, margin_right = 86, 24
    margin_top, margin_bottom = 40, 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_x = [x for _, points in series_map for x, _ in points]
    all_y = [y for _, points in series_map for _, y in points]
    x_min, x_max = padded_bounds(all_x)
    y_min, y_max = padded_bounds(all_y, pad_ratio=0.08)

    def sx(value: float) -> float:
        return margin_left + (value - x_min) * plot_width / max(x_max - x_min, 1e-12)

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - y_min) * plot_height / max(y_max - y_min, 1e-12)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="24" font-size="20" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
        f'<text x="{margin_left}" y="{height - 18}" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">step / epoch</text>',
        f'<text x="18" y="{margin_top + 12}" font-size="12" font-family="Arial, sans-serif" fill="#4b5563" transform="rotate(-90 18 {margin_top + 12})">{escape(y_label)}</text>',
    ]

    for tick_value in build_ticks(y_min, y_max, count=5):
        y = sy(tick_value)
        svg_lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg_lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="Arial, sans-serif" fill="#6b7280">{format_number(tick_value)}</text>')
    for tick_value in build_ticks(x_min, x_max, count=6):
        x = sx(tick_value)
        svg_lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#f3f4f6" stroke-width="1"/>')
        svg_lines.append(f'<text x="{x:.2f}" y="{height - 40}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" fill="#6b7280">{format_number(tick_value)}</text>')

    svg_lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d1d5db" stroke-width="1"/>')

    legend_y = 26
    legend_x = width - margin_right - 180
    for index, (name, points) in enumerate(series_map):
        color = SVG_PALETTE[index % len(SVG_PALETTE)]
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        svg_lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        if points:
            last_x, last_y = points[-1]
            svg_lines.append(f'<circle cx="{sx(last_x):.2f}" cy="{sy(last_y):.2f}" r="3.5" fill="{color}"/>')
        line_y = legend_y + index * 18
        svg_lines.append(f'<line x1="{legend_x}" y1="{line_y}" x2="{legend_x + 18}" y2="{line_y}" stroke="{color}" stroke-width="3"/>')
        svg_lines.append(f'<text x="{legend_x + 24}" y="{line_y + 4}" font-size="11" font-family="Arial, sans-serif" fill="#374151">{escape(name)}</text>')

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def write_bar_chart_svg(path: Path, title: str, values: Sequence[Tuple[str, float]], *, signed: bool = False):
    width, height = 980, 460
    margin_left, margin_right = 86, 24
    margin_top, margin_bottom = 40, 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_y = [value for _, value in values]
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    if signed:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    y_min, y_max = padded_bounds([y_min, y_max], pad_ratio=0.12)

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - y_min) * plot_height / max(y_max - y_min, 1e-12)

    bar_count = max(len(values), 1)
    slot_width = plot_width / bar_count
    bar_width = min(56.0, slot_width * 0.6)
    base_y = zero_y = sy(0.0 if signed else y_min)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="24" font-size="20" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
    ]

    for tick_value in build_ticks(y_min, y_max, count=5):
        y = sy(tick_value)
        svg_lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg_lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="Arial, sans-serif" fill="#6b7280">{format_number(tick_value)}</text>')

    svg_lines.append(f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width - margin_right}" y2="{zero_y:.2f}" stroke="#9ca3af" stroke-width="1.5"/>')
    svg_lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d1d5db" stroke-width="1"/>')

    for index, (label, value) in enumerate(values):
        color = "#059669" if (signed and value >= 0.0) else "#dc2626" if signed else SVG_PALETTE[index % len(SVG_PALETTE)]
        center_x = margin_left + slot_width * index + slot_width / 2.0
        x = center_x - bar_width / 2.0
        if signed:
            top = sy(max(value, 0.0))
            bottom = sy(min(value, 0.0))
            rect_y = min(top, bottom)
            rect_h = max(abs(bottom - top), 1.0)
        else:
            rect_y = sy(value)
            rect_h = max(base_y - rect_y, 1.0)
        svg_lines.append(f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width:.2f}" height="{rect_h:.2f}" fill="{color}" rx="4"/>')
        value_y = rect_y - 8 if value >= 0 else rect_y + rect_h + 14
        svg_lines.append(f'<text x="{center_x:.2f}" y="{value_y:.2f}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" fill="#374151">{format_number(value, signed=signed)}</text>')
        svg_lines.append(f'<text x="{center_x:.2f}" y="{height - 48}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" fill="#4b5563">{escape(label)}</text>')

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def padded_bounds(values: Sequence[float], pad_ratio: float = 0.05) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        delta = 1.0 if lower == 0.0 else abs(lower) * 0.1
        return lower - delta, upper + delta
    pad = (upper - lower) * pad_ratio
    return lower - pad, upper + pad


def build_ticks(lower: float, upper: float, count: int) -> List[float]:
    if count <= 1:
        return [lower]
    step = (upper - lower) / float(count - 1)
    return [lower + step * index for index in range(count)]


def nested_get(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def serialize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def format_cell(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format_number(float(value), signed=signed)
    return str(value)


def format_number(value: float, signed: bool = False) -> str:
    if math.isnan(value) or math.isinf(value):
        return "-"
    if abs(value) >= 1000:
        formatted = f"{value:,.1f}"
    elif abs(value) >= 100:
        formatted = f"{value:.2f}"
    elif abs(value) >= 1:
        formatted = f"{value:.4f}"
    else:
        formatted = f"{value:.6f}"
    if signed and value > 0 and not formatted.startswith("+"):
        return f"+{formatted}"
    return formatted


def slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "run"


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return None


if __name__ == "__main__":
    main()
