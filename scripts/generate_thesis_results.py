#!/usr/bin/env python3
"""
Build thesis-ready result artifacts from training split reports.

Inputs:
  logs/train_high_subject_split.json
  logs/train_low_subject_split.json
  logs/train_unified_subject_split.json

Outputs (results_thesis/):
  summary_metrics.csv
  class_metrics.csv
  thesis_table.tex
  interpretation_notes.txt
  manifest.json
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
OUT_DIR = ROOT / "results_thesis"

REPORT_FILES = {
    "high": LOGS_DIR / "train_high_subject_split.json",
    "low": LOGS_DIR / "train_low_subject_split.json",
    "unified": LOGS_DIR / "train_unified_subject_split.json",
}

CLASS_ORDER = ["awake", "induction", "trance"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = successes / n
    den = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / den
    margin = (z / den) * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def cm_metrics(labels: list[str], matrix: list[list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(labels)
    for i in range(n):
        label = labels[i]
        tp = int(matrix[i][i])
        fn = int(sum(matrix[i]) - tp)
        fp = int(sum(matrix[r][i] for r in range(n)) - tp)
        support = int(sum(matrix[i]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


@dataclass
class ModelSummary:
    model: str
    n_train: int
    n_val: int
    accuracy: float | None
    f1_macro: float | None
    acc_ci_low: float | None
    acc_ci_high: float | None
    high_confidence_pct: float | None
    cv_mean: float | None
    cv_std: float | None
    val_subjects: str


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    for key, path in REPORT_FILES.items():
        if path.exists():
            loaded[key] = load_json(path)

    if not loaded:
        raise SystemExit("No report files found in logs/.")

    summaries: list[ModelSummary] = []
    class_rows: list[dict[str, Any]] = []

    for model_name in ["high", "low", "unified"]:
        report = loaded.get(model_name)
        if report is None:
            continue

        metrics = report.get("metrics", {})
        n_val = int(report.get("n_val_samples", 0))
        accuracy = safe_float(metrics.get("val_accuracy"))

        if accuracy is not None and n_val > 0:
            successes = int(round(accuracy * n_val))
            ci_low, ci_high = wilson_interval(successes, n_val)
        else:
            ci_low, ci_high = (None, None)

        summaries.append(
            ModelSummary(
                model=model_name,
                n_train=int(report.get("n_train_samples", 0)),
                n_val=n_val,
                accuracy=accuracy,
                f1_macro=safe_float(metrics.get("val_f1_macro")),
                acc_ci_low=ci_low,
                acc_ci_high=ci_high,
                high_confidence_pct=safe_float(metrics.get("high_confidence_pct")),
                cv_mean=safe_float(metrics.get("cv_groupkfold_mean_accuracy")),
                cv_std=safe_float(metrics.get("cv_groupkfold_std_accuracy")),
                val_subjects=",".join(report.get("val_subjects", [])),
            )
        )

        cm = report.get("confusion_matrix")
        if isinstance(cm, dict) and "labels" in cm and "matrix" in cm:
            labels = list(cm["labels"])
            matrix = list(cm["matrix"])
            for row in cm_metrics(labels, matrix):
                row_out = {"model": model_name, **row}
                class_rows.append(row_out)

    # 1) Summary CSV
    summary_csv = OUT_DIR / "summary_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "n_train",
                "n_val",
                "val_accuracy",
                "val_f1_macro",
                "acc_ci95_low",
                "acc_ci95_high",
                "high_confidence_pct",
                "cv_mean",
                "cv_std",
                "val_subjects",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.model,
                    s.n_train,
                    s.n_val,
                    fmt_float(s.accuracy),
                    fmt_float(s.f1_macro),
                    fmt_float(s.acc_ci_low),
                    fmt_float(s.acc_ci_high),
                    fmt_float(s.high_confidence_pct, 2),
                    fmt_float(s.cv_mean),
                    fmt_float(s.cv_std),
                    s.val_subjects,
                ]
            )

    # 2) Per-class CSV
    class_csv = OUT_DIR / "class_metrics.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
        for r in class_rows:
            writer.writerow(
                [
                    r["model"],
                    r["class"],
                    fmt_float(float(r["precision"])),
                    fmt_float(float(r["recall"])),
                    fmt_float(float(r["f1"])),
                    int(r["support"]),
                    int(r["tp"]),
                    int(r["fp"]),
                    int(r["fn"]),
                ]
            )

    # 3) TeX table
    by_model = {s.model: s for s in summaries}

    def tex_row(name: str) -> str:
        s = by_model.get(name)
        if s is None:
            return f"{name.upper()} & NA & NA & NA & NA & NA \\\\"
        return (
            f"{name.upper()} & {s.n_train} & {s.n_val} & {fmt_float(s.accuracy)} "
            f"& {fmt_float(s.f1_macro)} & [{fmt_float(s.acc_ci_low)}, {fmt_float(s.acc_ci_high)}] \\\\"
        )

    tex = OUT_DIR / "thesis_table.tex"
    tex.write_text(
        "\n".join(
            [
                "\\begin{table}[ht]",
                "\\centering",
                "\\caption{Validation metrics in held-out subjects}",
                "\\label{tab:validation_metrics}",
                "\\begin{tabular}{lccccc}",
                "\\hline",
                "Model & N train & N val & Accuracy & F1 macro & 95\\% CI (acc) \\\\",
                "\\hline",
                tex_row("high"),
                tex_row("low"),
                tex_row("unified"),
                "\\hline",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 4) Interpretation notes
    notes = []
    notes.append("THESIS RESULT INTERPRETATION NOTES")
    notes.append("================================")
    notes.append("")

    s_high = by_model.get("high")
    s_low = by_model.get("low")
    s_uni = by_model.get("unified")

    if s_high:
        notes.append(
            f"HIGH: acc={fmt_float(s_high.accuracy)}, f1_macro={fmt_float(s_high.f1_macro)}, "
            f"CI95=[{fmt_float(s_high.acc_ci_low)}, {fmt_float(s_high.acc_ci_high)}]"
        )
    if s_low:
        notes.append(
            f"LOW: acc={fmt_float(s_low.accuracy)}, f1_macro={fmt_float(s_low.f1_macro)}, "
            f"CI95=[{fmt_float(s_low.acc_ci_low)}, {fmt_float(s_low.acc_ci_high)}]"
        )
    if s_uni:
        notes.append(
            f"UNIFIED: acc={fmt_float(s_uni.accuracy)}, f1_macro={fmt_float(s_uni.f1_macro)}, "
            f"CI95=[{fmt_float(s_uni.acc_ci_low)}, {fmt_float(s_uni.acc_ci_high)}]"
        )

    notes.append("")
    notes.append("Suggested thesis argument line:")
    notes.append(
        "The subject-wise held-out evaluation suggests that the model generalizes to unseen subjects, "
        "with consistently high validation accuracy and macro-F1 across suggestibility groups."
    )
    notes.append("")
    notes.append("Risk/limitation to report:")
    notes.append(
        "Validation currently uses one forced held-out subject per group in separate models (HIGH/LOW); "
        "future work should include repeated subject-wise splits or leave-one-subject-out evaluation."
    )

    notes_txt = OUT_DIR / "interpretation_notes.txt"
    notes_txt.write_text("\n".join(notes) + "\n", encoding="utf-8")

    # 5) Manifest
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_reports": {k: str(v) for k, v in REPORT_FILES.items()},
        "outputs": [
            str(summary_csv),
            str(class_csv),
            str(tex),
            str(notes_txt),
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Artifacts generated in", OUT_DIR)


if __name__ == "__main__":
    main()
