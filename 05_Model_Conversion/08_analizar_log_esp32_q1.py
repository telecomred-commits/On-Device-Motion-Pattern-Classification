#!/usr/bin/env python3
"""
Convierte uno o varios registros seriales del firmware Edge IA en tablas.

Solo las filas RESULT con scored=1 participan en accuracy, Macro-F1,
recall y matriz de confusión. Las dos primeras inferencias después de cada
cambio de etiqueta aparecen con scored=0 y se conservan para latencia.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "record_type",
    "model",
    "window",
    "time_ms",
    "expected_label",
    "scored",
    "predicted_label",
    "predicted_class",
    "p_reposo",
    "p_suave",
    "p_brusco",
    "feature_us",
    "inference_us",
    "total_us",
    "free_heap",
    "min_free_heap",
    "dropped_deadlines",
]
NUMERIC_RESULT_COLUMNS = [
    "window",
    "time_ms",
    "expected_label",
    "scored",
    "predicted_label",
    "p_reposo",
    "p_suave",
    "p_brusco",
    "feature_us",
    "inference_us",
    "total_us",
    "free_heap",
    "min_free_heap",
    "dropped_deadlines",
]
CLASS_NAMES = {0: "reposo", 1: "suave", 2: "brusco"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza registros seriales del benchmark ESP32."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def key_value_record(tokens: list[str]) -> dict[str, str]:
    if len(tokens) < 3 or len(tokens[1:]) % 2 != 0:
        return {}
    return {
        tokens[index]: tokens[index + 1]
        for index in range(1, len(tokens), 2)
    }


def parse_log(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    result_rows: list[list[str]] = []
    metadata: dict[str, str] = {"source_log": path.name}

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()
        result_position = line.find("RESULT,")
        if result_position >= 0:
            tokens = line[result_position:].split(",")
            if len(tokens) == len(RESULT_COLUMNS):
                result_rows.append(tokens)
            continue

        for prefix in ("CONFIG,", "RESOURCES,"):
            position = line.find(prefix)
            if position >= 0:
                tokens = line[position:].split(",")
                values = key_value_record(tokens)
                metadata.update(
                    {f"{tokens[0].lower()}_{key}": value for key, value in values.items()}
                )
                break

    frame = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    if frame.empty:
        raise ValueError(f"{path}: no contiene filas RESULT válidas.")
    for column in NUMERIC_RESULT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["source_log"] = path.name
    return frame, metadata


def percentile_95(values: pd.Series) -> float:
    return float(np.quantile(values.to_numpy(dtype=np.float64), 0.95))


def summarize_model(
    model: str,
    frame: pd.DataFrame,
    metadata_rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    scored = frame[frame["scored"].eq(1)].copy()
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    correct = (
        scored["expected_label"].eq(scored["predicted_label"])
        if not scored.empty
        else pd.Series(dtype=bool)
    )
    f1_values: list[float] = []
    recalls: list[float] = []

    for label in (0, 1, 2):
        true_positive = int(
            (
                scored["expected_label"].eq(label)
                & scored["predicted_label"].eq(label)
            ).sum()
        )
        false_positive = int(
            (
                scored["expected_label"].ne(label)
                & scored["predicted_label"].eq(label)
            ).sum()
        )
        false_negative = int(
            (
                scored["expected_label"].eq(label)
                & scored["predicted_label"].ne(label)
            ).sum()
        )
        support = int(scored["expected_label"].eq(label).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        recalls.append(recall)
        class_rows.append(
            {
                "model": model,
                "label": label,
                "class_name": CLASS_NAMES[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        for predicted in (0, 1, 2):
            count = int(
                (
                    scored["expected_label"].eq(label)
                    & scored["predicted_label"].eq(predicted)
                ).sum()
            )
            confusion_rows.append(
                {
                    "model": model,
                    "true_label": label,
                    "true_class": CLASS_NAMES[label],
                    "predicted_label": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "count": count,
                }
            )

    source_logs = set(frame["source_log"].unique())
    metadata_for_model = [
        row
        for row in metadata_rows
        if row.get("source_log") in source_logs
    ]
    sketch_values = [
        int(row["resources_sketch_bytes"])
        for row in metadata_for_model
        if row.get("resources_sketch_bytes", "").isdigit()
    ]

    summary = {
        "model": model,
        "source_logs": frame["source_log"].nunique(),
        "all_windows": len(frame),
        "scored_windows": len(scored),
        "accuracy": float(correct.mean()) if len(correct) else np.nan,
        "macro_f1": float(np.mean(f1_values)) if len(scored) else np.nan,
        "recall_reposo": recalls[0] if len(scored) else np.nan,
        "recall_suave": recalls[1] if len(scored) else np.nan,
        "recall_brusco": recalls[2] if len(scored) else np.nan,
        "feature_us_mean": frame["feature_us"].mean(),
        "feature_us_std": frame["feature_us"].std(ddof=1),
        "feature_us_p95": percentile_95(frame["feature_us"]),
        "feature_us_max": frame["feature_us"].max(),
        "inference_us_mean": frame["inference_us"].mean(),
        "inference_us_std": frame["inference_us"].std(ddof=1),
        "inference_us_p95": percentile_95(frame["inference_us"]),
        "inference_us_max": frame["inference_us"].max(),
        "total_us_mean": frame["total_us"].mean(),
        "total_us_p95": percentile_95(frame["total_us"]),
        "minimum_free_heap": frame["min_free_heap"].min(),
        "maximum_dropped_deadlines": frame["dropped_deadlines"].max(),
        "sketch_bytes": max(sketch_values) if sketch_values else np.nan,
    }
    return summary, class_rows, confusion_rows


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, str]] = []
    for path in args.inputs:
        frame, metadata = parse_log(path)
        frames.append(frame)
        metadata_rows.append(metadata)

    combined = pd.concat(frames, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for model, group in combined.groupby("model", sort=True):
        summary, classes, confusion = summarize_model(
            model,
            group,
            metadata_rows,
        )
        summary_rows.append(summary)
        class_rows.extend(classes)
        confusion_rows.extend(confusion)

    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "on_device_windows.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        args.output / "on_device_summary.csv",
        index=False,
    )
    pd.DataFrame(class_rows).to_csv(
        args.output / "on_device_class_metrics.csv",
        index=False,
    )
    pd.DataFrame(confusion_rows).to_csv(
        args.output / "on_device_confusion.csv",
        index=False,
    )
    pd.DataFrame(metadata_rows).to_csv(
        args.output / "on_device_resources.csv",
        index=False,
    )

    print("\n=== RESUMEN ESP32 ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nResultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
