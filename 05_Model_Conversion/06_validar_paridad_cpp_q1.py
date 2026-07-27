#!/usr/bin/env python3
"""
Compila los núcleos C++ y compara sus salidas con Python/scikit-learn.

La prueba cubre:
  - las 5.355 ventanas de características para cada uno de los cinco modelos;
  - las 5.355 ventanas reconstruidas desde los 45 archivos crudos;
  - la invariancia de una ventana constante.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_IDS = {
    "logistic_regression": 1,
    "decision_tree_d6": 2,
    "random_forest_compact": 3,
    "mlp_compact_32_16": 4,
    "mlp_reference_64_32": 5,
}
SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz"]
METADATA = [
    "subject",
    "session",
    "rep",
    "class_name",
    "label",
    "source_file",
    "start_idx",
    "end_idx",
    "start_time",
    "end_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida paridad entre Python y C++."
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--feature-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compile_cpp(
    source: Path,
    output: Path,
    project: Path,
    model_id: int | None = None,
) -> None:
    command = [
        "g++",
        "-std=c++11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(project),
    ]
    if model_id is not None:
        command.append(f"-DEDGEIA_ACTIVE_MODEL={model_id}")
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True)


def run_binary(binary: Path, input_text: str) -> np.ndarray:
    result = subprocess.run(
        [str(binary)],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(
        [[float(value) for value in line.split(",")] for line in lines],
        dtype=np.float64,
    )


def csv_text(values: np.ndarray) -> str:
    return "\n".join(
        ",".join(format(float(value), ".9g") for value in row)
        for row in values
    )


def validate_features(
    binary: Path,
    raw: pd.DataFrame,
    expected: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float | int]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    compared = 0

    for source_file, group in raw.groupby("source_file", sort=True):
        ordered = group.sort_values("time", kind="stable")
        actual = run_binary(
            binary,
            csv_text(ordered[SIGNALS].to_numpy(dtype=np.float64)),
        )
        reference = (
            expected[expected["source_file"].eq(source_file)]
            .sort_values("start_idx", kind="stable")[feature_columns]
            .to_numpy(dtype=np.float64)
        )
        if actual.shape != reference.shape:
            raise AssertionError(
                f"{source_file}: C++ {actual.shape}, Python {reference.shape}"
            )
        difference = np.abs(actual - reference)
        scale = np.maximum(np.abs(reference), 1.0e-9)
        maximum_absolute = max(maximum_absolute, float(difference.max()))
        maximum_relative = max(
            maximum_relative,
            float((difference / scale).max()),
        )
        compared += len(reference)

    constant = np.tile(
        np.array([4.0, -7.0, 9.80665, 13.0, -21.0, 2.0]),
        (50, 1),
    )
    constant_features = run_binary(binary, csv_text(constant))
    constant_maximum = float(np.abs(constant_features).max())
    if constant_maximum > 1.0e-6:
        raise AssertionError(
            "La señal constante no produjo características C++ nulas."
        )
    if maximum_absolute > 2.0e-3 or maximum_relative > 2.0e-4:
        raise AssertionError(
            "La extracción C++ excedió la tolerancia: "
            f"abs={maximum_absolute}, rel={maximum_relative}"
        )
    return {
        "windows_compared": compared,
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "constant_vector_maximum": constant_maximum,
    }


def validate_model(
    binary: Path,
    model_path: Path,
    features: np.ndarray,
    feature_columns: list[str],
) -> dict[str, float | int]:
    estimator = joblib.load(model_path)
    named_features = pd.DataFrame(features, columns=feature_columns)
    python_probabilities = estimator.predict_proba(named_features)
    python_prediction = estimator.predict(named_features).astype(int)

    actual = run_binary(binary, csv_text(features))
    cpp_prediction = actual[:, 0].astype(int)
    cpp_probabilities = actual[:, 1:4]
    difference = np.abs(cpp_probabilities - python_probabilities)
    mismatches = int(np.count_nonzero(cpp_prediction != python_prediction))
    probability_sum_error = float(
        np.abs(cpp_probabilities.sum(axis=1) - 1.0).max()
    )

    zero = np.zeros((1, features.shape[1]), dtype=np.float64)
    zero_actual = run_binary(binary, csv_text(zero))
    zero_prediction = int(zero_actual[0, 0])

    if mismatches != 0:
        raise AssertionError(
            f"{model_path.name}: {mismatches} etiquetas diferentes."
        )
    if float(difference.max()) > 5.0e-4:
        raise AssertionError(
            f"{model_path.name}: error de probabilidad "
            f"{float(difference.max())}."
        )
    if probability_sum_error > 5.0e-6:
        raise AssertionError(
            f"{model_path.name}: probabilidades no normalizadas."
        )
    if zero_prediction != 0:
        raise AssertionError(
            f"{model_path.name}: el vector nulo no produjo reposo."
        )

    return {
        "windows_compared": len(features),
        "class_mismatches": mismatches,
        "maximum_probability_absolute_error": float(difference.max()),
        "mean_probability_absolute_error": float(difference.mean()),
        "maximum_probability_sum_error": probability_sum_error,
        "zero_dynamic_prediction": zero_prediction,
    }


def main() -> None:
    args = parse_args()
    project = args.project_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_dataset)
    feature_frame = pd.read_csv(args.feature_dataset)
    feature_columns = [
        column for column in feature_frame.columns if column not in METADATA
    ]
    features = feature_frame[feature_columns].to_numpy(dtype=np.float64)

    report_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="edgeia_parity_") as temporary:
        build = Path(temporary)
        feature_binary = build / "feature_runner"
        compile_cpp(
            project / "tests" / "feature_runner.cpp",
            feature_binary,
            project,
        )
        feature_report = validate_features(
            feature_binary,
            raw,
            feature_frame,
            feature_columns,
        )
        summary["feature_extraction"] = feature_report
        report_rows.append(
            {"component": "dynamic_features", **feature_report}
        )

        model_reports: dict[str, object] = {}
        for slug, model_id in MODEL_IDS.items():
            binary = build / f"model_{model_id}"
            compile_cpp(
                project / "tests" / "model_runner.cpp",
                binary,
                project,
                model_id,
            )
            model_report = validate_model(
                binary,
                args.models_dir / f"{slug}.joblib",
                features,
                feature_columns,
            )
            model_reports[slug] = model_report
            report_rows.append({"component": slug, **model_report})
            print(
                f"{slug}: 0 diferencias de clase; "
                f"error máximo de probabilidad="
                f"{model_report['maximum_probability_absolute_error']:.3e}"
            )
        summary["models"] = model_reports

    pd.DataFrame(report_rows).to_csv(
        args.output_dir / "parity_report.csv",
        index=False,
    )
    (args.output_dir / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "Características: "
        f"{feature_report['windows_compared']} ventanas; "
        f"error absoluto máximo="
        f"{feature_report['maximum_absolute_error']:.3e}"
    )
    print("PARIDAD C++ VALIDADA")


if __name__ == "__main__":
    main()
