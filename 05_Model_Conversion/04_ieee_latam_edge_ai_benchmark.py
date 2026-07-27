#!/usr/bin/env python3
"""
Comparación reproducible de modelos Edge AI para IEEE Latin America Transactions.

El script:
1. valida el dataset de 36 características;
2. ejecuta validación externa Leave-One-Subject-Out (LOSO);
3. opcionalmente selecciona hiperparámetros mediante LOSO interno;
4. calcula métricas globales, por clase y por participante;
5. realiza comparaciones pareadas agrupadas por participante;
6. entrena los modelos finales con todos los participantes;
7. exporta modelos joblib y cabeceras C++ para ESP32;
8. comprueba la paridad entre scikit-learn y las representaciones exportadas;
9. genera vectores de prueba para el futuro firmware de benchmarking.

Modos:
    quick: hiperparámetros fijos; útil para verificar el flujo.
    full:  selección anidada de hiperparámetros; usar para el artículo.

Ejemplos:
    python 04_ieee_latam_edge_ai_benchmark.py \
        --input "dataset_features_q1(1).csv" \
        --output resultados_ieee_latam \
        --mode quick

    python 04_ieee_latam_edge_ai_benchmark.py \
        --input "dataset_features_q1(1).csv" \
        --output resultados_ieee_latam_full \
        --mode full --jobs -1 --bootstrap 10000

Las métricas de tiempo generadas en Python se identifican expresamente como
tiempos del computador. No sustituyen las mediciones de latencia en el ESP32.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

_matplotlib_cache = Path(tempfile.gettempdir()) / "ieee_latam_matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


META_COLUMNS = [
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

EXPECTED_SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz"]
EXPECTED_STATISTICS = ["mean", "std", "min", "max", "rms", "ptp"]
EXPECTED_FEATURES = [
    f"{signal}_{stat}"
    for signal in EXPECTED_SIGNALS
    for stat in EXPECTED_STATISTICS
]

PRIMARY_METRIC = "f1_macro"
FLOAT_BYTES = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]
    deployment_family: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparación LOSO y exportación ESP32 de modelos Edge AI."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_features_q1.csv"),
        help="CSV con metadatos y 36 características.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados_ieee_latam_edge_ai"),
        help="Directorio de resultados.",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="full",
        help="full (predeterminado) ejecuta selección LOSO anidada; quick usa parámetros fijos.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Procesos para GridSearchCV y Random Forest.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Remuestreos agrupados por participante para intervalos descriptivos.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benchmark-per-subject-class",
        type=int,
        default=20,
        help="Ventanas por combinación participante-clase para vectores ESP32.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    candidates = [
        path,
        Path.cwd() / path,
        Path.cwd() / "upload" / path.name,
        Path(__file__).resolve().parent.parent / "upload" / path.name,
        Path(__file__).resolve().parent.parent
        / "upload"
        / "dataset_features_q1(1).csv",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"No se encontró el dataset. Rutas revisadas:\n{checked}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_directories(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
        "headers": root / "esp32_headers",
        "benchmark": root / "esp32_benchmark",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_and_validate_dataset(path: Path) -> tuple[pd.DataFrame, list[str], list[int]]:
    df = pd.read_csv(path)

    required = {"subject", "session", "class_name", "label"}
    missing_meta = sorted(required.difference(df.columns))
    if missing_meta:
        raise ValueError(f"Faltan columnas obligatorias: {missing_meta}")

    missing_features = sorted(set(EXPECTED_FEATURES).difference(df.columns))
    if missing_features:
        raise ValueError(
            "El dataset no contiene las 36 características esperadas. "
            f"Faltan: {missing_features}"
        )

    feature_cols = EXPECTED_FEATURES.copy()
    non_numeric = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(f"Características no numéricas: {non_numeric}")

    if df[feature_cols].isna().any().any():
        locations = np.argwhere(df[feature_cols].isna().to_numpy())
        first = locations[0]
        raise ValueError(
            "Hay valores faltantes. Primer caso: "
            f"fila={int(first[0])}, columna={feature_cols[int(first[1])]}"
        )

    values = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("El dataset contiene valores infinitos.")

    labels = sorted(int(x) for x in df["label"].unique())
    if labels != list(range(len(labels))):
        raise ValueError(
            f"Las etiquetas deben ser enteros consecutivos desde cero: {labels}"
        )
    if len(labels) != 3:
        raise ValueError(f"Se esperaban tres clases; se encontraron {len(labels)}.")

    subjects = sorted(df["subject"].unique())
    if len(subjects) < 3:
        raise ValueError("LOSO requiere al menos tres participantes.")

    mapping_counts = (
        df[["label", "class_name"]].drop_duplicates().groupby("label").size()
    )
    if not (mapping_counts == 1).all():
        raise ValueError("Cada etiqueta debe corresponder a un único nombre de clase.")

    return df.reset_index(drop=True), feature_cols, labels


def build_model_specs(seed: int, jobs: int) -> list[ModelSpec]:
    common_mlp = dict(
        activation="relu",
        solver="adam",
        batch_size=16,
        max_iter=1200,
        early_stopping=False,
        random_state=seed,
    )

    return [
        ModelSpec(
            name="Logistic Regression",
            slug="logistic_regression",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            max_iter=4000,
                            solver="lbfgs",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            parameter_grid={"model__C": [0.1, 1.0, 10.0]},
            deployment_family="linear",
            description="Referencia lineal de bajo costo.",
        ),
        ModelSpec(
            name="Decision Tree d6",
            slug="decision_tree_d6",
            estimator=DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=5,
                random_state=seed,
            ),
            parameter_grid={
                "max_depth": [4, 6, 8],
                "min_samples_leaf": [1, 5],
            },
            deployment_family="tree",
            description="Árbol compacto basado en comparaciones.",
        ),
        ModelSpec(
            name="Random Forest compact",
            slug="random_forest_compact",
            estimator=RandomForestClassifier(
                n_estimators=25,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=jobs,
            ),
            parameter_grid={
                "n_estimators": [15, 25],
                "max_depth": [6, 8],
                "min_samples_leaf": [1, 3],
            },
            deployment_family="forest",
            description="Ensamble restringido para despliegue.",
        ),
        ModelSpec(
            name="MLP compact 32-16",
            slug="mlp_compact_32_16",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(32, 16),
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            **common_mlp,
                        ),
                    ),
                ]
            ),
            parameter_grid={
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [5e-4, 1e-3],
            },
            deployment_family="mlp",
            description="Red neuronal compacta.",
        ),
        ModelSpec(
            name="MLP reference 64-32",
            slug="mlp_reference_64_32",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            **common_mlp,
                        ),
                    ),
                ]
            ),
            parameter_grid={
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [5e-4, 1e-3],
            },
            deployment_family="mlp",
            description="Arquitectura neuronal utilizada como referencia.",
        ),
    ]


def fit_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    mode: str,
    jobs: int,
) -> tuple[BaseEstimator, dict[str, Any], float | None]:
    estimator = clone(spec.estimator)

    if mode == "quick":
        estimator.fit(X_train, y_train)
        return estimator, estimator.get_params(deep=True), None

    inner_logo = LeaveOneGroupOut()
    inner_cv = list(inner_logo.split(X_train, y_train, groups_train))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=spec.parameter_grid,
        scoring="f1_macro",
        cv=inner_cv,
        refit=True,
        n_jobs=jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def aligned_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    labels: list[int],
) -> np.ndarray:
    raw = estimator.predict_proba(X)
    classes = [int(x) for x in estimator.classes_]
    aligned = np.zeros((len(raw), len(labels)), dtype=np.float64)
    label_position = {label: idx for idx, label in enumerate(labels)}
    for source_idx, label in enumerate(classes):
        aligned[:, label_position[label]] = raw[:, source_idx]
    return aligned


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, labels: list[int]) -> float:
    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    positions = {label: idx for idx, label in enumerate(labels)}
    for row, label in enumerate(y_true):
        one_hot[row, positions[int(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def evaluate_outer_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[int],
    specs: list[ModelSpec],
    mode: str,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df[feature_cols]
    y = df["label"].astype(int)
    groups = df["subject"]
    logo = LeaveOneGroupOut()

    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    splits = list(logo.split(X, y, groups))
    total_fits = len(specs) * len(splits)
    completed = 0

    for spec in specs:
        for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
            held_out_subjects = sorted(df.iloc[test_idx]["subject"].unique())
            if len(held_out_subjects) != 1:
                raise RuntimeError("Cada fold LOSO debe contener un participante.")
            held_out_subject = held_out_subjects[0]

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            group_train = groups.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            fit_start = time.perf_counter()
            estimator, best_params, inner_best_f1 = fit_model(
                spec,
                X_train,
                y_train,
                group_train,
                mode,
                jobs,
            )
            fit_time = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred = estimator.predict(X_test).astype(int)
            probabilities = aligned_probabilities(estimator, X_test, labels)
            predict_time = time.perf_counter() - predict_start

            y_test_array = y_test.to_numpy(dtype=int)
            fold_rows.append(
                {
                    "model": spec.name,
                    "model_slug": spec.slug,
                    "outer_fold": fold_number,
                    "held_out_subject": held_out_subject,
                    "n_train_windows": len(train_idx),
                    "n_test_windows": len(test_idx),
                    "accuracy": accuracy_score(y_test_array, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_test_array, y_pred
                    ),
                    "precision_macro": precision_score(
                        y_test_array, y_pred, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_test_array, y_pred, average="macro", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_test_array, y_pred, average="macro", zero_division=0
                    ),
                    "log_loss": log_loss(
                        y_test_array, probabilities, labels=labels
                    ),
                    "brier_multiclass": multiclass_brier(
                        y_test_array, probabilities, labels
                    ),
                    "fit_time_pc_s": fit_time,
                    "predict_fold_time_pc_s": predict_time,
                    "predict_per_window_pc_us": predict_time
                    / len(test_idx)
                    * 1e6,
                    "inner_best_f1_macro": inner_best_f1,
                    "selected_parameters": json.dumps(
                        best_params, sort_keys=True, default=str
                    ),
                }
            )

            precision, recall, f1, support = precision_recall_fscore_support(
                y_test_array,
                y_pred,
                labels=labels,
                zero_division=0,
            )
            for idx, label in enumerate(labels):
                class_rows.append(
                    {
                        "model": spec.name,
                        "outer_fold": fold_number,
                        "held_out_subject": held_out_subject,
                        "label": label,
                        "precision": precision[idx],
                        "recall": recall[idx],
                        "f1": f1[idx],
                        "support": int(support[idx]),
                    }
                )

            metadata_cols = [
                col
                for col in [
                    "subject",
                    "session",
                    "class_name",
                    "source_file",
                    "start_idx",
                    "end_idx",
                    "start_time",
                    "end_time",
                ]
                if col in df.columns
            ]
            pred_df = df.iloc[test_idx][metadata_cols].copy()
            pred_df.insert(0, "row_id", test_idx)
            pred_df.insert(0, "model_slug", spec.slug)
            pred_df.insert(0, "model", spec.name)
            pred_df["true_label"] = y_test_array
            pred_df["predicted_label"] = y_pred
            pred_df["confidence"] = np.max(probabilities, axis=1)
            for probability_idx, label in enumerate(labels):
                pred_df[f"probability_label_{label}"] = probabilities[
                    :, probability_idx
                ]
            prediction_frames.append(pred_df)

            completed += 1
            print(
                f"[{completed:02d}/{total_fits:02d}] "
                f"{spec.name} | sujeto externo={held_out_subject} | "
                f"F1={fold_rows[-1]['f1_macro']:.5f}"
            )

    folds = pd.DataFrame(fold_rows)
    classes = pd.DataFrame(class_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return folds, classes, predictions


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def summarize_folds(
    folds: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "brier_multiclass",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in folds.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model_name,
            "n_independent_subjects": group["held_out_subject"].nunique(),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(
                values, bootstrap_repetitions, rng
            )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
            row[f"{metric}_cluster_bootstrap_ci95_low"] = low
            row[f"{metric}_cluster_bootstrap_ci95_high"] = high
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )


def confusion_tables(
    predictions: pd.DataFrame,
    labels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        cm = confusion_matrix(
            group["true_label"],
            group["predicted_label"],
            labels=labels,
        )
        row_totals = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm,
            row_totals,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_totals != 0,
        )
        for true_idx, true_label in enumerate(labels):
            for pred_idx, pred_label in enumerate(labels):
                count_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(cm[true_idx, pred_idx]),
                    }
                )
                normalized_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "row_normalized_rate": cm_norm[true_idx, pred_idx],
                    }
                )
    return pd.DataFrame(count_rows), pd.DataFrame(normalized_rows)


def exact_paired_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.allclose(differences, 0.0):
        return 1.0
    observed = abs(float(np.mean(differences)))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistics.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(
        np.mean(np.asarray(statistics) >= observed - np.finfo(float).eps * 10)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted.tolist()


def paired_model_comparisons(folds: pd.DataFrame) -> pd.DataFrame:
    pivot = folds.pivot(
        index="held_out_subject",
        columns="model",
        values="f1_macro",
    )
    rows: list[dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(pivot.columns, 2):
        difference = (
            pivot[model_a].to_numpy(dtype=np.float64)
            - pivot[model_b].to_numpy(dtype=np.float64)
        )
        tolerance = 1e-12
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_f1_difference_a_minus_b": float(np.mean(difference)),
                "median_f1_difference_a_minus_b": float(np.median(difference)),
                "wins_a": int(np.sum(difference > tolerance)),
                "ties": int(np.sum(np.abs(difference) <= tolerance)),
                "wins_b": int(np.sum(difference < -tolerance)),
                "n_subjects": len(difference),
                "exact_sign_flip_p_unadjusted": exact_paired_sign_flip_pvalue(
                    difference
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["exact_sign_flip_p_holm"] = holm_adjust(
        result["exact_sign_flip_p_unadjusted"].tolist()
    )
    return result.sort_values(
        ["exact_sign_flip_p_holm", "exact_sign_flip_p_unadjusted"]
    ).reset_index(drop=True)


def class_name_map(df: pd.DataFrame) -> dict[int, str]:
    mapping = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )
    return {int(key): str(value) for key, value in mapping.items()}


def c_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("No se pueden exportar NaN o infinito a C++.")
    value_32 = np.float32(value)
    if abs(float(value_32)) < float(np.finfo(np.float32).tiny):
        value_32 = np.float32(0.0)
    return f"{float(value_32):.9e}f"


def format_c_array(
    name: str,
    values: np.ndarray,
    c_type: str,
    values_per_line: int = 8,
    formatter: Callable[[Any], str] | None = None,
) -> str:
    flat = np.asarray(values).reshape(-1)
    if formatter is None:
        formatter = str
    lines = []
    for start in range(0, len(flat), values_per_line):
        chunk = ", ".join(formatter(value) for value in flat[start : start + values_per_line])
        lines.append(f"    {chunk}")
    body = ",\n".join(lines)
    return f"static const {c_type} {name}[{len(flat)}] = {{\n{body}\n}};\n"


def unwrap_pipeline(estimator: BaseEstimator) -> tuple[StandardScaler | None, BaseEstimator]:
    if isinstance(estimator, Pipeline):
        scaler = estimator.named_steps.get("scaler")
        core = estimator.named_steps["model"]
        return scaler, core
    return None, estimator


def named_reference_input(
    estimator: BaseEstimator,
    X: np.ndarray,
) -> pd.DataFrame | np.ndarray:
    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None:
        _, core = unwrap_pipeline(estimator)
        feature_names = getattr(core, "feature_names_in_", None)
    if feature_names is None:
        return X
    return pd.DataFrame(X, columns=list(feature_names))


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def export_logistic_regression(
    estimator: Pipeline,
    X: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    scaler, model = unwrap_pipeline(estimator)
    if scaler is None or not isinstance(model, LogisticRegression):
        raise TypeError("Se esperaba Pipeline(StandardScaler, LogisticRegression).")

    weights = np.asarray(model.coef_, dtype=np.float64)
    bias = np.asarray(model.intercept_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    fused_weights = (weights / scale[np.newaxis, :]).astype(np.float32)
    fused_bias = (
        bias - np.sum(weights * (mean / scale)[np.newaxis, :], axis=1)
    ).astype(np.float32)

    namespace = "edge_logistic_regression"
    text = [
        "#pragma once\n#include <math.h>\n#include <stdint.h>\n",
        f"namespace {namespace} {{\n",
        f"static constexpr int INPUT_SIZE = {fused_weights.shape[1]};\n",
        f"static constexpr int CLASS_COUNT = {fused_weights.shape[0]};\n",
        format_c_array("WEIGHTS", fused_weights, "float", formatter=c_float),
        format_c_array("BIAS", fused_bias, "float", formatter=c_float),
        """
inline int predict(const float* x, float* probabilities = nullptr) {
    float scores[CLASS_COUNT];
    int best_class = 0;
    for (int c = 0; c < CLASS_COUNT; ++c) {
        float value = BIAS[c];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            value += x[i] * WEIGHTS[c * INPUT_SIZE + i];
        }
        scores[c] = value;
        if (scores[c] > scores[best_class]) best_class = c;
    }
    if (probabilities != nullptr) {
        float maximum = scores[0];
        for (int c = 1; c < CLASS_COUNT; ++c) {
            if (scores[c] > maximum) maximum = scores[c];
        }
        float total = 0.0f;
        for (int c = 0; c < CLASS_COUNT; ++c) {
            probabilities[c] = expf(scores[c] - maximum);
            total += probabilities[c];
        }
        for (int c = 0; c < CLASS_COUNT; ++c) probabilities[c] /= total;
    }
    return best_class;
}
}  // namespace edge_logistic_regression
""",
    ]
    output_path.write_text("".join(text), encoding="utf-8")

    X_float32 = np.asarray(X, dtype=np.float32)
    logits = X_float32 @ fused_weights.T + fused_bias
    probabilities = softmax_numpy(logits)
    exported_predictions = np.argmax(probabilities, axis=1)
    reference_input = named_reference_input(estimator, X)
    reference_predictions = estimator.predict(reference_input).astype(int)
    reference_probabilities = estimator.predict_proba(reference_input)

    return {
        "parameter_count": int(fused_weights.size + fused_bias.size),
        "model_payload_bytes": int(
            (fused_weights.size + fused_bias.size) * FLOAT_BYTES
        ),
        "activation_ram_bytes": int(fused_bias.size * FLOAT_BYTES),
        "theoretical_operations": int(fused_weights.size),
        "parity_rate": float(np.mean(exported_predictions == reference_predictions)),
        "max_probability_error": float(
            np.max(np.abs(probabilities - reference_probabilities))
        ),
    }


def tree_flat_arrays(
    tree_model: DecisionTreeClassifier,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = tree_model.tree_
    left = tree.children_left.astype(np.int64)
    right = tree.children_right.astype(np.int64)
    feature = tree.feature.astype(np.int64)
    threshold = tree.threshold.astype(np.float32)
    probabilities = np.zeros((tree.node_count, class_count), dtype=np.float32)
    for node in range(tree.node_count):
        if left[node] == -1:
            counts = tree.value[node][0].astype(np.float32)
            total = counts.sum()
            probabilities[node, : len(counts)] = counts / total if total else 0.0
    return left, right, feature, threshold, probabilities


def predict_exported_tree(
    X: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    outputs = np.zeros((len(X), probabilities.shape[1]), dtype=np.float32)
    for row_idx, row in enumerate(X):
        node = 0
        while left[node] != -1:
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        outputs[row_idx] = probabilities[node]
    return np.argmax(outputs, axis=1), outputs


def export_decision_tree(
    estimator: DecisionTreeClassifier,
    X: np.ndarray,
    output_path: Path,
    class_count: int,
) -> dict[str, Any]:
    left, right, feature, threshold, probabilities = tree_flat_arrays(
        estimator, class_count
    )
    node_count = len(left)
    integer_type = "int16_t" if node_count < np.iinfo(np.int16).max else "int32_t"

    text = [
        "#pragma once\n#include <stdint.h>\n",
        "namespace edge_decision_tree {\n",
        f"static constexpr int INPUT_SIZE = {X.shape[1]};\n",
        f"static constexpr int CLASS_COUNT = {class_count};\n",
        f"static constexpr int NODE_COUNT = {node_count};\n",
        format_c_array("LEFT", left, integer_type, formatter=lambda x: str(int(x))),
        format_c_array("RIGHT", right, integer_type, formatter=lambda x: str(int(x))),
        format_c_array(
            "FEATURE", feature, "int16_t", formatter=lambda x: str(int(x))
        ),
        format_c_array("THRESHOLD", threshold, "float", formatter=c_float),
        format_c_array(
            "LEAF_PROBABILITY", probabilities, "float", formatter=c_float
        ),
        """
inline int predict(const float* x, float* output_probability = nullptr) {
    int node = 0;
    while (LEFT[node] != -1) {
        node = (x[FEATURE[node]] <= THRESHOLD[node]) ? LEFT[node] : RIGHT[node];
    }
    int best_class = 0;
    for (int c = 0; c < CLASS_COUNT; ++c) {
        const float probability = LEAF_PROBABILITY[node * CLASS_COUNT + c];
        if (output_probability != nullptr) output_probability[c] = probability;
        if (probability > LEAF_PROBABILITY[node * CLASS_COUNT + best_class]) {
            best_class = c;
        }
    }
    return best_class;
}
}  // namespace edge_decision_tree
""",
    ]
    output_path.write_text("".join(text), encoding="utf-8")

    exported_predictions, exported_probabilities = predict_exported_tree(
        X, left, right, feature, threshold, probabilities
    )
    reference_input = named_reference_input(estimator, X)
    reference_predictions = estimator.predict(reference_input).astype(int)
    reference_probabilities = estimator.predict_proba(reference_input)
    index_bytes = 2 if integer_type == "int16_t" else 4

    return {
        "parameter_count": node_count,
        "model_payload_bytes": int(
            node_count
            * (index_bytes * 2 + 2 + FLOAT_BYTES + class_count * FLOAT_BYTES)
        ),
        "activation_ram_bytes": class_count * FLOAT_BYTES,
        "theoretical_operations": int(estimator.get_depth()),
        "node_count": node_count,
        "maximum_depth": int(estimator.get_depth()),
        "parity_rate": float(np.mean(exported_predictions == reference_predictions)),
        "max_probability_error": float(
            np.max(np.abs(exported_probabilities - reference_probabilities))
        ),
    }


def forest_flat_arrays(
    model: RandomForestClassifier,
    class_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    offsets: list[int] = []
    all_left: list[np.ndarray] = []
    all_right: list[np.ndarray] = []
    all_feature: list[np.ndarray] = []
    all_threshold: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []
    current_offset = 0

    for tree_estimator in model.estimators_:
        left, right, feature, threshold, probabilities = tree_flat_arrays(
            tree_estimator, class_count
        )
        left = np.where(left == -1, -1, left + current_offset)
        right = np.where(right == -1, -1, right + current_offset)
        offsets.append(current_offset)
        all_left.append(left)
        all_right.append(right)
        all_feature.append(feature)
        all_threshold.append(threshold)
        all_probability.append(probabilities)
        current_offset += len(left)

    return (
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(all_left),
        np.concatenate(all_right),
        np.concatenate(all_feature),
        np.concatenate(all_threshold),
        np.concatenate(all_probability),
    )


def predict_exported_forest(
    X: np.ndarray,
    offsets: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    leaf_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    outputs = np.zeros((len(X), leaf_probability.shape[1]), dtype=np.float32)
    for row_idx, row in enumerate(X):
        scores = np.zeros(leaf_probability.shape[1], dtype=np.float32)
        for offset in offsets:
            node = int(offset)
            while left[node] != -1:
                node = (
                    int(left[node])
                    if row[feature[node]] <= threshold[node]
                    else int(right[node])
                )
            scores += leaf_probability[node]
        outputs[row_idx] = scores / len(offsets)
    return np.argmax(outputs, axis=1), outputs


def export_random_forest(
    estimator: RandomForestClassifier,
    X: np.ndarray,
    output_path: Path,
    class_count: int,
) -> dict[str, Any]:
    (
        offsets,
        left,
        right,
        feature,
        threshold,
        probabilities,
    ) = forest_flat_arrays(estimator, class_count)
    node_count = len(left)
    integer_type = "int16_t" if node_count < np.iinfo(np.int16).max else "int32_t"
    index_bytes = 2 if integer_type == "int16_t" else 4

    text = [
        "#pragma once\n#include <stdint.h>\n",
        "namespace edge_random_forest {\n",
        f"static constexpr int INPUT_SIZE = {X.shape[1]};\n",
        f"static constexpr int CLASS_COUNT = {class_count};\n",
        f"static constexpr int TREE_COUNT = {len(offsets)};\n",
        f"static constexpr int NODE_COUNT = {node_count};\n",
        format_c_array(
            "TREE_OFFSET", offsets, integer_type, formatter=lambda x: str(int(x))
        ),
        format_c_array("LEFT", left, integer_type, formatter=lambda x: str(int(x))),
        format_c_array("RIGHT", right, integer_type, formatter=lambda x: str(int(x))),
        format_c_array(
            "FEATURE", feature, "int16_t", formatter=lambda x: str(int(x))
        ),
        format_c_array("THRESHOLD", threshold, "float", formatter=c_float),
        format_c_array(
            "LEAF_PROBABILITY", probabilities, "float", formatter=c_float
        ),
        """
inline int predict(const float* x, float* output_probability = nullptr) {
    float scores[CLASS_COUNT] = {0.0f};
    for (int tree = 0; tree < TREE_COUNT; ++tree) {
        int node = TREE_OFFSET[tree];
        while (LEFT[node] != -1) {
            node = (x[FEATURE[node]] <= THRESHOLD[node]) ? LEFT[node] : RIGHT[node];
        }
        for (int c = 0; c < CLASS_COUNT; ++c) {
            scores[c] += LEAF_PROBABILITY[node * CLASS_COUNT + c];
        }
    }
    int best_class = 0;
    for (int c = 0; c < CLASS_COUNT; ++c) {
        scores[c] /= TREE_COUNT;
        if (output_probability != nullptr) output_probability[c] = scores[c];
        if (scores[c] > scores[best_class]) best_class = c;
    }
    return best_class;
}
}  // namespace edge_random_forest
""",
    ]
    output_path.write_text("".join(text), encoding="utf-8")

    exported_predictions, exported_probabilities = predict_exported_forest(
        X, offsets, left, right, feature, threshold, probabilities
    )
    reference_input = named_reference_input(estimator, X)
    reference_predictions = estimator.predict(reference_input).astype(int)
    reference_probabilities = estimator.predict_proba(reference_input)
    depth_sum = sum(tree.get_depth() for tree in estimator.estimators_)

    return {
        "parameter_count": node_count,
        "model_payload_bytes": int(
            len(offsets) * index_bytes
            + node_count
            * (index_bytes * 2 + 2 + FLOAT_BYTES + class_count * FLOAT_BYTES)
        ),
        "activation_ram_bytes": class_count * FLOAT_BYTES,
        "theoretical_operations": int(depth_sum),
        "tree_count": int(len(offsets)),
        "node_count": int(node_count),
        "maximum_depth": int(
            max(tree.get_depth() for tree in estimator.estimators_)
        ),
        "parity_rate": float(np.mean(exported_predictions == reference_predictions)),
        "max_probability_error": float(
            np.max(np.abs(exported_probabilities - reference_probabilities))
        ),
    }


def export_mlp(
    estimator: Pipeline,
    X: np.ndarray,
    output_path: Path,
    namespace: str,
) -> dict[str, Any]:
    scaler, model = unwrap_pipeline(estimator)
    if scaler is None or not isinstance(model, MLPClassifier):
        raise TypeError("Se esperaba Pipeline(StandardScaler, MLPClassifier).")
    if len(model.coefs_) != 3:
        raise ValueError("El exportador admite exactamente dos capas ocultas.")

    coefs = [np.asarray(weight, dtype=np.float64).copy() for weight in model.coefs_]
    biases = [np.asarray(bias, dtype=np.float64).copy() for bias in model.intercepts_]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    coefs[0] = coefs[0] / scale[:, np.newaxis]
    biases[0] = biases[0] - (mean / scale) @ np.asarray(
        model.coefs_[0], dtype=np.float64
    )
    coefs = [weight.astype(np.float32) for weight in coefs]
    biases = [bias.astype(np.float32) for bias in biases]

    input_size, hidden_1 = coefs[0].shape
    hidden_2 = coefs[1].shape[1]
    class_count = coefs[2].shape[1]

    text = [
        "#pragma once\n#include <math.h>\n#include <stdint.h>\n",
        f"namespace {namespace} {{\n",
        f"static constexpr int INPUT_SIZE = {input_size};\n",
        f"static constexpr int HIDDEN_1 = {hidden_1};\n",
        f"static constexpr int HIDDEN_2 = {hidden_2};\n",
        f"static constexpr int CLASS_COUNT = {class_count};\n",
        format_c_array("W1", coefs[0], "float", formatter=c_float),
        format_c_array("B1", biases[0], "float", formatter=c_float),
        format_c_array("W2", coefs[1], "float", formatter=c_float),
        format_c_array("B2", biases[1], "float", formatter=c_float),
        format_c_array("W3", coefs[2], "float", formatter=c_float),
        format_c_array("B3", biases[2], "float", formatter=c_float),
        """
inline float relu(float value) { return value > 0.0f ? value : 0.0f; }

inline int predict(const float* x, float* probabilities = nullptr) {
    static float hidden1[HIDDEN_1];
    static float hidden2[HIDDEN_2];
    float logits[CLASS_COUNT];

    for (int j = 0; j < HIDDEN_1; ++j) {
        float value = B1[j];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            value += x[i] * W1[i * HIDDEN_1 + j];
        }
        hidden1[j] = relu(value);
    }
    for (int j = 0; j < HIDDEN_2; ++j) {
        float value = B2[j];
        for (int i = 0; i < HIDDEN_1; ++i) {
            value += hidden1[i] * W2[i * HIDDEN_2 + j];
        }
        hidden2[j] = relu(value);
    }
    int best_class = 0;
    for (int c = 0; c < CLASS_COUNT; ++c) {
        float value = B3[c];
        for (int i = 0; i < HIDDEN_2; ++i) {
            value += hidden2[i] * W3[i * CLASS_COUNT + c];
        }
        logits[c] = value;
        if (logits[c] > logits[best_class]) best_class = c;
    }
    if (probabilities != nullptr) {
        float maximum = logits[0];
        for (int c = 1; c < CLASS_COUNT; ++c) {
            if (logits[c] > maximum) maximum = logits[c];
        }
        float total = 0.0f;
        for (int c = 0; c < CLASS_COUNT; ++c) {
            probabilities[c] = expf(logits[c] - maximum);
            total += probabilities[c];
        }
        for (int c = 0; c < CLASS_COUNT; ++c) probabilities[c] /= total;
    }
    return best_class;
}
}  // namespace exported_mlp
""",
    ]
    output_path.write_text("".join(text), encoding="utf-8")

    X_float32 = np.asarray(X, dtype=np.float32)
    hidden1_values = np.maximum(
        np.float32(0.0), X_float32 @ coefs[0] + biases[0]
    )
    hidden2_values = np.maximum(0.0, hidden1_values @ coefs[1] + biases[1])
    logits = hidden2_values @ coefs[2] + biases[2]
    probabilities = softmax_numpy(logits)
    exported_predictions = np.argmax(probabilities, axis=1)
    reference_input = named_reference_input(estimator, X)
    reference_predictions = estimator.predict(reference_input).astype(int)
    reference_probabilities = estimator.predict_proba(reference_input)
    parameter_count = int(
        sum(weight.size for weight in coefs) + sum(bias.size for bias in biases)
    )
    macs = int(sum(weight.size for weight in coefs))

    return {
        "parameter_count": parameter_count,
        "model_payload_bytes": parameter_count * FLOAT_BYTES,
        "activation_ram_bytes": int(
            (hidden_1 + hidden_2 + class_count) * FLOAT_BYTES
        ),
        "theoretical_operations": macs,
        "hidden_layer_1": int(hidden_1),
        "hidden_layer_2": int(hidden_2),
        "parity_rate": float(np.mean(exported_predictions == reference_predictions)),
        "max_probability_error": float(
            np.max(np.abs(probabilities - reference_probabilities))
        ),
    }


def export_model_to_cpp(
    spec: ModelSpec,
    estimator: BaseEstimator,
    X: np.ndarray,
    output_path: Path,
    class_count: int,
) -> dict[str, Any]:
    _, core = unwrap_pipeline(estimator)
    if isinstance(core, LogisticRegression):
        return export_logistic_regression(estimator, X, output_path)
    if isinstance(core, DecisionTreeClassifier):
        return export_decision_tree(core, X, output_path, class_count)
    if isinstance(core, RandomForestClassifier):
        return export_random_forest(core, X, output_path, class_count)
    if isinstance(core, MLPClassifier):
        namespace = "edge_" + re.sub(r"[^a-z0-9_]", "_", spec.slug.lower())
        return export_mlp(estimator, X, output_path, namespace)
    raise TypeError(f"Modelo no soportado para exportación: {type(core).__name__}")


def train_final_models_and_export(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[int],
    specs: list[ModelSpec],
    mode: str,
    jobs: int,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    X_df = df[feature_cols]
    X = X_df.to_numpy(dtype=np.float64)
    y = df["label"].astype(int)
    groups = df["subject"]
    rows: list[dict[str, Any]] = []
    estimators: dict[str, BaseEstimator] = {}

    for spec in specs:
        print(f"Entrenando modelo final: {spec.name}")
        start = time.perf_counter()
        estimator, best_params, inner_best_f1 = fit_model(
            spec, X_df, y, groups, mode, jobs
        )
        fit_time = time.perf_counter() - start
        estimators[spec.slug] = estimator

        model_path = paths["models"] / f"{spec.slug}.joblib"
        joblib.dump(estimator, model_path, compress=3)

        header_path = paths["headers"] / f"{spec.slug}.h"
        export_info = export_model_to_cpp(
            spec,
            estimator,
            X,
            header_path,
            len(labels),
        )

        row: dict[str, Any] = {
            "model": spec.name,
            "model_slug": spec.slug,
            "deployment_family": spec.deployment_family,
            "description": spec.description,
            "selected_parameters": json.dumps(
                best_params, sort_keys=True, default=str
            ),
            "inner_loso_best_f1_macro": inner_best_f1,
            "final_fit_time_pc_s": fit_time,
            "joblib_file": model_path.name,
            "joblib_bytes": model_path.stat().st_size,
            "cpp_header_file": header_path.name,
            "cpp_header_text_bytes": header_path.stat().st_size,
        }
        row.update(export_info)
        rows.append(row)

        if export_info["parity_rate"] < 1.0:
            raise RuntimeError(
                f"La exportación de {spec.name} no conserva todas las clases "
                f"(paridad={export_info['parity_rate']:.8f})."
            )

    return pd.DataFrame(rows), estimators


def create_benchmark_vectors(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[int],
    final_estimators: dict[str, BaseEstimator],
    per_subject_class: int,
    seed: int,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    for (_, _), group in df.groupby(["subject", "label"], sort=True):
        n = min(per_subject_class, len(group))
        selected_indices.extend(
            group.sample(n=n, random_state=seed).index.astype(int).tolist()
        )
    selected_indices = sorted(selected_indices)
    metadata = [
        col
        for col in [
            "subject",
            "session",
            "class_name",
            "label",
            "source_file",
            "start_idx",
            "end_idx",
            "start_time",
            "end_time",
        ]
        if col in df.columns
    ]
    result = df.loc[selected_indices, metadata + feature_cols].copy()
    result.insert(0, "row_id", selected_indices)
    X = df.loc[selected_indices, feature_cols]
    for slug, estimator in final_estimators.items():
        probabilities = aligned_probabilities(estimator, X, labels)
        result[f"{slug}_predicted_label"] = estimator.predict(X).astype(int)
        for idx, label in enumerate(labels):
            result[f"{slug}_probability_label_{label}"] = probabilities[:, idx]
    return result.reset_index(drop=True)


def plot_f1_by_subject(folds: pd.DataFrame, output_dir: Path) -> None:
    pivot = folds.pivot(
        index="held_out_subject",
        columns="model",
        values="f1_macro",
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, model in enumerate(pivot.columns):
        ax.plot(
            pivot.index,
            pivot[model],
            marker=markers[idx % len(markers)],
            linewidth=1.6,
            markersize=6,
            label=model,
        )
    ax.set_xlabel("Held-out participant")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(max(0.0, float(pivot.min().min()) - 0.03), 1.005)
    ax.set_xticks(pivot.index)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "f1_by_held_out_participant.png", dpi=300)
    fig.savefig(output_dir / "f1_by_held_out_participant.pdf")
    plt.close(fig)


def plot_summary_f1(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("f1_macro_mean", ascending=True)
    means = ordered["f1_macro_mean"].to_numpy()
    lows = ordered["f1_macro_cluster_bootstrap_ci95_low"].to_numpy()
    highs = ordered["f1_macro_cluster_bootstrap_ci95_high"].to_numpy()
    errors = np.vstack((means - lows, highs - means))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(ordered))
    ax.errorbar(
        means,
        y,
        xerr=errors,
        fmt="o",
        capsize=4,
        color="#007B94",
        ecolor="#444444",
    )
    ax.set_yticks(y, ordered["model"])
    ax.set_xlabel("Subject-independent macro-F1")
    ax.set_xlim(max(0.0, float(lows.min()) - 0.02), 1.005)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "macro_f1_summary.png", dpi=300)
    fig.savefig(output_dir / "macro_f1_summary.pdf")
    plt.close(fig)


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    labels: list[int],
    names: dict[int, str],
    output_dir: Path,
) -> None:
    models = list(predictions["model"].drop_duplicates())
    n_cols = 3
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12.0, 4.2 * n_rows),
        squeeze=False,
    )
    tick_names = [names[label] for label in labels]

    for ax, model in zip(axes.flat, models):
        group = predictions[predictions["model"] == model]
        cm = confusion_matrix(
            group["true_label"],
            group["predicted_label"],
            labels=labels,
            normalize="true",
        )
        ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(len(labels)):
            for col in range(len(labels)):
                ax.text(
                    col,
                    row,
                    f"{cm[row, col]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > 0.5 else "black",
                    fontsize=9,
                )
        ax.set_title(model, fontsize=10)
        ax.set_xticks(range(len(labels)), tick_names, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), tick_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes.flat[len(models) :]:
        ax.axis("off")

    fig.tight_layout(pad=1.2, h_pad=3.5, w_pad=2.0)
    fig.savefig(output_dir / "normalized_confusion_matrices.png", dpi=300)
    fig.savefig(output_dir / "normalized_confusion_matrices.pdf")
    plt.close(fig)


def dataframe_to_markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    decimal_columns: set[str] | None = None,
) -> str:
    decimal_columns = decimal_columns or set()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, record in df[columns].iterrows():
        formatted = []
        for col in columns:
            value = record[col]
            if col in decimal_columns and pd.notna(value):
                formatted.append(f"{float(value):.5f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def write_report(
    path: Path,
    input_path: Path,
    df: pd.DataFrame,
    mode: str,
    summary: pd.DataFrame,
    complexity: pd.DataFrame,
) -> None:
    summary_table = summary.rename(
        columns={
            "f1_macro_mean": "Macro-F1 mean",
            "f1_macro_std": "Macro-F1 SD",
            "f1_macro_min": "Minimum",
            "f1_macro_max": "Maximum",
        }
    )
    complexity_table = complexity.rename(
        columns={
            "model_payload_bytes": "Payload bytes",
            "activation_ram_bytes": "Activation RAM bytes",
            "theoretical_operations": "Operations",
            "parity_rate": "Parity",
        }
    )
    text = f"""# IEEE Latin America Transactions — Edge AI comparison

## Run

- Input: `{input_path.name}`
- SHA-256: `{sha256_file(input_path)}`
- Mode: `{mode}`
- Windows: {len(df)}
- Participants: {df['subject'].nunique()}
- Sessions: {df[['subject', 'session']].drop_duplicates().shape[0]}
- Classes: {df['label'].nunique()}
- Features: {len(EXPECTED_FEATURES)}

## Subject-independent predictive results

{dataframe_to_markdown_table(
    summary_table,
    ["model", "Macro-F1 mean", "Macro-F1 SD", "Minimum", "Maximum"],
    {"Macro-F1 mean", "Macro-F1 SD", "Minimum", "Maximum"},
)}

The confidence intervals in the CSV summary use participant-clustered
bootstrap resampling. With five independent participants, they are descriptive
and must not be interpreted as high-powered population inference.

## Exported model representations

{dataframe_to_markdown_table(
    complexity_table,
    ["model", "Payload bytes", "Activation RAM bytes", "Operations", "Parity"],
    {"Parity"},
)}

`Payload bytes` represents model arrays exported to C++, not the final compiled
flash occupation. Flash, static RAM, minimum heap, latency and energy must be
measured in the ESP32 benchmarking firmware.

PC prediction times are retained only for reproducibility and must not be
reported as ESP32 inference latency.
"""
    path.write_text(text, encoding="utf-8")


def dataset_summary(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    return {
        "rows_windows": len(df),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "subjects": sorted(
            int(value) if isinstance(value, (int, np.integer)) else str(value)
            for value in df["subject"].unique()
        ),
        "subject_count": int(df["subject"].nunique()),
        "subject_session_count": int(
            df[["subject", "session"]].drop_duplicates().shape[0]
        ),
        "source_file_count": int(df["source_file"].nunique())
        if "source_file" in df.columns
        else None,
        "class_counts": {
            str(key): int(value)
            for key, value in df["class_name"].value_counts().sort_index().items()
        },
        "label_counts": {
            str(int(key)): int(value)
            for key, value in df["label"].value_counts().sort_index().items()
        },
    }


def save_run_configuration(
    path: Path,
    args: argparse.Namespace,
    input_path: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    specs: list[ModelSpec],
) -> None:
    configuration = {
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "mode": args.mode,
        "jobs": args.jobs,
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "benchmark_per_subject_class": args.benchmark_per_subject_class,
        "dataset": dataset_summary(df, feature_cols),
        "models": [
            {
                "name": spec.name,
                "slug": spec.slug,
                "deployment_family": spec.deployment_family,
                "parameter_grid": spec.parameter_grid,
            }
            for spec in specs
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
    }
    path.write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("once", category=ConvergenceWarning)

    input_path = resolve_input_path(args.input)
    output_root = args.output.resolve()
    paths = ensure_output_directories(output_root)

    print(f"Dataset: {input_path}")
    print(f"Resultados: {output_root}")
    print(f"Modo: {args.mode}")

    df, feature_cols, labels = load_and_validate_dataset(input_path)
    label_names = class_name_map(df)
    specs = build_model_specs(args.seed, args.jobs)

    save_run_configuration(
        paths["root"] / "run_configuration.json",
        args,
        input_path,
        df,
        feature_cols,
        specs,
    )

    folds, class_metrics, predictions = evaluate_outer_loso(
        df,
        feature_cols,
        labels,
        specs,
        args.mode,
        args.jobs,
    )
    summary = summarize_folds(folds, args.bootstrap, args.seed)
    confusion_counts, confusion_normalized = confusion_tables(predictions, labels)
    pairwise = paired_model_comparisons(folds)

    folds.to_csv(paths["tables"] / "outer_loso_fold_metrics.csv", index=False)
    summary.to_csv(paths["tables"] / "outer_loso_summary.csv", index=False)
    class_metrics.to_csv(
        paths["tables"] / "outer_loso_class_metrics.csv", index=False
    )
    predictions.to_csv(
        paths["tables"] / "outer_loso_predictions.csv", index=False
    )
    confusion_counts.to_csv(
        paths["tables"] / "confusion_matrix_counts.csv", index=False
    )
    confusion_normalized.to_csv(
        paths["tables"] / "confusion_matrix_row_normalized.csv", index=False
    )
    pairwise.to_csv(
        paths["tables"] / "paired_subject_level_comparisons.csv", index=False
    )

    complexity, final_estimators = train_final_models_and_export(
        df,
        feature_cols,
        labels,
        specs,
        args.mode,
        args.jobs,
        paths,
    )
    complexity.to_csv(
        paths["tables"] / "final_model_complexity_and_parity.csv", index=False
    )

    benchmark_vectors = create_benchmark_vectors(
        df,
        feature_cols,
        labels,
        final_estimators,
        args.benchmark_per_subject_class,
        args.seed,
    )
    benchmark_vectors.to_csv(
        paths["benchmark"] / "esp32_benchmark_vectors.csv", index=False
    )

    plot_f1_by_subject(folds, paths["figures"])
    plot_summary_f1(summary, paths["figures"])
    plot_confusion_matrices(predictions, labels, label_names, paths["figures"])

    write_report(
        paths["root"] / "README_results.md",
        input_path,
        df,
        args.mode,
        summary,
        complexity,
    )

    print("\n=== RESUMEN LOSO ===")
    print(
        summary[
            [
                "model",
                "f1_macro_mean",
                "f1_macro_std",
                "f1_macro_min",
                "f1_macro_max",
            ]
        ].to_string(index=False)
    )
    print("\n=== PARIDAD DE EXPORTACIÓN ===")
    print(
        complexity[
            [
                "model",
                "model_payload_bytes",
                "activation_ram_bytes",
                "parity_rate",
                "max_probability_error",
            ]
        ].to_string(index=False)
    )
    print(f"\nResultados guardados en: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEjecución cancelada.", file=sys.stderr)
        raise SystemExit(130)
