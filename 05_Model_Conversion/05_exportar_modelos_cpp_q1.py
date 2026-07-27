#!/usr/bin/env python3
"""
Exporta los cinco modelos scikit-learn validados a cabeceras C++ sin heap.

Los predictores generados reciben las 32 características dinámicas en el
orden documentado por dataset_features_dinamicas_q1.csv y devuelven las
probabilidades de reposo=0, suave=1 y brusco=2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = [
    f"{signal}_{statistic}"
    for signal in ("ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag")
    for statistic in ("std", "ptp", "mean_abs_diff", "rms_diff")
]
METADATA_COLUMNS = {
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
}

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree_d6": "decision_tree_d6.joblib",
    "random_forest_compact": "random_forest_compact.joblib",
    "mlp_compact_32_16": "mlp_compact_32_16.joblib",
    "mlp_reference_64_32": "mlp_reference_64_32.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta modelos dinámicos scikit-learn a C++."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--feature-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feature_order(path: Path) -> None:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    observed = [column for column in columns if column not in METADATA_COLUMNS]
    if observed != FEATURE_COLUMNS:
        raise ValueError(
            "El orden de características no coincide con el contrato C++.\n"
            f"Esperado: {FEATURE_COLUMNS}\nObservado: {observed}"
        )


def validate_classes(estimator: object) -> None:
    classes = np.asarray(getattr(estimator, "classes_"), dtype=int)
    if not np.array_equal(classes, np.array([0, 1, 2])):
        raise ValueError(f"Orden de clases inesperado: {classes.tolist()}")


def float_literal(value: float) -> str:
    number = np.float32(value)
    if not np.isfinite(number):
        raise ValueError(f"No se puede exportar el valor {value!r}.")
    text = format(float(number), ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text + "f"


def int_literal(value: int) -> str:
    return str(int(value))


def format_1d(
    values: Iterable[object],
    formatter,
    indent: str = "  ",
    per_line: int = 8,
) -> str:
    rendered = [formatter(value) for value in values]
    lines = [
        indent + ", ".join(rendered[index : index + per_line])
        for index in range(0, len(rendered), per_line)
    ]
    return ",\n".join(lines)


def format_2d(
    values: np.ndarray,
    formatter,
    indent: str = "  ",
    per_line: int = 8,
) -> str:
    rows: list[str] = []
    for row in np.asarray(values):
        body = format_1d(row, formatter, indent + "  ", per_line)
        rows.append(f"{indent}{{\n{body}\n{indent}}}")
    return ",\n".join(rows)


def header_preamble(description: str) -> str:
    return (
        "#pragma once\n\n"
        f"// {description}\n"
        "// Generado automáticamente; no editar manualmente.\n"
        '#include "model_common.h"\n\n'
    )


def scaler_parts(model: Pipeline) -> tuple[np.ndarray, np.ndarray, object]:
    if list(model.named_steps) != ["scaler", "model"]:
        raise TypeError("Se esperaba Pipeline(scaler, model).")
    scaler = model.named_steps["scaler"]
    estimator = model.named_steps["model"]
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if mean.shape != (32,) or scale.shape != (32,):
        raise ValueError("El escalador debe tener 32 entradas.")
    return mean, scale, estimator


def export_logistic(model: Pipeline, destination: Path) -> dict[str, int]:
    mean, scale, estimator = scaler_parts(model)
    if not isinstance(estimator, LogisticRegression):
        raise TypeError("El modelo esperado es LogisticRegression.")
    validate_classes(estimator)
    coefficients = np.asarray(estimator.coef_, dtype=np.float32)
    intercept = np.asarray(estimator.intercept_, dtype=np.float32)
    if coefficients.shape != (3, 32) or intercept.shape != (3,):
        raise ValueError("Dimensiones inesperadas en regresión logística.")

    parameter_bytes = int(
        mean.nbytes + scale.nbytes + coefficients.nbytes + intercept.nbytes
    )
    text = header_preamble("Regresión logística multiclase con StandardScaler")
    text += f"""namespace edgeia {{
namespace logistic_regression {{

static const char kName[] = "Logistic Regression";
static const char kShortName[] = "LOGREG";
static const size_t kParameterBytes = {parameter_bytes};

static const float kMean[kFeatureCount] = {{
{format_1d(mean, float_literal)}
}};

static const float kScale[kFeatureCount] = {{
{format_1d(scale, float_literal)}
}};

static const float kWeights[kClassCount][kFeatureCount] = {{
{format_2d(coefficients, float_literal)}
}};

static const float kBias[kClassCount] = {{
{format_1d(intercept, float_literal)}
}};

inline void predict(const float* input, float* probabilities) {{
  float scores[kClassCount];
  for (size_t output = 0; output < kClassCount; ++output) {{
    float value = kBias[output];
    for (size_t feature = 0; feature < kFeatureCount; ++feature) {{
      const float normalized = (input[feature] - kMean[feature]) /
                               kScale[feature];
      value += normalized * kWeights[output][feature];
    }}
    scores[output] = value;
  }}
  softmax3(scores, probabilities);
}}

}}  // namespace logistic_regression
}}  // namespace edgeia
"""
    destination.write_text(text, encoding="utf-8")
    return {"parameter_bytes": parameter_bytes}


def tree_arrays(estimator: DecisionTreeClassifier) -> dict[str, np.ndarray]:
    validate_classes(estimator)
    tree = estimator.tree_
    values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals != 0,
    ).astype(np.float32)
    return {
        "left": np.asarray(tree.children_left, dtype=np.int16),
        "right": np.asarray(tree.children_right, dtype=np.int16),
        "feature": np.asarray(tree.feature, dtype=np.int8),
        "threshold": np.asarray(tree.threshold, dtype=np.float32),
        "probabilities": probabilities,
    }


def tree_parameter_bytes(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(array.nbytes for array in arrays.values()))


def emit_tree_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> str:
    nodes = len(arrays["left"])
    return f"""
static const int16_t {prefix}Left[{nodes}] = {{
{format_1d(arrays["left"], int_literal, per_line=12)}
}};
static const int16_t {prefix}Right[{nodes}] = {{
{format_1d(arrays["right"], int_literal, per_line=12)}
}};
static const int8_t {prefix}Feature[{nodes}] = {{
{format_1d(arrays["feature"], int_literal, per_line=16)}
}};
static const float {prefix}Threshold[{nodes}] = {{
{format_1d(arrays["threshold"], float_literal)}
}};
static const float {prefix}Probability[{nodes}][kClassCount] = {{
{format_2d(arrays["probabilities"], float_literal)}
}};
"""


def export_decision_tree(
    estimator: DecisionTreeClassifier,
    destination: Path,
) -> dict[str, int]:
    if not isinstance(estimator, DecisionTreeClassifier):
        raise TypeError("El modelo esperado es DecisionTreeClassifier.")
    arrays = tree_arrays(estimator)
    nodes = len(arrays["left"])
    parameter_bytes = tree_parameter_bytes(arrays)

    text = header_preamble("Árbol de decisión compacto")
    text += """namespace edgeia {
namespace decision_tree_d6 {

static const char kName[] = "Decision Tree d6";
static const char kShortName[] = "TREE_D6";
"""
    text += (
        f"static const size_t kParameterBytes = {parameter_bytes};\n"
        f"static const size_t kNodeCount = {nodes};\n"
    )
    text += emit_tree_arrays("k", arrays)
    text += """
inline void predict(const float* input, float* probabilities) {
  const TreeView tree = {
    kLeft, kRight, kFeature, kThreshold, &kProbability[0][0], kNodeCount
  };
  predictTree(tree, input, probabilities);
}

}  // namespace decision_tree_d6
}  // namespace edgeia
"""
    destination.write_text(text, encoding="utf-8")
    return {"parameter_bytes": parameter_bytes, "nodes": nodes}


def export_random_forest(
    estimator: RandomForestClassifier,
    destination: Path,
) -> dict[str, int]:
    if not isinstance(estimator, RandomForestClassifier):
        raise TypeError("El modelo esperado es RandomForestClassifier.")
    validate_classes(estimator)
    arrays_by_tree = [tree_arrays(tree) for tree in estimator.estimators_]
    parameter_bytes = sum(tree_parameter_bytes(item) for item in arrays_by_tree)
    total_nodes = sum(len(item["left"]) for item in arrays_by_tree)

    text = header_preamble("Random Forest compacto")
    text += """namespace edgeia {
namespace random_forest_compact {

static const char kName[] = "Random Forest compact";
static const char kShortName[] = "RF_15";
"""
    text += (
        f"static const size_t kParameterBytes = {parameter_bytes};\n"
        f"static const size_t kTreeCount = {len(arrays_by_tree)};\n"
        f"static const size_t kTotalNodeCount = {total_nodes};\n"
    )
    for index, arrays in enumerate(arrays_by_tree):
        text += emit_tree_arrays(f"kTree{index}", arrays)

    text += "\nstatic const TreeView kTrees[kTreeCount] = {\n"
    for index, arrays in enumerate(arrays_by_tree):
        nodes = len(arrays["left"])
        text += (
            "  {"
            f"kTree{index}Left, kTree{index}Right, kTree{index}Feature, "
            f"kTree{index}Threshold, &kTree{index}Probability[0][0], "
            f"{nodes}"
            "},\n"
        )
    text += """};

inline void predict(const float* input, float* probabilities) {
  probabilities[0] = 0.0f;
  probabilities[1] = 0.0f;
  probabilities[2] = 0.0f;
  float treeProbability[kClassCount];
  for (size_t tree = 0; tree < kTreeCount; ++tree) {
    predictTree(kTrees[tree], input, treeProbability);
    for (size_t output = 0; output < kClassCount; ++output) {
      probabilities[output] += treeProbability[output];
    }
  }
  const float inverseTreeCount = 1.0f / static_cast<float>(kTreeCount);
  for (size_t output = 0; output < kClassCount; ++output) {
    probabilities[output] *= inverseTreeCount;
  }
}

}  // namespace random_forest_compact
}  // namespace edgeia
"""
    destination.write_text(text, encoding="utf-8")
    return {
        "parameter_bytes": parameter_bytes,
        "trees": len(arrays_by_tree),
        "nodes": total_nodes,
    }


def export_mlp(
    model: Pipeline,
    destination: Path,
    namespace: str,
    name: str,
    short_name: str,
) -> dict[str, int | list[int]]:
    mean, scale, estimator = scaler_parts(model)
    if not isinstance(estimator, MLPClassifier):
        raise TypeError("El modelo esperado es MLPClassifier.")
    validate_classes(estimator)
    layers = [np.asarray(array, dtype=np.float32) for array in estimator.coefs_]
    biases = [
        np.asarray(array, dtype=np.float32)
        for array in estimator.intercepts_
    ]
    if len(layers) != 3 or len(biases) != 3:
        raise ValueError("Se esperaba una MLP con dos capas ocultas.")
    input_size, hidden_1 = layers[0].shape
    hidden_1_check, hidden_2 = layers[1].shape
    hidden_2_check, output_size = layers[2].shape
    if (
        input_size != 32
        or hidden_1_check != hidden_1
        or hidden_2_check != hidden_2
        or output_size != 3
    ):
        raise ValueError("Dimensiones incompatibles en la MLP.")

    parameter_bytes = int(
        mean.nbytes
        + scale.nbytes
        + sum(array.nbytes for array in layers)
        + sum(array.nbytes for array in biases)
    )
    text = header_preamble(f"{name} con StandardScaler")
    text += f"""namespace edgeia {{
namespace {namespace} {{

static const char kName[] = "{name}";
static const char kShortName[] = "{short_name}";
static const size_t kParameterBytes = {parameter_bytes};
static const size_t kHidden1 = {hidden_1};
static const size_t kHidden2 = {hidden_2};

static const float kMean[kFeatureCount] = {{
{format_1d(mean, float_literal)}
}};
static const float kScale[kFeatureCount] = {{
{format_1d(scale, float_literal)}
}};
static const float kWeight1[kFeatureCount][kHidden1] = {{
{format_2d(layers[0], float_literal)}
}};
static const float kBias1[kHidden1] = {{
{format_1d(biases[0], float_literal)}
}};
static const float kWeight2[kHidden1][kHidden2] = {{
{format_2d(layers[1], float_literal)}
}};
static const float kBias2[kHidden2] = {{
{format_1d(biases[1], float_literal)}
}};
static const float kWeight3[kHidden2][kClassCount] = {{
{format_2d(layers[2], float_literal)}
}};
static const float kBias3[kClassCount] = {{
{format_1d(biases[2], float_literal)}
}};

inline void predict(const float* input, float* probabilities) {{
  float normalized[kFeatureCount];
  float hidden1[kHidden1];
  float hidden2[kHidden2];
  float output[kClassCount];

  for (size_t feature = 0; feature < kFeatureCount; ++feature) {{
    normalized[feature] =
        (input[feature] - kMean[feature]) / kScale[feature];
  }}
  for (size_t neuron = 0; neuron < kHidden1; ++neuron) {{
    float value = kBias1[neuron];
    for (size_t feature = 0; feature < kFeatureCount; ++feature) {{
      value += normalized[feature] * kWeight1[feature][neuron];
    }}
    hidden1[neuron] = relu(value);
  }}
  for (size_t neuron = 0; neuron < kHidden2; ++neuron) {{
    float value = kBias2[neuron];
    for (size_t previous = 0; previous < kHidden1; ++previous) {{
      value += hidden1[previous] * kWeight2[previous][neuron];
    }}
    hidden2[neuron] = relu(value);
  }}
  for (size_t outputIndex = 0; outputIndex < kClassCount; ++outputIndex) {{
    float value = kBias3[outputIndex];
    for (size_t previous = 0; previous < kHidden2; ++previous) {{
      value += hidden2[previous] * kWeight3[previous][outputIndex];
    }}
    output[outputIndex] = value;
  }}
  softmax3(output, probabilities);
}}

}}  // namespace {namespace}
}}  // namespace edgeia
"""
    destination.write_text(text, encoding="utf-8")
    return {
        "parameter_bytes": parameter_bytes,
        "layers": [32, hidden_1, hidden_2, 3],
    }


def main() -> None:
    args = parse_args()
    validate_feature_order(args.feature_dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        slug: joblib.load(args.models_dir / filename)
        for slug, filename in MODEL_FILES.items()
    }
    results: dict[str, dict[str, object]] = {}

    exporters = [
        (
            "logistic_regression",
            "logistic_regression_model.h",
            lambda model, path: export_logistic(model, path),
        ),
        (
            "decision_tree_d6",
            "decision_tree_d6_model.h",
            lambda model, path: export_decision_tree(model, path),
        ),
        (
            "random_forest_compact",
            "random_forest_compact_model.h",
            lambda model, path: export_random_forest(model, path),
        ),
        (
            "mlp_compact_32_16",
            "mlp_compact_32_16_model.h",
            lambda model, path: export_mlp(
                model,
                path,
                "mlp_compact_32_16",
                "MLP compact 32-16",
                "MLP_32_16",
            ),
        ),
        (
            "mlp_reference_64_32",
            "mlp_reference_64_32_model.h",
            lambda model, path: export_mlp(
                model,
                path,
                "mlp_reference_64_32",
                "MLP reference 64-32",
                "MLP_64_32",
            ),
        ),
    ]

    for slug, output_name, exporter in exporters:
        source = args.models_dir / MODEL_FILES[slug]
        destination = args.output_dir / output_name
        details = exporter(models[slug], destination)
        results[slug] = {
            "source_model": str(source),
            "source_sha256": sha256(source),
            "header": output_name,
            "header_sha256": sha256(destination),
            **details,
        }
        print(
            f"{slug}: {output_name} "
            f"({details['parameter_bytes']} bytes de parámetros)"
        )

    manifest = {
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "class_order": {"0": "reposo", "1": "suave", "2": "brusco"},
        "float_format": "IEEE-754 float32",
        "models": results,
    }
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
