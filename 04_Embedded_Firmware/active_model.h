#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "benchmark_config.h"
#include "feature_extractor.h"

// ============================================================
// Funciones compartidas por los cinco modelos
// ============================================================
// Se mantienen dentro de active_model.h para que Wokwi no dependa
// de una pestaña auxiliar adicional.

namespace edgeia {

static constexpr size_t kFeatureCount = 32;
static constexpr size_t kClassCount = 3;

inline float relu(float value) {
  return value > 0.0f ? value : 0.0f;
}

inline void softmax3(
    const float* scores,
    float* probabilities) {
  float maximum = scores[0];
  for (size_t index = 1; index < kClassCount; ++index) {
    if (scores[index] > maximum) maximum = scores[index];
  }

  float total = 0.0f;
  for (size_t index = 0; index < kClassCount; ++index) {
    probabilities[index] = expf(scores[index] - maximum);
    total += probabilities[index];
  }
  if (total < 1.0e-12f) total = 1.0f;
  for (size_t index = 0; index < kClassCount; ++index) {
    probabilities[index] /= total;
  }
}

inline int argmax3(const float* values) {
  int selected = 0;
  if (values[1] > values[selected]) selected = 1;
  if (values[2] > values[selected]) selected = 2;
  return selected;
}

struct TreeView {
  const int16_t* left;
  const int16_t* right;
  const int8_t* feature;
  const float* threshold;
  const float* probability;
  size_t nodeCount;
};

inline void predictTree(
    const TreeView& tree,
    const float* input,
    float* probabilities) {
  int16_t node = 0;
  while (
      node >= 0 &&
      static_cast<size_t>(node) < tree.nodeCount &&
      tree.feature[node] >= 0) {
    const int8_t feature = tree.feature[node];
    node = input[feature] <= tree.threshold[node]
               ? tree.left[node]
               : tree.right[node];
  }

  if (node < 0 ||
      static_cast<size_t>(node) >= tree.nodeCount) {
    probabilities[0] = 1.0f;
    probabilities[1] = 0.0f;
    probabilities[2] = 0.0f;
    return;
  }

  const size_t offset =
      static_cast<size_t>(node) * kClassCount;
  for (size_t output = 0; output < kClassCount; ++output) {
    probabilities[output] =
        tree.probability[offset + output];
  }
}

}  // namespace edgeia

// ============================================================
// Seleccion del modelo
// ============================================================

#if EDGE_MODEL_ID == EDGE_MODEL_LOGISTIC_REGRESSION

#include "logistic_regression.h"
namespace active_model_namespace = edgeia::logistic_regression;
static constexpr const char* ACTIVE_MODEL_NAME =
    "logistic_regression";

#elif EDGE_MODEL_ID == EDGE_MODEL_DECISION_TREE

#include "decision_tree_d6.h"
namespace active_model_namespace = edgeia::decision_tree_d6;
static constexpr const char* ACTIVE_MODEL_NAME =
    "decision_tree_d6";

#elif EDGE_MODEL_ID == EDGE_MODEL_RANDOM_FOREST

#include "random_forest_compact.h"
namespace active_model_namespace = edgeia::random_forest_compact;
static constexpr const char* ACTIVE_MODEL_NAME =
    "random_forest_compact";

#elif EDGE_MODEL_ID == EDGE_MODEL_MLP_COMPACT

#include "mlp_compact_32_16.h"
namespace active_model_namespace = edgeia::mlp_compact_32_16;
static constexpr const char* ACTIVE_MODEL_NAME =
    "mlp_compact_32_16";

#elif EDGE_MODEL_ID == EDGE_MODEL_MLP_REFERENCE

#include "mlp_reference_64_32.h"
namespace active_model_namespace = edgeia::mlp_reference_64_32;
static constexpr const char* ACTIVE_MODEL_NAME =
    "mlp_reference_64_32";

#else
#error "EDGE_MODEL_ID no corresponde a un modelo disponible."
#endif

static constexpr unsigned long ACTIVE_MODEL_PARAMETER_BYTES =
    static_cast<unsigned long>(
        active_model_namespace::kParameterBytes);

static_assert(
    edgeia::kFeatureCount == edge_features::FEATURE_COUNT,
    "El modelo y el extractor deben usar 32 caracteristicas.");
static_assert(
    edgeia::kClassCount == 3,
    "El firmware espera tres clases.");

inline int activeModelPredict(
    const float features[edge_features::FEATURE_COUNT],
    float probabilities[3]) {
  active_model_namespace::predict(features, probabilities);
  return edgeia::argmax3(probabilities);
}
