#pragma once

#include <math.h>
#include <stddef.h>

#include "benchmark_config.h"

namespace edge_features {

static constexpr int SIGNAL_COUNT = 6;
static constexpr int DYNAMIC_SERIES_COUNT = 8;
static constexpr int STATISTICS_PER_SERIES = 4;
static constexpr int FEATURE_COUNT =
    DYNAMIC_SERIES_COUNT * STATISTICS_PER_SERIES;

enum SignalIndex {
  AX = 0,
  AY = 1,
  AZ = 2,
  GX = 3,
  GY = 4,
  GZ = 5
};

// Orden exacto del entrenamiento:
// ax, ay, az, gx, gy, gz, magnitud de aceleracion y
// magnitud de giroscopio. Para cada serie:
// std, pico a pico, media del cambio absoluto y RMS del cambio.

inline void finishStatistics(
    float minimum,
    float maximum,
    float sum_absolute_difference,
    float sum_squared_difference,
    float sum_squared_deviation,
    float* output) {
  const float difference_count =
      static_cast<float>(EDGE_WINDOW_SIZE - 1);
  output[0] = sqrtf(
      sum_squared_deviation /
      static_cast<float>(EDGE_WINDOW_SIZE));
  output[1] = maximum - minimum;
  output[2] =
      sum_absolute_difference / difference_count;
  output[3] =
      sqrtf(sum_squared_difference / difference_count);
}

inline void computeCircularStatistics(
    const float* values,
    int oldest_index,
    float* output) {
  const float reference = values[oldest_index];
  float sum_offset = 0.0f;
  float minimum = reference;
  float maximum = reference;
  float sum_absolute_difference = 0.0f;
  float sum_squared_difference = 0.0f;
  float previous = reference;

  for (int ordered = 0;
       ordered < EDGE_WINDOW_SIZE;
       ++ordered) {
    const int physical =
        (oldest_index + ordered) % EDGE_WINDOW_SIZE;
    const float value = values[physical];
    sum_offset += value - reference;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    if (ordered > 0) {
      const float difference = value - previous;
      sum_absolute_difference += fabsf(difference);
      sum_squared_difference += difference * difference;
    }
    previous = value;
  }

  const float mean_offset =
      sum_offset / static_cast<float>(EDGE_WINDOW_SIZE);
  float sum_squared_deviation = 0.0f;
  for (int ordered = 0;
       ordered < EDGE_WINDOW_SIZE;
       ++ordered) {
    const int physical =
        (oldest_index + ordered) % EDGE_WINDOW_SIZE;
    const float deviation =
        (values[physical] - reference) - mean_offset;
    sum_squared_deviation += deviation * deviation;
  }

  finishStatistics(
      minimum,
      maximum,
      sum_absolute_difference,
      sum_squared_difference,
      sum_squared_deviation,
      output);
}

inline void computeContiguousStatistics(
    const float* values,
    float* output) {
  const float reference = values[0];
  float sum_offset = 0.0f;
  float minimum = reference;
  float maximum = reference;
  float sum_absolute_difference = 0.0f;
  float sum_squared_difference = 0.0f;

  for (int index = 0;
       index < EDGE_WINDOW_SIZE;
       ++index) {
    const float value = values[index];
    sum_offset += value - reference;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    if (index > 0) {
      const float difference = value - values[index - 1];
      sum_absolute_difference += fabsf(difference);
      sum_squared_difference += difference * difference;
    }
  }

  const float mean_offset =
      sum_offset / static_cast<float>(EDGE_WINDOW_SIZE);
  float sum_squared_deviation = 0.0f;
  for (int index = 0;
       index < EDGE_WINDOW_SIZE;
       ++index) {
    const float deviation =
        (values[index] - reference) - mean_offset;
    sum_squared_deviation += deviation * deviation;
  }

  finishStatistics(
      minimum,
      maximum,
      sum_absolute_difference,
      sum_squared_difference,
      sum_squared_deviation,
      output);
}

inline void computeFeatures(
    const float signal_buffer[SIGNAL_COUNT][EDGE_WINDOW_SIZE],
    int oldest_index,
    float output[FEATURE_COUNT]) {
  int feature_index = 0;

  for (int signal = 0; signal < SIGNAL_COUNT; ++signal) {
    computeCircularStatistics(
        signal_buffer[signal],
        oldest_index,
        &output[feature_index]);
    feature_index += STATISTICS_PER_SERIES;
  }

  float acceleration_magnitude[EDGE_WINDOW_SIZE];
  float gyroscope_magnitude[EDGE_WINDOW_SIZE];
  for (int ordered = 0;
       ordered < EDGE_WINDOW_SIZE;
       ++ordered) {
    const int physical =
        (oldest_index + ordered) % EDGE_WINDOW_SIZE;
    const float ax = signal_buffer[AX][physical];
    const float ay = signal_buffer[AY][physical];
    const float az = signal_buffer[AZ][physical];
    const float gx = signal_buffer[GX][physical];
    const float gy = signal_buffer[GY][physical];
    const float gz = signal_buffer[GZ][physical];

    acceleration_magnitude[ordered] =
        sqrtf(ax * ax + ay * ay + az * az);
    gyroscope_magnitude[ordered] =
        sqrtf(gx * gx + gy * gy + gz * gz);
  }

  computeContiguousStatistics(
      acceleration_magnitude,
      &output[feature_index]);
  feature_index += STATISTICS_PER_SERIES;
  computeContiguousStatistics(
      gyroscope_magnitude,
      &output[feature_index]);
}

inline bool allFinite(const float* values, int count) {
  for (int index = 0; index < count; ++index) {
    if (!isfinite(values[index])) return false;
  }
  return true;
}

}  // namespace edge_features
