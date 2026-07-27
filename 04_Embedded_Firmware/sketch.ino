/*
  ESP32 + MPU6050: firmware comun para comparar cinco modelos Edge AI.

  Protocolo reproducido:
    - Acelerometro: +/-2 g, convertido a m/s^2.
    - Giroscopio: +/-500 grados/s.
    - Calibracion inicial con el sensor inmovil y Z hacia arriba.
    - Periodo: 24 ms (41.67 Hz).
    - Ventana: 50 muestras.
    - Desplazamiento: 25 muestras.
    - 32 caracteristicas dinamicas en el orden del entrenamiento.

  El modelo se elige con EDGE_MODEL_ID en benchmark_config.h o mediante
  una bandera de compilacion. No se aplica una regla externa de reposo:
  cada resultado proviene exclusivamente del modelo seleccionado.
*/

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <stdint.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "benchmark_config.h"
#include "feature_extractor.h"
#include "active_model.h"

#if EDGE_ENABLE_OLED
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#endif

// ============================================================
// Hardware
// ============================================================

static constexpr uint8_t SDA_PIN = 21;
static constexpr uint8_t SCL_PIN = 22;
static constexpr uint8_t MPU_ADDRESS_LOW = 0x68;
static constexpr uint8_t MPU_ADDRESS_HIGH = 0x69;

static constexpr uint8_t REG_SAMPLE_RATE_DIVIDER = 0x19;
static constexpr uint8_t REG_CONFIGURATION = 0x1A;
static constexpr uint8_t REG_GYRO_CONFIGURATION = 0x1B;
static constexpr uint8_t REG_ACCEL_CONFIGURATION = 0x1C;
static constexpr uint8_t REG_ACCEL_XOUT_HIGH = 0x3B;
static constexpr uint8_t REG_POWER_MANAGEMENT_1 = 0x6B;
static constexpr uint8_t REG_WHO_AM_I = 0x75;

static constexpr float GRAVITY_MS2 = 9.80665f;
static constexpr float ACCEL_LSB_PER_G = 16384.0f;
static constexpr float GYRO_LSB_PER_DPS = 65.5f;

uint8_t mpu_address = MPU_ADDRESS_LOW;

#if EDGE_ENABLE_OLED
static constexpr int SCREEN_WIDTH = 128;
static constexpr int SCREEN_HEIGHT = 64;
static constexpr uint8_t OLED_ADDRESS = 0x3C;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
#endif

// ============================================================
// Estado de adquisicion y del modelo
// ============================================================

float accelerometer_offset_g[3] = {0.0f, 0.0f, 0.0f};
float gyroscope_offset_dps[3] = {0.0f, 0.0f, 0.0f};

float signal_buffer[edge_features::SIGNAL_COUNT][EDGE_WINDOW_SIZE] = {};
float features[edge_features::FEATURE_COUNT] = {};
float probabilities[3] = {};

int ring_index = 0;
int buffered_samples = 0;
int new_samples_since_inference = 0;
uint32_t next_sample_us = 0;
uint32_t acquired_samples = 0;
uint32_t processed_windows = 0;
uint32_t read_errors = 0;
uint32_t missed_deadlines = 0;

int current_class = -1;
float current_confidence = 0.0f;
float last_feature_us = 0.0f;
float last_inference_us = 0.0f;
float last_total_us = 0.0f;
bool benchmark_completed = false;

const char* const CLASS_NAMES[3] = {
    "reposo",
    "suave",
    "brusco"
};

float timing_samples[EDGE_BENCHMARK_REPETITIONS] = {};
volatile float benchmark_sink = 0.0f;

// ============================================================
// I2C y MPU6050
// ============================================================

bool i2cWriteByte(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(mpu_address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

bool i2cReadBytes(uint8_t reg, uint8_t count, uint8_t* destination) {
  Wire.beginTransmission(mpu_address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  const uint8_t received =
      Wire.requestFrom(
          static_cast<int>(mpu_address),
          static_cast<int>(count),
          static_cast<int>(true));
  if (received != count) return false;

  for (uint8_t i = 0; i < count; ++i) {
    destination[i] = static_cast<uint8_t>(Wire.read());
  }
  return true;
}

bool detectMPU() {
  const uint8_t candidates[2] = {
      MPU_ADDRESS_LOW,
      MPU_ADDRESS_HIGH
  };

  for (uint8_t candidate : candidates) {
    mpu_address = candidate;
    Wire.beginTransmission(mpu_address);
    if (Wire.endTransmission(true) != 0) continue;

    uint8_t identity = 0;
    if (i2cReadBytes(REG_WHO_AM_I, 1, &identity) &&
        (identity == 0x68 || identity == 0x70)) {
      return true;
    }
  }
  return false;
}

bool initializeMPU() {
  // Misma configuracion utilizada durante la adquisicion.
  if (!i2cWriteByte(REG_POWER_MANAGEMENT_1, 0x01)) return false;
  delay(10);
  if (!i2cWriteByte(REG_SAMPLE_RATE_DIVIDER, 0x04)) return false;
  if (!i2cWriteByte(REG_CONFIGURATION, 0x03)) return false;
  if (!i2cWriteByte(REG_GYRO_CONFIGURATION, 0x08)) return false;
  if (!i2cWriteByte(REG_ACCEL_CONFIGURATION, 0x00)) return false;
  return true;
}

bool readUncalibratedMPU(
    float acceleration_g[3],
    float gyroscope_dps[3]) {
  uint8_t bytes[14] = {};
  if (!i2cReadBytes(REG_ACCEL_XOUT_HIGH, 14, bytes)) return false;

  const int16_t raw_ax =
      static_cast<int16_t>((bytes[0] << 8) | bytes[1]);
  const int16_t raw_ay =
      static_cast<int16_t>((bytes[2] << 8) | bytes[3]);
  const int16_t raw_az =
      static_cast<int16_t>((bytes[4] << 8) | bytes[5]);
  const int16_t raw_gx =
      static_cast<int16_t>((bytes[8] << 8) | bytes[9]);
  const int16_t raw_gy =
      static_cast<int16_t>((bytes[10] << 8) | bytes[11]);
  const int16_t raw_gz =
      static_cast<int16_t>((bytes[12] << 8) | bytes[13]);

  acceleration_g[0] = static_cast<float>(raw_ax) / ACCEL_LSB_PER_G;
  acceleration_g[1] = static_cast<float>(raw_ay) / ACCEL_LSB_PER_G;
  acceleration_g[2] = static_cast<float>(raw_az) / ACCEL_LSB_PER_G;

  gyroscope_dps[0] = static_cast<float>(raw_gx) / GYRO_LSB_PER_DPS;
  gyroscope_dps[1] = static_cast<float>(raw_gy) / GYRO_LSB_PER_DPS;
  gyroscope_dps[2] = static_cast<float>(raw_gz) / GYRO_LSB_PER_DPS;
  return true;
}

bool calibrateMPU() {
  double acceleration_sum[3] = {0.0, 0.0, 0.0};
  double gyroscope_sum[3] = {0.0, 0.0, 0.0};
  uint32_t successful_samples = 0;

  for (uint32_t i = 0; i < EDGE_CALIBRATION_SAMPLES; ++i) {
    float acceleration_g[3] = {};
    float gyroscope_dps[3] = {};
    if (readUncalibratedMPU(acceleration_g, gyroscope_dps)) {
      for (int axis = 0; axis < 3; ++axis) {
        acceleration_sum[axis] += acceleration_g[axis];
        gyroscope_sum[axis] += gyroscope_dps[axis];
      }
      ++successful_samples;
    }
    delay(2);
  }

  if (successful_samples <
      static_cast<uint32_t>(EDGE_CALIBRATION_SAMPLES * 9 / 10)) {
    return false;
  }

  for (int axis = 0; axis < 3; ++axis) {
    accelerometer_offset_g[axis] =
        static_cast<float>(
            acceleration_sum[axis] / successful_samples);
    gyroscope_offset_dps[axis] =
        static_cast<float>(
            gyroscope_sum[axis] / successful_samples);
  }

  // Se conserva aproximadamente 1 g en Z, igual que en la adquisicion.
  accelerometer_offset_g[2] -= 1.0f;
  return true;
}

bool readCalibratedMPU(float output[edge_features::SIGNAL_COUNT]) {
  float acceleration_g[3] = {};
  float gyroscope_dps[3] = {};
  if (!readUncalibratedMPU(acceleration_g, gyroscope_dps)) return false;

  output[edge_features::AX] =
      (acceleration_g[0] - accelerometer_offset_g[0]) * GRAVITY_MS2;
  output[edge_features::AY] =
      (acceleration_g[1] - accelerometer_offset_g[1]) * GRAVITY_MS2;
  output[edge_features::AZ] =
      (acceleration_g[2] - accelerometer_offset_g[2]) * GRAVITY_MS2;
  output[edge_features::GX] =
      gyroscope_dps[0] - gyroscope_offset_dps[0];
  output[edge_features::GY] =
      gyroscope_dps[1] - gyroscope_offset_dps[1];
  output[edge_features::GZ] =
      gyroscope_dps[2] - gyroscope_offset_dps[2];

  return edge_features::allFinite(
      output,
      edge_features::SIGNAL_COUNT);
}

// ============================================================
// Ventanas y caracteristicas
// ============================================================

void addSample(const float sample[edge_features::SIGNAL_COUNT]) {
  for (int signal = 0;
       signal < edge_features::SIGNAL_COUNT;
       ++signal) {
    // Los archivos de adquisicion se guardaron con tres decimales.
    signal_buffer[signal][ring_index] =
        roundf(sample[signal] * 1000.0f) / 1000.0f;
  }

  ring_index = (ring_index + 1) % EDGE_WINDOW_SIZE;
  if (buffered_samples < EDGE_WINDOW_SIZE) ++buffered_samples;
  ++new_samples_since_inference;
  ++acquired_samples;
}

void computeCurrentFeatures() {
  // Las caracteristicas de cambio dependen del orden temporal.
  // ring_index apunta a la muestra mas antigua cuando la ventana
  // circular esta completa.
  edge_features::computeFeatures(
      signal_buffer,
      ring_index,
      features);
}

// ============================================================
// Metricas de memoria
// ============================================================

uint32_t stackHighWaterBytes() {
  return static_cast<uint32_t>(
      uxTaskGetStackHighWaterMark(nullptr) * sizeof(StackType_t));
}

void printMemoryRow(const char* stage) {
  Serial.print("MEMORY,");
  Serial.print(ACTIVE_MODEL_NAME);
  Serial.print(",");
  Serial.print(stage);
  Serial.print(",");
  Serial.print(ESP.getSketchSize());
  Serial.print(",");
  Serial.print(ESP.getFreeSketchSpace());
  Serial.print(",");
  Serial.print(ACTIVE_MODEL_PARAMETER_BYTES);
  Serial.print(",");
  Serial.print(ESP.getHeapSize());
  Serial.print(",");
  Serial.print(ESP.getFreeHeap());
  Serial.print(",");
  Serial.print(ESP.getMinFreeHeap());
  Serial.print(",");
  Serial.print(
      heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  Serial.print(",");
  Serial.println(stackHighWaterBytes());
}

// ============================================================
// Estadistica temporal
// ============================================================

void sortTimingSamples(float* values, int count) {
  for (int i = 1; i < count; ++i) {
    const float key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = key;
  }
}

void printBenchmarkStatistics(
    const char* stage,
    int inner_iterations) {
  sortTimingSamples(
      timing_samples,
      EDGE_BENCHMARK_REPETITIONS);

  double sum = 0.0;
  for (int i = 0; i < EDGE_BENCHMARK_REPETITIONS; ++i) {
    sum += timing_samples[i];
  }

  const int median_index =
      (EDGE_BENCHMARK_REPETITIONS - 1) / 2;
  const int p95_index =
      (95 * EDGE_BENCHMARK_REPETITIONS + 99) / 100 - 1;
  const float mean =
      static_cast<float>(
          sum / EDGE_BENCHMARK_REPETITIONS);

  Serial.print("BENCHMARK,");
  Serial.print(ACTIVE_MODEL_NAME);
  Serial.print(",");
  Serial.print(stage);
  Serial.print(",");
  Serial.print(EDGE_BENCHMARK_REPETITIONS);
  Serial.print(",");
  Serial.print(inner_iterations);
  Serial.print(",");
  Serial.print(timing_samples[0], 6);
  Serial.print(",");
  Serial.print(timing_samples[median_index], 6);
  Serial.print(",");
  Serial.print(timing_samples[p95_index], 6);
  Serial.print(",");
  Serial.print(mean, 6);
  Serial.print(",");
  Serial.println(
      timing_samples[EDGE_BENCHMARK_REPETITIONS - 1],
      6);
}

void benchmarkFeatureExtraction() {
  for (int repetition = 0;
       repetition < EDGE_BENCHMARK_REPETITIONS;
       ++repetition) {
    const int64_t start = esp_timer_get_time();
    computeCurrentFeatures();
    const int64_t end = esp_timer_get_time();
    timing_samples[repetition] =
        static_cast<float>(end - start);
    benchmark_sink += features[repetition %
        edge_features::FEATURE_COUNT] * 1.0e-9f;
  }
  printBenchmarkStatistics("feature_extraction", 1);
}

void benchmarkInference() {
  for (int warmup = 0;
       warmup < EDGE_WARMUP_INFERENCES;
       ++warmup) {
    current_class = activeModelPredict(features, probabilities);
    benchmark_sink += probabilities[current_class] * 1.0e-9f;
  }

  for (int repetition = 0;
       repetition < EDGE_BENCHMARK_REPETITIONS;
       ++repetition) {
    const int64_t start = esp_timer_get_time();
    for (int inner = 0;
         inner < EDGE_INFERENCES_PER_TIMING;
         ++inner) {
      current_class = activeModelPredict(features, probabilities);
      benchmark_sink += probabilities[current_class] * 1.0e-9f;
    }
    const int64_t end = esp_timer_get_time();
    timing_samples[repetition] =
        static_cast<float>(end - start) /
        static_cast<float>(EDGE_INFERENCES_PER_TIMING);
  }
  printBenchmarkStatistics(
      "model_inference",
      EDGE_INFERENCES_PER_TIMING);
}

void benchmarkEndToEnd() {
  for (int repetition = 0;
       repetition < EDGE_BENCHMARK_REPETITIONS;
       ++repetition) {
    const int64_t start = esp_timer_get_time();
    computeCurrentFeatures();
    current_class = activeModelPredict(features, probabilities);
    const int64_t end = esp_timer_get_time();
    timing_samples[repetition] =
        static_cast<float>(end - start);
    benchmark_sink += probabilities[current_class] * 1.0e-9f;
  }
  printBenchmarkStatistics("features_plus_inference", 1);
}

void runOneTimeBenchmark() {
  Serial.println(
      "BENCHMARK_HEADER,model,stage,repetitions,"
      "inner_iterations,min_us,median_us,p95_us,mean_us,max_us");
  printMemoryRow("before_benchmark");
  benchmarkFeatureExtraction();
  benchmarkInference();
  benchmarkEndToEnd();
  printMemoryRow("after_benchmark");
  Serial.print("BENCHMARK_SINK,");
  Serial.println(benchmark_sink, 9);
  benchmark_completed = true;
}

// ============================================================
// Salida
// ============================================================

void printPredictionRow() {
#if EDGE_ENABLE_LIVE_OUTPUT
  Serial.print("PREDICTION,");
  Serial.print(processed_windows);
  Serial.print(",");
  Serial.print(ACTIVE_MODEL_NAME);
  Serial.print(",");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(current_class);
  Serial.print(",");
  Serial.print(CLASS_NAMES[current_class]);
  Serial.print(",");
  Serial.print(probabilities[0], 9);
  Serial.print(",");
  Serial.print(probabilities[1], 9);
  Serial.print(",");
  Serial.print(probabilities[2], 9);
  Serial.print(",");
  Serial.print(last_feature_us, 3);
  Serial.print(",");
  Serial.print(last_inference_us, 3);
  Serial.print(",");
  Serial.print(last_total_us, 3);
  Serial.print(",");
  Serial.print(ESP.getFreeHeap());
  Serial.print(",");
  Serial.print(ESP.getMinFreeHeap());
  Serial.print(",");
  Serial.print(stackHighWaterBytes());
  Serial.print(",");
  Serial.print(read_errors);
  Serial.print(",");
  Serial.println(missed_deadlines);
#endif
}

#if EDGE_ENABLE_OLED
void showOledMessage(const char* first, const char* second = nullptr) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(first);
  if (second != nullptr) display.println(second);
  display.display();
}

void updateOledPrediction() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(ACTIVE_MODEL_NAME);
  display.print("Clase: ");
  display.println(CLASS_NAMES[current_class]);
  display.print("Conf: ");
  display.print(current_confidence * 100.0f, 1);
  display.println("%");
  display.print("Inf: ");
  display.print(last_inference_us, 1);
  display.println(" us");
  display.print("Heap: ");
  display.println(ESP.getFreeHeap());
  display.print("Ventana: ");
  display.println(processed_windows);
  display.display();
}
#endif

// ============================================================
// Inferencia en vivo
// ============================================================

void processCurrentWindow() {
  const int64_t total_start = esp_timer_get_time();

  const int64_t feature_start = total_start;
  computeCurrentFeatures();
  const int64_t feature_end = esp_timer_get_time();

  if (!edge_features::allFinite(
          features,
          edge_features::FEATURE_COUNT)) {
    Serial.println("ERROR,non_finite_features");
    return;
  }

  const int64_t inference_start = esp_timer_get_time();
  current_class = activeModelPredict(features, probabilities);
  const int64_t inference_end = esp_timer_get_time();

  last_feature_us =
      static_cast<float>(feature_end - feature_start);
  last_inference_us =
      static_cast<float>(inference_end - inference_start);
  last_total_us =
      static_cast<float>(inference_end - total_start);
  current_confidence = probabilities[current_class];
  ++processed_windows;

  printPredictionRow();

  if (!benchmark_completed) {
    runOneTimeBenchmark();
    // El benchmark bloquea deliberadamente la adquisicion. Se reinicia
    // la referencia temporal para que no se cuenten plazos artificiales.
    next_sample_us = micros() + EDGE_SAMPLE_PERIOD_US;
  }

#if EDGE_ENABLE_OLED
  updateOledPrediction();
  // La pantalla es solo para demostracion; se reinicia el periodo para
  // impedir que su transferencia I2C contamine el contador de plazos.
  next_sample_us = micros() + EDGE_SAMPLE_PERIOD_US;
#endif
}

// ============================================================
// Arduino
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(250);

  Serial.println("EDGE_AI_BENCHMARK,version,2.0.0");
  Serial.print("CONFIG,model,");
  Serial.print(ACTIVE_MODEL_NAME);
  Serial.print(",model_id,");
  Serial.print(EDGE_MODEL_ID);
  Serial.print(",parameter_bytes,");
  Serial.print(ACTIVE_MODEL_PARAMETER_BYTES);
  Serial.print(",sample_period_us,");
  Serial.print(EDGE_SAMPLE_PERIOD_US);
  Serial.print(",sample_rate_hz,");
  Serial.print(1000000.0f / EDGE_SAMPLE_PERIOD_US, 6);
  Serial.print(",window_size,");
  Serial.print(EDGE_WINDOW_SIZE);
  Serial.print(",step_size,");
  Serial.print(EDGE_STEP_SIZE);
  Serial.print(",acceleration_unit,m_s2,gyroscope_unit,deg_s,oled,");
  Serial.println(EDGE_ENABLE_OLED);

  Serial.println(
      "MEMORY_HEADER,model,stage,sketch_bytes,free_sketch_bytes,"
      "model_parameter_bytes,heap_total_bytes,heap_free_bytes,"
      "heap_min_free_bytes,largest_free_block_bytes,"
      "stack_high_water_bytes");
  Serial.println(
      "PREDICTION_HEADER,window,model,time_ms,class_id,class_name,"
      "prob_reposo,prob_suave,prob_brusco,feature_us,inference_us,"
      "total_us,heap_free_bytes,heap_min_free_bytes,"
      "stack_high_water_bytes,read_errors,missed_deadlines");

  Wire.begin(SDA_PIN, SCL_PIN, 400000);

#if EDGE_ENABLE_OLED
  if (!display.begin(
          SSD1306_SWITCHCAPVCC,
          OLED_ADDRESS)) {
    Serial.println("ERROR,oled_not_detected");
    while (true) delay(1000);
  }
  showOledMessage("Edge AI", "Inicializando");
#endif

  if (!detectMPU()) {
    Serial.println("ERROR,mpu6050_not_detected");
#if EDGE_ENABLE_OLED
    showOledMessage("ERROR MPU6050", "Revise I2C");
#endif
    while (true) delay(1000);
  }

  if (!initializeMPU()) {
    Serial.println("ERROR,mpu6050_configuration_failed");
    while (true) delay(1000);
  }

  Serial.println("STATUS,calibrating_keep_sensor_still_z_up");
#if EDGE_ENABLE_OLED
  showOledMessage("Calibrando", "Quieto, Z arriba");
#endif

  if (!calibrateMPU()) {
    Serial.println("ERROR,mpu6050_calibration_failed");
    while (true) delay(1000);
  }

  Serial.print("CALIBRATION,accel_offset_g,");
  Serial.print(accelerometer_offset_g[0], 9);
  Serial.print(",");
  Serial.print(accelerometer_offset_g[1], 9);
  Serial.print(",");
  Serial.print(accelerometer_offset_g[2], 9);
  Serial.print(",gyro_offset_deg_s,");
  Serial.print(gyroscope_offset_dps[0], 9);
  Serial.print(",");
  Serial.print(gyroscope_offset_dps[1], 9);
  Serial.print(",");
  Serial.println(gyroscope_offset_dps[2], 9);

  printMemoryRow("after_setup");
  Serial.println("STATUS,ready_waiting_for_50_samples");

#if EDGE_ENABLE_OLED
  showOledMessage("Listo", ACTIVE_MODEL_NAME);
#endif

  next_sample_us = micros() + EDGE_SAMPLE_PERIOD_US;
}

void loop() {
  const uint32_t now_us = micros();
  const int32_t time_until_sample =
      static_cast<int32_t>(next_sample_us - now_us);

  if (time_until_sample > 0) {
    if (time_until_sample > 1000) delayMicroseconds(500);
    return;
  }

  const uint32_t lateness_us = now_us - next_sample_us;
  if (lateness_us >= EDGE_SAMPLE_PERIOD_US) {
    missed_deadlines += lateness_us / EDGE_SAMPLE_PERIOD_US;
    next_sample_us = now_us + EDGE_SAMPLE_PERIOD_US;
  } else {
    next_sample_us += EDGE_SAMPLE_PERIOD_US;
  }

  float sample[edge_features::SIGNAL_COUNT] = {};
  if (!readCalibratedMPU(sample)) {
    ++read_errors;
    return;
  }

  addSample(sample);

  if (buffered_samples >= EDGE_WINDOW_SIZE &&
      new_samples_since_inference >= EDGE_STEP_SIZE) {
    new_samples_since_inference = 0;
    processCurrentWindow();
  }
}
