#pragma once

// ============================================================
// Seleccion del modelo
// ============================================================
// Cambie EDGE_MODEL_ID para compilar una configuracion distinta.
// PlatformIO lo define automaticamente en cada entorno.

#define EDGE_MODEL_LOGISTIC_REGRESSION 1
#define EDGE_MODEL_DECISION_TREE       2
#define EDGE_MODEL_RANDOM_FOREST       3
#define EDGE_MODEL_MLP_COMPACT         4
#define EDGE_MODEL_MLP_REFERENCE       5

#ifndef EDGE_MODEL_ID
#define EDGE_MODEL_ID EDGE_MODEL_RANDOM_FOREST
#endif

// ============================================================
// Configuracion experimental
// ============================================================
// El periodo de 24 ms reproduce la mediana observada en los
// archivos de adquisicion (41.67 Hz).

#ifndef EDGE_SAMPLE_PERIOD_US
#define EDGE_SAMPLE_PERIOD_US 24000UL
#endif

#ifndef EDGE_WINDOW_SIZE
#define EDGE_WINDOW_SIZE 50
#endif

#ifndef EDGE_STEP_SIZE
#define EDGE_STEP_SIZE 25
#endif

#ifndef EDGE_CALIBRATION_SAMPLES
#define EDGE_CALIBRATION_SAMPLES 600
#endif

// Repeticiones para el benchmark ejecutado sobre la primera
// ventana valida. Las inferencias se miden por lotes para que
// los modelos muy pequenos no queden limitados por la resolucion
// de un microsegundo del temporizador.

#ifndef EDGE_BENCHMARK_REPETITIONS
#define EDGE_BENCHMARK_REPETITIONS 200
#endif

#ifndef EDGE_INFERENCES_PER_TIMING
#define EDGE_INFERENCES_PER_TIMING 100
#endif

#ifndef EDGE_WARMUP_INFERENCES
#define EDGE_WARMUP_INFERENCES 20
#endif

// 0: firmware de benchmark, sin pantalla ni dependencia Adafruit.
// 1: demostracion con OLED SSD1306 128x64 en 0x3C.

#ifndef EDGE_ENABLE_OLED
#define EDGE_ENABLE_OLED 0
#endif

// 1 imprime una fila PREDICTION por cada ventana procesada.
#ifndef EDGE_ENABLE_LIVE_OUTPUT
#define EDGE_ENABLE_LIVE_OUTPUT 1
#endif

static_assert(EDGE_WINDOW_SIZE == 50,
              "El modelo fue entrenado con ventanas de 50 muestras.");
static_assert(EDGE_STEP_SIZE == 25,
              "El estudio utilizo un desplazamiento de 25 muestras.");
static_assert(EDGE_BENCHMARK_REPETITIONS > 4,
              "Se requieren al menos cinco repeticiones.");
static_assert(EDGE_INFERENCES_PER_TIMING > 0,
              "El lote de inferencias debe ser mayor que cero.");

