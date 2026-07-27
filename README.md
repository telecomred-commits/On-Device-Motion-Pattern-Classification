# On-Device-Motion-Pattern-Classification
On-Device Motion Pattern Classification: A Comparative Study of Compact Machine Learning Models on ESP32.
# On-Device Motion Pattern Classification – Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/XXXXXX.svg)](https://doi.org/XXXXXX)

## Repository Overview

This repository contains the complete research artifacts supporting the manuscript:

> **On-Device Motion Pattern Classification**

The repository has been released to ensure full experimental reproducibility by providing the datasets, embedded firmware, evaluation results, model implementations, and simulation projects used throughout the study.

All materials correspond to the experiments described in the manuscript and allow independent researchers to reproduce the reported results.

---

# Repository Contents

```
Repository/
│
├── README.md
├── LICENSE
├── CITATION.cff
│
├── 01_Raw_Data/
├── 02_Processed_Data/
├── 03_Model_Evaluation/
├── 04_Embedded_Firmware/
├── 05_Model_Conversion/
├── 06_Wokwi_Projects/
├── 07_Documentation/
└── figures/
```

---

# Repository Structure

## 01_Raw_Data

Contains the original experimental data acquired from the MPU6050 sensor before any preprocessing.

Contents include:

- Raw sensor recordings
- Original acquisition files
- Acquisition notes

These files represent the complete unprocessed dataset used in this research.

---

## 02_Processed_Data

Contains the cleaned and integrated dataset after preprocessing.

Included files:

- Complete cleaned dataset
- Training dataset
- Validation dataset
- Test dataset

These datasets were directly used during model training and evaluation.

---

## 03_Model_Evaluation

Contains the complete experimental results obtained during the comparison of all evaluated machine learning models.

Included files:

- Logistic Regression results
- Decision Tree results
- Random Forest results
- Compact MLP results
- Reference MLP results
- Performance metrics
- Confusion matrices
- Computational performance measurements

These results support the comparative analysis reported in the manuscript.

---

## 04_Embedded_Firmware

Contains the complete embedded firmware deployed on the ESP32 microcontroller.

Each folder corresponds to one evaluated machine learning model and includes all C source files required for deployment.

Included implementations:

- Logistic Regression
- Decision Tree
- Random Forest
- Compact MLP
- Reference MLP

The firmware corresponds exactly to the versions evaluated during the experimental validation.

---

## 05_Model_Conversion

Contains the scripts and intermediate files used to convert trained machine learning models into embedded C implementations suitable for ESP32 deployment.

Contents include:

- Conversion scripts
- Exported model parameters
- Conversion documentation

---

## 06_Wokwi_Projects

This directory contains the public Wokwi simulation projects used during the development and validation of the embedded implementations.

The complete firmware can also be executed directly in the Wokwi online simulator.

### Public Wokwi Projects

| Project | Description | Link |
|----------|-------------|------|
| Prototype 1 | Data acquisition | *(Insert link)* |
| Prototype 2 | Logistic Regression | *(Insert link)* |
| Prototype 3 | Decision Tree | *(Insert link)* |
| Prototype 4 | Random Forest | *(Insert link)* |
| Prototype 5 | Compact MLP | *(Insert link)* |
| Prototype 6 | Reference MLP | *(Insert link)* |

Researchers can clone any project directly from Wokwi to reproduce the experiments without additional hardware configuration.

---

## 07_Documentation

Contains supporting documentation including:

- Experimental setup
- Hardware description
- Supplementary material
- Additional implementation notes

---

# Experimental Workflow

The experimental workflow implemented in this study is summarized below.

```
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Feature Extraction
    │
    ▼
Model Training
    │
    ▼
Performance Evaluation
    │
    ▼
Model Selection
    │
    ▼
Conversion to Embedded C
    │
    ▼
ESP32 Firmware
    │
    ▼
Experimental Validation
```

---

# Machine Learning Models

The following compact machine learning models were experimentally evaluated.

| Model | Included | Embedded Firmware |
|--------|-----------|------------------|
| Logistic Regression | ✓ | ✓ |
| Decision Tree | ✓ | ✓ |
| Random Forest | ✓ | ✓ |
| Compact MLP | ✓ | ✓ |
| Reference MLP | ✓ | ✓ |

---

# Hardware Platform

The embedded implementation was validated using:

- ESP32 Development Board
- MPU6050 Inertial Measurement Unit (IMU)
- Embedded C firmware
- Arduino Framework

---

# Reproducibility

The repository contains every artifact required to reproduce the experiments reported in the manuscript.

To reproduce the study:

1. Download the raw dataset.
2. Generate the processed dataset.
3. Train all machine learning models.
4. Evaluate each model.
5. Compare computational performance.
6. Select the best-performing model.
7. Convert the selected model into embedded C.
8. Compile the firmware.
9. Upload the firmware to the ESP32.
10. Validate the embedded implementation.
11. Compare the obtained metrics with those reported in the manuscript.

---

# Software Requirements

Recommended software versions:

- Python
- Scikit-learn
- NumPy
- Pandas
- Arduino IDE
- ESP32 Arduino Core
- Wokwi Simulator

Specific package versions are provided in the corresponding directories when applicable.

---

# Citation

If you use this repository, please cite both the associated manuscript and this repository.

```
Citation information will be updated after publication.
```

---

# License

This repository is distributed under the MIT License unless otherwise stated.

---

# Contact

For questions regarding the repository or the experimental implementation, please contact the corresponding author through the information provided in the published manuscript.

---

# Acknowledgments

The authors acknowledge the support provided by Fundación Universitaria Los Libertadores during the development of this research.

---

© Authors, 2026.
