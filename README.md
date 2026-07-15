# ECG-Based Continuous Biometric Authentication System

## Overview

This project implements a real-time biometric authentication system using electrocardiogram (ECG) signals. The system leverages physiological uniqueness of cardiac activity to perform continuous identity verification without relying on passwords or traditional credentials.

The pipeline integrates hardware-level ECG acquisition, signal preprocessing, deep learning–based feature extraction, and similarity-based authentication. It supports cross-platform deployment across microcontrollers and edge devices.

---

## System Architecture

```
ECG Sensor → Microcontroller → Data Streaming → Signal Processing → CNN Model → Feature Embedding → Similarity Matching → Authentication Decision
```

---

## Hardware Components

* ECG Analog Front-End: MAX30003 (primary), AD8232 (alternative)
* Microcontrollers:

  * RP2040 (Raspberry Pi Pico 2)
  * ESP32
  * STM32 (F4/F7 series)
* Edge Device:

  * Raspberry Pi 4
* Communication:

  * SPI (sensor interface)
  * USB Serial / UART / BLE (data transmission)

---

## Data Acquisition

* ECG signals captured using electrode-based setup (3-lead configuration)
* Sampling rate: 200–500 Hz (configurable)
* Data streamed as structured binary packets from microcontroller to host system
* FIFO-based reading from MAX30003 with interrupt-driven acquisition

---

## Signal Processing

### Preprocessing Steps

* Bandpass filtering: 0.5–40 Hz
* Z-score normalization
* Noise reduction (baseline wander, motion artifacts)

### Segmentation Methods

* No Peak Detection (NPD): fixed-length windowing
* R-R interval segmentation
* P-T wave segmentation

Each ECG segment is standardized to a fixed length (e.g., 1000 samples) before model input.

---

## Machine Learning Model

### Model Type

* 1D Convolutional Neural Network (CNN)

### Input

* ECG segment (1 × 1000 samples)

### Output

* High-dimensional embedding vector (identity representation)

### Training Frameworks

* Siamese Network (pairwise learning)
* Triplet Network (anchor-positive-negative)

### Loss Functions

* Triplet Loss (primary)
* Contrastive Loss (optional)

---

## Feature Matching

Similarity between embeddings is computed using:

* Pearson Correlation
* Cosine Similarity
* Euclidean Distance

Authentication decision is based on a threshold determined using Equal Error Rate (EER).

---

## Continuous Authentication

* Sliding window inference (2–5 seconds)
* Periodic re-verification of user identity
* Enables persistent authentication rather than one-time login

---

## Datasets

The model is trained and evaluated on multiple ECG datasets:

* PTB-XL
* PTB Diagnostic ECG Database
* ECG-ID Database
* MIT-BIH Arrhythmia Dataset
* Custom collected ECG data (MAX30003)

### Data Processing

* Resampling to a uniform frequency
* Normalization across datasets
* Train/validation/test split
* Cross-dataset generalization evaluation

---

## Deployment

### Platforms

* Raspberry Pi (edge inference)
* ESP32 / STM32 / RP2040 (TinyML deployment)

### Model Optimization

* Quantization (INT8)
* Pruning
* TensorFlow Lite Micro conversion

---

## Edge vs Cloud Comparison

Three inference modes are implemented:

1. Cloud Inference

   * Data sent to remote server
   * High accuracy, higher latency

2. Edge Inference (Raspberry Pi)

   * Local processing
   * Balanced latency and performance

3. TinyML (Microcontroller)

   * On-device inference
   * Low power, constrained accuracy

---

## Benchmarking

Performance is evaluated across devices:

* Latency (inference time)
* Power consumption
* Memory usage
* Accuracy degradation

Devices tested:

* ESP32
* STM32
* RP2040 (Pico)
* Raspberry Pi 4

---

## Evaluation Metrics

* Accuracy
* False Acceptance Rate (FAR)
* False Rejection Rate (FRR)
* Equal Error Rate (EER)
* ROC Curve

---

## Ablation Studies

Experiments include:

* Similarity metric comparison (Pearson vs Cosine vs Euclidean)
* Segmentation method comparison (NPD vs R-R vs P-T)
* Noise robustness analysis

---

## Security Layer (Optional Extension)

### Basic

* AES encryption for stored embeddings

### Advanced

* ECG embedding → quantization → bitstring → cryptographic key
* Hashing (SHA-256)
* Challenge-response authentication

---

## Applications

* Secure device authentication
* Wearable identity verification
* Medical system access control
* Continuous authentication for IoT devices

---

## Tech Stack

* Programming:

  * Python
  * C/C++ (embedded)
* ML Frameworks:

  * PyTorch
  * TensorFlow Lite / TFLite Micro
* Signal Processing:

  * NumPy
  * SciPy
* Embedded Systems:

  * STM32 HAL
  * ESP-IDF / Arduino
* Data Handling:

  * CSV / NumPy arrays

---

## Future Work

* Cross-dataset generalization improvements
* Fully on-device TinyML inference
* Continuous authentication stability analysis
* Cryptographic key generation from ECG
* BLE/mobile integration

---

## Repository Layout

This repo currently implements the PTBDB-based reproduction (Wang et al., 2024,
*"ECG Biometric Authentication Using Self-Supervised Learning for IoT Edge Sensors"*)
plus a live capture path via **MAX30003** + **Raspberry Pi Pico 2**, as a step toward
the full vision described above.

```
mini_project_VI/
├── src/                    # Core Python package
│   ├── preprocessing.py    #   filtering + R2R / P2T / NPD segmentation
│   ├── model.py             #   1D-CNN encoder (ECGEncoder / TripletECGModel)
│   ├── data_loader.py       #   unified PTBDB / MITDB / ECGIDDB loaders (subject-level)
│   ├── ptbdb_benchmark.py   #   training + evaluation (Siamese & Triplet, EER/FAR/FRR)
│   ├── quantization.py      #   fixed-point quantization (Section V-A)
│   ├── inference.py         #   ECGAuthenticator: enroll / authenticate
│   └── live_auth.py         #   live serial authentication loop
├── firmware/
│   └── pico_max30003.py     # MicroPython firmware: MAX30003 -> UART stream (runs on Pico 2)
├── scripts/
│   ├── download_data.py     # fetch PTBDB records
│   ├── run_all.py           # train + evaluate all 6 seg×framework combos (Table I)
│   └── plots/                # figure-generation scripts
├── tests/                   # test_model.py, test_max30003.py
├── models/                  # trained weights (ecg_model.pth, enrolled_profiles.pt)
├── results/                 # generated figures + score arrays
├── docs/
│   ├── paper/                # source paper (PDF/DOCX)
│   └── diagrams/              # architecture / workflow flowcharts (HTML)
└── data/                    # datasets (gitignored) — data/ptbdb, data/users
```

### Running

All scripts are meant to be run **from the repository root**:

```bash
python scripts/download_data.py          # get PTBDB
python scripts/run_all.py --epochs 20    # train + benchmark all combos
python tests/test_model.py               # sanity-check the encoder
python -m src.live_auth                   # live authentication (needs Pico on serial)
```

### Notes / known work items

- `src/ptbdb_benchmark.py` is the **correct** training path (`1 - pearson` distance,
  subject-level split, threshold tuning). The older buggy trainer was removed.
- The live path samples at ~125 Hz but the model expects 200 Hz — resampling still TODO.
- The encoder's large final FC layer (~36 MB model) must be slimmed (global pooling)
  before it can fit on a Pico 2 / ESP32.

See `docs/paper/` for the reference paper.

---

## Summary

This project demonstrates a complete end-to-end system for ECG-based biometric authentication, combining embedded hardware, signal processing, machine learning, and system-level benchmarking. It is designed to evaluate real-world feasibility of continuous, secure, and low-power biometric authentication on edge devices.
