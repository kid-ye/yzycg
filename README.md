Here is a **clean, concise, technical README** suitable for GitHub or submission.

---

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

## Summary

This project demonstrates a complete end-to-end system for ECG-based biometric authentication, combining embedded hardware, signal processing, machine learning, and system-level benchmarking. It is designed to evaluate real-world feasibility of continuous, secure, and low-power biometric authentication on edge devices.
