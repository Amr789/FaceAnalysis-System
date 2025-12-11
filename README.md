# FaceAnalysis-System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UeZZ_gKgRCwE8rtvHGvEm5YXFAKg0ZPN#scrollTo=2FFmNsZtyNdS)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

##  Overview
**FaceAnalysis-System** is a robust computer vision pipeline designed for **Age Estimation** and **Face Verification**. Unlike standard baseline models, this system prioritizes demographic fairness by using **Inverse Frequency Weighted Loss** to handle class imbalance, ensuring better performance on minority age groups (children and seniors).

The system features:
* **Age Estimation:** A custom **EfficientNet-B0** regressor.
* **Face Verification:** A pre-trained **InceptionResnetV1** (FaceNet) pipeline.
* **Robust Inference:** Implements **Test Time Augmentation (TTA)** to stabilize predictions.

##  Key Features
* **Modern Architecture:** Uses `EfficientNet-B0` for efficient feature extraction.
* **Fair Training:** Implements **Weighted L1 Loss** to penalize errors on rare age groups more heavily than common ones.
* **Production-Ready Structure:** Modular code organized into training, inference, and data handling layers.
* **Secure:** Automated data handling scripts ensuring API keys are never hardcoded.

---

## Project Structure
```text
FaceAnalysis-System/
├── data/                  # Dataset storage (excluded from git)
├── models/                # Saved model checkpoints
├── scripts/               # Helper scripts (setup, download)
├── src/
│   ├── dataset.py         # Data loading & Albumentations
│   ├── networks.py        # EfficientNet model definition
│   ├── train.py           # Training loop with Weighted Loss
│   └── inference.py       # Inference pipeline with TTA
├── requirements.txt       # Dependencies
├── main.py       # CLI Entry point 
└── README.md              # Documentation
```

## Installation

### Prerequisites
* Python 3.8+
* GPU recommended (CUDA)
