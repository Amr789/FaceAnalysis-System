# Multi-Task Face Analysis System

## 📌 Overview
This project implements a robust computer vision pipeline that performs **Age Estimation** and **Face Verification**. It leverages a custom-trained ResNet18 model for age regression and a pre-trained InceptionResnetV1 (VGGFace2) for face matching.

The system is designed to take two input images and determine:
1. The estimated age of the person in each image.
2. Whether the two images represent the same person.

## 🚀 Key Features
* **Pipeline Architecture:** Combines MTCNN (Face Detection), ResNet18 (Age), and InceptionResnetV1 (Verification).
* **Custom Training:** Age estimator trained on the [UTKFace Dataset](https://susanqq.github.io/UTKFace/) with augmentation (Albumentations).
* **Modular Design:** Code is refactored into clear training, data, and inference modules.

## 📂 Project Structure
```text
├── data/              # Dataset directory
├── models/            # Saved model weights
├── src/
│   ├── networks.py    # PyTorch model definitions
│   ├── dataset.py     # Data loading and augmentation
│   ├── train.py       # Training loop
│   └── inference.py   # Inference pipeline
├── main.py            # CLI entry point
└── requirements.txt