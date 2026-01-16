# Phase 2: Model Building & Training Report

**Project:** Ge-ezVision - Amharic Character Recognition
**Phase:** 2 - CNN Implementation
**Date:** January 16, 2026

---

## 1. Introduction
Following the data preparation in Phase 1, Phase 2 focused on designing and training a Machine Learning model. The objective was to create a baseline Deep Learning model capable of classifying the 238 unique Amharic characters.

## 2. Methodology: Simplicity First
We chose a **Simple Convolutional Neural Network (CNN)** approach. Complex models (like ResNet or Transformers) are harder to debug and explain. A simple architecture like **LeNet-5** is sufficient for 32x32 grayscale images and serves as an excellent educational baseline.

### A. Model Architecture (`src/models/model.py`)
The model, `SimpleCNN`, consists of two main stages:
1.  **Feature Extractor (Convolutional Blocks)**:
    -   **Block 1**:
        -   `Conv2d`: 32 filters, 3x3 kernel. Detects basic edges and lines.
        -   `BatchNorm`: Stabilizes learning.
        -   `ReLU`: Adds non-linearity.
        -   `MaxPool`: Reduces 32x32 -> 16x16.
    -   **Block 2**:
        -   `Conv2d`: 64 filters, 3x3 kernel. Detects compound shapes (curves, loops).
        -   `BatchNorm` & `ReLU`.
        -   `MaxPool`: Reduces 16x16 -> 8x8.
2.  **Classifier (Fully Connected Layers)**:
    -   `Flatten`: Converts 2D feature maps (64x8x8) into a 1D vector (4096 inputs).
    -   `Linear (FC1)`: compressed to 128 neurons.
    -   `Dropout (0.5)`: Randomly turns off neurons during training to prevent overfitting (memorization).
    -   `Linear (FC2)`: Output layer with 238 neurons (one for each character class).

### B. Input Pipeline (`src/models/dataset.py`)
We implemented a custom PyTorch `Dataset` class (`AmharicDataset`) that:
-   Reads the `dataset_index.csv`.
-   Loads images on-the-fly using `Pillow`.
-   Converts images to Tensors (normalized 0-1).
-   Maps string labels (e.g., "001he") to integer IDs (0-237).

## 3. Experimental Setup (`src/models/train_model.py`)
-   **Framework**: PyTorch
-   **Optimizer**: Adam (Adaptive Moment Estimation) - handles learning rates automatically.
-   **Loss Function**: CrossEntropyLoss (standard for multi-class classification).
-   **Hyperparameters**:
    -   `Batch Size`: 32
    -   `Learning Rate`: 0.001
    -   `Epochs`: 5

## 4. Results
The model was trained for 5 epochs.

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 5.11 | 2.05% | - | - | Initial learning. Loss began high (5.43) and decreased to 4.33. |
| **2** | - | - | - | - | *Log truncated for brevity* |
| **5** | 4.58 | 3.75% | 3.78 | **16.21%** | Significant improvement over random chance (0.4%). |

### Analysis
-   **Performance**: 16.2% accuracy is low for production but expected for a shallow network trained for only 5 epochs on 238 classes.
-   **Comparison**: Random guessing would yield ~0.42% accuracy (1/238). Our model is **38x better** than random chance, proving it is learning structural features.

## 5. Conclusion & Next Steps
We successfully implemented and verified the end-to-end training pipeline. The low accuracy suggests we need:
1.  **More Epochs**: 5 is likely insufficient for convergence.
2.  **Deeper Model**: A 2-layer CNN may be too simple for 238 nuances classes.
3.  **Data Augmentation**: Rotations/shifts to generalize better.

Phase 2 is considered **Complete** as the infrastructure is functional.
