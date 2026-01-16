# Phase 3: Evaluation & Prediction Report

**Project:** Ge-ezVision - Amharic Character Recognition
**Phase:** 3 - Evaluation & Deployment
**Date:** January 16, 2026

---

## 1. Introduction
Phase 3 focused on creating the necessary tools to **evaluate** the trained model's performance on unseen data and to **deploy** it for making predictions on new images. This phase bridges the gap between a trained model file (`.pth`) and a usable software artifact.

## 2. Tools Created

### A. Prediction Engine (`src/models/predict_model.py`)
This script enables single-item inference.
-   **Usage**: `python src/models/predict_model.py path/to/image.jpg`
-   **Process**:
    1.  Loads the trained architecture (`SimpleCNN`) and weights.
    2.  Preprocesses the input image (Grayscale conversion, Resize to 32x32).
    3.  Runs the forward pass.
    4.  Outputs the **Top-3** predicted classes with confidence probabilities.

### B. Evaluation Suite (`src/models/evaluate_model.py`)
This script performs batch evaluation on the entire Test Set (5,871 images).
-   **Metrics Generated**:
    -   **Accuracy**: Overall percentage of correct predictions.
    -   **Classification Report**: Precision, Recall, and F1-Score for each of the 238 classes. (Saved to `reports/classification_report.csv`).
    -   **Confusion Matrix**: A grid showing which characters are confused with which. (Saved to `reports/confusion_matrix.csv`).

### C. Analysis Notebook (`notebooks/3.0-evaluation.ipynb`)
A Jupyter notebook designed to visualize the metrics generated above. It helps identify patterns in the errors (e.g., are "ha" and "hu" often confused?).

## 3. Evaluation Results
*Note: These results are based on the 5-epoch training run.*

-   **Test Accuracy**: ~16.21%
-   **Baseline Comparison**: Random guess is ~0.42%. The model is performing **~38x better than random**.
-   **Observation**: While 16% is low for a production OCR system, it confirms the pipeline is working correctly. The "SimpleCNN" architecture successfully learned to extract features, but requires more capacity (layers) and training time to master the subtle differences between 238 Amharic characters.

## 4. Final Project Summary
We have successfully built an End-to-End Machine Learning Pipeline:
1.  **Data**: Ingested, cleaned, and organized 37k+ Amharic character images.
2.  **Model**: Designed and implemented a custom CNN in PyTorch.
3.  **Training**: Established a training loop with validation monitoring.
4.  **Evaluation**: Created tools to rigorous test and analyze the model.

The project infrastructure is robust and ready for future improvements (Phase 4), such as Hyperparameter Tuning or Architecture Search.
