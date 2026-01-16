# Ge-ezVision: Amharic Character Recognition

## Overview
This project is an end-to-end Machine Learning system designed to recognize handwritten Amharic characters. It processes raw images, trains a Convolutional Neural Network (CNN), and provides tools for evaluation and prediction.

## Key Features
-   **Dataset**: 37,652 images of 238 unique Amharic characters (Source: Fetulhak Abdurahman).
-   **Architecture**: Custom "SimpleCNN" (PyTorch) optimized for 32x32 grayscale images.
-   **Pipeline**: Automated scripts for data preprocessing, training, and evaluation.

## Project Structure
```
├── data/               # Raw and processed datasets
├── docs/               # Phase reports (1, 2, 3)
├── models/             # Saved model artifacts (.pth)
├── notebooks/          # EDA and Evaluation notebooks
├── reports/            # Generated metrics (Confusion Matrix, etc.)
├── src/                # Source code
│   ├── data/           # Processing scripts
│   └── models/         # Model definition, training, loading
└── README.md
```

## Getting Started

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision pandas pillow scikit-learn seaborn matplotlib tqdm
```

### 2. Data Preparation
```bash
# Processes raw data into data/processed/
python src/data/process_data.py
```

### 3. Training
```bash
# Trains for 5 epochs and saves to models/amharic_cnn.pth
python src/models/train_model.py
```

### 4. Prediction
```bash
python src/models/predict_model.py path/to/image.png
```

## Results (Phase 3)
-   **Test Accuracy**: 16.47% (Baseline: 0.4%)
-   The model demonstrates learning capability but requires further tuning (more epochs, deeper architecture) for production use.
