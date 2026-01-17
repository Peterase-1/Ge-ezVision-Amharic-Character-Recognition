# Ge-ezVision: Amharic Character Recognition

## Overview
This project is an end-to-end Machine Learning system designed to recognize handwritten Amharic characters. It processes raw images, trains a Deep Convolutional Neural Network (CNN), and provides tools for evaluation and prediction.

## Project Documentation
Detailed documentation for each phase of the project is available here:
*   **[Final Project Documentation (Full Report)](docs/Final_Project_Documentation.md)**: The complete, professional documentation covering the entire lifecycle.
*   **[Phase 1 Report](docs/phase_1_report.md)**: Project Proposal and Data Collection.
*   **[Phase 2 Report](docs/phase_2_report.md)**: Baseline Modeling.
*   **[Phase 3 Report](docs/phase_3_report.md)**: Evaluation and Refinement.

## Key Features
-   **Dataset**: 37,652 images of 238 unique Amharic characters (Source: Fetulhak Abdurahman).
-   **Architecture**: Custom "DeepAmharicNet" (PyTorch) - 4-layer Deep CNN with Batch Norm and Dropout.
-   **Pipeline**: Automated scripts for data preprocessing, training (with Augmentation), and evaluation.

## Project Structure
```
├── data/               # Raw and processed datasets
├── docs/               # Documentation
│   ├── [Phase 1 Report](docs/phase_1_report.md)
│   ├── [Phase 2 Report](docs/phase_2_report.md)
│   ├── [Phase 3 Report](docs/phase_3_report.md)
│   └── [Final Project Documentation](docs/Final_Project_Documentation.md)
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
# Trains for 20 epochs (Target >85%) and saves to models/amharic_cnn.pth
# Note: Takes ~3-4 hours on CPU
python src/models/train_model.py
```

### 4. Prediction
```bash
python src/models/predict_model.py path/to/image.png
```

## Results (Phase 4)
-   **Test Accuracy**: **88.32%** (Exceeded goal of 65%)
-   **Baseline**: 210x better than random guessing (0.4%).
-   **Architecture**: DeepAmharicNet successfully captures complex features of 238 distinct characters.
