# Phase 1: Project Initialization & Data Engineering Report

**Project:** Ge-ezVision - Amharic Character Recognition
**Phase:** 1 - Initialization & Data Preprocessing
**Date:** January 16, 2026

---

## 1. Overview
The goal of Phase 1 was to establish a robust Machine Learning environment, acquire a high-quality dataset, and transform raw data into a format suitable for training Deep Learning models (CNNs). We adopted a standard modular project structure to ensure scalability and reproducibility.

## 2. Project Architecture
We used a **Cookiecutter Data Science**-inspired structure. This separates data, code, and documentation, preventing the common "spaghetti code" issues in ML projects.

### Folder Breakdown
- **`data/`**: Stores all data, segregated by stage.
    - `raw/`: The immutable, original dataset (**Fetulhak Abdurahman's dataset**). Never modified.
    - `processed/`: The cleaned, resized, and split data ready for training.
- **`src/`**: Source code for the project.
    - `data/`: Scripts to ingest and transform data.
- **`notebooks/`**: Jupyter notebooks for exploration (EDA) and prototyping.
- **`docs/`**: Documentation (like this file).

---

## 3. Key Components & Implementation Details

### A. Data Acquisition
We initially considered several datasets but selected **Fetulhak Abdurahman's Handwritten Amharic Character Dataset** because it offers a comprehensive set of **238 unique characters** with **37,652 samples**. Other datasets were either incomplete (few-shot) or focused on full OCR (sentence-level).

### B. `src/data/process_data.py`
This is the core engine of Phase 1.
- **Why**: Raw data often comes in irregular formats (folders, archives) and sizes. Neural networks require uniform input tensors (e.g., 32x32x1).
- **How it works**:
    1.  **Scanning**: It recursively scans the `data/raw` extraction folders.
    2.  **Class Extraction**: It parses filenames (e.g., `001he.1.jpg`) to determine the label (`001he`). This automates labeling without manual entry.
    3.  **Preprocessing Loop**:
        - **Grayscale Conversion**: Converts images to 1 channel (L mode). Color is irrelevant for character shape.
        - **Resizing**: Resamples images to **32x32 pixels** using `LANCZOS` resampling foundation. 32x32 is a standard benchmark size (like MNIST/CIFAR) efficient for training while retaining legibility.
    4.  **Splitting**: It divides the data into:
        - **Train (70%)**: For learning weights.
        - **Validation (15%)**: For tuning hyperparameters.
        - **Test (15%)**: For final evaluation.
    5.  **Output**:
        - Saves images to `data/processed/amharic_chars/{class_id}/`.
        - Generates `data/processed/dataset_index.csv`. This CSV acts as a "map" for our data loaders, containing paths, labels, and split assignments.

### C. `notebooks/1.0-eda-initial-exploration.ipynb`
- **Why**: "Garbage In, Garbage Out". We must understand our data before modeling.
- **How**: It loads the CSV index to visualize:
    - **Class Balance**: Are some characters rare? (This informs if we need augmentation).
    - **Sample Quality**: Displays random images to verify they are readable and correctly resized.

### D. `requirements.txt` / Virtual Environment
- **Why**: Dependency isolation.
- **Content**: `torch`, `torchvision` (for later), `pandas`, `pillow`, `tqdm`. Ensures the code runs identically on any machine.

---

## 4. Summary of Results
- **Total Images**: 37,652
- **Classes**: 238
- **Image Format**: 32x32 Grayscale PNG
- **Status**: Ready for Model Building (Phase 2).
