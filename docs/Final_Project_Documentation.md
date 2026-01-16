# FINAL PROJECT DOCUMENTATION
# GE-EZVISION: AMHARIC CHARACTER RECOGNITION USING DEEP CNN

**DEBRE BERHAN UNIVERSITY**
**COLLEGE OF COMPUTING**
**DEPARTMENT OF COMPUTER SCIENCE**

**Project Title:** Ge-ezVision: Amharic Handwritten Character Recognition
**By:** [Your Name]
**Submitted to:** [Instructor Name]
**Date:** January 16, 2026
**Location:** Debre Berhan, Ethiopia

**Project Repository:** [https://github.com/Peterase-1/Ge-ezVision-Amharic-Character-Recognition](https://github.com/Peterase-1/Ge-ezVision-Amharic-Character-Recognition)
**Dataset Source:** [Fetulhak Abdurahman - Handwritten Amharic Character Dataset](https://www.kaggle.com/datasets/fetulhak/handwritten-amharic-character-dataset)

---

# CHAPTER ONE: INTRODUCTION

## 1.1 Background
In the current "Age of AI," nations utilize Artificial Intelligence to drive economic and social progress. For Ethiopia to benefit from and participate in this Global AI revolution, it is imperative to train AI systems using **Local Data**. However, a significant bottleneck exists: the majority of Ethiopia's institutional knowledge—including Government records, Organizational archives, medical history, and legal documents—remains written on paper.

While the developed world has moved to "Big Data," Ethiopia currently faces a data accessibility crisis. There is no large-scale, digitized corpus available for developing advanced technologies like **Large Language Models (LLMs)** for Amharic. As long as this data remains locked in physical format, future AI developement in the country is stalled.

This project, "Ge-ezVision", addresses this critical infrastructure gap. By developing a robust Optical Character Recognition (OCR) system using **Deep Convolutional Neural Networks (CNNs)**, we provide the technological key to unlock these physical archives. This is not merely a character recognition tool; it is a foundational step toward mass digitization, enabling the generation of the massive datasets required to build the next generation of Ethiopian AI.

## 1.2 Motivation
The primary motivation for this research is to enable **Data Sovereignty and Future AI Development** in Ethiopia.
1.  **Enabling Future LLMs**: Modern AI (like GPT-4) requires billions of tokens of text training data. Currently, Amharic lacks this volume of digital text. By automating the conversion of paper records to text, we create the raw material needed to train future Amharic LLMs.
2.  **Digitizing Institutional Memory**: Government and private organizations in Ethiopia hold decades of data on paper. Digitizing this improves efficiency, transparency, and ensures that vital national statistics are machine-readable.
3.  **Cultural & Historical Preservation**: Preserving manuscripts and historical records in a digital text format ensures they are searchable and accessible to future generations, preventing the loss of heritage due to physical decay.
4.  **Accessibility**: Converting handwritten text to digital text allows assistive technologies (like Screen Readers) to serve the visually impaired, promoting digital inclusivity.

Developing an accurate, open-source model for Amharic handwriting is therefore a critical infrastructure project for the country's digital future.

## 1.3 Statement of the problem
Recognizing handwritten Amharic characters presents a unique set of computational challenges that standard "out-of-the-box" models often fail to address:
1.  **Visual Similarity (Inter-class Similarity)**: The Ge'ez script has high visual correlation. For example, the character 'በ' (Be) and 'ቤ' (Bie) differ only by a subtle vertical dash. 'ፀ' (Tse) and 'ፅ' (Tso) are mirror-like. Shallow models cannot distinguish these fine-grained features.
2.  **The Curse of Dimensionality**: Classification complexity grows with the number of classes. With 238 classes, the decision boundary is far more complex than the 10-class MNIST (Digits) or 26-class EMNIST (English) problems.
3.  **Data Scarcity**: Unlike English, which has datasets with millions of samples, Amharic handwriting datasets are relatively small and noisy, leading to overfitting.

The problem this study addresses is the lack of a high-accuracy, noise-robust automated system for identifying this large set of complex characters.

## 1.4 Objectives

### 1.4.1 General Objective
The overarching goal of this project is to research, design, develop, and test a Deep Learning-based system capable of recognizing isolated handwritten Amharic characters with a validation accuracy exceeding 85%, suitable for deployment in digitization workflows.

### 1.4.2 Specific objectives
To achieve the general objective, the following specific mile-stones were set:
1.  **Data Acquisition**: To acquire and clean the "Fetulhak Abdurahman" dataset, ensuring it is balanced and free of corrupt files.
2.  **Architecture Design**: To construct a custom Deep CNN (`DeepAmharicNet`) with specific layers (Convolution, Batch Normalization, Dropout) tailored to the feature density of Amharic characters.
3.  **Robustness Enhancement**: To implement "Data Augmentation" pipelines that mathematically perturb training images (rotation, shifting) to force the model to learn invariant features.
4.  **Evaluation**: To rigorously test the model using identifying metrics including Confusion Matrices, F1-Scores, and Top-K Accuracy to understand failure cases.

## 1.5 Significance of the study
The successful completion of Ge-ezVision has significant implications:
*   **Academic**: It serves as a benchmark for future researchers applying Deep Learning to African scripts, demonstrating effective hyperparameters and architecture choices.
*   **Practical**: It provides a pre-trained model artifact (`.pth` file) that software developers can immediately integrate into Android Apps or Web Services for OCR.
*   **Social**: It promotes the use of indigenous languages in the digital sphere, preventing the technological marginalization of the Amharic language.

## 1.6 Scope and limitations
**Scope**:
*   The study is limited to **Offline OCR** (processing static images), not Online OCR (tracking stylus movement).
*   It covers the **238 basic characters** commonly used in modern Amharic.
*   It assumes inputs are **isolated characters** (pre-segmented), meaning the system does not currently handle full pages of text layout analysis.

**Limitations**:
*   **Background Noise**: The model is trained on relatively clean backgrounds. It may struggle with paper that has heavy stains or ruled lines.
*   **Cursive Writing**: While Amharic is not typically cursive, connected handwriting styles may pose segmentation challenges not addressed here.

## 1.7 Methodology (Overview)
The methodology follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) model modified for Deep Learning:
1.  **Data**: Ingestion of 37,000+ images.
2.  **Preprocessing**: Resizing to 32x32 pixels, Grayscale conversion (1 channel), and Normalization (0-1 range).
3.  **Modeling**: Iterative development from specific baseline (SimpleCNN) to optimized (DeepAmharicNet).
4.  **Evaluation**: Quantitative testing on a held-out Test Set (15% of data).

---

# CHAPTER TWO: LITERATURE REVIEW

## 2.1 Introduction
Optical Character Recognition (OCR) has evolved from a rule-based engineering field into a sub-discipline of Artificial Intelligence. This chapter reviews the historical progression and the theoretical foundations of the technologies used in this project.

## 2.2 Approaches to Character Recognition

### 2.2.1 Statistical and Template Matching (1990s-2000s)
Early OCR systems for Amharic relied on "Template Matching". A database of perfect character images existed, and new inputs were compared pixel-by-pixel.
*   *Limitation*: This approach is extremely brittle. If a writer makes a character slightly wider or tilted, the pixel overlap fails.
*   *Feature Engineering*: Later, researchers used handcrafted features like **Gabor Filters** or **HOG (Histogram of Oriented Gradients)** to describe shapes. This improved results but required deep linguistic expertise to define what "makes" a character.

### 2.2.2 Artificial Neural Networks (ANNs)
Multilayer Perceptrons (MLPs) introduced the ability to "learn" from data. However, MLPs struggle with images because they flatten the 2D structure into a 1D vector immediately, losing spatial relationships (e.g., knowing that a pixel is "above" another).

### 2.2.3 Deep Convolutional Neural Networks (The Modern Era)
The breakthrough for this project is the **CNN**. Unlike MLPs, CNNs preserve the 2D structure of the image. They use small "filters" (kernels) that slide over the image to detect local patterns.
*   **Low-Level**: First layers detect lines and edges.
*   **Mid-Level**: Middle layers combine lines into corners and curves.
*   **High-Level**: Final layers combine curves into full identifying character shapes (e.g., the loop of 'ወ').
This hierarchical learning makes them perfect for the complex visual structure of Ge'ez.

## 2.3 Basic Components of Opinion Mining / OCR
*(Note: Adapting structure to OCR context)*
An effective OCR pipeline consists of four distinct modules:
1.  **Image Acquisition**: The physical-to-digital bridge. In our case, this is pre-handled by the dataset providers who scanned handwritten forms.
2.  **Preprocessing**: The cleaning phase. Shadows are removed, contrast is enhanced, and images are scaled to a standard size (32x32) to ensure the neural network receives consistent input.
3.  **Feature Extraction**: The core "Vision" component. In `DeepAmharicNet`, this is performed by the 4 Convolutional blocks which compress the image into a dense vector of "meaning".
4.  **Classification**: The decision maker. The Fully Connected layers take the feature vector and calculate a probability score for each of the 238 classes.

## 2.4 Workflow Steps
The data flow in our implemented system is linear and deterministic:
1.  **Read Image** $\rightarrow$ **Resize (32x32)** $\rightarrow$ **ToTensor**.
2.  **Forward Pass**: The tensor flows through the network layers.
3.  **Softmax**: The final raw numbers (logits) are converted into probabilities (0% to 100%).
4.  **Argmax**: The index with the highest probability is selected as the prediction.

## 2.5 Related works in Amharic OCR
*   **Meshesha & Jawahar (2008)**: Used Gabor filters and achieved ~80% accuracy on printed text.
*   **Assabie & Bigun (2011)**: Used direction field tensors for handwriting, showing promise but high computational cost.
*   **Recent Deep Learning**: Studies from 2020-2024 have begun applying ResNets to Amharic. Our project aligns with this modern wave, but specifically focuses on optimizing for *resource efficiency* (getting high accuracy with a relatively small, fast model) rather than using massive, slow pre-trained models like VGG-16.

---

# CHAPTER THREE: METHODOLOGY

## 3.1 Research Design
We employed an **Experimental Quantitative Research Design**.
*   **Independent Variables**: Neural Network Depth (2 layers vs 4 layers), Learning Rate, Data Augmentation Strategies.
*   **Dependent Variable**: Validation/Test Accuracy.
*   **Control Variables**: Image size (32x32), Batch Size (64), Dataset Split (70/15/15).
The study proceeded in iterative phases: Phase 1 (Data Setup), Phase 2 (Baseline Modeling), Phase 3 (Evaluation Tooling), and Phase 4 (Optimization).

## 3.2 Data Collection
The foundation of any AI project is data. We utilized the **Fetulhak Abdurahman Handwritten Amharic Character Dataset**.
*   **Source**: Open-access research repository (Kaggle).
*   **Volume**: 37,652 total images.
*   **Granularity**: 238 folders, one for each character class.
*   **Diversity**: Collected from multiple writers to ensure variation in style.
*   **Validation**: We wrote a script (`verify_data.py`) to programmatically verify that all 238 classes were present and that no empty files existed.

## 3.3 System Architecture / Framework
The core contribution is the **`DeepAmharicNet`**, implemented in PyTorch.

### 3.3.1 The Architecture
The network is a stack of 4 identical "Residual-style" blocks followed by a classifier.
*   **Input**: 1x32x32 Grayscale Image.
*   **Block 1**: 32 Filters (3x3 Kernel). Detects rudimentary edges.
    *   *Batch Norm*: Stabilizes activations.
    *   *ReLU*: Adds non-linearity.
    *   *MaxPool*: Compresses 32x32 $\rightarrow$ 16x16.
*   **Block 2**: 64 Filters. Detects corners. (Output: 8x8).
*   **Block 3**: 128 Filters. Detects character parts. (Output: 4x4).
*   **Block 4**: 256 Filters. Detects full character shapes. (Output: 2x2).
*   **Classifier**:
    *   Flatten (256*2*2 = 1024 neurons).
    *   Linear Layer $\rightarrow$ 512 neurons.
    *   *Dropout (0.5)*: Randomly shuts off 50% of neurons to prevent memorization.
    *   Final Linear Layer $\rightarrow$ 238 neurons (Classes).

## 3.4 Algorithms and Techniques

### 3.4.1 Learning Algorithm: Adam
We used the **Adam (Adaptive Moment Estimation)** optimizer. Unlike standard SGD, Adam maintains a per-parameter learning rate. This is crucial for our high-dimensional problem where some weights (rare features) need to update faster than others (common features).

### 3.4.2 Loss Function: Cross Entropy
We minimized the **Cross Entropy Loss**:
$$ Loss = - \sum (y_{actual} \cdot \log(y_{predicted})) $$
This penalizes the model heavily when it is confident but wrong, forcing it to learn distinct decision boundaries.

### 3.4.3 Data Augmentation
To solve the data scarcity problem, we expanded the dataset synthetically during training:
*   `RandomRotation(10)`: Rotates image $\pm 10$ degrees.
*   `RandomAffine`: Shifts image horizontally/vertically by 10%.
This implies the model sees a "new" variation of the dataset every epoch, effectively training on infinite variations.

## 3.5 Implementation Tools
*   **Python**: The glue language.
*   **PyTorch**: The Deep Learning framework chosen for its dynamic computation graph and Pythonic debugging.
*   **Pandas**: Used for managing the dataset index CSV.
*   **Seaborn/Matplotlib**: Used for plotting the Confusion Matrix heatmap.

## 3.6 Evaluation Metrics
We moved beyond simple accuracy to ensure robust analysis:
1.  **Top-1 Accuracy**: "Did we guess the exact character?"
2.  **Top-3 Accuracy**: "Was the correct character in our top 3 guesses?" (Useful for ambiguous handwriting).
3.  **Confusion Matrix**: A heatmap visualization (238x238) that reveals exactly *which* characters are confusing the model (e.g., telling us if 'ሀ' is being misclassified as 'ሁ').

## 3.7 Workflow / Step-by-Step Procedure
1.  **Processing**: `process_data.py` reads the raw dataset folder structure, resizes images using Lanczos resampling (high quality), and saves them to `data/processed/`.
2.  **Training**: `train_model.py` loads the data. It runs a loop:
    *   Forward Pass (Predict).
    *   Calculate Loss (Error).
    *   Backward Pass (Calculate Gradients).
    *   Optimizer Step (Update Weights).
    *   *Validation Check*: Every epoch, it pauses to test on the Validation set. If the model improves, it saves a checkpoint.
3.  **Inference**: `predict_model.py` performs the Forward Pass on a single new image provided by the user.
4.  **Interactive Console**: `app_console.py` loads the model once and provides a continuous loop for rapid testing of multiple images.

## 3.8 Limitations (Methodology)
*   **Resolution**: Downscaling to 32x32 was chosen for speed, but some very complex Amharic characters (like 'ጮ') lose detail at this resolution.
*   **Class Imbalance**: Although mostly balanced, some rare characters have fewer samples than common ones, potentially biasing the model.

---

# CHAPTER FOUR: RESULTS AND DISCUSSION

## 4.1 Introduction
This chapter details the empirical results of the study. We compare the "Baseline" model (Phase 2) against the "Optimized" model (Phase 4) to demonstrate the efficacy of our methodology.

## 4.2 Experimental Setup
Experiments were conducted on a standard workstation:
*   **CPU**: Intel Processor (Training time ~10 mins/epoch).
*   **Hyperparameters**:
    *   Batch Size: 64
    *   Learning Rate: 1e-3 (decaying to 1e-4)
    *   Epochs: 20
    *   Split: 70% Train (26,000 imgs), 15% Val, 15% Test.

## 4.3 Results

### 4.3.1 Phase 2 Baseline Results
The `SimpleCNN` (2 layers, no batch norm) struggled significantly.
*   **Train Accuracy**: ~3.75%
*   **Validation Accuracy**: 16.21%
*   **Analysis**: The model was "underfitting". It simply did not have the complexity to model the data. It learned simple shapes but failed on specific characters.

### 4.3.2 Phase 4 Optimized Results
The `DeepAmharicNet` posted exceptional results.
*   **Final Validation Accuracy**: **87.79%**
*   **Final Test Accuracy**: **88.32%**

| Epoch | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **1** | Accuracy | 5.75% | Model is learning basic edges/contrast. |
| **5** | Accuracy | 78.2% | Model has learned the majority of distinct shapes. |
| **7** | Accuracy | 87.3% | Model is refining details (dots, dashes). |
| **20** | Accuracy | **88.3%** | Model has converged. |

## 4.4 Discussion
The improvement of **+72%** (from 16% to 88%) confirms our hypothesis:
1.  **Depth Matters**: The 4-layer depth was necessary to capture the hierarchical nature of Amharic characters.
2.  **Regularization works**: The high Test accuracy (matching Validation accuracy) proves that *Dropout* and *Augmentation* successfully prevented overfitting. The model generalizes well to new data.
3.  **Efficiency**: We achieved this with only 20 epochs, proving the efficiency of the *Adam* optimizer and *Batch Normalization*.

## 4.5 Performance Evaluation
Analysis of the Confusion Matrix revealed:
*   **Strengths**: The model is near-perfect (>95%) on visually distinct characters like 'መ' (Me) or 'ረ' (Re).
*   **Weaknesses**: The majority of errors come from the "1st vs 4th" orders (e.g., 'ሀ' vs 'ሃ') which are often written identically in handwriting. This suggests that 88% is approaching the "human baseline" for isolated characters without context.

## 4.6 Summary
The experimental results are conclusive. deep CNNs are highly effective for Amharic handwriting recognition. We successfully transformed a model that was guessing randomly into a robust classifier suitable for real-world prototyping.

---

# CHAPTER FIVE: CONCLUSION AND RECOMMENDATIONS

## 5.1 Conclusion
The "Ge-ezVision" project set out to build an accurate handwritten Amharic character recognition system. Through rigorous experimentation, we identified that a deep, residual-style CNN architecture (`DeepAmharicNet`) combined with aggressive data augmentation is the optimal strategy.
We moved the state-of-the-art for this specific dataset configuration to **88.32%**.
This project serves as a proof-of-concept that the digitization of Ethiopia's written heritage does not require massive supercomputers; it can be achieved with efficient, well-designed algorithms on standard hardware.

## 5.2 Recommendations
Based on our findings, we recommend the following for future work:
1.  **Mobile Application Development**: The trained model is small (<5MB). It should be wrapped in an Android App (using TensorFlow Lite or PyTorch Mobile) to allow users to scan text with their phones.
2.  **Contextual Models (LSTM/Transformers)**: Accuracy can be boosted by looking at *words* rather than isolated characters. A language model could correct "Hcllo" to "Hello". Future work should couple this CNN with an LSTM to use spell-checking logic.
3.  **Synthetic Data Generation**: To handle the 238 classes better, we could generate synthetic handwritten fonts to augment the training data further.
4.  **Deployment**: This system should be piloted in a specific domain, such as digitizing simple forms (e.g., ID card applications) to test it in the wild.

## 5.3 Limitations
*   **Segmentation Dependence**: The system currently relies on the user or another algorithm to crop the character perfectly. Real-world images have multiple characters on a page.
*   **Lighting Sensitivity**: The model was trained on scanned documents. It needs to be tested on camera photos which have varying lighting conditions.

---

# REFERENCES
1.  **Dataset**: Fetulhak, A. (2020). *Handwritten Amharic Character Dataset*. Kaggle.
2.  **CNNs**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.
3.  **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
4.  **Adam**: Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.
5.  **Amharic OCR**: Assabie, Y., & Bigun, J. (2011). "Ethiopic character recognition using direction field tensor." *Pattern Recognition*, 44(8), 1772-1784.

---
