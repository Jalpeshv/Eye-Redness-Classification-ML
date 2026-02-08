# 👁️ Eye Redness Classification System
## Go to space[🔗](https://huggingface.co/spaces/12erp0/Redness-Classification)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

> A comparative study and implementation of **Traditional Machine Learning** vs. **Deep Learning** for automated eye disease classification.

---

## 👥 Authors & Contributors

This project was conceptualized and developed by:

* 👨‍💻 **Viken Hadavani**
* 👨‍💻 **Harsh Dhandha**
* 👨‍💻 **Jalpesh Vasa**

---

## 📖 Project Overview

Eye redness is a common symptom associated with various ocular conditions ranging from benign irritations to serious pathologies. This project aims to automate the classification of eye redness into specific categories using computer vision.

We have implemented a **Multi-Model System** that allows users to switch between three distinct architectures to see how different AI approaches handle medical imagery:
1.  **Random Forest**: Demonstrating the power of manual feature engineering.
2.  **EfficientNet-B3**: Showcasing state-of-the-art transfer learning accuracy.
3.  **MobileNet-V3**: Highlighting efficiency for mobile/edge deployment.

---

## 🏥 Classification Classes

The system is trained to distinguish between **4 specific clinical conditions**:

| Class Label | Description | Visual Characteristics |
| :--- | :--- | :--- |
| **🟢 Normal** | Healthy eye | Clear sclera, no significant vascularization. |
| **🔴 Bulbar Conjunctival Redness** | Inflammation of the bulbar conjunctiva | Diffuse redness over the white part of the eye. |
| **🟠 Palpebral Conjunctiva Redness** | Inflammation of the inner eyelid | Redness visible on the inner lining of the eyelids. |
| **🩸 Sub Conjunctival Hemorrhage** | Broken blood vessel | A bright, localized patch of blood on the sclera. |

---

## 🧠 Model Architectures & Technical Details

### 1. 🌲 Random Forest (Traditional ML)
* **Philosophy**: Uses "handcrafted" features based on image processing theory.
* **Preprocessing**: Images resized to `128x128` pixels.
* **Feature Extraction Pipeline**:
    * 🎨 **Color Histograms**: Captures distribution of Red, Green, and Blue intensities.
    * 🧶 **Texture Analysis (GLCM)**: Computes Gray-Level Co-occurrence Matrix features including *Contrast*, *Dissimilarity*, *Homogeneity*, *Energy*, and *Correlation*.
    * ✏️ **Edge Detection**: Uses Canny edge detection to calculate edge density.
    * 📊 **Statistical Moments**: Mean, Standard Deviation, Min, and Max of pixel values.
* **Classifier**: Scikit-Learn Random Forest Classifier.

### 2. ⚡ EfficientNet-B3 (Deep Learning)
* **Philosophy**: High-accuracy Transfer Learning.
* **Architecture**: EfficientNet-B3 pretrained on ImageNet.
* **Modifications**: The final fully connected layer is replaced to output 4 classes.
* **Input Spec**: `300x300` pixels.
* **Normalization**: ImageNet standards (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 3. 📱 MobileNet-V3 (Lightweight DL)
* **Philosophy**: Efficiency and Speed.
* **Architecture**: MobileNet-V3 Large.
* **Custom Head**:
    * Linear (Input -> 512) -> ReLU -> Dropout (0.3)
    * Linear (512 -> 256) -> ReLU -> Dropout (0.2)
    * Linear (256 -> 4 Output)
* **Input Spec**: `224x224` pixels.
* **Key Tech**: Uses Depthwise Separable Convolutions to reduce parameter count.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* Git LFS (Large File Storage) - *Crucial for downloading model files*

### Step 1: Clone the Repository
```bash
git clone https://github.com/jalpeshv/eye-redness-classification-ml.git

cd eye-redness-classification-ml
```

### Step 2: Initialize Git LFS
This project contains large model files (.pth, .joblib). You must pull them correctly:

```bash
git lfs install
git lfs pull
```
### Step 3: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```
### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### 🚀 Usage Guide

1. Start the Application
Run the main Python script to launch the Gradio web server:

```bash
python app.py
```

2. Access the Interface
Once the server starts, you will see a local URL in the terminal (typically http://127.0.0.1:7860). Open this link in your web browser.

3. Using the Tool
Select Model: Use the dropdown menu to choose between Random Forest, EfficientNet-B3, or MobileNet-V3.

4. Upload Image: Drag and drop an eye image into the upload box.

5. Analyze: Click the "Classify Image" button.

6. View Results: The predicted class and confidence scores will appear on the right.

### 📂 Project Structure
```Plaintext
eye-redness-classification-ml/
├── Models/                             # 📦 Model Artifacts
│   ├── efficientnet_b3_best_model.pth  # PyTorch State Dict
│   ├── mobilenet_best_model.pth        # PyTorch State Dict
│   ├── random_forest_model.joblib      # Sklearn Model
│   ├── feature_params.joblib           # Scaler/Selector params
│   └── feature_selector.joblib         # Feature selection logic
├── .gitattributes                      # ⚙️ Git LFS Config
├── app.py                              # 🚀 Main Application Logic
├── requirements.txt                    # 📋 Python Dependencies
└── README.md                           # 📄 Documentation
```
### 🛠️ Technology Stack

* Interface: Gradio

* Deep Learning: PyTorch, Torchvision

* Machine Learning: Scikit-learn, Joblib

* Image Processing: OpenCV, Pillow, Scikit-image

* Data Manipulation: NumPy, Pandas

### ⚠️ Medical Disclaimer
IMPORTANT NOTICE

This software is a prototype developed for educational and research purposes only.

❌ It is NOT a certified medical device.

❌ It is NOT intended for clinical diagnosis, treatment, or decision-making.

❌ Predictions should NEVER replace professional medical advice.

Always consult a qualified ophthalmologist or healthcare provider for any eye health concerns.
