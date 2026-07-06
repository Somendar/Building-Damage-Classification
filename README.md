# 🛰️ AI-Based Post-Disaster Building Damage Classification using Dual EfficientNet-B0

## 📌 Project Overview

Natural disasters such as earthquakes, floods, hurricanes, and wildfires cause severe damage to buildings and infrastructure. Manual damage assessment is often slow, expensive, and difficult immediately after a disaster.

This project presents a Deep Learning based Building Damage Classification System that automatically predicts the structural damage level of buildings using paired **Pre-Disaster** and **Post-Disaster** satellite images from the **xBD (xView2)** dataset.

The proposed framework extracts individual buildings from large satellite images, preprocesses them, and classifies each building into one of four damage categories using a **Dual EfficientNet-B0** architecture.

Finally, the trained model is deployed as an interactive **Streamlit Web Application** for real-time prediction.

---

# 🎯 Problem Statement

After a natural disaster, emergency response teams must quickly identify damaged buildings to prioritize rescue operations and allocate resources effectively.

Traditional field inspections are:

- Time-consuming
- Labour intensive
- Expensive
- Difficult in inaccessible areas

This project automates the damage assessment process using Artificial Intelligence and Satellite Imagery.

---

# 🎯 Objectives

The objectives of this project are:

- Develop an automated building damage classification system.
- Extract individual buildings from satellite imagery.
- Compare multiple Deep Learning models.
- Propose an improved Dual EfficientNet-B0 architecture.
- Deploy the trained model using Streamlit.

---

# 🛰 Dataset

## Dataset Used

**xBD (xView2 Building Damage Assessment Dataset)**

The xBD dataset is one of the largest publicly available satellite image datasets for post-disaster damage assessment.

It contains:

- High-resolution satellite images
- Pre-disaster images
- Post-disaster images
- Building annotations
- Damage labels

---

## Damage Classes

The model predicts one of the following four classes:

- No Damage
- Minor Damage
- Major Damage
- Destroyed

---

## Final Dataset

| Property | Value |
|-----------|--------|
| Dataset | xBD |
| Total Samples | 16,956 |
| Classes | 4 |
| Image Size | 224 × 224 |
| Input | Pre + Post Building Images |

---

# 🏗 Project Methodology

The complete workflow consists of the following stages.

```
xBD Dataset
        │
        ▼
Dataset Inspection
        │
        ▼
Building Extraction
        │
        ▼
Building Cropping
        │
        ▼
Image Preprocessing
        │
        ▼
Balanced Dataset Creation
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Dual EfficientNet-B0
        │
        ▼
Model Evaluation
        │
        ▼
Streamlit Deployment
```

---

# 🔍 Data Preprocessing

The original xBD satellite images contain multiple buildings and background objects.

Each building was extracted individually using the provided building annotations.

Preprocessing steps include:

- Building Extraction
- Bounding Box Cropping
- Image Resizing (224 × 224)
- RGB Conversion
- Tensor Transformation
- Dataset Balancing
- Train / Validation / Test Split

---

# ⚙ Model Development

Three different Deep Learning models were developed and compared.

---

## 1. Baseline CNN

A simple Convolutional Neural Network was implemented to establish the initial benchmark.

Performance:

- Accuracy below 60%

---

## 2. EfficientNet-B0

Transfer Learning was applied using EfficientNet-B0 pretrained on ImageNet.

Advantages:

- Better feature extraction
- Faster convergence
- Improved accuracy

Performance:

Approximately 75%

---

## 3. Proposed Dual EfficientNet-B0

The final proposed architecture consists of two EfficientNet-B0 feature extractors.

One processes:

- Pre-disaster image

The second processes:

- Post-disaster image

The extracted feature vectors are concatenated and passed through fully connected layers for final classification.

---

# 🏛 Proposed Architecture

```
                 PRE IMAGE
                     │
             EfficientNet-B0
                     │
               Feature Vector
                     │
                     │
                     ▼
               Feature Fusion
                     ▲
                     │
               Feature Vector
                     │
             EfficientNet-B0
                     │
                POST IMAGE
                     │
                     ▼
          Fully Connected Layer
                     │
                     ▼
          Damage Classification
```

---

# 📈 Training Configuration

| Parameter | Value |
|------------|--------|
| Framework | PyTorch |
| Backbone | EfficientNet-B0 |
| Optimizer | Adam |
| Loss Function | Cross Entropy |
| Batch Size | 32 |
| Input Size | 224 × 224 |
| Classes | 4 |

---

# 📊 Results

## Model Comparison

| Model | Accuracy |
|---------|-----------:|
| Baseline CNN | <60% |
| EfficientNet-B0 | ~75% |
| Proposed Dual EfficientNet-B0 | **78.1%** |

---

## Classification Performance

| Metric | Score |
|---------|---------:|
| Accuracy | 78.1% |
| Precision | 79% |
| Recall | 78% |
| F1 Score | 78% |

---

# 📉 Confusion Matrix

(Add Confusion Matrix Screenshot Here)

Key Observations

- Excellent performance on Destroyed buildings.
- Good classification of No Damage buildings.
- Minor Damage and Major Damage occasionally overlap.
- Overall balanced classification.

---

# 🌐 Streamlit Deployment

The trained model was deployed using Streamlit Community Cloud.

Application Features

- Upload PRE image
- Upload POST image
- Predict damage class
- Confidence score
- Real-time inference

Live Demo

https://building-damage-classification-iazexu9jafbeak7swaqffa.streamlit.app/



# 💻 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Building-Damage-Classification.git
```

Open project

```bash
cd Building-Damage-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

# 🛠 Technologies Used

Programming Language

- Python

Libraries

- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- PIL
- Matplotlib
- Scikit-Learn
- Streamlit

Development Environment

- Google Colab
- VS Code
- GitHub
- Streamlit Cloud

---

# 🚀 Future Improvements

- Train on complete xBD dataset
- Vision Transformer (ViT) implementation
- Semantic Segmentation before classification
- Cloud-based disaster monitoring
- Mobile application deployment
- Real-time satellite image integration

---

# 👨‍💻 Authors

Somendar Kumar Das

Prudhvi

Sunil

MSc Data Science

University of Europe for Applied Sciences

---

# 📚 References

1. Gupta R. et al., xBD: A Dataset for Assessing Building Damage from Satellite Imagery.
2. Tan M. & Le Q., EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
3. PyTorch Documentation.
4. Streamlit Documentation.
5. ImageNet Dataset.
