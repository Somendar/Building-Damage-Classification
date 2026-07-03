# 🛰️ AI Building Damage Classification

## Overview

This project is a Deep Learning based Building Damage Assessment system developed using the xBD Disaster Dataset.

The model compares Pre-Disaster and Post-Disaster satellite building images and predicts the level of structural damage.

---

## Features

- Dual Image Classification
- EfficientNet-B0 Backbone
- Automatic Building Crop Pipeline
- Streamlit Web Application
- Interactive Prediction Dashboard
- Probability Visualization

---

## Dataset

Dataset: xBD (xView2 Building Damage Assessment)

Classes

- No Damage
- Minor Damage
- Major Damage
- Destroyed

Training Samples

16,956

---

## Model

Architecture

Dual EfficientNet-B0

Input

- Pre Disaster Building Image
- Post Disaster Building Image

Output

- Damage Category

---

## Performance

| Metric | Score |
|---------|---------|
| Accuracy | 78.1% |
| Precision | 79% |
| Recall | 78% |

---

## Project Structure

```
Building_Damage_Grading/

│── streamlit_app.py
│── dual_efficientnet_final.pth
│── requirements.txt
│── README.md

│── sample_data/

│── screenshots/

│── notebooks/

│── train/

```

---

## Installation

```bash
git clone YOUR_REPOSITORY

cd Building_Damage_Grading

pip install -r requirements.txt

streamlit run streamlit_app.py
```

---

## Technologies

- Python
- PyTorch
- Streamlit
- OpenCV
- Pandas
- NumPy

---

User Interface :

https://building-damage-classification-iazexu9jafbeak7swaqffa.streamlit.app/

## Author

Somendar Kumar Das , prudhvi , sunil

MSc Data Science

University of Europe for Applied Sciences
