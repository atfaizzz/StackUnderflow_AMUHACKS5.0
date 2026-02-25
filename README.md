# 🎓 Student Skill Gap Analyzer & Career Recommendation System (AI/ML)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

The **Student Skill Gap Analyzer & Career Recommendation System** is an AI/ML-powered application that:

- Analyzes a student's current skill set  
- Predicts the most suitable tech career role  
- Identifies skill gaps for the predicted career  
- Provides actionable insights for upskilling  

This project demonstrates a complete **machine learning lifecycle**, including:

- Data preprocessing  
- Feature engineering (TF-IDF)  
- Model training & evaluation  
- Model serialization  
- Inference & recommendation logic  

---

## Problem Statement

Many students struggle with:

- Choosing the right career path  
- Understanding industry-required skills  
- Identifying gaps in their current skill set  
- Planning structured upskilling  

This system solves the problem using **Natural Language Processing (NLP)** and **Machine Learning classification techniques**.

---

## 🏗️ Project Architecture

```
User Input (Skills)
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Trained ML Model (Random Forest)
        ↓
Career Prediction
        ↓
Skill Gap Analysis
        ↓
Recommendation Report
```

---

## 📁 Repository Structure

```
Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-/

├── dataset/
│   └── skills_dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_demo_prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   └── predict.py
│
├── models/
│   ├── career_prediction_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── results/
│   ├── metrics.txt
│   └── feature_info.txt
│
├── README.md
└── .gitignore
```

---

## Dataset Description

The dataset maps **technical skills → job roles**.

### Structure

| Column     | Description |
|------------|------------|
| `skills`   | Space-separated technical skills |
| `job_role` | Target career label |

### Example

```
skills,job_role
python numpy pandas,data analyst
python tensorflow deep learning,ai engineer
java spring sql,backend developer
html css javascript,frontend developer
```

### Coverage

- ~40+ skill-role mappings  
- 11+ technical career roles  
- Focused on software & AI domains  

---

## Machine Learning Pipeline

### 1️⃣ Text Preprocessing

Implemented in `src/preprocess.py`:

- Convert text to lowercase  
- Remove special characters  
- Normalize whitespace  
- Clean skill strings  

### 2️⃣ Feature Engineering

- TF-IDF Vectorizer  
- N-grams: (1,2)  
- Converts skill text into numerical vectors  

### 3️⃣ Model Training

Models evaluated:

| Model | Accuracy |
|--------|----------|
| Logistic Regression | ~37.5% |
| Random Forest (Selected) | ~62.5% |

Random Forest Classifier was selected as the final model due to superior performance.

### 4️⃣ Model Artifacts

Saved inside `/models`:

- `tfidf_vectorizer.pkl`
- `career_prediction_model.pkl`
- `label_encoder.pkl`

---

## Core Functionalities

### Career Prediction

Input:

```
python machine learning tensorflow
```

Output:

```
Predicted Career: AI ENGINEER
Confidence: 72.38%
```

### Skill Gap Analysis

The system calculates:

- Matched skills  
- Missing skills  
- Skill coverage percentage  

Example:

```
Skills Coverage: 30.8% (4/13 skills)

Matched Skills:
• python
• deep
• learning
• tensorflow

Missing Skills:
• keras
• pytorch
• computer vision
• image processing
• neural networks
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sudharsanv06/Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-.git
cd Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

---

## ▶️ How to Run

### Option 1: Jupyter Notebook

```bash
jupyter notebook
```

Open:
```
notebooks/04_demo_prediction.ipynb
```

### Option 2: Run Python Script

```bash
python src/predict.py
```

### Option 3: Use as Python Module

```python
from src.predict import CareerPredictor

predictor = CareerPredictor()
predictor.display_recommendation("python deep learning tensorflow")
```

---

## Evaluation & Insights

- Demonstrates end-to-end ML workflow  
- Lightweight NLP-based classification  
- Interpretable results  
- Actionable recommendations  
- Ideal academic AI/ML portfolio project  

---

## Future Enhancements

- Expand dataset (500+ mappings)  
- Deploy as Web App (Streamlit/Flask)  
- Add industry trend analysis  
- Recommend structured learning paths  
- Integrate real-time job market APIs  
- Upgrade to transformer-based embeddings (e.g., Sentence-BERT)  

---

## Use Cases

- Career guidance platforms  
- College placement cells  
- EdTech products  
- Personal skill assessment tools  
- AI/ML portfolio demonstration  

---

## Tech Stack

- Python  
- Scikit-Learn  
- Pandas  
- NumPy  
- TF-IDF  
- Random Forest  
- Joblib  
- Jupyter Notebook  

---

## Key Takeaways

- Even small datasets can produce meaningful ML solutions  
- TF-IDF works well for skill-text classification  
- Random Forest handles sparse text features effectively  
- Simple ML systems can deliver high practical value  

---

## 📄 License

This project is open-source under the MIT License.

---
