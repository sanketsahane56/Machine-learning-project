# 🚨 AI-Based Disaster Tweet Classification System

An advanced Machine Learning and Natural Language Processing (NLP) web application for automated detection, classification, and analysis of emergency disaster tweets from social media.

---

## 📌 Project Overview

During natural disasters (e.g. earthquakes, floods, hurricanes, wildfires) and emergency situations, millions of tweets are published on social media platforms (such as Twitter/X). Some tweets communicate genuine emergencies, calls for help, and critical situation updates, while others use informal or metaphorical language (e.g., *"This song is pure fire"*, *"I am drowned in work"*).

This application leverages Natural Language Processing (NLP) and Machine Learning (ML) algorithms to accurately classify tweets into **Disaster (1)** and **Non-Disaster (0)** categories, enabling real-time situational awareness and rapid emergency response.

---

## 📁 Directory Structure

```text
ML project/
├── app.py                     # Main Streamlit Web Application (11 Modules Dashboard)
├── run_app.py                 # Runner script with compatibility patches
├── requirements.txt           # Python dependencies list
├── README.md                  # Comprehensive Project Documentation
├── train.csv                  # Labeled training dataset (7,613 tweets)
├── test.csv                   # Unlabeled test dataset (3,263 tweets)
├── sample_submission.csv      # Kaggle competition submission format
└── src/                       # Modular Python codebase
    ├── __init__.py
    ├── data_loader.py         # Dataset loading, validation, and schema profiling
    ├── preprocessor.py        # Text sanitization, regex, stopwords, and tokenization
    ├── visualizer.py          # Interactive Plotly visualizations and metrics
    └── model_trainer.py       # ML training, evaluation, and batch inference pipeline
```

---

## 📊 Authentic Dataset Statistics

| Feature / Metric | Train Dataset (`train.csv`) | Test Dataset (`test.csv`) |
| :--- | :--- | :--- |
| **Total Records (Rows)** | 7,613 | 3,263 |
| **Columns** | 5 (`id`, `keyword`, `location`, `text`, `target`) | 4 (`id`, `keyword`, `location`, `text`) |
| **Disaster Tweets (Target = 1)** | 3,271 (42.97%) | - (Unlabeled) |
| **Non-Disaster Tweets (Target = 0)**| 4,342 (57.03%) | - (Unlabeled) |
| **Missing Values** | `location`: 2,533 (33.3%), `keyword`: 61 (0.8%) | `location`: 1,105, `keyword`: 26 |
| **Missing Text (`text`)** | 0 (100% complete) | 0 (100% complete) |
| **Duplicate Texts** | 110 tweets | 20 tweets |

---

## 🚀 Key Modules & Capabilities (11 Modules)

1. **📊 Executive Dashboard:** System KPIs, quick live inference widget, and high-level workflow architecture.
2. **📂 Dataset Explorer:** In-depth schema inspection of `train.csv`, `test.csv`, missing value ratios, and interactive keyword search.
3. **🧹 Data Cleaning & Preprocessing:** Transparent handling of missing values, duplicate analysis, and isolated in-memory transformations.
4. **📈 Exploratory Data Analysis (EDA):** Target distribution bar/donut charts, character and word length histograms, and predictive keyword rankings.
5. **🔤 NLP Preprocessing Sandbox:** Interactive step-by-step text sanitization (HTML decode, lowercase, URL strip, user mentions strip, hashtag normalization, stopwords filtering).
6. **⚙️ Feature Engineering:** TF-IDF N-Gram matrix extraction, vocabulary search, and top predictive coefficients.
7. **🧠 Model Training Pipeline:** Zero-data-leakage training across Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest.
8. **📉 Model Evaluation & Diagnostics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix Heatmap, ROC Curve, and error analysis (False Positives / False Negatives).
9. **🔮 Live Tweet Prediction:** Real-time tweet classification with confidence gauge meter and key contributing token explanation.
10. **📑 Test Dataset Batch Inference:** Automated classification across all 3,263 test tweets with multi-criteria filters.
11. **💾 Export & Download Results:** Instant download of Kaggle-compliant `submission.csv`, detailed analytical CSV, and summary reports.

---

## 🛠️ How to Run the Application (Windows)

### Step 1: Open Terminal / PowerShell in Project Directory
```powershell
cd "c:\Users\sanke\OneDrive\Desktop\ML project"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Launch the Streamlit Web Application
```powershell
python run_app.py
```

After running the command, open your browser and navigate to:
👉 **`http://localhost:8501`**

---

## 🏆 Model Benchmark & Validation Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Default)** | **81.75%** | **83.94%** | **71.10%** | **76.99%** | **0.865** |
| **Multinomial Naive Bayes** | 81.16% | 85.63% | 67.43% | 75.45% | 0.858 |
| **Complement Naive Bayes** | 80.76% | 80.23% | 73.24% | 76.58% | 0.856 |
| **Linear SVM** | 78.92% | 77.34% | 72.02% | 74.58% | 0.849 |
| **Random Forest** | 80.63% | 81.77% | 70.64% | 75.80% | 0.852 |

---

## ⚠️ Disclaimer
This application is developed for educational, academic, and research purposes using NLP and Machine Learning techniques. Predictions indicate statistical linguistic patterns and do not serve as official verification of an actual emergency incident. Always refer to certified disaster management agencies for emergency response.
