# 🎓 Student Performance Prediction

> A machine learning project to predict student final grades using demographic, social, and academic features.
> Built for portfolio purposes — demonstrating end-to-end ML workflow using Python.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![pandas](https://img.shields.io/badge/pandas-2.0-green?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project applies supervised machine learning to predict student academic performance (final grade `G3`) using the **UCI Student Performance Dataset**. The dataset contains student information from two Portuguese secondary schools, including demographic data, family background, study habits, and social factors.

The project demonstrates a complete ML pipeline:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Training (3+ algorithms)
- Model Evaluation & Comparison
- Visualization & Reporting

---

## 🎯 Problem Statement

**Can we predict a student's final grade based on their background, habits, and academic history?**

- **Task Type:** Regression (predict exact grade) + Classification (Pass / Fail)
- **Target Variable:** `G3` — final grade on a 0–20 scale
- **Classes (for classification):** Pass (G3 ≥ 10) | Fail (G3 < 10)

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── raw/                    # Original, unmodified dataset files
│   │   └── student-mat.csv     # UCI Student Performance (Math)
│   └── processed/              # Cleaned and transformed data
│       └── student_clean.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning & feature engineering
│   └── 03_modeling.ipynb       # Model training, evaluation & comparison
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Functions to load and validate data
│   ├── preprocessing.py        # Preprocessing pipeline (encoding, scaling)
│   ├── train.py                # Model training scripts
│   ├── evaluate.py             # Evaluation metrics and reporting
│   └── visualize.py            # Reusable plotting functions
│
├── models/
│   ├── random_forest.pkl       # Saved trained Random Forest model
│   ├── logistic_regression.pkl # Saved Logistic Regression model
│   └── gradient_boosting.pkl   # Saved Gradient Boosting model
│
├── reports/
│   ├── figures/                # All saved plots and charts
│   │   ├── grade_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   └── model_results.md        # Summary of model performance metrics
│
├── tests/
│   └── test_preprocessing.py   # Unit tests for preprocessing functions
│
├── .gitignore                  # Files and folders excluded from Git
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## 📊 Dataset

| Property       | Details                                      |
|----------------|----------------------------------------------|
| **Source**     | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/student+performance) |
| **Records**    | 395 students                                 |
| **Features**   | 33 (demographic, social, academic)           |
| **Target**     | `G3` — Final grade (0–20)                   |
| **License**    | Public / Open Access                         |

**Key Features Used:**
- `studytime` — Weekly study time
- `failures` — Number of past class failures
- `absences` — Number of school absences
- `Medu` / `Fedu` — Mother's / Father's education level
- `G1`, `G2` — First and second period grades
- `internet`, `romantic`, `activities` — Lifestyle factors

---

##  Models Used

| Model                    | Type           | Task           |
|--------------------------|----------------|----------------|
| Logistic Regression      | Linear         | Classification |
| Random Forest            | Ensemble       | Both           |
| Gradient Boosting (GBM)  | Ensemble       | Both           |

---

## 📈 Results Summary

| Model               | Accuracy | F1-Score | RMSE  |
|---------------------|----------|----------|-------|
| Logistic Regression | ~XX%     | ~X.XX    | —     |
| Random Forest       | ~XX%     | ~X.XX    | ~X.XX |
| Gradient Boosting   | ~XX%     | ~X.XX    | ~X.XX |

> Results will be updated after full training. See `reports/model_results.md` for details.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download `student-mat.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/student+performance) and place it in `data/raw/`.

### 5. Run the Notebooks
```bash
jupyter notebook
```
Open notebooks in order: `01_EDA` → `02_preprocessing` → `03_modeling`

---

## 🔍 Key Insights (from EDA)

- Students with **previous failures** have significantly lower final grades
- **Study time** shows a moderate positive correlation with performance
- **First and second period grades (G1, G2)** are the strongest predictors of G3
- Students with **internet access at home** tend to perform slightly better
- **Alcohol consumption** (workday and weekend) negatively correlates with grades

---

## 🛠️ Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| Python 3.9+     | Core programming language        |
| pandas          | Data manipulation                |
| numpy           | Numerical computations           |
| matplotlib      | Base plotting library            |
| seaborn         | Statistical visualizations       |
| scikit-learn    | ML models and evaluation         |
| Jupyter         | Interactive notebooks            |
| joblib          | Model serialization              |

---

##  Testing

```bash
python -m pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##  Author

**[Fatma Najm Aldeen]**
- GitHub: [@Fatma-Najm](https://github.com/Fatma-Najm)
- LinkedIn: [Fatma Najm](https://www.linkedin.com/in/fatma-najm-5065bb176/)
- Email: fatmanagim@fcis.bsu.edu.eg

---

##  Acknowledgements

- Dataset: [Paulo Cortez, University of Minho](https://archive.ics.uci.edu/ml/datasets/student+performance)
- Inspired by real-world educational analytics use cases