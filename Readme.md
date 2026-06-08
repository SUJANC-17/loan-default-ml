# 🏦 Loan Default Prediction — End-to-End ML Pipeline

> Predicting loan defaulters using a full machine learning pipeline — from raw data to a deployed REST API.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-green)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Problem Statement

Financial institutions face significant losses due to loan defaults. This project builds a classification model to predict whether a borrower will default, with a focus on **maximizing recall for defaulters** — because missing a true defaulter is more costly than a false alarm.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 255,347 |
| Target | `loan_status` (Default / No Default) |
| Features | Credit score, income, loan amount, DTI ratio, employment length, etc. |

---

## 🔍 Project Structure

```
loan-default-prediction/
│
├── data/                    # Raw and processed datasets
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb    # Model training and comparison
├── app/
│   ├── app.py               # Flask REST API
│   └── model.pkl            # Serialized model
├── requirements.txt
└── README.md
```

---

## 🧪 Workflow

### 1. Exploratory Data Analysis
- Distribution of loan status (class imbalance analysis)
- Correlation heatmap and feature importance ranking
- Missing value treatment and outlier detection
- Key findings: DTI ratio, credit score, and loan-to-income ratio are the strongest predictors

### 2. Preprocessing
- Label encoding and one-hot encoding for categorical variables
- Feature scaling with `StandardScaler`
- Train/test split (80/20) with stratification

### 3. Model Comparison

| Model | Accuracy | Precision | Recall (Defaulters) | F1 Score |
|---|---|---|---|---|
| Logistic Regression | ~89% | ~85% | **Highest** | ~87% |
| Random Forest | ~91% | ~88% | Moderate | ~89% |
| XGBoost | ~92% | ~90% | Lower | ~91% |

> ✅ **Logistic Regression was selected** — despite XGBoost having higher overall accuracy, Logistic Regression achieved the best **recall for defaulters**, which is the business-critical metric in loan risk assessment.

### 4. Deployment
- Serialized model with `joblib`
- Flask REST API with `/predict` endpoint
- Accepts JSON input, returns `{ "prediction": "Default" / "No Default", "probability": float }`

---

## 🚀 API Usage

**Start the server:**
```bash
python app/app.py
```

**Sample request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"credit_score": 620, "income": 45000, "loan_amount": 15000, "dti_ratio": 0.35, "employment_length": 3}'
```

**Sample response:**
```json
{
  "prediction": "Default",
  "probability": 0.73
}
```

---

## 🛠️ Tech Stack

- **Data Analysis:** Python, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** Scikit-learn, XGBoost
- **Deployment:** Flask
- **Environment:** Jupyter Notebook, VS Code

---

## 📦 Installation

```bash
git clone https://github.com/SUJANC-17/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
```

---

## 📈 Key Takeaways

- Accuracy alone is a poor metric for imbalanced classification — **recall matters more** in financial risk
- Logistic Regression, despite being simpler, can outperform complex models on the metric that matters
- End-to-end deployment bridges the gap between a notebook experiment and a usable product

---

## 👤 Author

**Sujan C** — B.E. CSE, SKCET Coimbatore (2024–2028)

[![GitHub](https://img.shields.io/badge/GitHub-SUJANC--17-black?logo=github)](https://github.com/SUJANC-17)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sujan--c-blue?logo=linkedin)](https://linkedin.com/in/sujan-c-195834377)