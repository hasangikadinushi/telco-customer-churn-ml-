# Telco Customer Churn Binary Classification
Machine Learning Coursework - Individual Assignment

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline to predict customer churn in telecommunications using the Kaggle **Telco Customer Churn** dataset. 

The project covers:
1. **Task 1: Exploratory Data Analysis (EDA)** – Data quality assessment, class imbalance detection, and bivariate visualizations.
2. **Task 2: Machine Learning Pipeline** – Data preprocessing, feature scaling, model training, and hyperparameter tuning for **K-Nearest Neighbors (KNN)**, **Logistic Regression (LR)**, and **Random Forest (RF)** across two experimental settings (Baseline vs. 5-Fold Cross-Validated Tuned).
3. **Task 3: AI Ethics & Post-Deployment Engineering** – Demographic parity audits (Fairness across age and gender), explainability via feature importance, and MLOps strategies (drift monitoring and continuous retraining).

---

## 📂 Project Structure
```
├── Telco-Customer-Churn.csv          # Raw dataset (7,043 rows, 21 columns)
├── telco_churn_classification.py     # Main executable Python script
├── model_comparison_results.csv      # Exported benchmark evaluation metrics
├── requirements.txt                  # Python package dependencies
├── figures/                          # Exported high-resolution charts for coursework report
│   ├── fig1_churn_distribution.png
│   ├── fig2_contract_and_tenure_churn.png
│   ├── fig3_charges_and_internetservice.png
│   ├── fig4_confusion_matrices.png
│   ├── fig5_roc_curves.png
│   ├── fig6_feature_importance.png
│   └── fig7_ethics_fairness_audit.png
└── README.md                         # Project documentation
```

---

## ⚙️ Installation & Usage

1. **Clone this repository**:
   ```bash
   git clone <YOUR_GIT_URL>
   cd "ML coursework"
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the entire pipeline**:
   ```bash
   python telco_churn_classification.py
   ```

---

## 📊 Summary of Experimental Results

| Model | Setting | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN** | Baseline | 74.73% | 52.53% | 50.00% | 0.5123 | 0.7718 |
| **Logistic Regression** | Baseline | 80.70% | 65.84% | 56.68% | 0.6092 | 0.8418 |
| **Random Forest** | Baseline | 78.50% | 61.79% | 49.73% | 0.5511 | 0.8253 |
| **KNN** | Tuned ($k=21$, Manhattan) | 77.08% | 57.22% | 54.01% | 0.5557 | 0.8154 |
| **Logistic Regression** | Tuned ($C=10.0$, lbfgs) | 80.27% | 65.09% | 55.35% | 0.5983 | 0.8410 |
| **Random Forest** | Tuned (`balanced`, depth=6) | 74.10% | 50.75% | **81.55%** | **0.6256** | **0.8422** |

*Key finding: Tuning Random Forest with `class_weight='balanced'` boosted recall on churners from 49.7% to 81.6%, achieving the highest ROC-AUC (0.8422) and best F1-Score (0.6256).*
