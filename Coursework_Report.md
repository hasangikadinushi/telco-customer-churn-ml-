# Predicting Customer Churn in Telecommunications Using Supervised Machine Learning
**Module**: Machine Learning — Individual Coursework (Part 01)  
**Author**: Dinushi Hasangika  
**Project Git Repository**: [https://github.com/hasangikadinushi/telco-customer-churn-ml-](https://github.com/hasangikadinushi/telco-customer-churn-ml-)  

---

## Executive Summary
Customer retention is a paramount strategic objective in subscription telecommunications, where customer acquisition costs vastly exceed retention investments. This report presents an end-to-end supervised machine learning investigation predicting customer churn using the IBM Telco Customer Churn corpus ($N = 7,043$). Following exploratory data analysis (EDA) and data preprocessing, three distinct classifiers—K-Nearest Neighbors (KNN), Logistic Regression (LR), and Random Forest (RF)—were evaluated under baseline and 5-fold cross-validated hyperparameter-tuned settings. Tuned Random Forest with cost-sensitive class balancing demonstrated superior performance, achieving a high recall of 81.55%, an F1-score of 0.6256, and an ROC-AUC of 0.8422. Key churn catalysts include short tenure, month-to-month contracts, and high monthly charges. The solution is complemented by an AI ethics audit and a post-deployment MLOps framework.

---

## 1. Introduction & Problem Formulation
In saturated telecommunication markets, customer attrition ("churn") poses substantial revenue risk. Acquiring replacement subscribers typically costs five to twenty-five times more than retaining existing accounts. Accurate predictive modeling enables carriers to proactively deploy targeted retention incentives prior to contract termination.

Formally, churn prediction is formulated as a supervised binary classification task. Given a training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i \in \mathbb{R}^D$ denotes a $D$-dimensional feature vector of demographics, contracted services, and billing statistics, the objective is to learn a mapping $f: \mathbb{R}^D \to \{0, 1\}$. The target variable $y_i \in \{0, 1\}$ represents churn status ($y_i = 1$ for churned, $y_i = 0$ for retained). Because churn is economically asymmetric—a missed churner (False Negative) causes irreversible recurring revenue loss, whereas an unnecessary retention concession (False Positive) carries minor overhead—model evaluation must prioritize recall, F1-score, and ROC-AUC over naive accuracy.

---

## 2. Corpus Preparation & Exploratory Data Analysis (Task 1)

### 2.1 Dataset Composition & Data Quality Remediation
The Telco Customer Churn dataset comprises $N = 7,043$ instances and 21 attributes spanning demographics, subscribed services, and account billing parameters. During data auditing, `TotalCharges` was found stored as an object data type due to 11 blank whitespace entries (`" "`). Cross-referencing verified that all 11 records had `tenure = 0`, representing newly enrolled customers who had not yet received a monthly bill. To prevent data loss, these entries were coerced to numeric floats and imputed with $0.0$. The non-predictive identifier `customerID` was removed to avoid overfitting.

### 2.2 Class Distribution & Imbalance
The target variable exhibits notable class imbalance: 5,174 customers (73.46%) were retained, while 1,869 customers (26.54%) churned (Figure 1). 

![Figure 1: Target Class Distribution](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig1_churn_distribution.png)

This imbalance renders classification accuracy deceptive: a trivial majority-class classifier predicting "No Churn" yields 73.46% accuracy while identifying zero churners. Consequently, stratified sampling and threshold-independent discrimination metrics were mandated.

### 2.3 Exploratory Bivariate Insights
Visual exploration uncovered several core drivers of churn:

![Figure 2: Impact of Contract Structure and Customer Tenure on Churn](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig2_contract_and_tenure_churn.png)

* **Contract Duration**: Contractual commitment displayed the strongest association with churn (Figure 2, Left). Customers on Month-to-month contracts experienced an acute churn rate of 42.71% (1,655 / 3,875), compared to 11.27% for One-year and 2.83% for Two-year contracts.
* **Customer Tenure**: Kernel Density Estimation (Figure 2, Right) demonstrated that churn density is heavily concentrated in the first 1–6 months of tenure. Risk decreases monotonically as customers surpass 24 months.

![Figure 3: Financial and Service-Tier Drivers of Customer Churn](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig3_charges_and_internetservice.png)

* **Financial Burden**: Churned customers exhibited significantly higher median monthly charges (\$79.65) than retained subscribers (\$64.43) (Figure 3, Left).
* **Internet Architecture**: Fiber Optic subscribers experienced disproportionately higher attrition (41.89%) relative to DSL (18.96%) and non-internet users (7.40%) (Figure 3, Right), reflecting customer price sensitivity or service quality dissatisfaction.
* **Support Add-ons**: Customers lacking Online Security and Tech Support churned at rates exceeding 41%, whereas customers with active support churned at under 16%.

---

## 3. Solution Methodology & Preprocessing Pipeline (Task 2)

### 3.1 Preprocessing Pipeline
1. **Target Encoding**: `Churn` was mapped to binary values (`No` $\to 0$, `Yes` $\to 1$).
2. **One-Hot Encoding**: Multi-category nominal attributes were converted into dummy indicators using `drop_first=True` to eliminate multi-collinearity and preserve matrix rank, expanding the predictor space to 30 features.
3. **Stratified Splitting**: An 80/20 train-test partition ($N_{\text{train}} = 5,634$; $N_{\text{test}} = 1,409$) was generated via stratified sampling (`stratify=y`) to maintain the 26.54% churn ratio in both sets.
4. **Feature Standardization**: Continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler`:
   $$z = \frac{x - \mu}{\sigma}$$
   The scaler was fitted strictly on $X_{\text{train}}$ and transformed across $X_{\text{test}}$ to prevent data leakage.

### 3.2 Model Theoretical Formulations
Three distinct learning algorithms were investigated:
* **K-Nearest Neighbors (KNN)**: An instance-based non-parametric classifier assigning class labels based on majority vote among the $k$ nearest training vectors in metric space:
   $$D_{\text{Manhattan}}(\mathbf{x}_i, \mathbf{x}^*) = \sum_{j=1}^D |x_{ij} - x^*_j|$$
   Feature standardization is strictly required because unscaled features like `TotalCharges` (\$0–\$8,000) would dominate distance computations over binary features.
* **Logistic Regression (LR)**: A linear probabilistic model mapping feature combinations through the logistic function:
   $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
   Estimated by minimizing negative log-likelihood with $L_2$ regularization:
   $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln(P_i) + (1-y_i)\ln(1 - P_i) \right] + \frac{1}{2C}\|\mathbf{w}\|_2^2$$
* **Random Forest (RF)**: An ensemble of $B$ bootstrap decision trees utilizing Bagging and random feature subsets ($m \approx \sqrt{D}$) to minimize split impurity. Random Forest is non-parametric, robust to outliers, and invariant to monotonic feature scaling.

---

## 4. Experimental Setup, Hyperparameter Tuning & Model Evaluation (Task 2)

### 4.1 Experimental Protocols & Hyperparameter Tuning
Two experimental regimes were evaluated across all algorithms:
* **Setting 1 (Baseline)**: Default hyperparameter configurations ($k=5$ Euclidean for KNN; $C=1.0$, $L_2$ for LR; 100 trees, unconstrained depth for RF).
* **Setting 2 (Hyperparameter Tuned)**: Exhaustive search (`GridSearchCV`) evaluated under 5-Fold Stratified Cross-Validation (`StratifiedKFold`, $K=5$, `shuffle=True`) optimizing ROC-AUC:
  * *KNN*: $k \in \{5, 9, 13, 17, 21\}$, distance metric $\in \{\text{'euclidean'}, \text{'manhattan'}\}$, weights $\in \{\text{'uniform'}, \text{'distance'}\}$.
  * *Logistic Regression*: $C \in \{0.01, 0.1, 1.0, 10.0\}$, solver $\in \{\text{'liblinear'}, \text{'lbfgs'}\}$, class\_weight $\in \{\text{None}, \text{'balanced'}\}$.
  * *Random Forest*: $n\_estimators \in \{100, 200\}$, $max\_depth \in \{6, 10, \text{None}\}$, class\_weight $\in \{\text{None}, \text{'balanced'}\}$.

**Optimal Discovered Parameters**:
* *Tuned KNN*: $k = 21$, metric = `manhattan`, weights = `uniform`.
* *Tuned Logistic Regression*: $C = 10.0$, solver = `lbfgs`, class\_weight = `None`.
* *Tuned Random Forest*: $n\_estimators = 100$, $max\_depth = 6$, $min\_samples\_split = 2$, class\_weight = `'balanced'`.

### 4.2 Benchmark Results & Model Comparison
Performance metrics evaluated on the unseen test set ($N_{\text{test}} = 1,409$) are summarized in Table 1:

**Table 1: Benchmark Performance Comparison Across Experimental Settings**

| Model Architecture | Experimental Setting | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN** | Baseline ($k=5$, Euclidean) | 74.73% | 52.53% | 50.00% | 0.5123 | 0.7718 |
| **Logistic Regression** | Baseline (Default, $C=1.0$) | **80.70%** | **65.84%** | 56.68% | 0.6092 | 0.8418 |
| **Random Forest** | Baseline (Default, 100 trees) | 78.50% | 61.79% | 49.73% | 0.5511 | 0.8253 |
| **KNN** | Tuned ($k=21$, Manhattan) | 77.08% | 57.22% | 54.01% | 0.5557 | 0.8154 |
| **Logistic Regression** | Tuned ($C=10.0$, lbfgs) | 80.27% | 65.09% | 55.35% | 0.5983 | 0.8410 |
| **Random Forest** | **Tuned (`balanced`, depth=6)** | 74.10% | 50.75% | **81.55%** | **0.6256** | **0.8422** |

### 4.3 In-Depth Diagnostic Evaluation

![Figure 4: Confusion Matrix Comparison Across Models & Settings](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig4_confusion_matrices.png)

Confusion matrix analysis (Figure 4) underscores operational trade-offs:
* In Baseline settings, all models suffer high False Negative rates: Baseline Random Forest missed 188 of 374 churners (Recall = 49.73%), and Baseline Logistic Regression missed 162 churners.
* **Tuned Random Forest with cost-sensitive class balancing successfully detected 305 out of 374 churners**, reducing False Negatives to 69 (Recall = 81.55%). In subscriber retention, identifying 81.6% of vulnerable accounts substantially outweighs the cost of false-positive retention contacts.

![Figure 5: Receiver Operating Characteristic (ROC) Curves](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig5_roc_curves.png)

The ROC curves (Figure 5) verify that **Tuned Random Forest achieved the highest discriminative power (AUC = 0.8422)**, followed by Tuned Logistic Regression (AUC = 0.8410) and Tuned KNN (AUC = 0.8154).

---

## 5. AI Ethics & Post-Deployment Strategies (Task 3)

### 5.1 AI Ethics Strategies in Model Development

![Figure 6: Top 10 Most Important Features (Random Forest Tuned)](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig6_feature_importance.png)

1. **Explainability & Transparency**: Predictive interventions must not operate as uninterpretable black boxes. Gini feature importance (Figure 6) confirmed that the top predictors are contractual tenure (0.1727), two-year contracts (0.1509), total charges (0.1102), and fiber-optic subscriptions (0.0924). Decisions are driven by contractual engagement rather than sensitive demographics.
2. **Data Privacy & Governance**: In accordance with GDPR and CCPA, direct identifiers (`customerID`) were stripped. Deployment must preserve data minimization, feature tokenization, and customer opt-out rights.

![Figure 7: AI Ethics & Fairness Audit across Demographic Subgroups](C:/Users/Dinushi%20Hasangika/.gemini/antigravity/brain/acc4eabb-b2fa-4686-a846-564e8f5b625c/figures/fig7_ethics_fairness_audit.png)

3. **Demographic Fairness Audit**: A subgroup analysis across `gender` and `SeniorCitizen` was executed on test predictions (Figure 7):
   * **Gender Equity**: Actual churn rates (Female: 28.09%, Male: 25.07%) corresponded to balanced model predictions (42.21% vs 43.07%), demonstrating statistical parity (<0.9% deviation).
   * **Senior Citizen Protection**: Senior citizens exhibited higher baseline churn (44.14% vs 23.25%), yielding higher predicted churn probabilities (67.57% vs 37.99%). Ethical governance must ensure that senior accounts are not subjected to punitive pricing or denied standard service channels due to algorithmic scoring.

### 5.2 Post-Deployment MLOps Architecture
To maintain production model reliability, three operational protocols must be enacted:
1. **Drift Monitoring**:
   * *Data Drift*: Automated Kolmogorov-Smirnov tests for continuous metrics and Population Stability Index (PSI) for categorical features must run weekly to detect shifts in customer distributions.
   * *Concept Drift*: Changes in the relationship $P(y|\mathbf{x})$—such as aggressive competitor campaigns increasing churn among loyal subscribers—must be monitored via rolling F1 and Brier loss scores.
2. **Automated Retraining Pipelines**: Orchestrated workflows (e.g., MLflow) should retrain the champion model monthly on rolling ground-truth retention data, enforcing automated quality gates (test AUC $\ge 0.82$, demographic deviation $\le 5\%$) prior to deployment.
3. **Human-in-the-Loop & Shadow Deployment**: Predictions with churn probability $> 0.85$ should route to human retention agents for case-by-case review. Retrained models must run in shadow mode alongside production before serving live traffic, backed by immediate rollback capabilities.

---

## 6. Limitations & Further Enhancements

### 6.1 Limitations
* **Static Snapshot Nature**: The corpus represents a single historical slice, lacking longitudinal time-series observations, network outage logs, and customer service call transcripts.
* **Absence of Cost Matrix**: Models were optimized on statistical ROC-AUC rather than an exact financial loss matrix integrating customer lifetime value (CLV) against intervention costs.

### 6.2 Further Enhancements
* **Gradient Boosting**: Exploring XGBoost, LightGBM, and CatBoost to evaluate gradient-boosted decision trees.
* **Threshold Calibration**: Applying Platt Scaling and tuning classification probability thresholds away from 0.5 to align precisely with corporate retention budget constraints.

---

## 7. Conclusion
This study established an end-to-end machine learning pipeline for customer churn classification. The hyperparameter-tuned Random Forest model emerged as the optimal architecture, attaining 81.55% recall, an F1-score of 0.6256, and an ROC-AUC of 0.8422. Key churn catalysts—namely short tenure and month-to-month contracts—were isolated. Supported by demographic fairness audits, feature explainability, and post-deployment MLOps strategies, the system provides a technically sound and ethically responsible retention solution.

---

## 8. References
1. Breiman, L., 2001. Random forests. *Machine Learning*, 45(1), pp. 5–32.
2. Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P., 2002. SMOTE: synthetic minority over-sampling technique. *JAIR*, 16, pp. 321–357.
3. Cover, T. and Hart, P., 1967. Nearest neighbor pattern classification. *IEEE Trans. Inf. Theory*, 13(1), pp. 21–27.
4. Fawcett, T., 2006. An introduction to ROC analysis. *Pattern Recognit. Lett.*, 27(8), pp. 861–874.
5. Hosmer Jr, D.W., Lemeshow, S. and Sturdivant, R.X., 2013. *Applied Logistic Regression*. 3rd ed. John Wiley & Sons.
6. Pedregosa, F., et al., 2011. Scikit-learn: Machine learning in Python. *JMLR*, 12, pp. 2825–2830.

---

## Appendix: Complete Source Code (Text Format)

```python
"""
MACHINE LEARNING COURSEWORK: BINARY CLASSIFICATION FOR TELCO CUSTOMER CHURN
Author: Dinushi Hasangika
Repository: https://github.com/hasangikadinushi/telco-customer-churn-ml-
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# 1. SETUP OUTPUT DIRECTORIES & STYLE
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# 2. LOAD DATASET
data_path = "Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# 3. TASK 1: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# 3.1 Class Distribution
churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='Churn', hue='Churn', data=df, palette=['#4C72B0', '#C44E52'], legend=False)
plt.title("Figure 1: Target Class Distribution (Customer Churn)", fontsize=13, weight='bold')
plt.xlabel("Churn Status", fontsize=11)
plt.ylabel("Customer Count", fontsize=11)
plt.savefig("figures/fig1_churn_distribution.png", dpi=300)
plt.close()

# 3.2 Handling Whitespace in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

# Contract vs Churn & Tenure KDE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0], palette=['#4C72B0', '#C44E52'])
axes[0].set_title("Churn Distribution by Contract Type", fontsize=12, weight='bold')
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False,
            palette=['#4C72B0', '#C44E52'], ax=axes[1], alpha=0.5)
axes[1].set_title("Customer Tenure (Months) Density by Churn", fontsize=12, weight='bold')
plt.suptitle("Figure 2: Impact of Contract Structure and Customer Tenure on Churn", fontsize=14, weight='bold')
plt.savefig("figures/fig2_contract_and_tenure_churn.png", dpi=300)
plt.close()

# Monthly Charges & Internet Service
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', hue='Churn', data=df, ax=axes[0], palette=['#4C72B0', '#C44E52'], legend=False)
axes[0].set_title("Monthly Charges vs Churn Status", fontsize=12, weight='bold')
sns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[1], palette=['#4C72B0', '#C44E52'])
axes[1].set_title("Churn by Internet Service Type", fontsize=12, weight='bold')
plt.suptitle("Figure 3: Financial and Service-Tier Drivers of Customer Churn", fontsize=14, weight='bold')
plt.savefig("figures/fig3_charges_and_internetservice.png", dpi=300)
plt.close()

# 4. TASK 2: DATA PREPROCESSING PIPELINE
df_clean = df.drop(columns=['customerID'])
df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
df_encoded = pd.get_dummies(df_clean, drop_first=True, dtype=int)

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODEL IMPLEMENTATION & EXPERIMENTAL SETTINGS
results = []
trained_models = {}

def evaluate_model(model_name, setting_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    results.append({
        'Model': model_name, 'Setting': setting_name,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': roc_auc
    })

# EXPERIMENTAL SETTING 1: BASELINE MODELS
knn_base = KNeighborsClassifier().fit(X_train_scaled, y_train)
evaluate_model("KNN", "Baseline", y_test, knn_base.predict(X_test_scaled), knn_base.predict_proba(X_test_scaled)[:, 1])
trained_models["KNN_Baseline"] = (knn_base, knn_base.predict(X_test_scaled), knn_base.predict_proba(X_test_scaled)[:, 1])

lr_base = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)
evaluate_model("Logistic Regression", "Baseline", y_test, lr_base.predict(X_test_scaled), lr_base.predict_proba(X_test_scaled)[:, 1])
trained_models["LR_Baseline"] = (lr_base, lr_base.predict(X_test_scaled), lr_base.predict_proba(X_test_scaled)[:, 1])

rf_base = RandomForestClassifier(random_state=42).fit(X_train, y_train)
evaluate_model("Random Forest", "Baseline", y_test, rf_base.predict(X_test), rf_base.predict_proba(X_test)[:, 1])
trained_models["RF_Baseline"] = (rf_base, rf_base.predict(X_test), rf_base.predict_proba(X_test)[:, 1])

# EXPERIMENTAL SETTING 2: HYPERPARAMETER TUNING (5-Fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tuned KNN
grid_knn = GridSearchCV(KNeighborsClassifier(), {
    'n_neighbors': [5, 9, 13, 17, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}, cv=cv, scoring='roc_auc', n_jobs=-1).fit(X_train_scaled, y_train)
knn_best = grid_knn.best_estimator_
evaluate_model("KNN", "Tuned", y_test, knn_best.predict(X_test_scaled), knn_best.predict_proba(X_test_scaled)[:, 1])
trained_models["KNN_Tuned"] = (knn_best, knn_best.predict(X_test_scaled), knn_best.predict_proba(X_test_scaled)[:, 1])

# Tuned Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}, cv=cv, scoring='roc_auc', n_jobs=-1).fit(X_train_scaled, y_train)
lr_best = grid_lr.best_estimator_
evaluate_model("Logistic Regression", "Tuned", y_test, lr_best.predict(X_test_scaled), lr_best.predict_proba(X_test_scaled)[:, 1])
trained_models["LR_Tuned"] = (lr_best, lr_best.predict(X_test_scaled), lr_best.predict_proba(X_test_scaled)[:, 1])

# Tuned Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}, cv=cv, scoring='roc_auc', n_jobs=-1).fit(X_train, y_train)
rf_best = grid_rf.best_estimator_
evaluate_model("Random Forest", "Tuned", y_test, rf_best.predict(X_test), rf_best.predict_proba(X_test)[:, 1])
trained_models["RF_Tuned"] = (rf_best, rf_best.predict(X_test), rf_best.predict_proba(X_test)[:, 1])

# 6. RESULTS EXPORT & EVALUATION CHARTS
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)

# Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
models_to_plot = [
    ("KNN (Baseline)", "KNN_Baseline"), ("Logistic Reg (Baseline)", "LR_Baseline"), ("Random Forest (Baseline)", "RF_Baseline"),
    ("KNN (Tuned)", "KNN_Tuned"), ("Logistic Reg (Tuned)", "LR_Tuned"), ("Random Forest (Tuned)", "RF_Tuned")
]
for idx, (title, key) in enumerate(models_to_plot):
    row, col = idx // 3, idx % 3
    _, y_pred, _ = trained_models[key]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[row, col], annot_kws={'size': 12, 'weight': 'bold'})
    axes[row, col].set_title(title, fontsize=12, weight='bold')
    axes[row, col].set_xlabel("Predicted Label")
    axes[row, col].set_ylabel("Actual Label")
plt.suptitle("Figure 4: Confusion Matrix Comparison Across Models & Settings", fontsize=14, weight='bold')
plt.savefig("figures/fig4_confusion_matrices.png", dpi=300)
plt.close()

# ROC Curves
plt.figure(figsize=(9, 6))
colors = {
    'KNN_Baseline': ('#55A868', '--', 'KNN (Baseline)'),
    'LR_Baseline': ('#4C72B0', '--', 'Logistic Regression (Baseline)'),
    'RF_Baseline': ('#C44E52', '--', 'Random Forest (Baseline)'),
    'KNN_Tuned': ('#2ca02c', '-', 'KNN (Tuned)'),
    'LR_Tuned': ('#1f77b4', '-', 'Logistic Regression (Tuned)'),
    'RF_Tuned': ('#d62728', '-', 'Random Forest (Tuned)')
}
for key, (color, linestyle, label) in colors.items():
    _, _, y_prob = trained_models[key]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})", color=color, linestyle=linestyle, lw=2)
plt.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Random Chance (AUC = 0.500)')
plt.title("Figure 5: Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("figures/fig5_roc_curves.png", dpi=300)
plt.close()

# 7. TASK 3: FEATURE IMPORTANCE & ETHICS AUDIT
# Feature Importance
importances = rf_best.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_imp_df, palette='viridis', legend=False)
plt.title("Figure 6: Top 10 Most Important Features (Random Forest Tuned)", fontsize=13, weight='bold')
plt.savefig("figures/fig6_feature_importance.png", dpi=300)
plt.close()

# Ethics Demographic Audit
test_subgroup = df.loc[X_test.index].copy()
test_subgroup['Actual_Churn'] = y_test
test_subgroup['Pred_Churn'] = trained_models["RF_Tuned"][1]

senior_stats = test_subgroup.groupby('SeniorCitizen').agg(Actual_Churn_Rate=('Actual_Churn', 'mean'), Predicted_Churn_Rate=('Pred_Churn', 'mean')).reset_index()
senior_stats['SeniorCitizen'] = senior_stats['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})

gender_stats = test_subgroup.groupby('gender').agg(Actual_Churn_Rate=('Actual_Churn', 'mean'), Predicted_Churn_Rate=('Pred_Churn', 'mean')).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x='SeniorCitizen', y='Rate', hue='Metric', data=senior_stats.melt('SeniorCitizen', ['Actual_Churn_Rate', 'Predicted_Churn_Rate'], 'Metric', 'Rate'), ax=axes[0], palette=['#4C72B0', '#C44E52'])
axes[0].set_title("Actual vs Predicted Churn by Senior Citizen Status", fontsize=12, weight='bold')
sns.barplot(x='gender', y='Rate', hue='Metric', data=gender_stats.melt('gender', ['Actual_Churn_Rate', 'Predicted_Churn_Rate'], 'Metric', 'Rate'), ax=axes[1], palette=['#4C72B0', '#C44E52'])
axes[1].set_title("Actual vs Predicted Churn by Gender", fontsize=12, weight='bold')
plt.suptitle("Figure 7: AI Ethics & Fairness Audit across Demographic Subgroups", fontsize=14, weight='bold')
plt.savefig("figures/fig7_ethics_fairness_audit.png", dpi=300)
plt.close()
```
