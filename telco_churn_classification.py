"""
================================================================================
MACHINE LEARNING COURSEWORK: BINARY CLASSIFICATION FOR TELCO CUSTOMER CHURN
================================================================================
Student Coursework Script
Dataset: Telco Customer Churn (Kaggle)
Models: K-Nearest Neighbors (KNN), Logistic Regression, Random Forest
Tasks Covered:
  - Task 1: Exploratory Data Analysis (EDA) & Visualizations
  - Task 2: Data Preprocessing, Baseline Models, Hyperparameter Tuning & Comparison
  - Task 3: AI Ethics Audit (Fairness/Demographic Analysis) & Feature Importance
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules for modeling, tuning, and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ------------------------------------------------------------------------------
# 1. SETUP OUTPUT DIRECTORIES & STYLE
# ------------------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

print("=" * 80)
print("STARTING TELCO CUSTOMER CHURN MACHINE LEARNING PIPELINE")
print("=" * 80)

# ------------------------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------------------------
data_path = "Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

print(f"\n[INFO] Dataset loaded successfully.")
print(f"       Total Samples: {df.shape[0]}, Total Features: {df.shape[1]}")
print(f"\n[INFO] First 5 rows preview:")
print(df.head(3))

# ------------------------------------------------------------------------------
# 3. TASK 1: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 1: PERFORMING EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# 3.1 Class Distribution (Churn Imbalance)
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"\n[EDA] Churn Class Distribution:")
print(f"      No Churn  (0): {churn_counts['No']} ({churn_pct['No']:.2f}%)")
print(f"      Churned   (1): {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
print(f"      -> Note: Class imbalance exists (~73.5% vs ~26.5%).")

# Plot 1: Churn Target Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='Churn', data=df, palette=['#4C72B0', '#C44E52'])
plt.title("Figure 1: Target Class Distribution (Customer Churn)", fontsize=13, weight='bold')
plt.xlabel("Churn Status", fontsize=11)
plt.ylabel("Customer Count", fontsize=11)
for p in ax.patches:
    height = p.get_height()
    pct = (height / len(df)) * 100
    ax.annotate(f"{height} ({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', color='white', weight='bold', fontsize=11)
plt.savefig("figures/fig1_churn_distribution.png", dpi=300)
plt.close()
print("      [Saved] figures/fig1_churn_distribution.png")

# 3.2 Inspecting and Fixing TotalCharges Data Quality Issue
print("\n[EDA] Checking missing / whitespace values in numerical columns...")
whitespace_count = (df['TotalCharges'].str.strip() == '').sum()
print(f"      Found {whitespace_count} blank spaces in 'TotalCharges' column.")

# Convert TotalCharges to numeric, coercing spaces to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Impute NaN values with 0 (because all 11 rows have tenure = 0)
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
print(f"      Successfully converted 'TotalCharges' to float64 (imputed {whitespace_count} missing values with 0.0).")

# Plot 2: Churn by Contract Type & Tenure Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contract vs Churn
sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0], palette=['#4C72B0', '#C44E52'])
axes[0].set_title("Churn Distribution by Contract Type", fontsize=12, weight='bold')
axes[0].set_xlabel("Contract Type", fontsize=11)
axes[0].set_ylabel("Number of Customers", fontsize=11)

# Tenure KDE distribution by Churn
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False,
            palette=['#4C72B0', '#C44E52'], ax=axes[1], alpha=0.5)
axes[1].set_title("Customer Tenure (Months) Density by Churn", fontsize=12, weight='bold')
axes[1].set_xlabel("Tenure (Months)", fontsize=11)
axes[1].set_ylabel("Density", fontsize=11)

plt.suptitle("Figure 2: Impact of Contract Structure and Customer Tenure on Churn", fontsize=14, weight='bold')
plt.savefig("figures/fig2_contract_and_tenure_churn.png", dpi=300)
plt.close()
print("      [Saved] figures/fig2_contract_and_tenure_churn.png")

# Plot 3: Monthly Charges Boxplot & Internet Service Churn Rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Monthly Charges Boxplot
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[0], palette=['#4C72B0', '#C44E52'])
axes[0].set_title("Monthly Charges vs Churn Status", fontsize=12, weight='bold')
axes[0].set_xlabel("Churn Status", fontsize=11)
axes[0].set_ylabel("Monthly Charges ($)", fontsize=11)

# Internet Service vs Churn Count
sns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[1], palette=['#4C72B0', '#C44E52'])
axes[1].set_title("Churn by Internet Service Type", fontsize=12, weight='bold')
axes[1].set_xlabel("Internet Service Type", fontsize=11)
axes[1].set_ylabel("Number of Customers", fontsize=11)

plt.suptitle("Figure 3: Financial and Service-Tier Drivers of Customer Churn", fontsize=14, weight='bold')
plt.savefig("figures/fig3_charges_and_internetservice.png", dpi=300)
plt.close()
print("      [Saved] figures/fig3_charges_and_internetservice.png")


# ------------------------------------------------------------------------------
# 4. TASK 2: DATA PREPROCESSING
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: DATA PREPROCESSING PIPELINE")
print("=" * 80)

# 4.1 Remove non-predictive identifier
df_clean = df.copy()
df_clean = df_clean.drop(columns=['customerID'])
print("  [Step 1] Dropped identifier column 'customerID'.")

# 4.2 Encode Target Variable
df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
print("  [Step 2] Mapped target 'Churn' to binary: No -> 0, Yes -> 1.")

# 4.3 Feature Encoding
# Identify categorical columns (excluding target)
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print(f"  [Step 3] One-Hot Encoding {len(categorical_cols)} categorical features...")
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True, dtype=int)
print(f"           Dataset shape after One-Hot Encoding: {df_encoded.shape}")

# 4.4 Separate Features (X) and Target (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# 4.5 Train-Test Split (Stratified to maintain class proportion)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  [Step 4] Stratified Train-Test Split (80% Train, 20% Test):")
print(f"           X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"           Train Churn Rate: {y_train.mean():.4f}, Test Churn Rate: {y_test.mean():.4f}")

# 4.6 Feature Scaling using StandardScaler
# Crucial for distance-based algorithms (KNN) and regularized linear models (Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  [Step 5] Feature Scaling completed using StandardScaler (mean=0, std=1).")


# ------------------------------------------------------------------------------
# 5. MODEL IMPLEMENTATION & EXPERIMENTAL SETTINGS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: MODEL TRAINING & HYPERPARAMETER TUNING")
print("=" * 80)

# Dictionary to hold all trained models and their predictions
results = []
trained_models = {}

# Helper function to evaluate and record metrics
def evaluate_model(model_name, setting_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    results.append({
        'Model': model_name,
        'Setting': setting_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    print(f"  [{setting_name}] {model_name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

# ------------------------------------------------------------------------------
# EXPERIMENTAL SETTING 1: BASELINE MODELS (Default Hyperparameters)
# ------------------------------------------------------------------------------
print("\n--- EXPERIMENTAL SETTING 1: BASELINE MODELS (DEFAULT PARAMETERS) ---")

# 1. Baseline K-Nearest Neighbors (KNN)
knn_base = KNeighborsClassifier()
knn_base.fit(X_train_scaled, y_train)
y_pred_knn_base = knn_base.predict(X_test_scaled)
y_prob_knn_base = knn_base.predict_proba(X_test_scaled)[:, 1]
evaluate_model("KNN", "Baseline", y_test, y_pred_knn_base, y_prob_knn_base)
trained_models["KNN_Baseline"] = (knn_base, y_pred_knn_base, y_prob_knn_base)

# 2. Baseline Logistic Regression
lr_base = LogisticRegression(random_state=42, max_iter=1000)
lr_base.fit(X_train_scaled, y_train)
y_pred_lr_base = lr_base.predict(X_test_scaled)
y_prob_lr_base = lr_base.predict_proba(X_test_scaled)[:, 1]
evaluate_model("Logistic Regression", "Baseline", y_test, y_pred_lr_base, y_prob_lr_base)
trained_models["LR_Baseline"] = (lr_base, y_pred_lr_base, y_prob_lr_base)

# 3. Baseline Random Forest
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)  # Tree-based algorithms can use raw or scaled features
y_pred_rf_base = rf_base.predict(X_test)
y_prob_rf_base = rf_base.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", "Baseline", y_test, y_pred_rf_base, y_prob_rf_base)
trained_models["RF_Baseline"] = (rf_base, y_pred_rf_base, y_prob_rf_base)


# ------------------------------------------------------------------------------
# EXPERIMENTAL SETTING 2: HYPERPARAMETER TUNED MODELS (via 5-Fold Cross Validation)
# ------------------------------------------------------------------------------
print("\n--- EXPERIMENTAL SETTING 2: HYPERPARAMETER TUNING (GRID SEARCH CV) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Tuned K-Nearest Neighbors
print("  [Tuning] K-Nearest Neighbors (GridSearchCV)...")
knn_param_grid = {
    'n_neighbors': [5, 9, 13, 17, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
knn_best = grid_knn.best_estimator_
y_pred_knn_tuned = knn_best.predict(X_test_scaled)
y_prob_knn_tuned = knn_best.predict_proba(X_test_scaled)[:, 1]
evaluate_model("KNN", "Tuned", y_test, y_pred_knn_tuned, y_prob_knn_tuned)
trained_models["KNN_Tuned"] = (knn_best, y_pred_knn_tuned, y_prob_knn_tuned)
print(f"           Best KNN Params: {grid_knn.best_params_}")

# 2. Tuned Logistic Regression
print("  [Tuning] Logistic Regression (GridSearchCV)...")
lr_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced']
}
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), lr_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
lr_best = grid_lr.best_estimator_
y_pred_lr_tuned = lr_best.predict(X_test_scaled)
y_prob_lr_tuned = lr_best.predict_proba(X_test_scaled)[:, 1]
evaluate_model("Logistic Regression", "Tuned", y_test, y_pred_lr_tuned, y_prob_lr_tuned)
trained_models["LR_Tuned"] = (lr_best, y_pred_lr_tuned, y_prob_lr_tuned)
print(f"           Best LR Params: {grid_lr.best_params_}")

# 3. Tuned Random Forest
print("  [Tuning] Random Forest (GridSearchCV)...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_best = grid_rf.best_estimator_
y_pred_rf_tuned = rf_best.predict(X_test)
y_prob_rf_tuned = rf_best.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", "Tuned", y_test, y_pred_rf_tuned, y_prob_rf_tuned)
trained_models["RF_Tuned"] = (rf_best, y_pred_rf_tuned, y_prob_rf_tuned)
print(f"           Best RF Params: {grid_rf.best_params_}")


# ------------------------------------------------------------------------------
# 6. MODEL COMPARISON, METRICS EXPORT & EVALUATION PLOTS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: RESULTS SUMMARY & MODEL COMPARISON")
print("=" * 80)

# Build Results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n[SUMMARY TABLE: MODEL BENCHMARK COMPARISON]")
print(results_df.to_string(index=False))
print("\n[Exported] Saved metrics to 'model_comparison_results.csv'")

# Plot 4: Confusion Matrices Comparison (Baseline vs Tuned)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
models_to_plot = [
    ("KNN (Baseline)", "KNN_Baseline"),
    ("Logistic Reg (Baseline)", "LR_Baseline"),
    ("Random Forest (Baseline)", "RF_Baseline"),
    ("KNN (Tuned)", "KNN_Tuned"),
    ("Logistic Reg (Tuned)", "LR_Tuned"),
    ("Random Forest (Tuned)", "RF_Tuned")
]

for idx, (title, key) in enumerate(models_to_plot):
    row = idx // 3
    col = idx % 3
    _, y_pred, _ = trained_models[key]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[row, col],
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[row, col].set_title(title, fontsize=12, weight='bold')
    axes[row, col].set_xlabel("Predicted Label", fontsize=10)
    axes[row, col].set_ylabel("Actual Label", fontsize=10)
    axes[row, col].set_xticklabels(['No Churn (0)', 'Churn (1)'])
    axes[row, col].set_yticklabels(['No Churn (0)', 'Churn (1)'])

plt.suptitle("Figure 4: Confusion Matrix Comparison Across Models & Settings", fontsize=14, weight='bold')
plt.savefig("figures/fig4_confusion_matrices.png", dpi=300)
plt.close()
print("      [Saved] figures/fig4_confusion_matrices.png")

# Plot 5: ROC Curves Comparison
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
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
plt.title("Figure 5: Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.savefig("figures/fig5_roc_curves.png", dpi=300)
plt.close()
print("      [Saved] figures/fig5_roc_curves.png")


# ------------------------------------------------------------------------------
# 7. TASK 3: FEATURE IMPORTANCE, ETHICS AUDIT & POST-DEPLOYMENT INSIGHTS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: AI ETHICS AUDIT & FEATURE IMPORTANCE")
print("=" * 80)

# 7.1 Feature Importance from Best Random Forest Model
rf_best_model, _, _ = trained_models["RF_Tuned"]
importances = rf_best_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title("Figure 6: Top 10 Most Important Features (Random Forest Tuned)", fontsize=13, weight='bold')
plt.xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
plt.ylabel("Feature Name", fontsize=11)
plt.savefig("figures/fig6_feature_importance.png", dpi=300)
plt.close()
print("      [Saved] figures/fig6_feature_importance.png")
print("\n[INFO] Top 5 Most Predictive Features for Churn:")
for i, r in feature_imp_df.head(5).iterrows():
    print(f"       - {r['Feature']}: {r['Importance']:.4f}")

# 7.2 AI Ethics: Fairness & Demographic Subgroup Analysis
print("\n[ETHICS AUDIT] Evaluating Model Fairness across Demographic Subgroups...")
# We evaluate the tuned Random Forest model on test set protected attributes
test_indices = X_test.index
test_subgroup = df.loc[test_indices].copy()
test_subgroup['Actual_Churn'] = y_test
test_subgroup['Pred_Churn'] = trained_models["RF_Tuned"][1]

# Audit Senior Citizens vs Non-Senior Citizens
senior_stats = test_subgroup.groupby('SeniorCitizen').agg(
    Customer_Count=('Churn', 'count'),
    Actual_Churn_Rate=('Actual_Churn', 'mean'),
    Predicted_Churn_Rate=('Pred_Churn', 'mean')
).reset_index()
senior_stats['SeniorCitizen'] = senior_stats['SeniorCitizen'].map({0: 'Non-Senior Citizen', 1: 'Senior Citizen'})

# Audit Gender
gender_stats = test_subgroup.groupby('gender').agg(
    Customer_Count=('Churn', 'count'),
    Actual_Churn_Rate=('Actual_Churn', 'mean'),
    Predicted_Churn_Rate=('Pred_Churn', 'mean')
).reset_index()

print("\nDemographic Fairness Audit Table (Senior Citizen status):")
print(senior_stats.to_string(index=False))
print("\nDemographic Fairness Audit Table (Gender):")
print(gender_stats.to_string(index=False))

# Plot 7: Fairness Audit Visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Senior Citizen Plot
senior_melted = senior_stats.melt(id_vars='SeniorCitizen',
                                  value_vars=['Actual_Churn_Rate', 'Predicted_Churn_Rate'],
                                  var_name='Metric', value_name='Rate')
sns.barplot(x='SeniorCitizen', y='Rate', hue='Metric', data=senior_melted, ax=axes[0], palette=['#4C72B0', '#C44E52'])
axes[0].set_title("Actual vs Predicted Churn by Senior Citizen Status", fontsize=12, weight='bold')
axes[0].set_ylabel("Churn Proportion", fontsize=11)
axes[0].set_ylim(0, 0.6)

# Gender Plot
gender_melted = gender_stats.melt(id_vars='gender',
                                  value_vars=['Actual_Churn_Rate', 'Predicted_Churn_Rate'],
                                  var_name='Metric', value_name='Rate')
sns.barplot(x='gender', y='Rate', hue='Metric', data=gender_melted, ax=axes[1], palette=['#4C72B0', '#C44E52'])
axes[1].set_title("Actual vs Predicted Churn by Gender", fontsize=12, weight='bold')
axes[1].set_ylabel("Churn Proportion", fontsize=11)
axes[1].set_ylim(0, 0.6)

plt.suptitle("Figure 7: AI Ethics & Fairness Audit across Demographic Subgroups", fontsize=14, weight='bold')
plt.savefig("figures/fig7_ethics_fairness_audit.png", dpi=300)
plt.close()
print("      [Saved] figures/fig7_ethics_fairness_audit.png")

print("\n" + "=" * 80)
print("MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
print("All charts have been generated in the 'figures/' folder.")
print("All model comparison metrics are saved in 'model_comparison_results.csv'.")
print("=" * 80)
