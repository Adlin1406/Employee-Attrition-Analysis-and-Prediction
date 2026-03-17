# ============================================================
#  EMPLOYEE ATTRITION ANALYSIS & PREDICTION
#  Run this file directly: python employee_attrition.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import os
os.makedirs("outputs", exist_ok=True)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
print("=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)

df = pd.read_excel( r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\Employee_Attrition_Analysis.xlsx")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# STEP 2 - DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Data Preprocessing")
print("=" * 50)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Target variable distribution
print("\nAttrition value counts:")
print(df['Attrition'].value_counts())
print("Attrition Rate:", round((df['Attrition'] == 'Yes').mean() * 100, 2), "%")

# Drop columns that are useless (constant values)
df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'],
        inplace=True)
print("\nDropped constant/ID columns.")

# Encode target variable: Yes=1, No=0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode all other categorical columns using LabelEncoder
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("\nEncoding categorical columns:", cat_cols)
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nData after encoding:")
print(df.head())
print("Shape after preprocessing:", df.shape)

# Save cleaned data
df.to_csv(  r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\cleaned_data.csv", index=False)
print("\nCleaned data saved")


# ============================================================
# STEP 3 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: Exploratory Data Analysis")
print("=" * 50)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# --- Plot 1: Attrition Count ---
plt.figure(figsize=(5, 4))
df['Attrition'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Attrition Count (0=Stayed, 1=Left)')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/plot1_attrition_count.png")
plt.show()
print("Plot 1 saved: attrition count")

# --- Plot 2: Age Distribution ---
plt.figure(figsize=(8, 4))
df[df['Attrition'] == 0]['Age'].hist(bins=20, alpha=0.6, color='steelblue', label='Stayed')
df[df['Attrition'] == 1]['Age'].hist(bins=20, alpha=0.6, color='tomato', label='Left')
plt.title('Age Distribution by Attrition')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plot2_age_distribution.png")
plt.show()
print("Plot 2 saved: age distribution")

# --- Plot 3: Monthly Income vs Attrition ---
plt.figure(figsize=(6, 4))
df.groupby('Attrition')['MonthlyIncome'].mean().plot(
    kind='bar', color=['steelblue', 'tomato'])
plt.title('Average Monthly Income by Attrition')
plt.xlabel('Attrition (0=Stayed, 1=Left)')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/plot3_monthly_income.png")
plt.show()
print("Plot 3 saved: monthly income")

# --- Plot 4: Correlation Heatmap ---
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm',
            linewidths=0.3, annot_kws={'size': 6})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("outputs/plot4_correlation_heatmap.png")
plt.show()
print("Plot 4 saved: correlation heatmap")

# --- Plot 5: OverTime vs Attrition ---
# Reload original to get readable labels for this chart
df_orig = pd.read_excel(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\Employee_Attrition_Analysis.xlsx")
overtime_attrition = df_orig.groupby('OverTime')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100)
plt.figure(figsize=(5, 4))
overtime_attrition.plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Attrition Rate % by OverTime')
plt.xlabel('OverTime')
plt.ylabel('Attrition Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/plot5_overtime_attrition.png")
plt.show()
print("Plot 5 saved: overtime vs attrition")


# ============================================================
# STEP 4 - FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: Feature Engineering")
print("=" * 50)

# Create a few new useful features
df['IncomePerYear']        = df['MonthlyIncome'] / (df['TotalWorkingYears'].replace(0, 1))
df['LoyaltyScore']         = df['YearsAtCompany'] / (df['TotalWorkingYears'].replace(0, 1))
df['EngagementScore']      = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                               df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
df['PromotionStagnation']  = df['YearsSinceLastPromotion'] / (df['YearsInCurrentRole'].replace(0, 1))

print("New features created:")
print("  - IncomePerYear       (monthly income / total working years)")
print("  - LoyaltyScore        (years at company / total working years)")
print("  - EngagementScore     (average of 4 satisfaction scores)")
print("  - PromotionStagnation (years since promotion / years in current role)")
print("\nShape after feature engineering:", df.shape)


# ============================================================
# STEP 5 - TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: Train / Test Split")
print("=" * 50)

X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Test set size    :", X_test.shape)
print("Train attrition rate:", round(y_train.mean() * 100, 2), "%")
print("Test  attrition rate:", round(y_test.mean() * 100, 2), "%")


# ============================================================
# STEP 6 - MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: Model Training & Evaluation")
print("=" * 50)

# Define models
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# Store results
results = []

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        'Model'    : name,
        'Accuracy' : round(acc, 4),
        'Precision': round(prec, 4),
        'Recall'   : round(rec, 4),
        'F1-Score' : round(f1, 4),
        'AUC-ROC'  : round(auc, 4),
    })

# Print summary table
results_df = pd.DataFrame(results)
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
print(results_df.to_string(index=False))
results_df.to_csv("outputs/model_results.csv", index=False)
print("\nResults saved to outputs/model_results.csv")


# ============================================================
# STEP 7 - BEST MODEL & FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: Best Model & Feature Importance")
print("=" * 50)

# Pick best model by AUC-ROC
best_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_model = models[best_name]
print(f"Best Model: {best_name}")
print(f"AUC-ROC   : {results_df.loc[results_df['AUC-ROC'].idxmax(), 'AUC-ROC']}")

# Feature importance (works for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False).head(15)

    print("\nTop 15 Important Features:")
    print(feat_imp)

    plt.figure(figsize=(8, 6))
    feat_imp.sort_values().plot(kind='barh', color='steelblue')
    plt.title(f'Top 15 Feature Importances - {best_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig("outputs/plot6_feature_importance.png")
    plt.show()
    print("Plot 6 saved: feature importance")


# ============================================================
# STEP 8 - PREDICT AT-RISK EMPLOYEES
# ============================================================
print("\n" + "=" * 50)
print("STEP 8: Predicting At-Risk Employees")
print("=" * 50)

# Predict probability for all employees
all_probs = best_model.predict_proba(X)[:, 1]

# Load original data for readable output
df_orig = pd.read_excel(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\Employee_Attrition_Analysis.xlsx")
df_orig['Attrition_Probability_%'] = (all_probs * 100).round(1)
df_orig['Risk_Level'] = pd.cut(
    df_orig['Attrition_Probability_%'],
    bins=[0, 30, 60, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Sort by highest risk first
df_risk = df_orig.sort_values('Attrition_Probability_%', ascending=False)

# Show top 10 at-risk employees
print("\nTop 10 At-Risk Employees:")
print(df_risk[['EmployeeNumber', 'Age', 'Department', 'JobRole',
               'MonthlyIncome', 'OverTime', 'Attrition',
               'Attrition_Probability_%', 'Risk_Level']].head(10).to_string(index=False))

# Risk level summary
print("\nRisk Level Summary:")
print(df_risk['Risk_Level'].value_counts())

# Save at-risk list
df_risk.to_csv("outputs/at_risk_employees.csv", index=False)
print("\nAt-risk list saved to outputs/at_risk_employees.csv")


# ============================================================
# STEP 9 - SAVE SUMMARY PLOTS
# ============================================================
print("\n" + "=" * 50)
print("STEP 9: Final Summary Plot")
print("=" * 50)

# Bar chart comparing all model metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x      = np.arange(len(results_df))
width  = 0.15

fig, ax = plt.subplots(figsize=(12, 5))
colors  = ['steelblue', 'tomato', 'green', 'orange', 'purple']
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=color)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plot7_model_comparison.png")
plt.show()
print("Plot 7 saved: model comparison")


# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 50)
print("ALL STEPS COMPLETE!")
print("=" * 50)
print("\nFiles saved in outputs/ folder:")
print("  - cleaned_data.csv")
print("  - model_results.csv")
print("  - at_risk_employees.csv")
print("  - plot1_attrition_count.png")
print("  - plot2_age_distribution.png")
print("  - plot3_monthly_income.png")
print("  - plot4_correlation_heatmap.png")
print("  - plot5_overtime_attrition.png")
print("  - plot6_feature_importance.png")
print("  - plot7_model_comparison.png")
print(f"\nBest Model : {best_name}")
print(f"AUC-ROC    : {results_df.loc[results_df['AUC-ROC'].idxmax(), 'AUC-ROC']}")
