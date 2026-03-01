import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Analysis & Prediction Dashboard")
st.markdown("### HR Analytics for Smart Retention Strategy")

# -------------------------------------------------
# LOAD DATA (YOUR PATH INCLUDED)
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_Traffic_3_GUVI\Employee-Attrition-Cleaned.csv"
    )

df = load_data()
df.columns = df.columns.str.strip()

# -------------------------------------------------
# AUTO DETECT ATTRITION COLUMN
# -------------------------------------------------
possible_targets = ['Attrition', 'attrition', 'left', 'Exited']
target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    st.error("⚠ Attrition column not found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Convert Yes/No to 1/0 if needed
if df[target_column].dtype == 'object':
    df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

total_emp = df.shape[0]
total_left = df[target_column].sum()
attrition_rate = (total_left / total_emp) * 100

col1.metric("👥 Total Employees", total_emp)
col2.metric("🚪 Employees Left", int(total_left))
col3.metric("📉 Attrition Rate (%)", round(attrition_rate, 2))

# -------------------------------------------------
# ATTRITION BY DEPARTMENT
# -------------------------------------------------
if 'Department' in df.columns:
    st.subheader("🏢 Attrition by Department")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Department', hue=target_column, data=df, palette='Set2', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# -------------------------------------------------
# JOB SATISFACTION
# -------------------------------------------------
if 'JobSatisfaction' in df.columns:
    st.subheader("📊 Attrition by Job Satisfaction")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='JobSatisfaction', hue=target_column, data=df, palette='pastel', ax=ax2)
    st.pyplot(fig2)

# -------------------------------------------------
# SALARY VS ATTRITION
# -------------------------------------------------
if 'MonthlyIncome' in df.columns:
    st.subheader("💰 Salary Distribution vs Attrition")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x=target_column, y='MonthlyIncome',
                data=df, palette=['#2ECC71', '#E74C3C'], ax=ax3)
    st.pyplot(fig3)

# -------------------------------------------------
# CORRELATION HEATMAP
# -------------------------------------------------
st.subheader("📈 Correlation Heatmap")
numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.5, ax=ax4)
st.pyplot(fig4)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
st.subheader("🤖 Model Training & Evaluation")

df_model = pd.get_dummies(df, drop_first=True)

X = df_model.drop(target_column, axis=1)
y = df_model[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training time
start_time = time.time()

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# -------------------------------------------------
# DISPLAY METRICS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(accuracy, 3))
col2.metric("Precision", round(precision, 3))
col3.metric("Recall", round(recall, 3))

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", round(f1, 3))
col5.metric("AUC-ROC", round(auc, 3))
col6.metric("Training Time (sec)", round(training_time, 3))

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# -------------------------------------------------
# ROC CURVE
# -------------------------------------------------
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr)
ax_roc.plot([0, 1], [0, 1], linestyle='--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
st.pyplot(fig_roc)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader("📌 Top 10 Important Features")

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

fig5, ax5 = plt.subplots()
ax5.barh(range(len(indices)), importances[indices])
ax5.set_yticks(range(len(indices)))
ax5.set_yticklabels(X.columns[indices])
st.pyplot(fig5)

# -------------------------------------------------
# HIGH RISK EMPLOYEES
# -------------------------------------------------
df_model['Attrition_Probability'] = model.predict_proba(X)[:, 1]
high_risk = df_model[df_model['Attrition_Probability'] > 0.7]

st.subheader("🚨 High Risk Employees (Probability > 70%)")
st.dataframe(high_risk.head(20))

# -------------------------------------------------
# BUSINESS IMPACT
# -------------------------------------------------
st.subheader("💼 Business Impact Analysis")

st.markdown("""
### 📌 Attrition Rate Reduction
Identifies high-risk employees early and enables:
- Targeted retention programs
- Compensation adjustments
- Career growth planning
- Work-life balance improvements

### 💰 Cost Savings
Reduces:
- Recruitment costs
- Onboarding expenses
- Training investment losses

### 📊 Strategic HR Decisions
Transforms HR from reactive hiring to proactive retention strategy.
""")

st.markdown("---")
st.markdown("Developed as an HR Analytics Machine Learning Project")