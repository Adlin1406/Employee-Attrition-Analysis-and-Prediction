# ============================================================
#  STREAMLIT DASHBOARD - Employee Attrition
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(page_title="Attrition Dashboard", layout="wide")
st.title("👥 Employee Attrition Analysis Dashboard")
st.markdown("---")


# ── Load & Prepare Data ────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_excel(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\Employee_Attrition_Analysis.xlsx")

    # Keep original for display
    df_display = df.copy()

    # Preprocess for model
    df_model = df.copy()
    df_model.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'],
                  inplace=True)
    df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])

    # Feature engineering
    df_model['EngagementScore'] = (
        df_model['JobSatisfaction'] + df_model['EnvironmentSatisfaction'] +
        df_model['RelationshipSatisfaction'] + df_model['WorkLifeBalance']) / 4
    df_model['LoyaltyScore'] = df_model['YearsAtCompany'] / (
        df_model['TotalWorkingYears'].replace(0, 1))

    return df_display, df_model


@st.cache_resource
def train_model(df_model):
    X = df_model.drop(columns=['Attrition'])
    y = df_model['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 3),
        'F1-Score': round(f1_score(y_test, y_pred), 3),
        'AUC-ROC' : round(roc_auc_score(y_test, y_prob), 3),
    }
    return model, X, metrics


df_display, df_model = load_and_prepare()
model, X_all, metrics = train_model(df_model)


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📈 Overview",
    "📊 EDA Charts",
    "🤖 Model Results",
    "⚠️ At-Risk Employees",
    "🔮 Predict Employee"
])


# ═══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "📈 Overview":
    st.subheader("📈 Overview")

    total   = len(df_display)
    left    = (df_display['Attrition'] == 'Yes').sum()
    stayed  = total - left
    rate    = round(left / total * 100, 1)
    avg_inc = round(df_display['MonthlyIncome'].mean(), 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees",  total)
    c2.metric("Active Employees", stayed)
    c3.metric("Attrited",         left)
    c4.metric("Attrition Rate",   f"{rate}%")
    c5.metric("Avg Monthly $",    f"${avg_inc:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Department")
        dept = df_display.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        dept.columns = ['Department', 'Attrition Rate %']
        fig, ax = plt.subplots()
        ax.bar(dept['Department'], dept['Attrition Rate %'],
               color=['steelblue', 'tomato', 'green'])
        ax.set_ylabel('Attrition Rate (%)')
        ax.set_title('Attrition by Department')
        plt.xticks(rotation=10)
        st.pyplot(fig)

    with col2:
        st.subheader("Attrition by OverTime")
        ot = df_display.groupby('OverTime')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        ot.columns = ['OverTime', 'Attrition Rate %']
        fig, ax = plt.subplots()
        ax.bar(ot['OverTime'], ot['Attrition Rate %'], color=['steelblue', 'tomato'])
        ax.set_ylabel('Attrition Rate (%)')
        ax.set_title('Attrition by OverTime')
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════
# PAGE 2: EDA CHARTS
# ═══════════════════════════════════════════════════════════
elif page == "📊 EDA Charts":
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots()
        df_display[df_display['Attrition'] == 'No']['Age'].hist(
            bins=20, alpha=0.6, color='steelblue', label='Stayed', ax=ax)
        df_display[df_display['Attrition'] == 'Yes']['Age'].hist(
            bins=20, alpha=0.6, color='tomato', label='Left', ax=ax)
        ax.set_xlabel('Age')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("**Monthly Income vs Attrition**")
        fig, ax = plt.subplots()
        df_display.boxplot(column='MonthlyIncome', by='Attrition', ax=ax)
        ax.set_title('')
        ax.set_xlabel('Attrition')
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Job Satisfaction vs Attrition**")
        sat = df_display.groupby('JobSatisfaction')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        sat.columns = ['JobSatisfaction', 'Attrition Rate %']
        fig, ax = plt.subplots()
        ax.bar(sat['JobSatisfaction'].astype(str), sat['Attrition Rate %'],
               color='tomato')
        ax.set_xlabel('Job Satisfaction (1=Low, 4=Very High)')
        ax.set_ylabel('Attrition Rate (%)')
        st.pyplot(fig)

    with col4:
        st.markdown("**Marital Status vs Attrition**")
        ms = df_display.groupby('MaritalStatus')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index()
        ms.columns = ['MaritalStatus', 'Attrition Rate %']
        fig, ax = plt.subplots()
        ax.bar(ms['MaritalStatus'], ms['Attrition Rate %'],
               color=['steelblue', 'tomato', 'green'])
        ax.set_ylabel('Attrition Rate (%)')
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    num_cols = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance',
                'YearsAtCompany', 'TotalWorkingYears', 'DistanceFromHome',
                'EnvironmentSatisfaction', 'RelationshipSatisfaction']
    df_num = df_display[num_cols].copy()
    df_num['Attrition'] = (df_display['Attrition'] == 'Yes').astype(int)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_num.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ═══════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.subheader("🤖 Random Forest Model Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",  metrics['Accuracy'])
    col2.metric("F1-Score",  metrics['F1-Score'])
    col3.metric("AUC-ROC",   metrics['AUC-ROC'])

    st.markdown("---")
    st.subheader("Top 15 Feature Importances")

    feat_imp = pd.Series(model.feature_importances_, index=X_all.columns)
    feat_imp = feat_imp.sort_values(ascending=False).head(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title('Feature Importances - Random Forest')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

    st.subheader("Feature Importance Table")
    feat_df = feat_imp.sort_values(ascending=False).reset_index()
    feat_df.columns = ['Feature', 'Importance']
    feat_df['Importance'] = feat_df['Importance'].round(4)
    st.dataframe(feat_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: AT-RISK EMPLOYEES
# ═══════════════════════════════════════════════════════════
elif page == "⚠️ At-Risk Employees":
    st.subheader("⚠️ At-Risk Employee List")

    # Predict for all
    probs = model.predict_proba(X_all)[:, 1]
    df_risk = df_display.copy()
    df_risk['Attrition_Prob_%'] = (probs * 100).round(1)
    df_risk['Risk_Level'] = pd.cut(
        df_risk['Attrition_Prob_%'],
        bins=[0, 30, 60, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    df_risk = df_risk.sort_values('Attrition_Prob_%', ascending=False)

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk",   (df_risk['Risk_Level'] == 'High Risk').sum())
    col2.metric("🟠 Medium Risk", (df_risk['Risk_Level'] == 'Medium Risk').sum())
    col3.metric("🟢 Low Risk",    (df_risk['Risk_Level'] == 'Low Risk').sum())

    # Filter
    risk_filter = st.selectbox("Filter by Risk Level",
                                ["All", "High Risk", "Medium Risk", "Low Risk"])
    if risk_filter != "All":
        df_risk = df_risk[df_risk['Risk_Level'] == risk_filter]

    st.dataframe(
        df_risk[['EmployeeNumber', 'Age', 'Department', 'JobRole',
                 'MonthlyIncome', 'OverTime', 'Attrition',
                 'Attrition_Prob_%', 'Risk_Level']].head(50),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════
# PAGE 5: PREDICT SINGLE EMPLOYEE
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Predict Employee":
    st.subheader("🔮 Predict Attrition for a Single Employee")
    st.markdown("Fill in the details and click **Predict**")

    col1, col2, col3 = st.columns(3)

    with col1:
        age           = st.slider("Age", 18, 65, 30)
        monthly_inc   = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
        overtime      = st.selectbox("OverTime", ["Yes", "No"])
        job_sat       = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

    with col2:
        years_co      = st.slider("Years at Company", 0, 40, 5)
        total_yrs     = st.slider("Total Working Years", 0, 40, 8)
        distance      = st.slider("Distance from Home", 1, 30, 5)
        wlb           = st.slider("Work-Life Balance (1-4)", 1, 4, 3)

    with col3:
        env_sat       = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        num_companies = st.slider("No. of Companies Worked", 0, 9, 2)
        yrs_promo     = st.slider("Years Since Last Promotion", 0, 15, 1)
        stock_opt     = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    if st.button("🔍 Predict", use_container_width=True):
        # Build input row with zeros, then fill in what we know
        row = pd.DataFrame(np.zeros((1, len(X_all.columns))), columns=X_all.columns)

        field_map = {
            'Age'                    : age,
            'MonthlyIncome'          : monthly_inc,
            'OverTime'               : 1 if overtime == "Yes" else 0,
            'JobSatisfaction'        : job_sat,
            'YearsAtCompany'         : years_co,
            'TotalWorkingYears'      : total_yrs,
            'DistanceFromHome'       : distance,
            'WorkLifeBalance'        : wlb,
            'EnvironmentSatisfaction': env_sat,
            'NumCompaniesWorked'     : num_companies,
            'YearsSinceLastPromotion': yrs_promo,
            'StockOptionLevel'       : stock_opt,
            'EngagementScore'        : (job_sat + env_sat + wlb + 3) / 4,
            'LoyaltyScore'           : years_co / max(total_yrs, 1),
        }

        for field, val in field_map.items():
            if field in row.columns:
                row[field] = val

        prob = model.predict_proba(row)[0][1] * 100
        risk = "🔴 High Risk" if prob >= 60 else "🟠 Medium Risk" if prob >= 30 else "🟢 Low Risk"

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Attrition Probability", f"{prob:.1f}%")
        r2.metric("Risk Level", risk)

        if prob >= 60:
            st.error("⚠️ This employee is at HIGH RISK of leaving. Consider immediate retention action.")
        elif prob >= 30:
            st.warning("This employee is at MEDIUM RISK. Monitor and engage proactively.")
        else:
            st.success("✅ This employee is at LOW RISK of leaving.")
