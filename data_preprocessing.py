"""
STEP 1: Data Preprocessing & Cleaning
======================================
- Loads raw Excel data
- Handles missing values, duplicates, constants
- Encodes categorical variables
- Saves cleaned data to outputs/
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH  = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\Employee_Attrition_Analysis.xlsx"
CLEAN_CSV_PATH = r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\Project_EmployeeAttrition\cleaned_data.csv"
os.makedirs("outputs", exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("\n── Shape ──────────────────────────────")
    print(df.shape)

    print("\n── Data Types ─────────────────────────")
    print(df.dtypes.value_counts())

    print("\n── Missing Values ─────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found.")

    print("\n── Duplicate Rows ─────────────────────")
    print(f"Duplicates: {df.duplicated().sum()}")

    print("\n── Target Distribution (Attrition) ────")
    print(df["Attrition"].value_counts())
    print(f"Attrition Rate: {(df['Attrition'] == 'Yes').mean():.2%}")


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no information (single unique value)."""
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        print(f"\n[INFO] Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Attrition: Yes → 1, No → 0."""
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


def encode_binary_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode simple binary string columns."""
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["OverTime", "Gender"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
            print(f"[INFO] Binary-encoded: {col}")
    return df


def encode_ordinal_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordered string categories to integers."""
    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    if "BusinessTravel" in df.columns:
        df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)
        print("[INFO] Ordinal-encoded: BusinessTravel")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining nominal categoricals."""
    nominal_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if nominal_cols:
        print(f"[INFO] One-hot encoding: {nominal_cols}")
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"[INFO] Removed {removed} duplicate rows.")
    return df


def main():
    df = load_data(RAW_DATA_PATH)
    inspect_data(df)

    df = handle_duplicates(df)
    df = remove_constant_columns(df)   # drops EmployeeCount, StandardHours, Over18
    df = encode_target(df)
    df = encode_binary_categoricals(df)
    df = encode_ordinal_categoricals(df)
    df = one_hot_encode(df)            # encodes Department, EducationField, JobRole, MaritalStatus

    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"\n[DONE] Cleaned data saved → {CLEAN_CSV_PATH}")
    print(f"       Final shape: {df.shape}")


if __name__ == "__main__":
    main()
