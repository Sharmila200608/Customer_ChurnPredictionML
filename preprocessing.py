# ============================================================
# preprocessing.py
# Handles all data loading, cleaning, and feature engineering
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(filepath="data/churn.csv"):
    """
    Loads the dataset, cleans it, encodes features,
    and returns train/test splits ready for modeling.
    """

    # ── 1. Load Data ────────────────────────────────────────
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── 2. Drop Irrelevant Column ────────────────────────────
    # customerID is just an identifier — not useful for prediction
    df.drop(columns=["customerID"], inplace=True)

    # ── 3. Fix TotalCharges (may be stored as string) ────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ── 4. Handle Missing Values ─────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        print(f"   Missing values found:\n{missing[missing > 0]}")
    # Fill numeric nulls with median (robust to outliers)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # Fill any remaining numeric NaNs with column medians
    df = df.fillna(df.median(numeric_only=True))
    print("✅ Missing values handled.")

    # ── 5. Encode Target Column ──────────────────────────────
    # Churn: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ── 6. Encode Categorical Features ──────────────────────
    # Identify all object (string) columns except the target
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"   Encoding {len(categorical_cols)} categorical columns...")

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    print("✅ Categorical encoding done.")

    # ── 7. Split Features and Target ────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # ── 8. Train / Test Split ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 9. Scale Features (helps Logistic Regression converge) ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"✅ Data split: {X_train.shape[0]} train | {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test, X.columns.tolist()
