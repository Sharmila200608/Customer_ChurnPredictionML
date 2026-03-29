# ============================================================
# train.py
# Orchestrates the full training pipeline:
#   1. Preprocess data
#   2. Train & evaluate models
#   3. Save the best model to model.pkl
# ============================================================

import pickle
from preprocessing import load_and_preprocess
from model import train_and_select_best


def main():
    print("\n🚀 Starting Customer Churn Prediction Training...\n")

    # ── Step 1: Load and preprocess data ─────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        filepath="data/churn.csv"
    )

    # ── Step 2: Train models and get the best one ─────────────
    best_model, best_name = train_and_select_best(
        X_train, X_test, y_train, y_test
    )

    # ── Step 3: Save the best model using pickle ──────────────
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "features": feature_names}, f)

    print(f"\n✅ Model saved to '{model_path}'")
    print(f"   Model type  : {best_name}")
    print(f"   Features    : {len(feature_names)} input columns")
    print("\n🎉 Training complete! Run 'streamlit run app.py' to launch the app.\n")


if __name__ == "__main__":
    main()
