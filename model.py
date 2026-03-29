# ============================================================
# model.py
# Trains Logistic Regression & Random Forest, evaluates both,
# and returns the best performing model.
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(name, model, X_test, y_test):
    """
    Evaluates a trained model and prints a full report.
    Returns a dict of metrics.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  📊 Model: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {"name": name, "model": model, "f1": f1, "accuracy": acc}


def train_and_select_best(X_train, X_test, y_train, y_test):
    """
    Trains Logistic Regression and Random Forest.
    Evaluates both and returns the best model (by F1-score).
    """

    # ── 1. Define Models ─────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = []

    # ── 2. Train & Evaluate Each Model ───────────────────────
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    # ── 3. Select Best by F1-Score ───────────────────────────
    best = max(results, key=lambda x: x["f1"])
    print(f"\n🏆 Best Model: {best['name']}  (F1 = {best['f1']:.4f})")

    return best["model"], best["name"]
