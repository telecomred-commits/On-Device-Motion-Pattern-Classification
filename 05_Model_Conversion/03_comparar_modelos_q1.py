import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

INPUT_FILE = "dataset_features_q1.csv"
SUMMARY_OUT = "comparacion_modelos_q1_resumen.csv"
FOLDS_OUT = "comparacion_modelos_q1_folds.csv"

# Puedes cambiar a "subject" si tienes suficientes sujetos y quieres prueba más dura
GROUP_MODE = "subject_session"

def build_groups(df):
    if GROUP_MODE == "subject":
        return df["subject"].astype(str)
    elif GROUP_MODE == "subject_session":
        return df["subject"].astype(str) + "_sess" + df["session"].astype(str)
    else:
        raise ValueError("GROUP_MODE invalido")

def get_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=16,
                learning_rate_init=1e-3,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20
            ))
        ]),
    }

def main():
    df = pd.read_csv(INPUT_FILE)

    feature_cols = [c for c in df.columns if c not in [
        "subject", "session", "rep", "class_name", "label", "source_file",
        "start_idx", "end_idx", "start_time", "end_time"
    ]]

    X = df[feature_cols]
    y = df["label"]
    groups = build_groups(df)

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)

    if n_splits < 2:
        raise ValueError("No hay suficientes grupos para validación por grupos.")

    cv = GroupKFold(n_splits=n_splits)

    fold_rows = []
    models = get_models()

    for model_name, model in models.items():
        fold_num = 0
        for train_idx, test_idx in cv.split(X, y, groups):
            fold_num += 1

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t1

            fold_rows.append({
                "model": model_name,
                "fold": fold_num,
                "acc": accuracy_score(y_test, y_pred),
                "bal_acc": balanced_accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "train_time_s": train_time,
                "infer_time_s": infer_time,
                "n_test": len(test_idx),
            })

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(FOLDS_OUT, index=False)

    summary = (
        df_folds
        .groupby("model")
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            bal_acc_mean=("bal_acc", "mean"),
            bal_acc_std=("bal_acc", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            train_time_mean=("train_time_s", "mean"),
            infer_time_mean=("infer_time_s", "mean"),
        )
        .reset_index()
        .sort_values(by="f1_macro_mean", ascending=False)
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== COMPARACION Q1 ===")
    print(summary)
    print(f"\nArchivo generado: {SUMMARY_OUT}")
    print(f"Archivo generado: {FOLDS_OUT}")

    # entrenar y guardar el mejor modelo en TODO el dataset
    best_model_name = summary.iloc[0]["model"]
    best_model = get_models()[best_model_name]
    best_model.fit(X, y)

    joblib.dump(best_model, f"best_model_{best_model_name}_q1.joblib")
    print(f"Mejor modelo guardado: best_model_{best_model_name}_q1.joblib")

if __name__ == "__main__":
    main()