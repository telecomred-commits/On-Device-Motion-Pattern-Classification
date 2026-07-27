import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

INPUT_FILE = "dataset_features_q1.csv"

MODEL_FILE = "modelo_mlp_q1.joblib"
SCALER_FILE = "scaler_mlp_q1.joblib"
CONF_MATRIX_FILE = "confusion_matrix_q1.png"

RANDOM_STATE = 42

def main():
    # =========================
    # 1. Cargar dataset
    # =========================
    df = pd.read_csv(INPUT_FILE)

    # Columnas que NO deben entrar al modelo
    drop_cols = [
        "subject",
        "session",
        "rep",
        "class_name",
        "label",
        "source_file",
        "start_idx",
        "end_idx",
        "start_time",
        "end_time"
    ]

    X = df.drop(columns=drop_cols)
    y = df["label"]

    print("\n=== DATASET CARGADO ===")
    print("Filas:", len(df))
    print("Columnas de entrada:", X.shape[1])
    print("Clases:")
    print(y.value_counts().sort_index())

    # =========================
    # 2. División train / val / test
    # =========================
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y
    )

    val_relative_size = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    print("\n=== DIVISIÓN ===")
    print("Train:", len(X_train))
    print("Val  :", len(X_val))
    print("Test :", len(X_test))

    # =========================
    # 3. Escalado
    # =========================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # =========================
    # 4. Modelo MLP
    # =========================
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20
    )

    model.fit(X_train_s, y_train)

    # =========================
    # 5. Evaluación
    # =========================
    y_train_pred = model.predict(X_train_s)
    y_val_pred   = model.predict(X_val_s)
    y_test_pred  = model.predict(X_test_s)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_val   = accuracy_score(y_val, y_val_pred)
    acc_test  = accuracy_score(y_test, y_test_pred)

    print("\n=== ACCURACY ===")
    print(f"Train: {acc_train:.4f}")
    print(f"Val  : {acc_val:.4f}")
    print(f"Test : {acc_test:.4f}")

    print("\n=== REPORTE TEST ===")
    print(classification_report(y_test, y_test_pred, digits=4))

    # =========================
    # 6. Matriz de confusión
    # =========================
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["reposo", "suave", "brusco"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de confusión - Test Q1")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_FILE, dpi=200)
    plt.close()

    print(f"\nMatriz de confusión guardada en: {CONF_MATRIX_FILE}")

    # =========================
    # 7. Guardar modelo y scaler
    # =========================
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"Modelo guardado en: {MODEL_FILE}")
    print(f"Scaler guardado en: {SCALER_FILE}")

if __name__ == "__main__":
    main()