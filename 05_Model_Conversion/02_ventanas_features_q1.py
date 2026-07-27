import pandas as pd
import numpy as np

INPUT_FILE = "dataset_raw_unido_q1.csv"
OUTPUT_FILE = "dataset_features_q1.csv"

WINDOW_SIZE = 50
STEP = 25

SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz"]
GROUP_COLS = ["subject", "session", "rep", "class_name", "label", "source_file"]

def extract_features(window_df):
    feats = {}
    for col in SIGNALS:
        x = window_df[col].values.astype(float)
        feats[f"{col}_mean"] = np.mean(x)
        feats[f"{col}_std"]  = np.std(x)
        feats[f"{col}_min"]  = np.min(x)
        feats[f"{col}_max"]  = np.max(x)
        feats[f"{col}_rms"]  = np.sqrt(np.mean(x**2))
        feats[f"{col}_ptp"]  = np.ptp(x)
    return feats

def main():
    df = pd.read_csv(INPUT_FILE)

    rows = []

    grouped = df.groupby(GROUP_COLS, sort=True)

    for keys, g in grouped:
        g = g.sort_values("time").reset_index(drop=True)
        n = len(g)

        for start in range(0, n - WINDOW_SIZE + 1, STEP):
            end = start + WINDOW_SIZE
            w = g.iloc[start:end]

            feats = extract_features(w)

            row = {
                "subject": keys[0],
                "session": keys[1],
                "rep": keys[2],
                "class_name": keys[3],
                "label": keys[4],
                "source_file": keys[5],
                "start_idx": start,
                "end_idx": end - 1,
                "start_time": w["time"].iloc[0],
                "end_time": w["time"].iloc[-1],
            }
            row.update(feats)
            rows.append(row)

    df_feat = pd.DataFrame(rows)
    df_feat.to_csv(OUTPUT_FILE, index=False)

    print("\n=== DATASET FEATURES Q1 ===")
    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"Ventanas totales: {len(df_feat)}")
    print("\n=== Conteo por clase ===")
    print(df_feat["class_name"].value_counts())
    print("\n=== Conteo por sujeto ===")
    print(df_feat["subject"].value_counts().sort_index())
    print("\n=== Conteo por sesion ===")
    print(df_feat["session"].value_counts().sort_index())

if __name__ == "__main__":
    main()