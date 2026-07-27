import re
from pathlib import Path
from io import StringIO
import pandas as pd

RAW_DIR = Path("data_raw")
OUT_FILE = "dataset_raw_unido_q1.csv"

LABEL_MAP = {
    "reposo": 0,
    "suave": 1,
    "brusco": 2,
}

FILENAME_RE = re.compile(
    r"^subj(?P<subject>\d+)_sess(?P<session>\d+)_(?P<class_name>reposo|suave|brusco)_rep(?P<rep>\d+)\.csv$",
    re.IGNORECASE
)

BASE_COLS = ["time", "ax", "ay", "az", "gx", "gy", "gz", "label"]

def detect_delimiter(lines):
    candidates = [",", ";", "\t"]
    best = ","
    best_score = -1

    for d in candidates:
        score = 0
        for line in lines[:30]:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("ets ", "rst:", "load:", "entry ")):
                continue
            score += line.count(d)
        if score > best_score:
            best_score = score
            best = d
    return best

def clean_one_file(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Nombre inválido: {path.name}")

    subject = int(m.group("subject"))
    session = int(m.group("session"))
    class_name = m.group("class_name").lower()
    rep = int(m.group("rep"))
    forced_label = LABEL_MAP[class_name]

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip().replace("\x00", "") for ln in text.splitlines()]

    delim = detect_delimiter(lines)
    clean_rows = []

    for line in lines:
        if not line:
            continue
        if line.startswith(("ets ", "rst:", "load:", "entry ")):
            continue

        parts = [p.strip() for p in line.split(delim)]

        # cabecera
        joined = ",".join(parts).lower().replace(" ", "")
        if "time" in joined and "ax" in joined and "ay" in joined and "az" in joined and "gx" in joined and "gy" in joined and "gz" in joined:
            continue

        # aceptar 7 u 8 columnas
        if len(parts) < 7:
            continue

        parts = parts[:8]

        try:
            values = [float(parts[i]) for i in range(7)]
        except ValueError:
            continue

        row = {
            "subject": subject,
            "session": session,
            "rep": rep,
            "class_name": class_name,
            "label": forced_label,
            "time": values[0],
            "ax": values[1],
            "ay": values[2],
            "az": values[3],
            "gx": values[4],
            "gy": values[5],
            "gz": values[6],
            "source_file": path.name,
        }
        clean_rows.append(row)

    if not clean_rows:
        raise ValueError(f"No quedaron datos útiles en {path.name}")

    return pd.DataFrame(clean_rows)

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No hay archivos CSV en data_raw/")

    dfs = []
    for f in files:
        df = clean_one_file(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data = data.sort_values(
        by=["subject", "session", "rep", "class_name", "time"]
    ).reset_index(drop=True)

    data.to_csv(OUT_FILE, index=False)

    print("\n=== DATASET RAW UNIDO ===")
    print(f"Archivo generado: {OUT_FILE}")
    print(f"Filas totales: {len(data)}")
    print("\n=== Conteo por clase ===")
    print(data["class_name"].value_counts())
    print("\n=== Conteo por sujeto ===")
    print(data["subject"].value_counts().sort_index())
    print("\n=== Conteo por sesion ===")
    print(data["session"].value_counts().sort_index())

if __name__ == "__main__":
    main()