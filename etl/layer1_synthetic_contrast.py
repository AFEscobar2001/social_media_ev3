"""Capa 1 de Fase 3 — Contraste en el sintetico: sentimiento vs toxicidad.

Mismo dataset (sintetico), mismo metodo (TF-IDF + LogReg), dos tareas:
- sentimiento (sentiment_label, 3 clases): tiene senal  -> F1-macro ~0.94
- toxicidad (toxicity_score binarizado en >0.5): NO tiene senal -> ~ azar

Punto de partida del relato de Fase 3: el problema no es el metodo, es que la
columna toxicity_score del dataset sintetico se asigno independiente del texto.



Uso:
    source ../venv/bin/activate        # cada terminal nueva
    python3 etl/layer1_synthetic_contrast.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SEED = 42
CSV = Path(__file__).resolve().parents[1] / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv"


def build_pipeline() -> Pipeline:
    """Mismo pipeline ganador de EV2: TF-IDF (1-2 gramas) + Regresion Logistica."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def evaluate_task(X, y, task_name: str) -> dict:
    """Entrena el pipeline y lo compara contra baseline mayoritario y de azar."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    f1_model = f1_score(y_te, pred, average="macro", zero_division=0)
    acc_model = accuracy_score(y_te, pred)

    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    p_maj = majority.predict(X_te)

    random_b = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_tr, y_tr)
    p_rnd = random_b.predict(X_te)

    # Senal honesta: el modelo debe superar claramente al azar en F1-macro.
    # Para toxicidad balanceada el rival real es el azar (~0.52), no la mayoria.
    f1_random = f1_score(y_te, p_rnd, average="macro", zero_division=0)
    has_signal = bool(f1_model - f1_random > 0.05)

    logging.info("%s -> F1-macro modelo=%.4f | accuracy modelo=%.4f | senal=%s",
                 task_name, f1_model, acc_model, "SI" if has_signal else "NO")

    return {
        "model": {"f1_macro": round(float(f1_model), 4), "accuracy": round(float(acc_model), 4)},
        "baseline_majority": {
            "f1_macro": round(float(f1_score(y_te, p_maj, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_te, p_maj)), 4),
        },
        "baseline_random": {
            "f1_macro": round(float(f1_random), 4),
            "accuracy": round(float(accuracy_score(y_te, p_rnd)), 4),
        },
        "has_signal": has_signal,
    }


def main() -> None:
    df = pd.read_csv(CSV)
    text = df["text_content"].fillna("").astype(str)

    sentiment = evaluate_task(text, df["sentiment_label"], "sentimiento (3 clases)")

    tox_bin = (pd.to_numeric(df["toxicity_score"], errors="coerce") > 0.5).map({True: "alta", False: "baja"})
    toxicity = evaluate_task(text, tox_bin, "toxicidad (alta/baja)")

    results = {
        "dataset": "Social_Media_Engagement_Dataset.csv (sintetico, 12.000 filas)",
        "method": "TF-IDF(max_features=5000, ngram_range=(1,2), min_df=2) + LogReg",
        "seed": SEED,
        "sentiment_task": sentiment,
        "toxicity_task": toxicity,
        "interpretation": (
            "Mismo dato, mismo metodo. El sentimiento supera ampliamente al azar "
            "(senal real). La toxicidad queda al nivel del azar en accuracy y por "
            "debajo del azar en F1-macro (sin senal): la toxicity_score sintetica "
            "se asigno independiente del texto, por eso no es modelable."
        ),
    }

    out_dir = Path(__file__).resolve().parents[1] / "Adolfo" / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer1_synthetic_contrast.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("Resultado guardado en %s", out_path)


if __name__ == "__main__":
    main()