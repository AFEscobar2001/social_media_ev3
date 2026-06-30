"""Benchmark del modelo de sentimiento EV2 contra datos reales (no sinteticos).

Pregunta que responde: el F1-macro ~0.94 del modelo entrenado en el dataset
sintetico, es real o es un artefacto de como se generaron los datos?

Fuente externa: cardiffnlp/tweet_sentiment_multilingual, config "english".
Se usa SOLO la porcion en ingles para aislar el efecto "sintetico vs real" sin
contaminarlo con el idioma (el TF-IDF se entreno con vocabulario en ingles).
Texto real de redes sociales, etiquetado por humanos. No se mezcla con el
dataset sintetico de EV2: se usa solo para evaluar, nunca para entrenar.

Uso:
    python3 etl/benchmark_real_data.py

Requiere conexion a internet (descarga el dataset la primera vez y lo deja
cacheado en ~/.cache/huggingface).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Permite importar src/ de Adolfo sin reinstalar nada.
ADOLFO_SRC = Path(__file__).resolve().parents[1] / "Adolfo" / "src"
sys.path.insert(0, str(ADOLFO_SRC))

SEED = 42
HF_DATASET = "cardiffnlp/tweet_sentiment_multilingual"
HF_CONFIG = "english"  # solo ingles: comparacion justa de mismo idioma

# Mapeo esperado segun la tarjeta del dataset (0=negative,1=neutral,2=positive).
# Se valida en runtime contra ds.features antes de usarlo, por si cambia.
EXPECTED_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def load_synthetic_training_data() -> tuple[pd.Series, pd.Series]:
    """Carga el dataset sintetico de EV2 (texto + sentiment_label)."""
    csv_path = Path(__file__).resolve().parents[1] / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv"
    df = pd.read_csv(csv_path)
    return df["text_content"], df["sentiment_label"]


def train_logreg_on_synthetic() -> Pipeline:
    """Entrena el mismo pipeline ganador de EV2 (TF-IDF + LogReg) sobre datos sinteticos."""
    X_text, y = load_synthetic_training_data()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipeline.fit(X_text, y)
    logging.info("Pipeline LogReg entrenado sobre %s posts sinteticos.", len(X_text))
    return pipeline


def load_real_benchmark_data() -> pd.DataFrame:
    """Descarga el split de test de cardiffnlp/tweet_sentiment_multilingual (config english).

    Valida el esquema de labels en runtime (no se asume a ciegas) y mapea
    a Negative/Neutral/Positive para que sea comparable con sentiment_label.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, HF_CONFIG)
    test_split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]

    label_feature = test_split.features["label"]
    names = getattr(label_feature, "names", None)
    if names:
        # El dataset trae sus propios nombres de clase; los normalizamos a
        # Title Case para que calcen con sentiment_label (Negative/Neutral/Positive).
        runtime_map = {i: name.capitalize() for i, name in enumerate(names)}
    else:
        runtime_map = EXPECTED_LABEL_MAP

    logging.info("Mapeo de labels detectado en el dataset real: %s", runtime_map)

    df = test_split.to_pandas()
    df["sentiment_label_real"] = df["label"].map(runtime_map)

    unmapped = df["sentiment_label_real"].isna().sum()
    if unmapped:
        logging.warning(
            "%s filas con label sin mapeo conocido. Revisar EXPECTED_LABEL_MAP "
            "contra ds['test'].features['label'].names.", unmapped,
        )

    return df.dropna(subset=["sentiment_label_real", "text"])


def evaluate(pipeline: Pipeline, df_real: pd.DataFrame) -> dict:
    """Evalua el pipeline entrenado en sintetico contra el benchmark real."""
    X_real = df_real["text"]
    y_real = df_real["sentiment_label_real"]

    y_pred = pipeline.predict(X_real)
    f1_macro_real = f1_score(y_real, y_pred, average="macro", zero_division=0)

    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(X_real, y_real)
    y_dummy = dummy.predict(X_real)
    f1_macro_baseline = f1_score(y_real, y_dummy, average="macro", zero_division=0)

    report = classification_report(y_real, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_real, y_pred, labels=sorted(y_real.unique()))

    return {
        "benchmark_config": HF_CONFIG,
        "n_samples_real": int(len(df_real)),
        "f1_macro_synthetic_reported_ev2": 0.9366,
        "f1_macro_on_real_data": round(float(f1_macro_real), 4),
        "f1_macro_baseline_dummy": round(float(f1_macro_baseline), 4),
        "gap_synthetic_vs_real": round(0.9366 - float(f1_macro_real), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": sorted(y_real.unique()),
    }


def main() -> None:
    pipeline = train_logreg_on_synthetic()
    df_real = load_real_benchmark_data()
    results = evaluate(pipeline, df_real)

    out_dir = Path(__file__).resolve().parents[1] / "Adolfo" / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_real_data.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {k: v for k, v in results.items() if k not in ("classification_report", "confusion_matrix")},
        indent=2, ensure_ascii=False,
    ))
    logging.info("Resultado completo guardado en %s", out_path)


if __name__ == "__main__":
    main()
