from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config

logger = logging.getLogger(__name__)


class SentimentModelEV2:
    """Wrapper del pipeline de sentimiento de EV2 con carga perezosa."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self.source: str = "uninitialized"  # 'joblib' | 'retrained_from_csv'
        self.model_path: Path | None = None
        self.csv_path: Path | None = None
        self.classes_: list[str] = []
        self.n_train_samples: int | None = None

    # -- Construcción del modelo ------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        """Mismo pipeline ganador de EV2 (idéntico al benchmark del ETL)."""
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000, random_state=config.SEED)),
            ]
        )

    def _retrain_from_csv(self) -> Pipeline:
        csv_path = config.resolve_ev2_csv_path()
        if not csv_path:
            raise FileNotFoundError(
                "No se encontró un .joblib de EV2 ni el CSV para reentrenar. "
                "Define EV2_MODEL_PATH o EV2_CSV_PATH como variable de entorno."
            )
        logger.info("Reentrenando modelo EV2 desde CSV: %s", csv_path)
        df = pd.read_csv(csv_path)
        missing = {"text_content", "sentiment_label"} - set(df.columns)
        if missing:
            raise ValueError(f"El CSV no tiene las columnas requeridas: {missing}")

        pipeline = self._build_pipeline()
        pipeline.fit(df["text_content"].astype(str), df["sentiment_label"])
        self.source = "retrained_from_csv"
        self.csv_path = csv_path
        self.n_train_samples = int(len(df))
        return pipeline

    def load(self) -> None:
        """Carga el modelo: joblib si existe, si no reentrena desde CSV."""
        model_path = config.resolve_ev2_model_path()
        if model_path:
            logger.info("Cargando modelo EV2 serializado: %s", model_path)
            self._pipeline = joblib.load(model_path)
            self.source = "joblib"
            self.model_path = model_path
        else:
            self._pipeline = self._retrain_from_csv()

        self.classes_ = [str(c) for c in self._pipeline.classes_]
        logger.info(
            "Modelo EV2 listo (source=%s, clases=%s)", self.source, self.classes_
        )

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    # -- Inferencia --------------------------------------------------------
    def predict(self, texts: Sequence[str]) -> list[dict]:
        """Predice etiqueta y probabilidades por clase para cada texto."""
        if self._pipeline is None:
            raise RuntimeError("El modelo EV2 no está cargado. Llama a load() primero.")

        clean = [str(t) for t in texts]
        labels = self._pipeline.predict(clean)
        results: list[dict] = []

        if hasattr(self._pipeline, "predict_proba"):
            probas = self._pipeline.predict_proba(clean)
            for label, row in zip(labels, probas):
                results.append(
                    {
                        "label": str(label),
                        "probabilities": {
                            cls: round(float(p), 4)
                            for cls, p in zip(self.classes_, row)
                        },
                    }
                )
        else:
            for label in labels:
                results.append({"label": str(label), "probabilities": None})
        return results

    def info(self) -> dict:
        return {
            "name": "EV2 — TF-IDF + Regresión Logística",
            "task": "sentiment_label (Negative/Neutral/Positive)",
            "source": self.source,
            "model_path": str(self.model_path) if self.model_path else None,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "n_train_samples": self.n_train_samples,
            "classes": self.classes_,
            "seed": config.SEED,
            "note": (
                "Entrenado sobre datos sintéticos. F1-macro ≈ 0.94 reportado en "
                "EV2; usar /benchmark contra texto real para validar si se sostiene."
            ),
        }


# Instancia única reutilizada por la app (se carga en el lifespan de FastAPI).
ev2_model = SentimentModelEV2()