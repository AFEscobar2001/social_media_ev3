"""Servicio del modelo de referencia real (Cardiff NLP) vía `tweetnlp`.

Fuente: https://github.com/cardiffnlp/tweetnlp
Modelo por defecto: cardiffnlp/twitter-roberta-base-sentiment-latest
(RoBERTa fine-tuneado en ~124M tweets, 3 clases en inglés: Negative/Neutral/
Positive — el mismo espacio de etiquetas que `sentiment_label` de EV2).

"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from . import config

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")


def preprocess_tweet(text: str) -> str:
    """Normaliza menciones y URLs como espera twitter-roberta de Cardiff NLP."""
    text = _MENTION_RE.sub("@user", str(text))
    text = _URL_RE.sub("http", text)
    return text


def _normalize_label(raw: str) -> str:
    """Lleva cualquier etiqueta del modelo de referencia al espacio canónico.

    tweetnlp suele devolver 'positive'/'neutral'/'negative' en minúscula.
    Algunos checkpoints devuelven 'LABEL_0/1/2'. Normalizamos ambos casos.
    """
    raw_l = str(raw).strip().lower()
    mapping = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }
    return mapping.get(raw_l, raw_l.capitalize())


class ReferenceModel:
    """Wrapper perezoso del modelo de sentimiento de Cardiff NLP (tweetnlp)."""

    def __init__(self) -> None:
        self._model = None
        self._import_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import tweetnlp  # import perezoso: puede no estar instalado
        except ImportError as exc:  # pragma: no cover - depende del entorno
            self._import_error = str(exc)
            raise RuntimeError(
                "El modelo de referencia requiere la librería 'tweetnlp', que no "
                "está instalada. Instálala con: pip install tweetnlp "
                "(arrastra torch + transformers y descarga el modelo la 1ª vez). "
                "El endpoint /predict con el modelo EV2 funciona sin esta dependencia."
            ) from exc

        logger.info("Cargando modelo de referencia (tweetnlp): %s", config.REFERENCE_MODEL_ID)
        # tweetnlp.load_model('sentiment') usa twitter-roberta-base-sentiment-latest
        # por defecto; permitimos sobreescribir el checkpoint vía env.
        self._model = tweetnlp.load_model(
            config.REFERENCE_TASK, model_name=config.REFERENCE_MODEL_ID
        )
        logger.info("Modelo de referencia cargado.")

    def predict(self, texts: Sequence[str]) -> list[dict]:
        """Predice sentimiento con el modelo real para cada texto."""
        self._ensure_loaded()
        clean = [preprocess_tweet(t) for t in texts]

        # tweetnlp acepta str o list[str]; pedimos la distribución completa.
        raw = self._model.sentiment(clean, return_probability=True)
        if isinstance(raw, dict):  # un solo texto -> un solo dict
            raw = [raw]

        results: list[dict] = []
        for item in raw:
            label = _normalize_label(item.get("label", ""))
            prob = item.get("probability")
            if isinstance(prob, dict):
                probabilities = {
                    _normalize_label(k): round(float(v), 4) for k, v in prob.items()
                }
            elif isinstance(prob, (int, float)):
                probabilities = {label: round(float(prob), 4)}
            else:
                probabilities = None
            results.append({"label": label, "probabilities": probabilities})
        return results

    def info(self) -> dict:
        return {
            "name": "Cardiff NLP (tweetnlp)",
            "model_id": config.REFERENCE_MODEL_ID,
            "task": config.REFERENCE_TASK,
            "loaded": self.loaded,
            "source": "https://github.com/cardiffnlp/tweetnlp",
            "note": (
                "Modelo real entrenado en ~124M tweets. Se usa SOLO para evaluar "
                "el modelo EV2 contra texto real; nunca para reentrenarlo."
            ),
        }


reference_model = ReferenceModel()