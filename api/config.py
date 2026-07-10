from __future__ import annotations

import logging
import os
from pathlib import Path

# --- Reproducibilidad (consistente con el resto del proyecto EV2/EV3) --------
SEED = int(os.environ.get("SEED", "42"))


# api/config.py -> api/ -> raíz del repo
REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*candidates: Path) -> Path | None:
    """Devuelve la primera ruta candidata que exista, o None."""
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


# --- Modelo de EV2 (TF-IDF + Regresión Logística) ----------------------------

def resolve_ev2_model_path() -> Path | None:
    env_path = os.environ.get("EV2_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return _first_existing(
        REPO_ROOT / "Adolfo" / "models" / "trained_models" / "sentiment_logreg.joblib",
        REPO_ROOT / "Adolfo" / "models" / "trained_models" / "best_sentiment_model.joblib",
        REPO_ROOT / "models" / "trained_models" / "sentiment_logreg.joblib",
    )


def resolve_ev2_csv_path() -> Path | None:
    env_path = os.environ.get("EV2_CSV_PATH")
    if env_path:
        return Path(env_path)
    return _first_existing(
        REPO_ROOT / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv",
        REPO_ROOT / "data" / "raw" / "Social_Media_Engagement_Dataset.csv",
        REPO_ROOT / "Social_Media_Engagement_Dataset.csv",
    )


# --- Modelo de referencia (Cardiff NLP, vía tweetnlp) ------------------------

REFERENCE_TASK = os.environ.get("REFERENCE_TASK", "sentiment")
REFERENCE_MODEL_ID = os.environ.get(
    "REFERENCE_MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Todo (EV2 y referencia) se normaliza a estas tres clases para que las comparaciones sean válidas.
CANONICAL_LABELS = ["Negative", "Neutral", "Positive"]

# --- Metadatos de la API -----------------------------------------------------
API_TITLE = os.environ.get("API_TITLE", "API de Sentimiento — EV3 SCY1101")
API_VERSION = os.environ.get("API_VERSION", "1.0.0")
API_DESCRIPTION = (
    "Expone el modelo de sentimiento entrenado en EV2 (TF-IDF + Regresión "
    "Logística sobre datos sintéticos) y lo contrasta contra el modelo real de "
    "Cardiff NLP (tweetnlp / twitter-roberta-base-sentiment-latest) para "
    "validar si su desempeño se sostiene fuera del dataset sintético."
)


def setup_logging() -> None:
    """Configura logging profesional a consola (idempotente)."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )