"""
config.py
=========
Configuracion central del proyecto: semilla global, rutas y deteccion de entorno.

Este modulo es la unica fuente de verdad para la reproducibilidad. Todos los
notebooks y modulos importan SEED y las rutas desde aqui, de modo que cambiar
un parametro en un solo lugar afecta a todo el pipeline.

Pensado para ejecutarse en Google Colab: la ruta raiz del proyecto apunta a
/content/proyecto_modelado por defecto y puede ajustarse con la variable de
entorno PROJECT_ROOT (util si se sube el proyecto a Google Drive).
"""

from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija la semilla en todas las fuentes de aleatoriedad usadas.

    Llamar una vez al inicio de cada notebook garantiza que los splits,
    la inicializacion de KMeans y los modelos den siempre el mismo resultado.

    Parameters
    ----------
    seed : int
        Valor de la semilla. Por defecto usa la constante global SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Ruta del proyecto en Google Colab
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """Resuelve la carpeta raiz del proyecto en Google Colab.

    Por defecto asume que el proyecto se subio a /content/proyecto_modelado.
    Si se subio a otra ruta (por ejemplo a Google Drive), basta con definir
    la variable de entorno PROJECT_ROOT antes de importar este modulo:

        import os
        os.environ["PROJECT_ROOT"] = "/content/drive/MyDrive/proyecto_modelado"

    Returns
    -------
    pathlib.Path
        Ruta absoluta a la raiz del proyecto.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path("/content/proyecto_modelado")


# ---------------------------------------------------------------------------
# Rutas del proyecto (derivadas de la raiz)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models" / "trained_models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
REPORTS_DIR: Path = RESULTS_DIR / "reports"

DATASET_PATH: Path = DATA_DIR / "Social_Media_Engagement_Dataset.csv"


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen (idempotente)."""
    for d in (DATA_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Columnas: que se usa y que se descarta (ver 01_exploratory_analysis.ipynb)
# ---------------------------------------------------------------------------
# Columna objetivo de la tarea supervisada de este insight.
TARGET_COLUMN: str = "sentiment_label"

# Unica feature de entrada para el modelado de sentimiento: el texto del post.
# Se clasifica desde el TEXTO y no desde sentiment_score para evitar fuga de
# datos (data leakage), ya que sentiment_label se deriva directamente del score.
TEXT_COLUMN: str = "text_content"

# Columnas descartadas para el modelado, con su justificacion. Se documentan
# aqui ademas de en el notebook 01 para que la decision quede en el codigo.
DROPPED_COLUMNS: dict[str, str] = {
    "post_id": "Identificador unico (12000/12000). No aporta senal, solo memoriza.",
    "user_id": "Identificador unico por fila. Mismo problema que post_id.",
    "sentiment_score": "FUGA DE DATOS: sentiment_label se deriva de este score "
                       "(Neg<-0.2<=Neu<=0.2<Pos). Usarlo daria ~100% trampa.",
    "timestamp": "Fecha sin patron temporal util para clasificar el tono del texto.",
    "language": "Metadato del usuario; el texto real esta en ingles. No describe el post.",
    "location": "Alta cardinalidad (33) sin relacion con el sentimiento del texto.",
    "brand_name": "Identidad de marca; introduce sesgo y no es senal de tono.",
    "product_name": "Alta cardinalidad (70). Memoriza productos, no generaliza.",
    "campaign_name": "Alta cardinalidad (23); nombre de campana no determina el tono.",
    "hashtags": "Etiquetas tematicas (#Food); ruido para clasificar sentimiento.",
    "mentions": "39% nulos; menciones a cuentas no indican tono.",
    "keywords": "Palabras ya contenidas en text_content; redundante con la feature.",
    "likes_count": "Metrica de engagement, no de tono. Ademas es post-publicacion.",
    "shares_count": "Idem likes: resultado posterior, no causa del sentimiento.",
    "comments_count": "Idem: metrica de resultado, no de contenido del texto.",
    "impressions": "Metrica de alcance; denominador del engagement, no del tono.",
    "engagement_rate": "Variable de otro insight; sin relacion con clasificar tono.",
    "buzz_change_rate": "Metrica de campana; sin senal para el tono del post.",
    "user_past_sentiment_avg": "Historial del usuario; el objetivo es clasificar ESTE post.",
    "user_engagement_growth": "Metrica de crecimiento del usuario; no describe el texto.",
    "day_of_week": "Dia de publicacion; sin relacion con el tono del mensaje.",
    "topic_category": "Util como variable de CRUCE en clustering, no como feature de tono.",
    "emotion_type": "Etiqueta alternativa correlacionada con el objetivo; se evita filtrar.",
    "campaign_phase": "Fase de campana; metadato sin senal de tono del texto.",
}
