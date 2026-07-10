"""
data_preprocessing.py
======================
Funciones de carga, validacion y preparacion de datos.

Mantiene toda la logica de lectura del CSV y la separacion de
features/objetivo en un solo lugar, de modo que los notebooks solo orquesten.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import DATASET_PATH, TEXT_COLUMN, TARGET_COLUMN


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset de redes sociales desde un CSV.

    Parameters
    ----------
    path : Path or str
        Ruta al archivo CSV. Por defecto la del proyecto (config.DATASET_PATH).

    Returns
    -------
    pandas.DataFrame
        El dataset completo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe (mensaje guia para Colab).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {path}.\n"
            "En Google Colab, sube el proyecto o monta Drive y verifica "
            "que data/Social_Media_Engagement_Dataset.csv exista, "
            "o define la variable de entorno PROJECT_ROOT."
        )
    return pd.read_csv(path)


def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un reporte rapido de calidad de datos.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset a inspeccionar.

    Returns
    -------
    pandas.DataFrame
        Tabla con tipo, nulos, % nulos y cardinalidad por columna.
    """
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_nulos": df.isnull().sum(),
        "pct_nulos": (df.isnull().mean() * 100).round(2),
        "n_unicos": df.nunique(),
    })
    return rep.sort_values("pct_nulos", ascending=False)


def get_text_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Extrae la feature de texto y el objetivo para el modelado supervisado.

    Solo se usa el texto del post como entrada. El resto de columnas se
    descarta de forma deliberada (ver config.DROPPED_COLUMNS y notebook 01).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo.

    Returns
    -------
    (X, y) : tuple of pandas.Series
        X = texto del post (sin nulos, reemplazados por cadena vacia).
        y = etiqueta de sentimiento.

    Raises
    ------
    KeyError
        Si faltan las columnas esperadas en el DataFrame.
    """
    for col in (TEXT_COLUMN, TARGET_COLUMN):
        if col not in df.columns:
            raise KeyError(f"Falta la columna requerida '{col}' en el dataset.")
    X = df[TEXT_COLUMN].fillna("")
    y = df[TARGET_COLUMN]
    return X, y
