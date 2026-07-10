"""
model_training.py
=================
Definicion y entrenamiento de modelos supervisados (clasificacion de
sentimiento) y no supervisados (clustering de temas) sobre el texto.

Cada constructor devuelve un Pipeline de scikit-learn para que la
vectorizacion TF-IDF viaje siempre junto al estimador. Esto evita fugas de
datos entre train/test y hace el entregable 100% reproducible.
"""

from __future__ import annotations
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from config import SEED


# ---------------------------------------------------------------------------
# Supervisado: clasificacion de sentimiento desde el texto
# ---------------------------------------------------------------------------
def build_tfidf(max_features: int = 5000, ngram_range=(1, 2)) -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF compartido por los modelos supervisados."""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)


def get_supervised_models() -> Dict[str, Pipeline]:
    """Devuelve los pipelines supervisados a comparar.

    Se eligen tres familias distintas a proposito:
    - LogisticRegression: lineal, rapida e interpretable (referencia fuerte en texto).
    - MultinomialNB: probabilistico, muy rapido, clasico para texto.
    - RandomForest: no lineal basado en arboles, para contrastar capacidad.

    Returns
    -------
    dict[str, Pipeline]
        Nombre del modelo -> pipeline (TF-IDF + clasificador).
    """
    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=SEED),
        "NaiveBayes": MultinomialNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1
        ),
    }
    return {
        name: Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
        for name, clf in models.items()
    }


# ---------------------------------------------------------------------------
# No supervisado: descubrimiento de temas en el texto
# ---------------------------------------------------------------------------
def build_text_features(max_features: int = 3000, n_components: int = 50):
    """Crea el pipeline de features para clustering: TF-IDF + TruncatedSVD.

    TruncatedSVD es el equivalente de PCA para matrices dispersas (texto).
    Reduce la dimensionalidad antes de agrupar, lo que acelera y estabiliza
    los algoritmos de clustering.

    Parameters
    ----------
    max_features : int
        Vocabulario maximo del TF-IDF.
    n_components : int
        Dimensiones tras la reduccion.

    Returns
    -------
    Pipeline
        Pipeline (TF-IDF -> TruncatedSVD) sin estimador final.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features,
                                  ngram_range=(1, 2), stop_words="english")),
        ("svd", TruncatedSVD(n_components=n_components, random_state=SEED)),
    ])


def get_unsupervised_models(n_clusters: int = 6) -> dict:
    """Devuelve los modelos de clustering a comparar.

    - KMeans: centroides, rapido, escala a todo el dataset.
    - Agglomerative: jerarquico, util para validar estructura (no escala bien).
    - DBSCAN: por densidad, decide el numero de grupos por si mismo.

    Parameters
    ----------
    n_clusters : int
        Numero de grupos para los metodos que lo requieren.

    Returns
    -------
    dict
        Nombre -> estimador de clustering ya configurado.
    """
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=10),
    }
