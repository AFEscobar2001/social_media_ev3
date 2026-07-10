"""
model_evaluation.py
===================
Funciones de evaluacion y comparacion para modelos supervisados y no
supervisados. Centraliza el calculo de metricas para que los notebooks
solo muestren resultados.
"""

from __future__ import annotations
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
)


# ---------------------------------------------------------------------------
# Supervisado
# ---------------------------------------------------------------------------
def evaluate_supervised(models: Dict[str, Pipeline],
                        X_train, y_train, X_test, y_test,
                        cv: int = 5) -> pd.DataFrame:
    """Evalua y compara modelos supervisados con validacion cruzada + test.

    Parameters
    ----------
    models : dict[str, Pipeline]
        Modelos a evaluar.
    X_train, y_train, X_test, y_test :
        Particiones de entrenamiento y prueba.
    cv : int
        Numero de folds para la validacion cruzada.

    Returns
    -------
    pandas.DataFrame
        Tabla comparativa: F1 CV (media y std), accuracy y F1 en test, y tiempo.
    """
    rows = {}
    for name, pipe in models.items():
        t0 = time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows[name] = {
            "cv_f1_mean": round(scores.mean(), 4),
            "cv_f1_std": round(scores.std(), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
            "train_time_s": round(time.time() - t0, 2),
        }
    return pd.DataFrame(rows).T.sort_values("test_f1_macro", ascending=False)


def detailed_report(pipe: Pipeline, X_test, y_test) -> str:
    """Devuelve el classification_report textual de un modelo ya entrenado."""
    pred = pipe.predict(X_test)
    return classification_report(y_test, pred, digits=3)


def get_confusion(pipe: Pipeline, X_test, y_test):
    """Devuelve (matriz_confusion, etiquetas_ordenadas) de un modelo entrenado."""
    pred = pipe.predict(X_test)
    labels = sorted(y_test.unique())
    return confusion_matrix(y_test, pred, labels=labels), labels


def top_terms_per_class(pipe: Pipeline, n: int = 10) -> Dict[str, list]:
    """Extrae las palabras mas influyentes por clase (solo modelos lineales).

    Funciona con LogisticRegression dentro de un pipeline con TF-IDF.

    Returns
    -------
    dict[str, list[str]]
        Clase -> lista de terminos mas asociados.
    """
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("top_terms_per_class requiere un modelo lineal con coef_.")
    names = np.array(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[-n:][::-1]
        out[str(cls)] = names[idx].tolist()
    return out


# ---------------------------------------------------------------------------
# No supervisado
# ---------------------------------------------------------------------------
def evaluate_clustering(labels, X_reduced, y_true) -> dict:
    """Calcula metricas internas y externas para un clustering.

    - silhouette: cohesion/separacion interna (no usa etiquetas).
    - ARI / NMI: coincidencia con una etiqueta de referencia (aqui, sentimiento).

    Parameters
    ----------
    labels : array-like
        Asignacion de cluster por muestra (-1 = ruido en DBSCAN).
    X_reduced : array-like
        Features reducidas usadas para agrupar.
    y_true : array-like
        Etiqueta de referencia para las metricas externas.

    Returns
    -------
    dict
        n_clusters, silhouette, ARI y NMI.
    """
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1 and mask.sum() > n_clusters:
        sil = silhouette_score(np.asarray(X_reduced)[mask], labels[mask])
    else:
        sil = float("nan")
    return {
        "n_clusters": n_clusters,
        "silhouette": round(sil, 4),
        "ARI_vs_sentiment": round(adjusted_rand_score(y_true, labels), 4),
        "NMI_vs_sentiment": round(normalized_mutual_info_score(y_true, labels), 4),
    }
