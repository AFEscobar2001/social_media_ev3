"""
hyperparameter_tuning.py
========================
Funciones para optimizar hiperparametros con GridSearchCV y
RandomizedSearchCV, documentando el proceso y el impacto.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import SEED


def tune_logreg_grid(X_train, y_train, cv: int = 5) -> GridSearchCV:
    """Optimiza la Regresion Logistica (mejor supervisado) con GridSearchCV.

    Explora la fuerza de regularizacion C, el rango de n-gramas del TF-IDF
    y el tamano del vocabulario. Busqueda exhaustiva sobre una rejilla acotada.

    Parameters
    ----------
    X_train, y_train :
        Datos de entrenamiento.
    cv : int
        Folds de validacion cruzada.

    Returns
    -------
    GridSearchCV
        Objeto ya ajustado; usar .best_params_ y .best_score_.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    grid = {
        "tfidf__max_features": [3000, 5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro",
                          n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search


def tune_logreg_random(X_train, y_train, n_iter: int = 15,
                       cv: int = 5) -> RandomizedSearchCV:
    """Optimiza la Regresion Logistica con RandomizedSearchCV.

    Muestrea C de una distribucion log-uniforme (mas eficiente que la rejilla
    cuando el espacio es amplio). Util para comparar contra GridSearchCV.

    Returns
    -------
    RandomizedSearchCV
        Objeto ya ajustado.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    dist = {
        "tfidf__max_features": [3000, 5000, 8000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": loguniform(1e-2, 1e2),
    }
    search = RandomizedSearchCV(pipe, dist, n_iter=n_iter, cv=cv,
                                scoring="f1_macro", n_jobs=-1,
                                random_state=SEED, verbose=0)
    search.fit(X_train, y_train)
    return search


def summarize_search(search) -> dict:
    """Resume un objeto de busqueda en un diccionario legible."""
    return {
        "best_score_cv_f1": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "n_candidates": len(search.cv_results_["params"]),
    }
