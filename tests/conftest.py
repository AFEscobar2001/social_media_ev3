"""Configuración compartida de pytest para los tests de la API.

Apunta el modelo EV2 a un CSV de fixture pequeño (entrena en <1s) en lugar del
dataset completo de 12.000 filas, para que los tests sean rápidos y no dependan
de rutas del repo. tweetnlp NO se instala en CI, así que los endpoints que lo
usan se prueban en su modo de degradación elegante (503 con mensaje claro).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE_CSV = Path(__file__).parent / "fixtures_sentiment.csv"
# Debe definirse ANTES de importar la app (se lee al cargar el modelo).
os.environ["EV2_CSV_PATH"] = str(FIXTURE_CSV)
# Aseguramos que no se use un .joblib del repo durante los tests.
os.environ.pop("EV2_MODEL_PATH", None)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:  # dispara el lifespan (carga el modelo)
        yield test_client