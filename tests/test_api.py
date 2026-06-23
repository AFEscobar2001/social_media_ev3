"""Tests automatizados de la API REST de sentimiento (EV3 SCY1101).

Cubren:
  - salud y metadatos del servicio,
  - predicción individual y batch con el modelo EV2,
  - validación de entrada (Pydantic / FastAPI),
  - degradación elegante de /compare y /benchmark cuando tweetnlp no está.

Se ejecutan con:  pytest -v
No requieren internet ni tweetnlp instalado.
"""
from __future__ import annotations

import importlib.util

CANONICAL = {"Negative", "Neutral", "Positive"}
TWEETNLP_INSTALLED = importlib.util.find_spec("tweetnlp") is not None


# --- meta --------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ev2_model_loaded"] is True
    assert body["ev2_model_source"] == "retrained_from_csv"


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "ev2" in body and "reference" in body
    assert set(body["ev2"]["classes"]) <= CANONICAL


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 308)
    assert resp.headers["location"] == "/docs"


# --- predicción EV2 ----------------------------------------------------------
def test_predict_positive(client):
    resp = client.post("/predict", json={"text": "Best purchase ever, highly recommend!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"]["label"] in CANONICAL
    # las probabilidades deben sumar ~1
    probs = body["prediction"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_predict_negative(client):
    resp = client.post("/predict", json={"text": "Not worth the money, very disappointed."})
    assert resp.status_code == 200
    assert resp.json()["prediction"]["label"] in CANONICAL


def test_predict_batch(client):
    payload = {"texts": ["Best purchase ever!", "Worst experience, garbage."]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all(p["prediction"]["label"] in CANONICAL for p in preds)


# --- validación de entrada ---------------------------------------------------
def test_predict_rejects_empty_text(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # Pydantic min_length


def test_predict_rejects_missing_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


# --- comparación / referencia (degradación elegante sin tweetnlp) ------------
def test_compare_without_tweetnlp_returns_503(client):
    if TWEETNLP_INSTALLED:
        import pytest

        pytest.skip("tweetnlp instalado: este test valida el modo sin la dependencia")
    resp = client.post("/compare", json={"text": "Great product, love it!"})
    assert resp.status_code == 503
    assert "tweetnlp" in resp.json()["detail"].lower()


def test_benchmark_without_tweetnlp_returns_503(client):
    if TWEETNLP_INSTALLED:
        import pytest

        pytest.skip("tweetnlp instalado: este test valida el modo sin la dependencia")
    payload = {"items": [{"text": "Great product!", "true_label": "Positive"}]}
    resp = client.post("/benchmark", json=payload)
    assert resp.status_code == 503