"""API REST de sentimiento 

    GET  /health         estado del servicio y de los modelos
    GET  /model-info     metadatos de ambos modelos
    POST /predict        etiqueta un texto con el modelo EV2
    POST /predict/batch  etiqueta varios textos con el modelo EV2
    POST /compare        EV2 vs modelo real sobre el mismo texto
    POST /benchmark       evaluación batch: acuerdo + F1 contra etiquetas reales

Documentación interactiva auto-generada en /docs (Swagger) y /redoc.

Ejecutar localmente:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from sklearn.metrics import f1_score

from . import config, schemas
from .model_service import ev2_model
from .tweetnlp_service import reference_model

config.setup_logging()
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo EV2 al arrancar (el de referencia se carga perezoso)."""
    logger.info("Iniciando API: cargando modelo EV2...")
    try:
        ev2_model.load()
    except Exception as exc:  # log claro pero no impedimos el arranque
        logger.error("No se pudo cargar el modelo EV2 al inicio: %s", exc)
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan,
)


def _reference_available() -> bool:
    """¿Está instalada la librería tweetnlp? (sin descargar el modelo)."""
    import importlib.util

    return importlib.util.find_spec("tweetnlp") is not None


# --- Endpoints ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=schemas.HealthOut, tags=["meta"])
def health():
    """Chequeo de salud: estado del servicio y de cada modelo."""
    return schemas.HealthOut(
        status="ok" if ev2_model.ready else "degraded",
        ev2_model_loaded=ev2_model.ready,
        ev2_model_source=ev2_model.source,
        reference_model_available=_reference_available(),
        reference_model_loaded=reference_model.loaded,
    )


@app.get("/model-info", tags=["meta"])
def model_info():
    """Metadatos de ambos modelos (para documentación y defensa oral)."""
    return {"ev2": ev2_model.info(), "reference": reference_model.info()}


@app.post("/predict", response_model=schemas.PredictOut, tags=["ev2"])
def predict(payload: schemas.TextIn):
    """Etiqueta un texto con el modelo de EV2 (TF-IDF + Regresión Logística)."""
    if not ev2_model.ready:
        raise HTTPException(status_code=503, detail="Modelo EV2 no disponible.")
    pred = ev2_model.predict([payload.text])[0]
    return schemas.PredictOut(text=payload.text, prediction=schemas.Prediction(**pred))


@app.post("/predict/batch", response_model=schemas.BatchPredictOut, tags=["ev2"])
def predict_batch(payload: schemas.BatchIn):
    """Etiqueta varios textos de una vez con el modelo de EV2."""
    if not ev2_model.ready:
        raise HTTPException(status_code=503, detail="Modelo EV2 no disponible.")
    preds = ev2_model.predict(payload.texts)
    out = [
        schemas.PredictOut(text=t, prediction=schemas.Prediction(**p))
        for t, p in zip(payload.texts, preds)
    ]
    return schemas.BatchPredictOut(predictions=out)


@app.post("/compare", response_model=schemas.CompareOut, tags=["comparación"])
def compare(payload: schemas.TextIn):
    """Compara la etiqueta del modelo EV2 contra el modelo real de Cardiff NLP.

    Este es el corazón de la integración: prueba si el modelo entrenado en datos
    sintéticos coincide con un modelo real entrenado en tweets reales.
    """
    if not ev2_model.ready:
        raise HTTPException(status_code=503, detail="Modelo EV2 no disponible.")
    ev2_pred = ev2_model.predict([payload.text])[0]
    try:
        ref_pred = reference_model.predict([payload.text])[0]
    except RuntimeError as exc:
        # tweetnlp no instalado / sin internet: error claro, no 500 críptico.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return schemas.CompareOut(
        text=payload.text,
        ev2=schemas.Prediction(**ev2_pred),
        reference=schemas.Prediction(**ref_pred),
        agree=ev2_pred["label"] == ref_pred["label"],
    )


@app.post("/benchmark", response_model=schemas.BenchmarkOut, tags=["comparación"])
def benchmark(payload: schemas.BenchmarkIn):
    """Evalúa EV2 vs el modelo real sobre un lote de textos.

    Responde la pregunta abierta desde EV2: ¿el F1-macro ≈ 0.94 del modelo
    sintético es real o un artefacto del dato? Reporta:
      - agreement_rate: cuánto coinciden ambos modelos.
      - f1_macro_ev2 / f1_macro_reference: si se entregan etiquetas reales,
        el F1-macro de cada modelo contra esa verdad.
    """
    if not ev2_model.ready:
        raise HTTPException(status_code=503, detail="Modelo EV2 no disponible.")

    texts = [it.text for it in payload.items]
    ev2_labels = [p["label"] for p in ev2_model.predict(texts)]

    try:
        ref_labels = [p["label"] for p in reference_model.predict(texts)]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    agreements = sum(a == b for a, b in zip(ev2_labels, ref_labels))
    agreement_rate = round(agreements / len(texts), 4)

    f1_ev2 = f1_ref = None
    true_labels = [it.true_label for it in payload.items]
    if all(t is not None for t in true_labels):
        f1_ev2 = round(
            float(f1_score(true_labels, ev2_labels, average="macro", zero_division=0)), 4
        )
        f1_ref = round(
            float(f1_score(true_labels, ref_labels, average="macro", zero_division=0)), 4
        )
        note = (
            "F1-macro calculado contra las etiquetas reales entregadas. "
            "Un gap grande entre EV2 y la referencia sugiere que el alto F1 de "
            "EV2 es en parte artefacto del dataset sintético."
        )
    else:
        note = (
            "No se entregaron true_label en todos los items, así que solo se "
            "reporta agreement_rate (acuerdo entre ambos modelos)."
        )

    return schemas.BenchmarkOut(
        n_samples=len(texts),
        agreement_rate=agreement_rate,
        f1_macro_ev2=f1_ev2,
        f1_macro_reference=f1_ref,
        note=note,
    )