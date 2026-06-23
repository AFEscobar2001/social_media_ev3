
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["Best purchase ever, highly recommend!"])


class BatchIn(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, examples=[["Best purchase ever!", "Not worth the money."]]
    )


class Prediction(BaseModel):
    label: str = Field(..., description="Negative / Neutral / Positive")
    probabilities: Optional[dict[str, float]] = None


class PredictOut(BaseModel):
    text: str
    model: str = "ev2"
    prediction: Prediction


class BatchPredictOut(BaseModel):
    model: str = "ev2"
    predictions: list[PredictOut]


class CompareOut(BaseModel):
    text: str
    ev2: Prediction = Field(..., description="Modelo de EV2 (datos sintéticos)")
    reference: Prediction = Field(..., description="Modelo real de Cardiff NLP")
    agree: bool = Field(..., description="¿Coinciden ambos modelos en la etiqueta?")


class BenchmarkItem(BaseModel):
    text: str
    true_label: Optional[str] = Field(
        None, description="Etiqueta real (opcional): Negative/Neutral/Positive"
    )


class BenchmarkIn(BaseModel):
    items: list[BenchmarkItem] = Field(..., min_length=1)


class BenchmarkOut(BaseModel):
    n_samples: int
    agreement_rate: float = Field(
        ..., description="Proporción de textos donde EV2 y la referencia coinciden"
    )
    f1_macro_ev2: Optional[float] = Field(
        None, description="F1-macro de EV2 vs true_label (si se entregaron etiquetas)"
    )
    f1_macro_reference: Optional[float] = Field(
        None, description="F1-macro del modelo de referencia vs true_label"
    )
    note: str


class HealthOut(BaseModel):
    status: str
    ev2_model_loaded: bool
    ev2_model_source: str
    reference_model_available: bool
    reference_model_loaded: bool