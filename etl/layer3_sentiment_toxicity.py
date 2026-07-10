#!/usr/bin/env python3
"""
Capa 3 — Correlación sentimiento × toxicidad dentro de Measuring Hate Speech
============================================================================
Mide la correlación de Pearson entre la columna continua de anotadores
('sentiment') y 'hate_speech_score' (ambas en escala IRT comparable), global
y por plataforma. Excluye el code 1 ('reference'), que no es una red social.

Regla de escalas (CRÍTICA): la Capa 3 usa la escala CONTINUA de anotadores de
MHS, nunca el sentiment_label categórico (Positive/Neutral/Negative) de las
Capas 1-2. No se mezclan.

Salida: Adolfo/results/metrics/layer3_sentiment_toxicity.json
        (con "_provenance": "medido")

Ejecutar en VS Code (requiere red a MongoDB Atlas):
    source ../venv/bin/activate
    python etl/layer3_sentiment_toxicity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.stats import pearsonr

# --- rutas ---
HERE = Path(__file__).resolve().parent          # etl/
ROOT = HERE.parent                              # social_media_ev3/
OUT = ROOT / "Adolfo/results/metrics/layer3_sentiment_toxicity.json"

PLATFORM_MAP = {0: "reddit", 1: "reference", 2: "twitter", 3: "youtube"}
PLATFORMS = ["reddit", "twitter", "youtube"]    # code 1 'reference' excluido

# Posibles nombres de columna en distintos dumps de MHS
SENT_CANDIDATES = ["sentiment", "sentiment_score", "annotator_sentiment"]
PLAT_CANDIDATES = ["platform", "platform_int", "platform_code"]
SCORE_COL = "hate_speech_score"


def connect():
    # En contexto stdin/heredoc, load_dotenv() sin ruta falla: pasar la ruta.
    load_dotenv(ROOT / ".env")
    uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME", "Ev3")
    if not uri:
        sys.exit("Falta MONGODB_URI en .env")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")                # falla rápido si no hay red
    return client[db_name]["measuringhatespeech"]


def detect(sample: dict, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in sample:
            return c
    sys.exit(f"No encontré la columna de {label}. Claves disponibles: {list(sample)}")


def fetch(collection) -> pd.DataFrame:
    one = collection.find_one()
    if one is None:
        sys.exit("La colección measuringhatespeech está vacía.")
    sent_col = detect(one, SENT_CANDIDATES, "sentimiento")
    plat_col = detect(one, PLAT_CANDIDATES, "plataforma")
    if SCORE_COL not in one:
        sys.exit(f"No encontré '{SCORE_COL}'. Claves: {list(one)}")

    proj = {"_id": 0, sent_col: 1, plat_col: 1, SCORE_COL: 1}
    docs = list(collection.find({}, proj))
    df = pd.DataFrame(docs).rename(
        columns={sent_col: "sentiment", plat_col: "platform_code"})
    df = df.dropna(subset=["sentiment", "platform_code", SCORE_COL])
    df["platform"] = df["platform_code"].astype(int).map(PLATFORM_MAP)
    df = df[df["platform"].isin(PLATFORMS)]      # excluye reference (1)
    print(f"Documentos usados (sin reference): {len(df):,}")
    return df


def corr_block(s: pd.Series, t: pd.Series) -> dict:
    r, p = pearsonr(s, t)
    return {"r": round(float(r), 4), "p_value": float(p), "n": int(len(s))}


def main():
    col = connect()
    df = fetch(col)

    overall = corr_block(df["sentiment"], df[SCORE_COL])
    by_platform = {
        p: corr_block(g["sentiment"], g[SCORE_COL])
        for p, g in df.groupby("platform")
    }
    rs = [v["r"] for v in by_platform.values()]

    result = {
        "source": "Ev3.measuringhatespeech",
        "_provenance": "medido",
        "method": "Pearson r entre 'sentiment' (escala continua de anotadores MHS) "
                  "y 'hate_speech_score' (escala IRT), global y por plataforma.",
        "scales_note": "Capa 3 usa la escala continua de anotadores; NO el "
                       "sentiment_label categorico de las Capas 1-2.",
        "platform_map_used": {str(k): v for k, v in PLATFORM_MAP.items()},
        "reference_code_excluded": 1,
        "overall": overall,
        "by_platform": by_platform,
        "business_reading": {
            "finding": ("Relacion negatividad-toxicidad consistente entre "
                        "plataformas" if (max(rs) - min(rs)) < 0.1
                        else "La fuerza de la relacion varia entre plataformas"),
            "consistent_across_platforms": (max(rs) - min(rs)) < 0.1,
            "range_r_min_max": [round(min(rs), 4), round(max(rs), 4)],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Escrito: {OUT}")
    print(f"  overall r = {overall['r']}  (p = {overall['p_value']:.2e}, n = {overall['n']:,})")
    for p in PLATFORMS:
        if p in by_platform:
            b = by_platform[p]
            print(f"  {p:<8} r = {b['r']}  (n = {b['n']:,})")


if __name__ == "__main__":
    main()