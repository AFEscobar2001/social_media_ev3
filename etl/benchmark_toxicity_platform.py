"""Benchmark de toxicidad por plataforma sobre datos reales (Measuring Hate Speech).

Pregunta de investigacion:
    Es Twitter mas toxico que Reddit y YouTube, medido con la misma regla
    (hate_speech_score) y con datos reales etiquetados por humanos?

Fuente: coleccion `measuringhatespeech` en MongoDB Atlas (corpus Measuring
Hate Speech, UC Berkeley D-Lab). Cada comentario trae un hate_speech_score
continuo (mayor = mas toxico) y un codigo de plataforma.

Mapeo de plataforma (confirmado con el D-Lab):
    0 = reddit
    1 = reference   <- NO es red social, texto de control. SE EXCLUYE SIEMPRE.
    2 = twitter
    3 = youtube

Metodo:
    - Deduplicar a un comentario por comment_id (hate_speech_score y platform
      son a nivel de comentario; el corpus trae una fila por anotador).
    - Excluir el codigo 1 (reference).
    - Estadistica descriptiva de hate_speech_score por plataforma.
    - Kruskal-Wallis (no parametrico) para decidir si las diferencias entre
      plataformas son significativas: la conclusion depende del p-valor.
    - Comparaciones pareadas (Mann-Whitney U) con correccion de Holm para
      ubicar que plataforma difiere de cual.

Uso:
    source ../venv/bin/activate
    python3 etl/benchmark_toxicity_platform.py

Salida:
    Adolfo/results/metrics/toxicity_by_platform.json
"""
from __future__ import annotations

import json
import logging
import os
from itertools import combinations
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.stats import kruskal, mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
load_dotenv()

DB_NAME = os.environ.get("MONGODB_DB_NAME", "Ev3")
TOX_COLLECTION = os.environ.get("MONGODB_TOX_COLLECTION", "measuringhatespeech")

# Mapeo confirmado con el D-Lab. El codigo 1 (reference) se excluye SIEMPRE.
PLATFORM_MAP = {0: "reddit", 1: "reference", 2: "twitter", 3: "youtube"}
PLATFORMS_FOR_ANALYSIS = ["reddit", "twitter", "youtube"]

SCORE_COL = "hate_speech_score"
ID_COL = "comment_id"
PLATFORM_COL = "platform"


def get_mongo_client(timeout_ms: int = 8000) -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MONGODB_URI no esta definida. Crea social_media_ev3/.env "
            "(ver .env.example) con MONGODB_URI antes de correr este script."
        )
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def load_toxicity_corpus() -> pd.DataFrame:
    """Carga comment_id, platform y hate_speech_score desde la coleccion real.

    Proyecta solo las columnas necesarias para no traer el corpus completo
    (cada documento tiene decenas de campos de anotacion que no se usan aqui).
    """
    client = get_mongo_client()
    coll = client[DB_NAME][TOX_COLLECTION]

    total = coll.count_documents({})
    logging.info("Coleccion %s.%s: %s documentos en total.", DB_NAME, TOX_COLLECTION, total)
    if total == 0:
        raise ValueError(
            f"La coleccion {DB_NAME}.{TOX_COLLECTION} esta vacia. "
            "Corre primero la Fase 1 (load_mhs_to_mongo.py --drop --dedup)."
        )

    projection = {"_id": 0, ID_COL: 1, PLATFORM_COL: 1, SCORE_COL: 1}
    documents = list(coll.find({}, projection))
    df = pd.DataFrame(documents)

    missing = {ID_COL, PLATFORM_COL, SCORE_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas esperadas en la coleccion: {sorted(missing)}. "
            f"Columnas presentes: {sorted(df.columns)}."
        )
    return df


def deduplicate_by_comment(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa a un comentario por comment_id.

    hate_speech_score y platform son a nivel de comentario (se repiten por
    cada anotador), asi que tomar la primera fila por comment_id es correcto.
    Es defensivo: si la coleccion ya viene deduplicada desde la Fase 1, esto
    no cambia nada; si no, deja un comentario por fila igualmente.
    """
    before = len(df)
    df = df.dropna(subset=[ID_COL, PLATFORM_COL, SCORE_COL])
    df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)
    after = len(df)
    logging.info("Deduplicado por %s: %s -> %s filas.", ID_COL, before, after)
    return df


def map_platforms(df: pd.DataFrame) -> pd.DataFrame:
    """Traduce la plataforma a nombre y excluye reference (codigo 1).

    Robusto a las dos formas en que puede venir el campo `platform` desde
    Atlas: como codigo numerico (0..3) o como texto ('reddit', 'twitter'...).
    """
    df = df.copy()
    numeric = pd.to_numeric(df[PLATFORM_COL], errors="coerce")

    if numeric.notna().mean() > 0.5:
        observed = sorted(numeric.dropna().unique().tolist())
        logging.info("Plataforma detectada como codigo numerico: %s", observed)
        unexpected = set(observed) - set(PLATFORM_MAP)
        if unexpected:
            logging.warning("Codigos de plataforma no esperados (se ignoran): %s", unexpected)
        df["platform_name"] = numeric.map(PLATFORM_MAP)
    else:
        logging.info("Plataforma detectada como texto; se normaliza a minusculas.")
        df["platform_name"] = df[PLATFORM_COL].astype(str).str.strip().str.lower()

    n_reference = int((df["platform_name"] == "reference").sum())
    logging.info("Excluyendo %s filas de 'reference'.", n_reference)

    df = df[df["platform_name"].isin(PLATFORMS_FOR_ANALYSIS)].reset_index(drop=True)
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df = df.dropna(subset=[SCORE_COL]).reset_index(drop=True)
    return df


def describe_by_platform(df: pd.DataFrame) -> dict:
    stats = {}
    for name in PLATFORMS_FOR_ANALYSIS:
        scores = df.loc[df["platform_name"] == name, SCORE_COL]
        stats[name] = {
            "n": int(scores.shape[0]),
            "mean": round(float(scores.mean()), 4),
            "median": round(float(scores.median()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
        }
    return stats


def run_tests(df: pd.DataFrame) -> dict:
    groups = {
        name: df.loc[df["platform_name"] == name, SCORE_COL].to_numpy()
        for name in PLATFORMS_FOR_ANALYSIS
    }

    h_stat, p_global = kruskal(*groups.values())

    raw = []
    for a, b in combinations(PLATFORMS_FOR_ANALYSIS, 2):
        u_stat, p_uncorr = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        raw.append((a, b, u_stat, p_uncorr))

    # Correccion de Holm sobre los p-valores pareados (paso descendente).
    ordered = sorted(raw, key=lambda x: x[3])
    m = len(ordered)
    holm = {}
    running_max = 0.0
    for i, (a, b, u_stat, p_uncorr) in enumerate(ordered):
        adj = min(1.0, (m - i) * p_uncorr)
        running_max = max(running_max, adj)
        holm[(a, b)] = running_max

    pairwise = []
    for a, b, u_stat, p_uncorr in raw:
        pairwise.append({
            "pair": f"{a}_vs_{b}",
            "u_statistic": round(float(u_stat), 4),
            "p_value_uncorrected": float(f"{p_uncorr:.3e}"),
            "p_value_holm": float(f"{holm[(a, b)]:.3e}"),
            "significant_holm_0_05": bool(holm[(a, b)] < 0.05),
        })

    return {
        "kruskal_wallis": {
            "h_statistic": round(float(h_stat), 4),
            "p_value": float(f"{p_global:.3e}"),
            "significant_0_05": bool(p_global < 0.05),
        },
        "pairwise_mann_whitney": pairwise,
    }


def build_business_reading(stats: dict, tests: dict) -> dict:
    ranking = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    ranking_names = [name for name, _ in ranking]

    return {
        "ranking_most_to_least_toxic": ranking_names,
        "most_toxic_platform": ranking_names[0],
        "least_toxic_platform": ranking_names[-1],
        "all_means_negative": bool(all(v["mean"] < 0 for v in stats.values())),
        "hypothesis_twitter_most_toxic_confirmed": bool(ranking_names[0] == "twitter"),
        "global_difference_significant": tests["kruskal_wallis"]["significant_0_05"],
    }


def main() -> None:
    df = load_toxicity_corpus()
    df = deduplicate_by_comment(df)
    df = map_platforms(df)

    if df.empty:
        raise ValueError("No quedaron filas tras el mapeo de plataformas. Revisa el campo platform.")

    stats = describe_by_platform(df)
    tests = run_tests(df)
    reading = build_business_reading(stats, tests)

    results = {
        "source": f"{DB_NAME}.{TOX_COLLECTION}",
        "score_column": SCORE_COL,
        "score_note": "hate_speech_score: mayor = mas toxico (escala IRT, puede ser negativa)",
        "platform_map_used": {str(k): v for k, v in PLATFORM_MAP.items()},
        "reference_code_excluded": 1,
        "n_comments_analyzed": int(df.shape[0]),
        "descriptive_by_platform": stats,
        "statistical_tests": tests,
        "business_reading": reading,
    }

    out_path = Path(__file__).resolve().parents[1] / "Adolfo" / "results" / "metrics" / "toxicity_by_platform.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("Resultado guardado en %s", out_path)


if __name__ == "__main__":
    main()