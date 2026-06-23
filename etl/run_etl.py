"""Pipeline ETL principal para la Evaluación Parcial 3 SCY1101.

Integra las fuentes disponibles del proyecto y deja preparado el punto de
entrada para una API o base de datos externa. El objetivo es generar un dataset
final único para dashboard, análisis y presentación.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_STATE = 42

REQUIRED_RAW_COLUMNS = {
    "post_id",
    "timestamp",
    "platform",
    "text_content",
    "sentiment_score",
    "sentiment_label",
    "toxicity_score",
    "likes_count",
    "shares_count",
    "comments_count",
    "impressions",
    "engagement_rate",
    "brand_name",
    "campaign_name",
    "campaign_phase",
}

LEAKAGE_COLUMNS = {
    "likes_count",
    "shares_count",
    "comments_count",
    "impressions",
    "engagement_rate",
    "total_interactions",
    "er_recalc",
    "er_normalized",
    "composite_score",
    "sentiment_score",
    "toxicity_score",
    "sentiment_label",
    "toxicity_bin",
    "log_engagement",
    "authentic_success",
    "authentic_success_ev3",
}


@dataclass
class EtlPaths:
    root: Path
    raw_adolfo: Path
    enriched_felipe: Path
    external_campaign_csv: Path
    external_api_json: Path
    external_sqlite: Path
    processed_dir: Path
    reports_dir: Path
    logs_dir: Path


def resolve_paths(project_root: str | None = None) -> EtlPaths:
    root = Path(project_root or os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
    return EtlPaths(
        root=root,
        raw_adolfo=root / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv",
        enriched_felipe=root / "Felipe" / "social_media_enriched (1).csv",
        external_campaign_csv=root / "data" / "external" / "campaign_metadata.csv",
        external_api_json=root / "data" / "external" / "api_campaign_metadata.json",
        external_sqlite=root / "data" / "external" / "campaign_metadata.sqlite",
        processed_dir=root / "data" / "processed",
        reports_dir=root / "data" / "processed" / "reports",
        logs_dir=root / "logs",
    )


def setup_logging(paths: EtlPaths) -> None:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs_dir / "etl.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def ensure_directories(paths: EtlPaths) -> None:
    for directory in [paths.processed_dir, paths.reports_dir, paths.external_campaign_csv.parent, paths.logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def assert_csv_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe la fuente {label}: {path}")
    with path.open("rb") as file:
        signature = file.read(8)
    if signature.startswith(b"\x89PNG"):
        raise ValueError(f"La fuente {label} tiene extensión CSV, pero realmente parece ser una imagen PNG: {path}")


def read_csv_source(path: Path, label: str) -> pd.DataFrame:
    assert_csv_file(path, label)
    logging.info("Cargando fuente %s desde %s", label, path)
    df = pd.read_csv(path)
    df["source_loaded_from"] = label
    return df


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    missing = sorted(set(required) - set(df.columns))
    for col in missing:
        issues.append({"source": label, "severity": "error", "check": "missing_column", "detail": col})
    if missing:
        raise ValueError(f"La fuente {label} no tiene columnas obligatorias: {missing}")
    return issues


def build_campaign_metadata_if_missing(base: pd.DataFrame, path: Path) -> None:
    if path.exists():
        logging.info("Fuente externa de campañas encontrada: %s", path)
        return

    logging.info("No existe fuente externa de campañas. Creando fuente provisional: %s", path)
    campaigns = sorted(base.get("campaign_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    objectives = ["awareness", "conversion", "retention", "launch", "support"]
    segments = ["masivo", "joven", "premium", "tecnico", "fidelizacion"]
    rows = []
    for i, campaign in enumerate(campaigns):
        rows.append(
            {
                "campaign_name": campaign,
                "campaign_objective": objectives[i % len(objectives)],
                "target_segment": segments[i % len(segments)],
                "planned_budget_usd": int(8000 + (i * 1750) % 42000),
                "priority_level": ["alta", "media", "baja"][i % 3],
                "external_source_type": "simulated_csv_until_api_db_ready",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def load_external_campaign_metadata(paths: EtlPaths, base: pd.DataFrame) -> pd.DataFrame:
    """Carga metadata externa desde CSV, JSON API exportado o SQLite si existen.

    Por ahora el CSV funciona como fuente provisional. Cuando el compañero agregue
    API o base de datos, basta con generar alguno de los archivos esperados o
    adaptar esta función al endpoint real.
    """
    build_campaign_metadata_if_missing(base, paths.external_campaign_csv)

    frames: list[pd.DataFrame] = []
    csv_df = pd.read_csv(paths.external_campaign_csv)
    csv_df["metadata_source"] = "external_csv"
    frames.append(csv_df)

    if paths.external_api_json.exists():
        logging.info("Cargando metadata tipo API exportada desde %s", paths.external_api_json)
        api_df = pd.read_json(paths.external_api_json)
        api_df["metadata_source"] = "api_json"
        frames.append(api_df)

    if paths.external_sqlite.exists():
        logging.info("Cargando metadata desde SQLite %s", paths.external_sqlite)
        with sqlite3.connect(paths.external_sqlite) as conn:
            sql_df = pd.read_sql_query("SELECT * FROM campaign_metadata", conn)
        sql_df["metadata_source"] = "sqlite"
        frames.append(sql_df)

    metadata = pd.concat(frames, ignore_index=True, sort=False)
    if "campaign_name" not in metadata.columns:
        raise ValueError("La metadata externa debe incluir campaign_name para poder integrarse.")
    return metadata.drop_duplicates(subset=["campaign_name"], keep="last")


def choose_base_dataset(raw_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Usa el dataset enriquecido como base y valida contra el CSV original."""
    raw_ids = set(raw_df["post_id"].astype(str))
    enriched_ids = set(enriched_df["post_id"].astype(str))
    missing_from_enriched = raw_ids - enriched_ids
    if missing_from_enriched:
        logging.warning("Hay %s post_id del CSV original que no están en el enriquecido.", len(missing_from_enriched))
    base = enriched_df.copy()
    base["exists_in_raw_source"] = base["post_id"].astype(str).isin(raw_ids)
    return base


def transform_posts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "year" not in out.columns:
        out["year"] = out["timestamp"].dt.year
    if "month" not in out.columns:
        out["month"] = out["timestamp"].dt.month
    if "hour" not in out.columns:
        out["hour"] = out["timestamp"].dt.hour

    for col in ["likes_count", "shares_count", "comments_count", "impressions", "engagement_rate", "sentiment_score", "toxicity_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"likes_count", "shares_count", "comments_count"}.issubset(out.columns):
        out["total_interactions"] = out[["likes_count", "shares_count", "comments_count"]].fillna(0).sum(axis=1)

    if {"total_interactions", "impressions"}.issubset(out.columns):
        out["er_recalc_ev3"] = np.where(out["impressions"].fillna(0) > 0, out["total_interactions"] / out["impressions"], np.nan)

    if "engagement_rate" in out.columns:
        out["log_engagement"] = np.log10(out["engagement_rate"].clip(lower=0).fillna(0) + 1)
        out["fake_engagement_flag"] = (out["engagement_rate"] > 1).astype(int)
    else:
        out["fake_engagement_flag"] = 0

    if {"engagement_rate", "sentiment_score", "toxicity_score"}.issubset(out.columns):
        organic = out["fake_engagement_flag"] == 0
        engagement_threshold = out.loc[organic, "engagement_rate"].quantile(0.75)
        toxicity_threshold = out.loc[organic, "toxicity_score"].quantile(0.40)
        out["authentic_success_ev3"] = (
            organic
            & (out["engagement_rate"] >= engagement_threshold)
            & (out["sentiment_score"] > 0)
            & (out["toxicity_score"] <= toxicity_threshold)
        ).astype(int)
        out["authentic_success_rule"] = "organic + engagement_p75 + positive_sentiment + toxicity_p40"

    text = out.get("text_content", pd.Series("", index=out.index)).fillna("").astype(str)
    hashtags = out.get("hashtags", pd.Series("", index=out.index)).fillna("").astype(str)
    mentions = out.get("mentions", pd.Series("", index=out.index)).fillna("").astype(str)
    out["text_len"] = text.str.len()
    out["word_count"] = text.str.split().str.len().fillna(0)
    out["hashtag_count"] = hashtags.str.count("#")
    out["mention_count"] = mentions.str.count("@")
    out["has_question"] = text.str.contains(r"\?", regex=True).astype(int)
    out["has_exclamation"] = text.str.contains("!", regex=False).astype(int)

    return out


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 3),
                "unique_values": int(df[col].nunique(dropna=True)),
                "leakage_for_prediction": col in LEAKAGE_COLUMNS,
            }
        )
    return pd.DataFrame(rows).sort_values(["leakage_for_prediction", "missing_pct"], ascending=[False, False])


def main(project_root: str | None = None) -> None:
    paths = resolve_paths(project_root)
    ensure_directories(paths)
    setup_logging(paths)

    logging.info("Iniciando ETL EV3 en %s", paths.root)
    raw_df = read_csv_source(paths.raw_adolfo, "raw_adolfo_csv")
    enriched_df = read_csv_source(paths.enriched_felipe, "enriched_felipe_csv")

    validate_required_columns(raw_df, REQUIRED_RAW_COLUMNS, "raw_adolfo_csv")
    validate_required_columns(enriched_df, REQUIRED_RAW_COLUMNS, "enriched_felipe_csv")

    base = choose_base_dataset(raw_df, enriched_df)
    campaign_metadata = load_external_campaign_metadata(paths, base)

    final_df = transform_posts(base)
    final_df = final_df.merge(campaign_metadata, on="campaign_name", how="left")
    final_df["etl_processed_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    output_csv = paths.processed_dir / "social_media_ev3_final.csv"
    quality_csv = paths.reports_dir / "etl_quality_report.csv"
    summary_json = paths.reports_dir / "etl_summary.json"

    quality = build_quality_report(final_df)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")
    quality.to_csv(quality_csv, index=False, encoding="utf-8")

    summary = {
        "rows": int(len(final_df)),
        "columns": int(len(final_df.columns)),
        "output_csv": str(output_csv),
        "quality_report": str(quality_csv),
        "sources": {
            "raw_adolfo_csv": str(paths.raw_adolfo),
            "enriched_felipe_csv": str(paths.enriched_felipe),
            "external_campaign_csv": str(paths.external_campaign_csv),
            "external_api_json_exists": paths.external_api_json.exists(),
            "external_sqlite_exists": paths.external_sqlite.exists(),
        },
        "fake_engagement_posts": int(final_df.get("fake_engagement_flag", pd.Series(dtype=int)).sum()),
        "authentic_success_ev3_posts": int(final_df.get("authentic_success_ev3", pd.Series(dtype=int)).sum()),
        "leakage_columns_flagged": sorted([c for c in final_df.columns if c in LEAKAGE_COLUMNS]),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info("ETL finalizado. Filas=%s Columnas=%s", summary["rows"], summary["columns"])
    logging.info("Archivo final: %s", output_csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline ETL EV3.")
    parser.add_argument("--project-root", default=None, help="Ruta raíz del proyecto. Si se omite, usa PROJECT_ROOT o la raíz del repo.")
    args = parser.parse_args()
    main(args.project_root)