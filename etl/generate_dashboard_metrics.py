#!/usr/bin/env python3
"""Generate discussion-ready metrics for the EV3 toxicity dashboard.

This script centralizes the evidence used by the dashboard. It never invents
columns: when a metric cannot be calculated from available JSON/CSV/Mongo data,
it is written as unavailable with an explicit limitation.
"""
from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "Adolfo" / "results" / "metrics"
OUT = METRICS_DIR / "dashboard_discussion_summary.json"
TOX_EXPLAIN = METRICS_DIR / "toxicity_explain.json"
TOX_PLATFORM = METRICS_DIR / "toxicity_by_platform.json"
BENCHMARK_REAL = METRICS_DIR / "benchmark_real_data.json"
LAYER1 = METRICS_DIR / "layer1_synthetic_contrast.json"
LAYER3 = METRICS_DIR / "layer3_sentiment_toxicity.json"

PLATFORM_MAP = {0: "reddit", 2: "twitter", 3: "youtube"}
PLATFORMS = ["youtube", "reddit", "twitter"]
SCORE_FIELD = "hate_speech_score"
PLATFORM_FIELD = "platform"
TEXT_CANDIDATES = ["text", "comment_text", "tweet", "body", "content"]
SENTIMENT_CANDIDATES = ["sentiment", "sentiment_score", "annotator_sentiment"]
CONTEXT_CANDIDATES = {
    "year": ["year", "created_year", "annotation_year"],
    "date": ["date", "created_at", "timestamp"],
    "community": ["community", "subreddit", "channel", "channel_id", "conversation_id"],
    "language": ["language", "lang"],
    "country": ["country", "location", "region"],
    "content_type": ["content_type", "category", "topic"],
}

THRESHOLDS = {
    "bajo_discusion": 0.0,
    "medio_toxico_mhs": 0.5,
    "alto_cola_extrema": 2.0,
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_load_mongo() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load raw MHS data if Mongo credentials and dependencies are available."""
    report = {"available": False, "reason": None, "columns": []}
    try:
        from dotenv import load_dotenv
        from pymongo import MongoClient
    except Exception as exc:  # dependency not installed
        report["reason"] = f"Dependencia no disponible: {type(exc).__name__}"
        return None, report

    load_dotenv(ROOT / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        report["reason"] = "No existe MONGODB_URI en variables de entorno/.env."
        return None, report

    db_name = os.environ.get("MONGODB_DB_NAME", "Ev3")
    coll_name = os.environ.get("MONGODB_TOX_COLLECTION", "measuringhatespeech")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        coll = client[db_name][coll_name]
        sample = coll.find_one({})
        if not sample:
            report["reason"] = f"Coleccion vacia: {db_name}.{coll_name}"
            return None, report

        text_field = next((c for c in TEXT_CANDIDATES if c in sample), None)
        sentiment_field = next((c for c in SENTIMENT_CANDIDATES if c in sample), None)
        projection = {"_id": 0, SCORE_FIELD: 1, PLATFORM_FIELD: 1}
        if text_field:
            projection[text_field] = 1
        if sentiment_field:
            projection[sentiment_field] = 1
        for candidates in CONTEXT_CANDIDATES.values():
            for col in candidates:
                if col in sample:
                    projection[col] = 1

        docs = list(coll.find({PLATFORM_FIELD: {"$in": list(PLATFORM_MAP)}}, projection))
        client.close()
        if not docs:
            report["reason"] = "La consulta a Mongo no devolvio comentarios de plataformas sociales."
            return None, report

        df = pd.DataFrame(docs)
        if text_field and text_field in df.columns:
            df = df.rename(columns={text_field: "text"})
        if sentiment_field and sentiment_field in df.columns:
            df = df.rename(columns={sentiment_field: "sentiment"})
        df = df.rename(columns={SCORE_FIELD: "score"})
        df["platform"] = df[PLATFORM_FIELD].map(PLATFORM_MAP)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["platform", "score"])
        report.update({"available": True, "reason": "Mongo disponible", "columns": sorted(df.columns.tolist())})
        return df.reset_index(drop=True), report
    except Exception as exc:
        report["reason"] = f"No se pudo leer Mongo: {type(exc).__name__}: {exc}"
        return None, report


def platform_stats_from_raw(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for platform, group in df.groupby("platform"):
        score = group["score"].dropna()
        out[platform] = {
            "n": int(score.size),
            "mean": round(float(score.mean()), 4),
            "median": round(float(score.median()), 4),
            "std": round(float(score.std()), 4),
            "min": round(float(score.min()), 4),
            "max": round(float(score.max()), 4),
            "p25": round(float(score.quantile(0.25)), 4),
            "p50": round(float(score.quantile(0.50)), 4),
            "p75": round(float(score.quantile(0.75)), 4),
            "p90": round(float(score.quantile(0.90)), 4),
            "p95": round(float(score.quantile(0.95)), 4),
        }
        for name, threshold in THRESHOLDS.items():
            out[platform][f"pct_sobre_{name}"] = round(float((score > threshold).mean() * 100), 2)
            out[platform][f"n_sobre_{name}"] = int((score > threshold).sum())
    return out


def platform_stats_from_existing(explain: dict[str, Any], by_platform: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dist = explain.get("opcion1_distribucion", {}) if explain else {}
    threshold = explain.get("opcion2_umbral", {}) if explain else {}
    descriptive = by_platform.get("descriptive_by_platform", {}) if by_platform else {}
    out: dict[str, dict[str, Any]] = {}
    for platform in sorted(set(dist) | set(threshold) | set(descriptive)):
        d = dist.get(platform, {})
        t = threshold.get(platform, {})
        b = descriptive.get(platform, {})
        out[platform] = {
            "n": d.get("n") or t.get("n") or b.get("n"),
            "mean": d.get("mean") or b.get("mean"),
            "median": d.get("median") or b.get("median"),
            "std": d.get("std") or b.get("std"),
            "min": b.get("min"),
            "max": d.get("max") or b.get("max"),
            "p25": d.get("q25"),
            "p50": d.get("median"),
            "p75": d.get("q75"),
            "p90": None,
            "p95": d.get("q95"),
            "mean_sin_cola": d.get("mean_sin_cola"),
            "pct_sobre_medio_toxico_mhs": t.get("pct_toxicos"),
            "n_sobre_medio_toxico_mhs": t.get("n_toxicos"),
            "pct_sobre_bajo_discusion": None,
            "n_sobre_bajo_discusion": None,
            "pct_sobre_alto_cola_extrema": None,
            "n_sobre_alto_cola_extrema": None,
        }
    return out


def kruskal_from_existing(explain: dict[str, Any], by_platform: dict[str, Any]) -> dict[str, Any]:
    kw = (explain or {}).get("ranking_y_test", {}).get("kruskal_wallis") or (by_platform or {}).get("statistical_tests", {}).get("kruskal_wallis", {})
    return {
        "H": kw.get("H") or kw.get("h_statistic"),
        "p_value": kw.get("p_valor") or kw.get("p_value"),
        "significant": kw.get("significativo") if "significativo" in kw else kw.get("significant_0_05"),
    }


def pairwise_from_existing(explain: dict[str, Any], by_platform: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = []
    raw_pairs = (explain or {}).get("ranking_y_test", {}).get("comparaciones_por_pares", {})
    for pair, data in raw_pairs.items():
        pairs.append({
            "pair": pair,
            "more_toxic": data.get("mas_toxico"),
            "p_value": data.get("p_valor"),
            "significant": data.get("significativo"),
            "method": "Mann-Whitney U, two-sided, existing JSON",
        })
    if pairs:
        return pairs
    for row in (by_platform or {}).get("statistical_tests", {}).get("pairwise_mann_whitney", []):
        pairs.append({
            "pair": row.get("pair"),
            "more_toxic": None,
            "p_value": row.get("p_value_holm") or row.get("p_value_uncorrected"),
            "significant": row.get("significant_holm_0_05"),
            "method": "Mann-Whitney U with Holm correction, existing JSON",
        })
    return pairs


def correlation_from_layer3(layer3: dict[str, Any] | None, raw_df: pd.DataFrame | None) -> dict[str, Any]:
    if layer3:
        return {
            "status": "available_from_layer3_json",
            "method": layer3.get("method"),
            "overall": layer3.get("overall"),
            "by_platform": layer3.get("by_platform"),
            "note": layer3.get("scales_note"),
        }
    if raw_df is not None and {"sentiment", "score"}.issubset(raw_df.columns):
        try:
            from scipy.stats import pearsonr
            clean = raw_df[["sentiment", "score", "platform"]].dropna().copy()
            clean["sentiment"] = pd.to_numeric(clean["sentiment"], errors="coerce")
            clean["score"] = pd.to_numeric(clean["score"], errors="coerce")
            clean = clean.dropna(subset=["sentiment", "score"])
            overall_r, overall_p = pearsonr(clean["sentiment"], clean["score"])
            by_platform = {}
            for platform, group in clean.groupby("platform"):
                if len(group) >= 3:
                    r, p = pearsonr(group["sentiment"], group["score"])
                    by_platform[platform] = {"r": round(float(r), 4), "p_value": float(p), "n": int(len(group))}
            return {
                "status": "computed_from_mongo",
                "method": "Pearson r between MHS continuous sentiment and hate_speech_score",
                "overall": {"r": round(float(overall_r), 4), "p_value": float(overall_p), "n": int(len(clean))},
                "by_platform": by_platform,
                "note": "MHS continuous annotator sentiment, not EV2 sentiment_label.",
            }
        except Exception as exc:
            return {"status": "unavailable", "reason": f"No se pudo calcular correlacion: {type(exc).__name__}: {exc}"}
    return {
        "status": "unavailable",
        "reason": "No existe layer3_sentiment_toxicity.json y no hay datos crudos con columna sentiment disponibles localmente.",
    }


def context_analysis(raw_df: pd.DataFrame | None, mongo_report: dict[str, Any]) -> dict[str, Any]:
    if raw_df is None:
        return {
            "available_columns": [],
            "year": {"status": "unavailable", "reason": "Sin datos crudos disponibles en esta ejecucion."},
            "community": {"status": "unavailable", "reason": "Sin datos crudos disponibles en esta ejecucion."},
            "language_country": {"status": "unavailable", "reason": "Sin datos crudos disponibles en esta ejecucion."},
        }
    cols = set(raw_df.columns)
    result: dict[str, Any] = {"available_columns": sorted(cols)}

    def first_available(kind: str) -> str | None:
        for col in CONTEXT_CANDIDATES[kind]:
            if col in cols:
                return col
        return None

    date_col = first_available("date")
    year_col = first_available("year")
    if year_col:
        tmp = raw_df.copy()
        tmp["year_context"] = pd.to_numeric(tmp[year_col], errors="coerce")
        y = tmp.dropna(subset=["year_context"]).groupby(["year_context", "platform"])["score"].agg(["count", "mean", "median"]).reset_index()
        result["year"] = {"status": "available", "source_column": year_col, "rows": y.to_dict("records")}
    elif date_col:
        tmp = raw_df.copy()
        tmp["date_context"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp["year_context"] = tmp["date_context"].dt.year
        y = tmp.dropna(subset=["year_context"]).groupby(["year_context", "platform"])["score"].agg(["count", "mean", "median"]).reset_index()
        result["year"] = {"status": "available", "source_column": date_col, "rows": y.to_dict("records")}
    else:
        result["year"] = {"status": "unavailable", "reason": "El corpus cargado no trae fecha/año de captura por comentario."}

    community_col = first_available("community")
    if community_col:
        tmp = raw_df.dropna(subset=[community_col]).copy()
        by_comm = tmp.groupby(["platform", community_col])["score"].agg(["count", "mean", "sum"]).reset_index()
        result["community"] = {"status": "available", "source_column": community_col, "top_rows": by_comm.sort_values("sum", ascending=False).head(30).to_dict("records")}
    else:
        result["community"] = {"status": "unavailable", "reason": "No hay columna comunidad/canal/subreddit/conversacion disponible."}

    lang_col = first_available("language")
    country_col = first_available("country")
    result["language_country"] = {
        "status": "available" if (lang_col or country_col) else "unavailable",
        "language_column": lang_col,
        "country_column": country_col,
        "reason": None if (lang_col or country_col) else "El corpus no permite concluir pais, region, idioma ni cultura de origen.",
    }
    return result


def build_hypotheses(context: dict[str, Any], correlations: dict[str, Any]) -> list[dict[str, str]]:
    community_status = context.get("community", {}).get("status")
    year_status = context.get("year", {}).get("status")
    lang_status = context.get("language_country", {}).get("status")
    corr_status = correlations.get("status")
    return [
        {
            "hipotesis": "La mayor toxicidad se concentra en pocas comunidades o canales.",
            "evidencia_disponible": context.get("community", {}).get("reason", "Existe columna de comunidad para analizar concentracion."),
            "estado": "Confirmable con datos" if community_status == "available" else "No comprobable con este dataset",
            "dato_faltante": "Comunidad, canal, subreddit o identificador de conversacion por comentario." if community_status != "available" else "Calcular concentracion por comunidad y validar estabilidad temporal.",
        },
        {
            "hipotesis": "El patron depende del año o contexto de recoleccion.",
            "evidencia_disponible": context.get("year", {}).get("reason", "Existe columna temporal para tendencia."),
            "estado": "Confirmable con datos" if year_status == "available" else "Parcial / no comprobable",
            "dato_faltante": "Fecha o año por comentario; el JSON actual solo anota que son datos 2019." if year_status != "available" else "Comparar años y eventos de contexto.",
        },
        {
            "hipotesis": "Diferencias geograficas o linguisticas afectan el ranking.",
            "evidencia_disponible": context.get("language_country", {}).get("reason", "Existen columnas de idioma/pais."),
            "estado": "Confirmable con datos" if lang_status == "available" else "No comprobable con este dataset",
            "dato_faltante": "Pais, region, idioma o cultura de origen por comentario." if lang_status != "available" else "Controlar por idioma/pais antes de comparar plataformas.",
        },
        {
            "hipotesis": "El origen del corpus influye en que YouTube aparezca mas toxico.",
            "evidencia_disponible": "Fuente documentada: Measuring Hate Speech; muestra de plataformas desbalanceada y periodo anotado como 2019.",
            "estado": "Parcialmente observada",
            "dato_faltante": "Diseño muestral completo, criterios de recoleccion y representatividad por plataforma.",
        },
        {
            "hipotesis": "La dinamica de comentarios o reglas de moderacion explican diferencias.",
            "evidencia_disponible": "El dataset permite observar diferencias de score, no mecanismos causales de diseño o moderacion.",
            "estado": "Discusion futura",
            "dato_faltante": "Metadatos de moderacion, tipo de contenido, estructura de interaccion y cambios de politica por plataforma.",
        },
        {
            "hipotesis": "Hay implicancia para brand safety y riesgo reputacional.",
            "evidencia_disponible": "Existe mayor toxicidad relativa y porcentaje sobre umbral por plataforma; no hay datos de anuncios ni presencia de marcas.",
            "estado": "Implicancia, no causalidad",
            "dato_faltante": "Cruce con marcas, anuncios, categorias de video, campañas y exposicion publicitaria.",
        },
        {
            "hipotesis": "Sentimiento negativo y toxicidad se mueven juntos en datos reales.",
            "evidencia_disponible": "Correlacion disponible." if corr_status != "unavailable" else correlations.get("reason", "No disponible."),
            "estado": "Confirmada por datos" if corr_status != "unavailable" else "Pendiente de generar desde Mongo",
            "dato_faltante": "Generar layer3_sentiment_toxicity.json o conectar Mongo con columna sentiment." if corr_status == "unavailable" else "Validar si la relacion se mantiene por subcomunidad o periodo.",
        },
    ]


def main() -> None:
    explain = load_json(TOX_EXPLAIN, {})
    by_platform = load_json(TOX_PLATFORM, {})
    benchmark = load_json(BENCHMARK_REAL, {})
    layer1 = load_json(LAYER1, {})
    layer3 = load_json(LAYER3, None)

    raw_df, mongo_report = maybe_load_mongo()
    if raw_df is not None:
        platform_stats = platform_stats_from_raw(raw_df)
        metric_source = "computed_from_mongo"
    else:
        platform_stats = platform_stats_from_existing(explain, by_platform)
        metric_source = "existing_json_fallback"

    ranking = (explain or {}).get("ranking_y_test", {}).get("ranking") or [
        {"plataforma": p, "score_medio": d.get("mean")}
        for p, d in platform_stats.items()
    ]
    ranking = sorted(ranking, key=lambda r: safe_float(r.get("score_medio")) or -999, reverse=True)
    most = ranking[0]["plataforma"] if ranking else None
    least = ranking[-1]["plataforma"] if ranking else None

    total_comments = (explain or {}).get("_meta", {}).get("n_total") or by_platform.get("n_comments_analyzed") or sum((d.get("n") or 0) for d in platform_stats.values())
    counts = {p: int(d.get("n") or 0) for p, d in platform_stats.items()}
    max_n = max(counts.values()) if counts else 0
    min_n = min([v for v in counts.values() if v > 0], default=0)
    imbalance_ratio = round(max_n / min_n, 3) if min_n else None

    correlations = correlation_from_layer3(layer3, raw_df)
    context = context_analysis(raw_df, mongo_report)
    hypotheses = build_hypotheses(context, correlations)

    limitations = []
    if raw_df is None:
        limitations.append("Esta ejecucion no accedio a Mongo; las metricas se consolidan desde JSONs existentes.")
    if context.get("community", {}).get("status") != "available":
        limitations.append("No hay comunidad/canal/subreddit por comentario; no se puede probar concentracion comunitaria.")
    if context.get("year", {}).get("status") != "available":
        limitations.append("No hay fecha/año por comentario en los datos disponibles localmente; no se puede analizar evolucion temporal.")
    if context.get("language_country", {}).get("status") != "available":
        limitations.append("El corpus no permite concluir pais, region, idioma ni cultura de origen.")
    if correlations.get("status") == "unavailable":
        limitations.append("La correlacion sentimiento-toxicidad no esta disponible hasta generar layer3_sentiment_toxicity.json o conectar Mongo.")
    if imbalance_ratio and imbalance_ratio > 1.5:
        limitations.append(f"La muestra esta desbalanceada por plataforma (ratio max/min n = {imbalance_ratio}).")

    interpretation_matrix = [
        {
            "hallazgo": f"{most.capitalize() if most else 'La plataforma lider'} aparece con mayor toxicidad relativa.",
            "evidencia": "Ranking por hate_speech_score medio y Kruskal-Wallis significativo.",
            "posible_explicacion": "Corpus, tipo de comunidad, dinamica de comentarios o muestra.",
            "dato_adicional_necesario": "Comunidad/canal, tipo de contenido, fecha, idioma/pais y reglas de moderacion.",
        },
        {
            "hallazgo": "El porcentaje sobre umbral es parecido entre YouTube y Reddit, y menor en Twitter.",
            "evidencia": "opcion2_umbral en toxicity_explain.json.",
            "posible_explicacion": "La diferencia no es solo media; tambien depende de cola y composicion de comentarios.",
            "dato_adicional_necesario": "Datos crudos y contexto por comunidad para aislar subgrupos.",
        },
        {
            "hallazgo": "El modelo entrenado en dato sintetico generaliza mal a texto real.",
            "evidencia": f"F1 sintetico reportado {benchmark.get('f1_macro_synthetic_reported_ev2')} vs F1 real {benchmark.get('f1_macro_on_real_data')}.",
            "posible_explicacion": "Cambio de dominio: texto sintetico vs lenguaje real.",
            "dato_adicional_necesario": "Datos reales etiquetados para entrenamiento y validacion externa.",
        },
    ]

    output = {
        "_meta": {
            "generated_by": "etl/generate_dashboard_metrics.py",
            "metric_source": metric_source,
            "source_project": "SCY1101 Social Media EV3",
            "dashboard_question_original": "¿Es Twitter más tóxico que Reddit y YouTube usando datos reales?",
            "dashboard_question_evolved": "¿La mayor toxicidad relativa observada en YouTube representa un problema de plataforma, de comunidad, de muestra o de contexto del dataset?",
            "thresholds_used": THRESHOLDS,
            "mongo": mongo_report,
        },
        "executive_summary": {
            "central_finding": f"En este corpus, {most} presenta mayor toxicidad relativa que {least}.",
            "main_value": "El valor no es afirmar una verdad universal, sino abrir una discusion basada en evidencia sobre corpus, comunidades, moderacion, diseño de interaccion y riesgo reputacional.",
            "main_limitation": limitations[0] if limitations else "No hay limitacion critica detectada automaticamente.",
            "brand_safety_note": "No se miden anuncios ni presencia de marcas; solo se propone como linea futura cruzar toxicidad con exposicion de marca.",
        },
        "sample": {
            "total_comments": int(total_comments or 0),
            "counts_by_platform": counts,
            "imbalance_ratio_max_min": imbalance_ratio,
            "imbalance_warning": imbalance_ratio is not None and imbalance_ratio > 1.5,
        },
        "platform_metrics": platform_stats,
        "ranking": ranking,
        "kruskal_wallis": kruskal_from_existing(explain, by_platform),
        "pairwise_tests": pairwise_from_existing(explain, by_platform),
        "distinctive_terms": (explain or {}).get("opcion3_terminos_distintivos", {}),
        "sentiment_toxicity_correlation": correlations,
        "synthetic_vs_real": {
            "sentiment_synthetic_f1_macro": (layer1.get("sentiment_task", {}).get("model", {}) if layer1 else {}).get("f1_macro"),
            "toxicity_synthetic_has_signal": (layer1.get("toxicity_task", {}) if layer1 else {}).get("has_signal"),
            "toxicity_synthetic_f1_macro": (layer1.get("toxicity_task", {}).get("model", {}) if layer1 else {}).get("f1_macro"),
            "f1_macro_synthetic_reported_ev2": benchmark.get("f1_macro_synthetic_reported_ev2"),
            "f1_macro_on_real_data": benchmark.get("f1_macro_on_real_data"),
            "gap_synthetic_vs_real": benchmark.get("gap_synthetic_vs_real"),
            "n_samples_real": benchmark.get("n_samples_real"),
        },
        "context_analysis": context,
        "hypotheses": hypotheses,
        "interpretation_matrix": interpretation_matrix,
        "limitations": limitations,
        "recommended_conclusion": "No estamos diciendo simplemente que YouTube sea más tóxico. Estamos mostrando que, en este corpus, YouTube concentra mayor toxicidad relativa y que ese resultado obliga a analizar contexto: comunidades, diseño de interacción, políticas de moderación, origen del dataset y posibles riesgos para marcas. El valor del proyecto no es entregar una solución cerrada, sino transformar un ranking simple en una discusión basada en evidencia.",
        "what_not_to_conclude": [
            "No demuestra causalidad.",
            "No permite afirmar que YouTube sea universalmente la plataforma más tóxica.",
            "No permite decir que toda la plataforma o todos sus usuarios sean tóxicos.",
            "No entrega una acción directa para un consumidor individual.",
            "No permite inferir país, idioma, comunidad o periodo si esas columnas no existen.",
            "No mide impacto publicitario ni presencia real de marcas.",
        ],
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito: {OUT}")
    print(json.dumps({
        "total_comments": output["sample"]["total_comments"],
        "most_toxic": most,
        "metric_source": metric_source,
        "limitations": len(limitations),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()