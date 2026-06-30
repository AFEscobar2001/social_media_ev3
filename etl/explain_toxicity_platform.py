#!/usr/bin/env python3
"""
explain_toxicity_platform.py
============================================================================
Responde el "POR QUÉ" del ranking de toxicidad por plataforma, no solo el
"cuál". Tres capas, todas honestas (asociación, NO causalidad):

  Opción 1  Distribución por plataforma (no solo el promedio): media,
            mediana, cuartiles, desviación y peso de la cola tóxica.
  Opción 2  % de comentarios sobre el umbral de toxicidad por plataforma.
  Opción 3  Términos distintivos por plataforma en los comentarios tóxicos
            (log-odds ratio con prior de Dirichlet, método Monroe et al.).

Fuente: colección `measuringhatespeech` en MongoDB Atlas (Ev3).
Mapeo de plataforma:  0=reddit  1=reference(EXCLUIR)  2=twitter  3=youtube
Escala: `hate_speech_score` continuo (mayor = más tóxico).

Salida:
  Adolfo/results/metrics/toxicity_explain.json
  Adolfo/results/figures/toxicity_distribution.png

Uso:
  source ../venv/bin/activate
  python3 etl/explain_toxicity_platform.py
  python3 etl/explain_toxicity_platform.py --threshold 0.5 --top 15
============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PLATFORM_MAP = {0: "reddit", 2: "twitter", 3: "youtube"}  # 1=reference excluido
SCORE_FIELD = "hate_speech_score"
PLATFORM_FIELD = "platform"
DEFAULT_THRESHOLD = 0.5          # cutoff "hateful" usado en la doc de MHS
DEFAULT_TOP = 15                 # términos distintivos por plataforma

ROOT = Path(__file__).resolve().parent.parent     # .../social_media_ev3
METRICS_DIR = ROOT / "Adolfo" / "results" / "metrics"
FIGURES_DIR = ROOT / "Adolfo" / "results" / "figures"


# ---------------------------------------------------------------------------
# 1. Carga desde MongoDB
# ---------------------------------------------------------------------------
def load_data_from_mongo() -> pd.DataFrame:
    """Lee score + texto + plataforma de `measuringhatespeech`, excluye reference."""
    from dotenv import load_dotenv
    from pymongo import MongoClient

    # .env en la raíz del repo; ruta explícita por el quirk conocido
    load_dotenv(ROOT / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("ERROR: falta MONGODB_URI en el .env de la raíz del repo.")

    db_name = os.environ.get("MONGODB_DB_NAME", "Ev3")
    coll_name = os.environ.get("MONGODB_TOX_COLLECTION", "measuringhatespeech")

    client = MongoClient(uri)
    coll = client[db_name][coll_name]
    total = coll.count_documents({})
    print(f"[mongo] {db_name}.{coll_name}: {total:,} documentos")

    # Detectar el campo de texto (MHS suele usar 'text')
    sample = coll.find_one({})
    if sample is None:
        sys.exit("ERROR: la colección está vacía.")
    text_field = next(
        (f for f in ("text", "comment_text", "tweet", "body", "content")
         if f in sample),
        None,
    )
    if text_field is None:
        sys.exit(f"ERROR: no encuentro campo de texto. Campos: {list(sample)}")
    print(f"[mongo] campo de texto detectado: '{text_field}'")

    cursor = coll.find(
        {PLATFORM_FIELD: {"$in": list(PLATFORM_MAP)}},      # excluye 1=reference
        {SCORE_FIELD: 1, PLATFORM_FIELD: 1, text_field: 1, "_id": 0},
    )
    df = pd.DataFrame(list(cursor))
    client.close()

    df = df.rename(columns={text_field: "text", SCORE_FIELD: "score"})
    df["platform"] = df[PLATFORM_FIELD].map(PLATFORM_MAP)
    df = df.dropna(subset=["platform", "score", "text"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    print(f"[mongo] filas tras excluir reference y nulos: {len(df):,}")
    print(df["platform"].value_counts().to_string())
    return df[["platform", "score", "text"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RANKING + TEST ESTADÍSTICO (el "cuál", con su p-valor)
# ---------------------------------------------------------------------------
def ranking_and_test(df: pd.DataFrame) -> dict:
    from scipy import stats

    groups = {p: g["score"].values for p, g in df.groupby("platform")}
    # ranking por media (mayor = más tóxico)
    ranking = sorted(
        ((p, float(v.mean())) for p, v in groups.items()),
        key=lambda x: x[1], reverse=True,
    )
    # Kruskal-Wallis global
    h, p_global = stats.kruskal(*groups.values())
    # comparaciones por pares (Mann-Whitney U, una cola: mayor es más tóxico)
    pares = {}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            u, pv = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            pares[f"{a}_vs_{b}"] = {
                "mas_toxico": a if groups[a].mean() > groups[b].mean() else b,
                "p_valor": float(pv),
                "significativo": bool(pv < 0.05),
            }
    return {
        "ranking": [{"plataforma": p, "score_medio": round(m, 4)} for p, m in ranking],
        "kruskal_wallis": {
            "H": round(float(h), 4),
            "p_valor": float(p_global),
            "significativo": bool(p_global < 0.05),
        },
        "comparaciones_por_pares": pares,
    }


# ---------------------------------------------------------------------------
# OPCIÓN 1 — Distribución por plataforma (no solo el promedio)
# ---------------------------------------------------------------------------
def analyze_distribution(df: pd.DataFrame, threshold: float) -> dict:
    out = {}
    for plat, g in df.groupby("platform"):
        s = g["score"]
        out[plat] = {
            "n": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "q75": round(float(s.quantile(0.75)), 4),
            "q95": round(float(s.quantile(0.95)), 4),   # la cola tóxica
            "max": round(float(s.max()), 4),
            # cuánto del promedio lo empuja la cola > umbral
            "mean_sin_cola": round(float(s[s <= threshold].mean()), 4),
        }
    return out


# ---------------------------------------------------------------------------
# OPCIÓN 2 — % de comentarios sobre el umbral por plataforma
# ---------------------------------------------------------------------------
def analyze_threshold(df: pd.DataFrame, threshold: float) -> dict:
    out = {}
    for plat, g in df.groupby("platform"):
        n = len(g)
        n_tox = int((g["score"] > threshold).sum())
        out[plat] = {
            "n": int(n),
            "n_toxicos": n_tox,
            "pct_toxicos": round(100.0 * n_tox / n, 2) if n else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# OPCIÓN 3 — Términos distintivos por plataforma (log-odds + prior Dirichlet)
# Método Monroe, Colaresi & Quinn (2008), "Fightin' Words".
# Compara los comentarios TÓXICOS (> umbral) de una plataforma contra los
# tóxicos del resto. Es correlación, NO causa: "qué se dice distinto", no "por qué".
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    import re
    toks = re.findall(r"[a-záéíóúñü]+", str(text).lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


def distinctive_terms(df: pd.DataFrame, threshold: float, top_n: int) -> dict:
    toxic = df[df["score"] > threshold]
    if toxic.empty:
        return {"_aviso": "ningún comentario supera el umbral; baja --threshold"}

    # Conteo de tokens por plataforma sobre el subconjunto tóxico
    counts: dict[str, Counter] = {}
    for plat, g in toxic.groupby("platform"):
        c: Counter = Counter()
        for txt in g["text"]:
            c.update(_tokenize(txt))
        counts[plat] = c

    vocab = set().union(*[set(c) for c in counts.values()])
    total = Counter()
    for c in counts.values():
        total.update(c)
    alpha0 = sum(total.values())          # prior informativo = corpus tóxico global

    out: dict[str, list] = {}
    for plat, c_i in counts.items():
        n_i = sum(c_i.values())
        # rest = todas las otras plataformas juntas
        c_rest = Counter()
        for p2, c2 in counts.items():
            if p2 != plat:
                c_rest.update(c2)
        n_j = sum(c_rest.values())

        scored = []
        for w in vocab:
            a_w = total[w]                # prior del término
            y_i, y_j = c_i.get(w, 0), c_rest.get(w, 0)
            # log-odds con prior
            num_i = y_i + a_w
            den_i = n_i + alpha0 - num_i
            num_j = y_j + a_w
            den_j = n_j + alpha0 - num_j
            if den_i <= 0 or den_j <= 0:
                continue
            delta = math.log(num_i / den_i) - math.log(num_j / den_j)
            var = 1.0 / num_i + 1.0 / num_j
            z = delta / math.sqrt(var)
            scored.append((w, round(z, 3), y_i))
        # solo términos sobrerrepresentados (z>0), con frecuencia mínima
        scored = [s for s in scored if s[1] > 0 and s[2] >= 5]
        scored.sort(key=lambda x: x[1], reverse=True)
        out[plat] = [
            {"termino": w, "z": z, "frecuencia": n} for w, z, n in scored[:top_n]
        ]
    return out


# Stopwords inglesas mínimas (MHS es corpus en inglés)
_STOPWORDS = set("""
the a an and or but if then else when while for to of in on at by with from
is are was were be been being do does did have has had this that these those
i you he she it we they me him her them my your his its our their as so not no
yes can will would could should may might must just like get got than too very
about into out up down over under again more most some any all what who which
""".split())


# ---------------------------------------------------------------------------
# Figura — boxplot de distribución por plataforma
# ---------------------------------------------------------------------------
def make_figure(df: pd.DataFrame, threshold: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["youtube", "reddit", "twitter"]            # más → menos tóxico
    order = [p for p in order if p in df["platform"].unique()]
    data = [df.loc[df["platform"] == p, "score"].values for p in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=order, vert=True, showfliers=False, widths=0.55)
    ax.axhline(threshold, color="crimson", ls="--", lw=1,
               label=f"umbral tóxico ({threshold})")
    ax.set_ylabel("hate_speech_score (mayor = más tóxico)")
    ax.set_title("Distribución de toxicidad por plataforma (Measuring Hate Speech)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"[fig] guardada en {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = ap.parse_args()

    df = load_data_from_mongo()

    result = {
        "_meta": {
            "fuente": "measuringhatespeech (MongoDB Atlas, Ev3)",
            "umbral_toxico": args.threshold,
            "n_total": int(len(df)),
            "nota": "Asociacion, NO causalidad. Datos 2019, no verdad universal.",
        },
        "ranking_y_test": ranking_and_test(df),
        "opcion1_distribucion": analyze_distribution(df, args.threshold),
        "opcion2_umbral": analyze_threshold(df, args.threshold),
        "opcion3_terminos_distintivos": distinctive_terms(
            df, args.threshold, args.top
        ),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "toxicity_explain.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[json] guardado en {out_path}")

    make_figure(df, args.threshold, FIGURES_DIR / "toxicity_distribution.png")

    # Resumen en consola para verificar
    print("\n=== RESUMEN ===")
    for plat, d in result["opcion2_umbral"].items():
        print(f"{plat:8s}  {d['pct_toxicos']:5.2f}% sobre umbral  (n={d['n']:,})")


if __name__ == "__main__":
    main()