"""
dashboard_toxicity_why.py
============================================================================
Bloque para la VISTA TÉCNICA del dashboard. Responde el "POR QUÉ" del ranking
de toxicidad, no solo el "cuál". Lee Adolfo/results/metrics/toxicity_explain.json
(generado por etl/explain_toxicity_platform.py).

CÓMO INTEGRARLO en dashboards/app.py:
  1. Copia este archivo a dashboards/dashboard_toxicity_why.py
  2. Arriba de app.py:   from dashboard_toxicity_why import render_toxicity_why
  3. Dentro del bloque de la vista "Técnica":   render_toxicity_why()
============================================================================
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# raíz del repo desde dashboards/
_JSON = Path(__file__).resolve().parent.parent / "Adolfo" / "results" / "metrics" / "toxicity_explain.json"
_FIG = Path(__file__).resolve().parent.parent / "Adolfo" / "results" / "figures" / "toxicity_distribution.png"
_ORDER = ["youtube", "reddit", "twitter"]   # más → menos tóxico


def render_toxicity_why() -> None:
    st.header("¿Por qué una plataforma es más tóxica?")

    if not _JSON.exists():
        st.warning(
            "Falta toxicity_explain.json. Corre primero:\n\n"
            "`python3 etl/explain_toxicity_platform.py`"
        )
        return

    data = json.loads(_JSON.read_text())
    thr = data["_meta"]["umbral_toxico"]

    st.caption(
        f"Fuente: {data['_meta']['fuente']} · n={data['_meta']['n_total']:,} · "
        f"umbral tóxico = {thr}. "
        "⚠️ Asociación, no causalidad: muestra patrones de 2019, no una verdad universal."
    )

    # ---- OPCIÓN 1: ranking interpretable -------------------------------
    st.subheader("1 · Score medio de toxicidad y significancia")
    rt = data["ranking_y_test"]
    df_rank = pd.DataFrame(rt["ranking"]).copy()
    df_rank["score_medio"] = pd.to_numeric(df_rank["score_medio"], errors="coerce")
    df_rank = df_rank.sort_values("score_medio", ascending=False)
    df_rank["toxicidad_relativa"] = df_rank["score_medio"] - df_rank["score_medio"].min()
    fig_rank = px.scatter(
        df_rank,
        x="score_medio",
        y="plataforma",
        size="toxicidad_relativa",
        size_max=24,
        text=df_rank["score_medio"].map(lambda x: f"{x:.3f}"),
        title="Score medio de toxicidad por plataforma",
    )
    fig_rank.update_traces(textposition="middle right")
    fig_rank.update_layout(xaxis_title="hate_speech_score medio (más a la derecha = más tóxico)", yaxis_title="")
    st.plotly_chart(fig_rank, use_container_width=True)

    d1 = data["opcion1_distribucion"]
    rows = []
    for p in _ORDER:
        if p in d1:
            v = d1[p]
            rows.append({
                "plataforma": p,
                "mediana": v["median"],
                "media": v["mean"],
                "media sin cola": v["mean_sin_cola"],
                "p95 (cola tóxica)": v["q95"],
                "máximo": v["max"],
            })
    df_dist = pd.DataFrame(rows)

    # ---- OPCIÓN 2: % sobre el umbral --------------------------------------
    st.subheader(f"2 · Comentarios que cruzan el umbral tóxico (> {thr})")
    d2 = data["opcion2_umbral"]
    rows2 = [
        {"plataforma": p, "% tóxicos": d2[p]["pct_toxicos"],
         "n tóxicos": d2[p]["n_toxicos"], "n total": d2[p]["n"]}
        for p in _ORDER if p in d2
    ]
    df2 = pd.DataFrame(rows2).set_index("plataforma")
    fig_thr = px.bar(df2.reset_index(), x="plataforma", y="% tóxicos", text="% tóxicos", title=f"Comentarios sobre el umbral tóxico (> {thr})")
    fig_thr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_thr.update_layout(xaxis_title="", yaxis_title="% de comentarios tóxicos")
    st.plotly_chart(fig_thr, use_container_width=True)
    st.dataframe(df2, use_container_width=True)
    st.markdown("**Resumen de la distribución**")
    st.dataframe(df_dist.set_index("plataforma"), use_container_width=True)
    st.markdown(
        "Esta vista es más clara para presentar que un histograma o boxplot: muestra qué proporción de comentarios supera el límite tóxico. "
        "La tabla de distribución queda como apoyo para explicar mediana, p95 y cola extrema."
    )

    # ---- OPCIÓN 3: términos distintivos -----------------------------------
    st.subheader("3 · Qué se dice distinto en los comentarios tóxicos")
    d3 = data["opcion3_terminos_distintivos"]
    if "_aviso" in d3:
        st.info(d3["_aviso"])
    else:
        cols = st.columns(len([p for p in _ORDER if p in d3]))
        for col, p in zip(cols, [p for p in _ORDER if p in d3]):
            with col:
                st.markdown(f"**{p}**")
                terms = pd.DataFrame(d3[p])
                if not terms.empty:
                    st.dataframe(terms.set_index("termino")[["z", "frecuencia"]],
                                 use_container_width=True)
        st.caption(
            "Términos sobrerrepresentados por plataforma (log-odds con prior, "
            "Monroe et al. 2008). z alto = más característico de esa plataforma. "
            "Es *qué se dice distinto*, no *por qué* — correlación, no causa."
        )