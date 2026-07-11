"""
Dashboard EV3 - Auditoria de toxicidad multiplataforma.

El objetivo no es cerrar la discusion en "YouTube es mas toxico", sino mostrar
que el ranking abre preguntas sobre corpus, comunidades, moderacion, contexto y
riesgo reputacional. Mantiene `server` exportado para Render/gunicorn.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "Adolfo" / "results" / "metrics"
DISCUSSION_JSON = METRICS_DIR / "dashboard_discussion_summary.json"
EXPLAIN_JSON = METRICS_DIR / "toxicity_explain.json"
DIST_FIG = ROOT / "Adolfo" / "results" / "figures" / "toxicity_distribution.png"

ORDER = ["youtube", "reddit", "twitter"]
COLORS = {"youtube": "#d62728", "reddit": "#ff7f0e", "twitter": "#2ca02c"}
CARD = {
    "padding": "14px",
    "borderRadius": "8px",
    "background": "#f8fafc",
    "border": "1px solid #e5e7eb",
}
NOTE = {
    "background": "#fff7e6",
    "borderLeft": "4px solid #f59e0b",
    "padding": "12px",
    "borderRadius": "6px",
    "margin": "10px 0",
}
BLUE_NOTE = {
    "background": "#eff6ff",
    "borderLeft": "4px solid #2563eb",
    "padding": "12px",
    "borderRadius": "6px",
    "margin": "10px 0",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def encode_fig(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


SUMMARY = load_json(DISCUSSION_JSON)
EXPLAIN = load_json(EXPLAIN_JSON, {})
DIST_B64 = encode_fig(DIST_FIG)


def platform_label(value: str | None) -> str:
    return str(value or "N/D").capitalize()


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/D"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None:
        return "N/D"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "N/D"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_p(value: Any) -> str:
    if value is None:
        return "N/D"
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def ordered_platforms(items: list[str] | None = None) -> list[str]:
    present = set(items or platform_metrics().keys())
    ordered = [p for p in ORDER if p in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def platform_metrics() -> dict[str, dict[str, Any]]:
    if SUMMARY and SUMMARY.get("platform_metrics"):
        return SUMMARY["platform_metrics"]
    # Fallback for older deployments.
    dist = EXPLAIN.get("opcion1_distribucion", {})
    thr = EXPLAIN.get("opcion2_umbral", {})
    out = {}
    for p in sorted(set(dist) | set(thr)):
        d = dist.get(p, {})
        t = thr.get(p, {})
        out[p] = {
            "n": d.get("n") or t.get("n"),
            "mean": d.get("mean"),
            "median": d.get("median"),
            "std": d.get("std"),
            "p25": d.get("q25"),
            "p50": d.get("median"),
            "p75": d.get("q75"),
            "p90": None,
            "p95": d.get("q95"),
            "max": d.get("max"),
            "pct_sobre_medio_toxico_mhs": t.get("pct_toxicos"),
            "n_sobre_medio_toxico_mhs": t.get("n_toxicos"),
        }
    return out


def platform_df() -> pd.DataFrame:
    rows = []
    metrics = platform_metrics()
    for p in ordered_platforms(list(metrics)):
        d = metrics[p]
        rows.append({
            "plataforma": p,
            "n": d.get("n"),
            "media": d.get("mean"),
            "mediana": d.get("median"),
            "std": d.get("std"),
            "p25": d.get("p25"),
            "p50": d.get("p50"),
            "p75": d.get("p75"),
            "p90": d.get("p90"),
            "p95": d.get("p95"),
            "max": d.get("max"),
            "% sobre umbral": d.get("pct_sobre_medio_toxico_mhs"),
            "n sobre umbral": d.get("n_sobre_medio_toxico_mhs"),
        })
    return pd.DataFrame(rows)


def ranking_df() -> pd.DataFrame:
    ranking = SUMMARY.get("ranking", []) if SUMMARY else EXPLAIN.get("ranking_y_test", {}).get("ranking", [])
    if not ranking:
        return pd.DataFrame(columns=["plataforma", "score_medio"])
    df = pd.DataFrame(ranking).copy()
    df["score_medio"] = pd.to_numeric(df["score_medio"], errors="coerce")
    return df.sort_values("score_medio", ascending=False)


def threshold_value() -> float:
    if SUMMARY:
        return SUMMARY.get("_meta", {}).get("thresholds_used", {}).get("medio_toxico_mhs", 0.5)
    return EXPLAIN.get("_meta", {}).get("umbral_toxico", 0.5)


def data_table(df: pd.DataFrame, page_size: int = 8):
    display_df = df.copy()
    return dash_table.DataTable(
        columns=[{"name": str(c), "id": str(c)} for c in display_df.columns],
        data=display_df.to_dict("records"),
        page_size=page_size,
        style_cell={"fontFamily": "system-ui", "fontSize": 13, "padding": "7px", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f3f4f6"},
        style_table={"overflowX": "auto"},
    )


def kpi_card(title: str, value: str, note: str = "", color: str = "#111827"):
    return html.Div([
        html.Div(value, style={"fontSize": "25px", "fontWeight": "700", "color": color}),
        html.Div(title, style={"fontSize": "13px", "fontWeight": "600", "color": "#374151"}),
        html.Div(note, style={"fontSize": "12px", "color": "#6b7280", "marginTop": "4px"}) if note else None,
    ], style={**CARD, "flex": 1, "minWidth": "170px"})


def section_title(title: str, subtitle: str | None = None):
    return html.Div([
        html.H3(title, style={"marginBottom": "4px"}),
        html.P(subtitle, style={"color": "#6b7280", "marginTop": 0}) if subtitle else None,
    ])


def missing_metrics_view():
    return html.Div([
        html.H2("Faltan métricas para el dashboard"),
        html.P("Genera primero el JSON consolidado:"),
        html.Code("python etl/generate_dashboard_metrics.py"),
        html.P("Si solo existe toxicity_explain.json, el dashboard puede mostrar parte de la evidencia, pero la vista crítica queda incompleta."),
    ], style={"padding": "20px", "background": "#fff7ed", "borderRadius": "8px"})


def view_ejecutiva() -> list:
    if not SUMMARY:
        return [missing_metrics_view()]
    sample = SUMMARY.get("sample", {})
    kw = SUMMARY.get("kruskal_wallis", {})
    most = ranking_df().iloc[0]["plataforma"] if not ranking_df().empty else None
    youtube_n = sample.get("counts_by_platform", {}).get("youtube")
    main_limitation = SUMMARY.get("executive_summary", {}).get("main_limitation", "Limitaciones documentadas en metodología.")

    matrix = pd.DataFrame(SUMMARY.get("interpretation_matrix", []))
    not_conclude = SUMMARY.get("what_not_to_conclude", [])

    return [
        html.H2("Auditoría de toxicidad multiplataforma"),
        html.Div([
            html.B("Pregunta original: "),
            html.Span(SUMMARY["_meta"].get("dashboard_question_original")),
            html.Br(),
            html.B("Segunda capa interpretativa: "),
            html.Span(SUMMARY["_meta"].get("dashboard_question_evolved")),
        ], style=BLUE_NOTE),
        html.Div([
            kpi_card("Mayor toxicidad relativa", platform_label(most), "En este corpus, no universal", COLORS.get(str(most), "#111827")),
            kpi_card("Comentarios analizados", fmt_int(sample.get("total_comments")), "Measuring Hate Speech"),
            kpi_card("Kruskal-Wallis p", fmt_p(kw.get("p_value")), "Diferencias estadísticamente significativas"),
            kpi_card("n YouTube", fmt_int(youtube_n), "Muestra menor que Reddit/Twitter"),
            kpi_card("Limitación principal", "Contexto", main_limitation),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "14px 0"}),
        html.Div([
            html.H3("Entonces, ¿de qué sirve saber esto?"),
            html.P("No sirve para afirmar que YouTube sea universalmente más tóxico ni entrega una acción directa al consumidor individual."),
            html.P("Sí sirve como señal de riesgo para equipos de análisis, moderación o marcas: indica dónde conviene investigar con más detalle antes de tomar decisiones de exposición, monitoreo o moderación."),
        ], style=NOTE),
        html.Div([
            html.H3("Hallazgo central"),
            html.P(SUMMARY.get("executive_summary", {}).get("central_finding")),
            html.P(SUMMARY.get("executive_summary", {}).get("main_value")),
        ], style={**CARD, "margin": "12px 0"}),
        section_title("Discusión abierta", "El dato no cierra una causa: abre líneas de análisis defendibles."),
        html.Div([
            kpi_card("Comunidades", "Hipótesis", "¿La toxicidad viene de pocos canales o grupos?"),
            kpi_card("Origen del corpus", "Parcial", "Muestra, recolección y periodo pueden influir."),
            kpi_card("Diseño de interacción", "Futuro", "Comentarios, recomendación y moderación no están medidos."),
            kpi_card("Marcas/anunciantes", "Riesgo", "No mide anuncios, pero sugiere monitoreo de brand safety."),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "14px"}),
        section_title("Matriz de interpretación", "Cada hallazgo se separa de la explicación posible y del dato faltante."),
        data_table(matrix, page_size=5) if not matrix.empty else html.P("No hay matriz disponible."),
        section_title("Qué NO podemos concluir"),
        html.Ul([html.Li(item) for item in not_conclude]),
    ]


def score_chart():
    df = ranking_df()
    if df.empty:
        return html.P("Ranking no disponible.")
    fig = px.scatter(
        df,
        x="score_medio",
        y="plataforma",
        color="plataforma",
        color_discrete_map=COLORS,
        text=df["score_medio"].map(lambda x: f"{x:.3f}"),
        title="Score medio de toxicidad por plataforma",
    )
    fig.update_traces(marker={"size": 18}, textposition="middle right")
    fig.update_layout(
        showlegend=False,
        height=330,
        xaxis_title="hate_speech_score medio (más a la derecha = más tóxico)",
        yaxis_title="",
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(df["plataforma"].tolist()))},
        margin=dict(t=55, b=20),
    )
    return dcc.Graph(figure=fig)


def threshold_chart():
    df = platform_df()
    if df.empty or "% sobre umbral" not in df:
        return html.P("Métrica de umbral no disponible.")
    fig = px.bar(
        df,
        x="plataforma",
        y="% sobre umbral",
        color="plataforma",
        color_discrete_map=COLORS,
        text="% sobre umbral",
        title=f"Comentarios sobre el umbral tóxico (> {threshold_value()})",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, height=340, xaxis_title="", yaxis_title="% de comentarios", margin=dict(t=55, b=20))
    return dcc.Graph(figure=fig)


def mean_median_chart():
    df = platform_df()
    if df.empty:
        return html.P("Distribución no disponible.")
    long = df[["plataforma", "media", "mediana", "p95"]].melt("plataforma", var_name="métrica", value_name="score")
    fig = px.bar(long, x="plataforma", y="score", color="métrica", barmode="group", title="Media, mediana y p95 por plataforma")
    fig.update_layout(height=360, xaxis_title="", yaxis_title="hate_speech_score", margin=dict(t=55, b=20))
    return dcc.Graph(figure=fig)


def sample_chart():
    df = platform_df()
    if df.empty:
        return html.P("Tamaño muestral no disponible.")
    fig = px.bar(df, x="plataforma", y="n", color="plataforma", color_discrete_map=COLORS, text="n", title="Tamaño muestral por plataforma")
    fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="comentarios", margin=dict(t=55, b=20))
    return dcc.Graph(figure=fig)


def synthetic_real_chart():
    if not SUMMARY:
        return html.P("Métricas sintético vs real no disponibles.")
    m = SUMMARY.get("synthetic_vs_real", {})
    rows = [
        {"métrica": "F1 sintético EV2", "valor": m.get("f1_macro_synthetic_reported_ev2")},
        {"métrica": "F1 texto real", "valor": m.get("f1_macro_on_real_data")},
        {"métrica": "F1 toxicidad sintética", "valor": m.get("toxicity_synthetic_f1_macro")},
    ]
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return html.P("No hay benchmark sintético vs real disponible.")
    fig = px.bar(df, x="métrica", y="valor", text="valor", title="Generalización: dato sintético vs texto real")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=330, xaxis_title="", yaxis_title="F1-macro", yaxis={"range": [0, 1]}, margin=dict(t=55, b=20))
    return dcc.Graph(figure=fig)


def view_tecnica() -> list:
    if not SUMMARY:
        return [missing_metrics_view()]
    kw = SUMMARY.get("kruskal_wallis", {})
    pairwise = pd.DataFrame(SUMMARY.get("pairwise_tests", []))
    df_metrics = platform_df()
    corr = SUMMARY.get("sentiment_toxicity_correlation", {})
    limitations = SUMMARY.get("limitations", [])

    metric_table = df_metrics.copy()
    if not metric_table.empty:
        for col in ["media", "mediana", "std", "p25", "p50", "p75", "p90", "p95", "max"]:
            if col in metric_table:
                metric_table[col] = metric_table[col].map(lambda v: fmt_num(v, 3) if pd.notna(v) else "N/D")
        metric_table["% sobre umbral"] = metric_table["% sobre umbral"].map(fmt_pct)

    return [
        html.H2("Vista técnica · Evidencia estadística"),
        html.P("Cada gráfico responde una pregunta distinta: ranking, proporción sobre umbral, distribución y validez de modelos."),
        section_title("1 · Ranking de score medio", "Más a la derecha significa mayor hate_speech_score promedio."),
        score_chart(),
        html.P(f"Kruskal-Wallis: H = {fmt_num(kw.get('H'), 4)}, p = {fmt_p(kw.get('p_value'))}. Resultado significativo: {kw.get('significant')}.", style={"color": "#6b7280"}),
        data_table(pairwise, page_size=5) if not pairwise.empty else html.P("Comparaciones por pares no disponibles."),
        section_title(f"2 · Comentarios que cruzan el umbral tóxico (> {threshold_value()})"),
        threshold_chart(),
        section_title("3 · Distribución", "La media resume el centro; p95 y máximo muestran la cola extrema."),
        mean_median_chart(),
        html.Img(src=DIST_B64, style={"maxWidth": "100%", "marginTop": "8px"}) if DIST_B64 else html.P("Boxplot no disponible. Ejecutar etl/explain_toxicity_platform.py para regenerarlo."),
        data_table(metric_table, page_size=5),
        section_title("4 · Tamaño muestral y calidad de lectura"),
        sample_chart(),
        html.Div(f"Advertencia de desbalance: {SUMMARY.get('sample', {}).get('imbalance_warning')} · ratio max/min n = {SUMMARY.get('sample', {}).get('imbalance_ratio_max_min')}", style=NOTE),
        section_title("5 · Sentimiento-toxicidad"),
        html.Div([
            html.B("Estado: "), html.Span(corr.get("status", "N/D")), html.Br(),
            html.Span(corr.get("reason") or corr.get("note") or "Sin observación adicional."),
        ], style=BLUE_NOTE),
        section_title("6 · Sintético vs real"),
        synthetic_real_chart(),
        html.P("El F1 alto en datos sintéticos no garantiza generalización. Esta brecha explica por qué el dashboard separa evidencia real de evidencia sintética.", style={"color": "#6b7280"}),
        section_title("7 · Limitaciones técnicas detectadas"),
        html.Ul([html.Li(x) for x in limitations]),
    ]


def terms_block():
    terms = SUMMARY.get("distinctive_terms", {}) if SUMMARY else {}
    if not terms:
        return html.P("No hay términos distintivos disponibles.")
    blocks = []
    for p in ordered_platforms(list(terms)):
        data = terms.get(p)
        if not isinstance(data, list):
            continue
        df = pd.DataFrame(data).head(8)
        blocks.append(html.Div([
            html.H4(platform_label(p)),
            data_table(df, page_size=8) if not df.empty else html.P("Sin términos."),
        ], style={**CARD, "flex": 1, "minWidth": "260px"}))
    return html.Div(blocks, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})


def view_contexto() -> list:
    if not SUMMARY:
        return [missing_metrics_view()]
    hypotheses = pd.DataFrame(SUMMARY.get("hypotheses", []))
    context = SUMMARY.get("context_analysis", {})
    limitations = SUMMARY.get("limitations", [])

    return [
        html.H2("Vista contexto · Hipótesis y discusión"),
        html.P("Esta vista separa lo comprobado de lo que solo puede discutirse como hipótesis futura."),
        section_title("Hipótesis evaluadas"),
        data_table(hypotheses, page_size=10) if not hypotheses.empty else html.P("No hay hipótesis disponibles."),
        section_title("Qué permite explicar el corpus"),
        html.Div([
            kpi_card("Año/fecha", context.get("year", {}).get("status", "N/D"), context.get("year", {}).get("reason", "")),
            kpi_card("Comunidad/canal", context.get("community", {}).get("status", "N/D"), context.get("community", {}).get("reason", "")),
            kpi_card("País/idioma", context.get("language_country", {}).get("status", "N/D"), context.get("language_country", {}).get("reason", "")),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        section_title("Términos distintivos en comentarios tóxicos", "Puede contener lenguaje ofensivo porque proviene del corpus analizado. Sirve como evidencia exploratoria, no causal."),
        terms_block(),
        section_title("Implicancia para marcas y anunciantes"),
        html.Div([
            html.P("Si una marca aparece junto a conversaciones de alta toxicidad, podría existir riesgo reputacional."),
            html.P("Este proyecto no mide anuncios ni presencia de marcas; por lo tanto no se puede afirmar impacto publicitario directo."),
            html.P("Sí se propone como línea futura cruzar toxicidad con categorías de video, presencia de marca, campañas y cambios de moderación."),
        ], style=NOTE),
        section_title("Limitaciones que quedan abiertas"),
        html.Ul([html.Li(x) for x in limitations]),
    ]


def view_metodologia() -> list:
    if not SUMMARY:
        return [missing_metrics_view()]
    syn = SUMMARY.get("synthetic_vs_real", {})
    meta = SUMMARY.get("_meta", {})

    return [
        html.H2("Vista metodología · Cómo leer el resultado"),
        section_title("Tres capas de evidencia"),
        html.Div([
            kpi_card("Capa 1", "Sintético", "Sirve para probar pipeline, pero la toxicidad sintética no tiene señal fuerte."),
            kpi_card("Capa 2", "Texto real", "Measuring Hate Speech permite auditar toxicidad anotada por personas."),
            kpi_card("Capa 3", "Discusión", "Hipótesis, limitaciones y datos faltantes para no caer en causalidad falsa."),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        section_title("Por qué no mezclamos todas las fuentes por fila"),
        html.P("Las fuentes no miden exactamente lo mismo ni comparten una llave común de comentario. Por eso se integran por capas: el sintético valida técnicas, Measuring Hate Speech responde toxicidad real y el dashboard conecta ambas lecturas sin forzar joins artificiales."),
        section_title("Qué aprendimos del dataset sintético"),
        html.Ul([
            html.Li(f"Sentimiento sintético: F1-macro ≈ {fmt_num(syn.get('sentiment_synthetic_f1_macro'), 3)}."),
            html.Li(f"Sentimiento al probar en texto real: F1-macro ≈ {fmt_num(syn.get('f1_macro_on_real_data'), 3)}."),
            html.Li(f"Toxicidad sintética con señal: {syn.get('toxicity_synthetic_has_signal')}."),
        ]),
        section_title("Cómo interpretar el resultado de YouTube"),
        html.Div([
            html.P("Significa: dentro de este corpus, YouTube aparece con mayor toxicidad relativa."),
            html.P("No significa: que YouTube sea universalmente más tóxico, que toda la plataforma sea tóxica o que exista una causa probada."),
            html.P("Abre hipótesis: comunidades, corpus, tipo de contenido, moderación, diseño de interacción y sesgo de muestra."),
            html.P("Datos faltantes: comunidad/canal, fecha por comentario, país/idioma, categoría de contenido, exposición a marcas y reglas de moderación."),
        ], style=BLUE_NOTE),
        section_title("Reproducibilidad"),
        html.Ul([
            html.Li("Generar métricas: python etl/generate_dashboard_metrics.py"),
            html.Li("Dashboard local: python dashboards/app_dash.py"),
            html.Li("API local: uvicorn api.main:app --reload --port 8000"),
            html.Li("Docker: docker compose up --build"),
            html.Li(f"Fuente de métricas en esta ejecución: {meta.get('metric_source')}"),
        ]),
        section_title("Conclusión defendible"),
        html.Div(SUMMARY.get("recommended_conclusion"), style=NOTE),
    ]


def story_block(number: str, title: str, lines: list[str], phrase: str, visual=None, note: str | None = None):
    text_children = [
        html.Div(number, style={
            "width": "34px", "height": "34px", "borderRadius": "50%", "background": "#0f172a",
            "color": "white", "display": "flex", "alignItems": "center", "justifyContent": "center",
            "fontWeight": "700", "flexShrink": 0,
        }),
        html.Div([
            html.H3(title, style={"margin": "0 0 6px"}),
            *[html.P(line, style={"margin": "6px 0", "lineHeight": "1.45"}) for line in lines],
            html.Div([html.B("Frase para decir: "), html.Span(phrase)], style={
                "background": "#ecfdf5", "borderLeft": "4px solid #10b981", "padding": "10px",
                "borderRadius": "6px", "marginTop": "10px", "fontSize": "14px",
            }),
            html.Div(note, style={"color": "#6b7280", "fontSize": "13px", "marginTop": "8px"}) if note else None,
        ], style={"flex": 1}),
    ]
    children = [html.Div(text_children, style={"display": "flex", "gap": "12px", "alignItems": "flex-start"})]
    if visual is not None:
        children.append(html.Div(visual, style={"marginTop": "12px"}))
    return html.Div(children, style={**CARD, "margin": "14px 0", "background": "#ffffff"})


def compact_hypotheses_table():
    if not SUMMARY:
        return html.P("Matriz de hipotesis no disponible.")
    rows = SUMMARY.get("hypotheses", [])
    if not rows:
        return html.P("Matriz de hipotesis no disponible.")
    df = pd.DataFrame(rows)
    keep = [c for c in ["hipotesis", "evidencia_disponible", "estado", "dato_faltante"] if c in df.columns]
    return data_table(df[keep], page_size=7)


def implication_cards():
    return html.Div([
        kpi_card("Consumidor", "No directo", "El ranking no dice que debe hacer una persona individual."),
        kpi_card("Marcas", "Monitoreo", "Puede orientar analisis de brand safety, no medir impacto publicitario."),
        kpi_card("Moderacion", "Hipotesis", "Sugiere revisar contexto, reglas y comunidades."),
        kpi_card("Investigacion", "Siguiente paso", "Cruzar con fecha, comunidad, idioma y tipo de contenido."),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})


def view_presentacion() -> list:
    if not SUMMARY:
        return [missing_metrics_view()]
    rank = ranking_df()
    most_local = rank.iloc[0]["plataforma"] if not rank.empty else "youtube"
    second_local = rank.iloc[1]["plataforma"] if len(rank) > 1 else "reddit"
    third_local = rank.iloc[2]["plataforma"] if len(rank) > 2 else "twitter"
    kw_local = SUMMARY.get("kruskal_wallis", {})
    syn_local = SUMMARY.get("synthetic_vs_real", {})
    sample_local = SUMMARY.get("sample", {})
    conclusion = SUMMARY.get("recommended_conclusion", "Un ranking aislado no sirve; un ranking con limites abre una discusion util.")

    intro_visual = html.Div([
        kpi_card("Pregunta", "Toxicidad", "Es Twitter la plataforma mas toxica?"),
        kpi_card("Hipotesis", "Twitter", "Intuicion inicial del equipo, no conclusion."),
        kpi_card("Evidencia", fmt_int(sample_local.get("total_comments")), "Comentarios del corpus MHS."),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})

    evidence_note = (
        f"Resultado estadistico: Kruskal-Wallis H = {fmt_num(kw_local.get('H'), 4)}, "
        f"p = {fmt_p(kw_local.get('p_value'))}. Aplica al corpus analizado."
    )

    return [
        html.H2("Modo Presentacion - De intuicion a aprendizaje"),
        html.P("Esta vista esta ordenada para exponer el proyecto sin saltar entre graficos. Cada bloque tiene una idea, una evidencia y una frase para decir en voz alta.", style={"color": "#4b5563"}),
        story_block(
            "1",
            "Pregunta inicial",
            [
                "El punto de partida fue una pregunta simple: que plataforma aparece con mayor toxicidad en los datos disponibles.",
                "La pregunta parece de ranking, pero en ciencia de datos el ranking solo es el inicio de la discusion.",
            ],
            "Partimos con una intuicion comun, pero decidimos tratarla como hipotesis y no como verdad.",
            intro_visual,
        ),
        story_block(
            "2",
            "Hipotesis del equipo",
            [
                "La hipotesis inicial era que Twitter seria la plataforma mas toxica. Parecia razonable por su uso publico, discusiones rapidas y alta exposicion de opiniones.",
                "Esa idea no se podia defender con percepcion: habia que contrastarla con datos y dejar claro el alcance del corpus.",
            ],
            "En datos, una intuicion se respeta, pero se prueba.",
        ),
        story_block(
            "3",
            "Primer problema: el dato sintetico no bastaba",
            [
                "El dataset sintetico sirvio para construir pipeline, API y modelos, pero no era suficiente para concluir toxicidad real.",
                f"La senal de toxicidad sintetica fue debil (F1-macro cercano a {fmt_num(syn_local.get('toxicity_synthetic_f1_macro'), 3)}) y el salto a texto real bajo el rendimiento a {fmt_num(syn_local.get('f1_macro_on_real_data'), 3)}.",
            ],
            "Teniamos un sistema que funcionaba, pero todavia no teniamos evidencia fuerte para sostener una conclusion real sobre toxicidad.",
            synthetic_real_chart(),
        ),
        story_block(
            "4",
            "Evidencia real: la hipotesis se refuto",
            [
                f"Con el corpus Measuring Hate Speech, {platform_label(most_local)} aparece con mayor toxicidad relativa, seguido de {platform_label(second_local)} y luego {platform_label(third_local)}.",
                "El resultado fue contrario a la intuicion inicial: Twitter no quedo primero dentro de este corpus.",
            ],
            "La hipotesis inicial no se confirmo; ese tambien es un resultado valido.",
            score_chart(),
            note=evidence_note,
        ),
        story_block(
            "5",
            "Discusion: por que podria pasar?",
            [
                "Decir que YouTube aparece mas toxico no basta. La pregunta util es que condiciones del dato podrian explicarlo.",
                "El proyecto separa lo observado de lo no comprobable: comunidades, ano, idioma, pais, moderacion y tipo de contenido requieren mas datos.",
            ],
            "Un ranking aislado no sirve; un ranking con evidencia, limites e hipotesis abre una discusion util.",
            compact_hypotheses_table(),
        ),
        story_block(
            "6",
            "Implicancia: consumidor, marcas, moderacion e investigacion",
            [
                "Para un consumidor individual, el dato no entrega una accion directa ni prueba una verdad universal.",
                "Para una marca o equipo de moderacion, si puede funcionar como senal para investigar exposicion, comunidades y reglas de interaccion.",
            ],
            "El valor no esta en culpar una plataforma; esta en detectar donde conviene mirar con mas contexto.",
            implication_cards(),
        ),
        story_block(
            "7",
            "Aprendizaje final",
            [
                conclusion,
                "La conclusion mas defendible no fuerza una solucion cerrada: reconoce que se puede afirmar, que falta y que deberia investigarse despues.",
            ],
            "Una buena solucion de ciencia de datos no fuerza respuestas: convierte una intuicion en una discusion basada en evidencia.",
        ),
    ]


app = Dash(__name__, title="EV3 · Toxicidad multiplataforma")
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("EV3 · Auditoría de toxicidad multiplataforma", style={"margin": "0 0 4px"}),
        html.Div("De ranking simple a discusión basada en evidencia", style={"color": "#6b7280", "fontStyle": "italic"}),
        dcc.RadioItems(
            id="audiencia",
            options=[
                {"label": " Modo Presentación", "value": "Presentación"},
                {"label": " Ejecutiva", "value": "Ejecutiva"},
                {"label": " Técnica", "value": "Técnica"},
                {"label": " Contexto / Hipótesis", "value": "Contexto"},
                {"label": " Metodología", "value": "Metodología"},
            ],
            value="Presentación",
            inline=True,
            inputStyle={"marginLeft": "16px", "marginRight": "4px"},
            style={"marginTop": "12px"},
        ),
    ], style={"borderBottom": "1px solid #e5e7eb", "paddingBottom": "14px"}),
    html.Div(id="content", style={"maxWidth": "1080px", "margin": "18px auto"}),
], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "22px", "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif", "color": "#111827"})


@app.callback(Output("content", "children"), Input("audiencia", "value"))
def render(audiencia: str):
    if audiencia == "Presentación":
        return view_presentacion()
    if audiencia == "Técnica":
        return view_tecnica()
    if audiencia == "Contexto":
        return view_contexto()
    if audiencia == "Metodología":
        return view_metodologia()
    return view_ejecutiva()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)