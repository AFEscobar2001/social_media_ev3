"""
app_dash.py — Dashboard EV3 · Auditoría de toxicidad multiplataforma (Plotly Dash)
====================================================================================
Versión ampliada, orientada a presentación en clase (SCY1101).

La pregunta central es siempre la misma:

    ¿Es una red social más tóxica que otra? ¿Y por qué?

El Dash organiza la respuesta en 5 vistas, pensadas para recorrerse en este orden:

    1. Modo Presentación   → los 7 bloques de la historia (hilo conductor de la exposición)
    2. Vista Ejecutiva      → resumen para audiencia no técnica: KPIs, semáforo, límites
    3. Vista Técnica         → ranking, pruebas estadísticas, distribución, benchmark
    4. Contexto / Hipótesis  → qué se puede discutir, qué falta, qué NO se puede concluir
    5. Metodología            → capas de evidencia, reproducibilidad, comandos

Fuente principal de la investigación: Adolfo/results/metrics/toxicity_explain.json
(generado por etl/explain_toxicity_platform.py). El CSV de campañas aporta solo
contexto de volumen de negocio. Cuando faltan datos que el guion menciona pero
que no vienen en el JSON (p. ej. comparación F1 sintético vs. real), se usan los
valores de referencia documentados en la guía de estudio como respaldo, marcados
siempre como "valor de referencia" en pantalla.

Ejecutar:
    python3 dashboards/app_dash.py        # http://127.0.0.1:8050
"""
import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../social_media_ev3
EXPLAIN_JSON = ROOT / "Adolfo" / "results" / "metrics" / "toxicity_explain.json"
DIST_FIG = ROOT / "Adolfo" / "results" / "figures" / "toxicity_distribution.png"
F1_JSON_CANDIDATES = [
    ROOT / "Adolfo" / "results" / "metrics" / "sentiment_ev2_vs_real.json",
    ROOT / "results" / "metrics" / "f1_sintetico_vs_real.json",
]
CSV_CANDIDATES = [
    ROOT / "data" / "processed" / "social_media_ev3_final.csv",
    ROOT / "data" / "processed" / "social_media_enriched.csv",
    ROOT / "Felipe" / "social_media_enriched (1).csv",
    ROOT / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv",
]

ORDER = ["youtube", "reddit", "twitter"]                # más → menos tóxico
RANK_COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]         # 🔴 más  🟡 medio  🟢 menos
ACCENT = "#1c64f2"
NAVY = "#152238"

# Valores de referencia documentados en la guía de estudio (respaldo si el
# JSON de resultados no trae alguna métrica todavía). Se muestran siempre con
# la etiqueta "valor de referencia" para no aparentar que provienen del run.
REF = {
    "n_total": 39495,
    "score_medio": {"youtube": -0.7155, "reddit": -0.9343, "twitter": -1.0794},
    "kruskal": {"H": 167.3949, "p": 4.47e-37},
    "umbral_pct": {"youtube": 28.39, "reddit": 28.17, "twitter": 22.83},
    "muestra": {"reddit": 15842, "twitter": 15475, "youtube": 8178},
    "f1_sintetico": 0.9366,
    "f1_real": 0.3442,
    "umbral_valor": 0.5,
}

HIPOTESIS_INICIAL = "Twitter sería la plataforma más tóxica."

FALTANTES = [
    {"tema": "Comunidades", "detalle": "No se puede aislar el efecto de una comunidad "
     "específica: cada comentario no incluye canal, subreddit u otra identificación "
     "equivalente."},
    {"tema": "Evolución temporal", "detalle": "No hay fecha ni año por comentario, por lo "
     "que no es posible analizar si el resultado cambia con el tiempo."},
    {"tema": "Ubicación e idioma", "detalle": "No se registran país, región, idioma ni "
     "contexto cultural de origen del comentario."},
    {"tema": "Origen del corpus", "detalle": "La forma en que se construyó la muestra "
     "podría influir en que YouTube aparezca primero. Es una explicación posible, no una "
     "causa demostrada."},
    {"tema": "Diseño de interacción y moderación", "detalle": "No hay información sobre "
     "reglas de moderación ni sobre la estructura de las conversaciones; queda como línea "
     "de investigación futura."},
    {"tema": "Marcas y anunciantes", "detalle": "El resultado permite hablar de riesgo "
     "reputacional, pero no de impacto publicitario directo: no hay datos sobre anuncios "
     "ni presencia de marcas."},
]

PREGUNTAS_PROFESOR = [
    ("¿Cuál es la principal novedad del nuevo Dash?",
     "No se limita a mostrar gráficos: organiza una historia que conecta el hallazgo con "
     "su evidencia, sus límites y las hipótesis que quedan abiertas."),
    ("¿Por qué YouTube aparece primero?",
     "Porque presenta el mayor hate_speech_score promedio dentro de este corpus. Con los "
     "datos disponibles no es posible demostrar la causa."),
    ("¿Esto prueba que YouTube es siempre más tóxico?",
     "No. El resultado solo describe el corpus analizado y no puede generalizarse a todos "
     "los contenidos o contextos."),
    ("¿Por qué Twitter no quedó primero?",
     "Porque la evidencia no confirmó la intuición inicial. Que una hipótesis sea "
     "refutada también es un resultado válido."),
    ("¿Qué aporta Kruskal-Wallis?",
     "Permite evaluar si las diferencias entre plataformas son estadísticamente "
     "significativas."),
    ("¿Cuál es la limitación más importante?",
     "Faltan variables de contexto: comunidad, fecha, país, idioma, tipo de contenido y "
     "moderación."),
    ("¿Qué utilidad tiene para las marcas?",
     "Puede orientar el monitoreo de riesgo reputacional, pero no mide directamente el "
     "impacto de la publicidad ni la exposición de una marca."),
    ("¿Qué mejorarían en una siguiente versión?",
     "Incorporar los datos crudos completos, agregar variables de contexto, analizar "
     "comunidades y validar si el resultado se mantiene a lo largo del tiempo."),
]


# ---------------------------------------------------------------------------
# Carga de datos (al iniciar la app)
# ---------------------------------------------------------------------------
def load_explain() -> dict | None:
    if EXPLAIN_JSON.exists():
        return json.loads(EXPLAIN_JSON.read_text())
    return None


def load_csv() -> pd.DataFrame | None:
    for path in CSV_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_f1() -> dict:
    """F1 sintético (EV2) vs. F1 sobre texto real. Usa el JSON si existe;
    si no, cae al valor de referencia documentado en la guía de estudio."""
    for path in F1_JSON_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return {
                    "sintetico": float(data.get("f1_sintetico", REF["f1_sintetico"])),
                    "real": float(data.get("f1_real", REF["f1_real"])),
                    "fuente": "run",
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                break
    return {"sintetico": REF["f1_sintetico"], "real": REF["f1_real"], "fuente": "referencia"}


def encode_fig(path: Path) -> str | None:
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return None


EX = load_explain()
DF_CSV = load_csv()
DIST_B64 = encode_fig(DIST_FIG)
F1 = load_f1()


def ordered(d: dict) -> list[str]:
    return [p for p in ORDER if p in d]


def color_map(platforms: list[str]) -> dict:
    return {p: RANK_COLORS[i] if i < len(RANK_COLORS) else "#777"
            for i, p in enumerate(platforms)}


def get_score_medio() -> dict:
    """{'youtube': -0.71, ...} desde el JSON si está, si no desde REF."""
    if EX is not None:
        try:
            rows = EX["ranking_y_test"]["ranking"]
            return {r["plataforma"]: float(r["score_medio"]) for r in rows}
        except (KeyError, TypeError):
            pass
    return dict(REF["score_medio"])


def get_kw() -> dict:
    if EX is not None:
        try:
            kw = EX["ranking_y_test"]["kruskal_wallis"]
            return {"H": kw["H"], "p": kw["p_valor"], "significativo": kw["significativo"]}
        except (KeyError, TypeError):
            pass
    return {"H": REF["kruskal"]["H"], "p": REF["kruskal"]["p"], "significativo": True}


def get_umbral_pct() -> dict:
    if EX is not None:
        try:
            d2 = EX["opcion2_umbral"]
            return {p: d2[p]["pct_toxicos"] for p in ordered(d2)}
        except (KeyError, TypeError):
            pass
    return dict(REF["umbral_pct"])


def get_muestra() -> dict:
    if EX is not None:
        try:
            d1 = EX["opcion1_distribucion"]
            return {p: d1[p]["n"] for p in ordered(d1)}
        except (KeyError, TypeError):
            pass
    return dict(REF["muestra"])


def get_n_total() -> int:
    if EX is not None:
        try:
            return int(EX["_meta"]["n_total"])
        except (KeyError, TypeError):
            pass
    return REF["n_total"]


def get_umbral_valor() -> float:
    if EX is not None:
        try:
            return float(EX["_meta"]["umbral_toxico"])
        except (KeyError, TypeError):
            pass
    return REF["umbral_valor"]


def toxicity_rank_df(rt: dict | None = None) -> pd.DataFrame:
    """Ranking ordenado de mayor a menor toxicidad, con un índice positivo
    fácil de leer (el score original de measuringhatespeech puede ser negativo)."""
    if rt is not None:
        df = pd.DataFrame(rt["ranking"]).copy()
        df["score_medio"] = pd.to_numeric(df["score_medio"], errors="coerce")
    else:
        scores = get_score_medio()
        df = pd.DataFrame({"plataforma": list(scores.keys()),
                            "score_medio": list(scores.values())})
    df = df.sort_values("score_medio", ascending=False)
    min_score = df["score_medio"].min()
    df["toxicidad_relativa"] = df["score_medio"] - min_score
    df["score_original"] = df["score_medio"].map(lambda x: f"{x:.4f}")
    df["lectura"] = df.apply(lambda r: f"{r['plataforma']} ({r['score_original']})", axis=1)
    return df


# ---------------------------------------------------------------------------
# Componentes visuales reutilizables
# ---------------------------------------------------------------------------
def table(df: pd.DataFrame, idx_name: str | None = None):
    if idx_name:
        df = df.reset_index().rename(columns={"index": idx_name})
    return dash_table.DataTable(
        columns=[{"name": str(c), "id": str(c)} for c in df.columns],
        data=df.to_dict("records"),
        style_cell={"fontFamily": "system-ui", "fontSize": 14, "padding": "8px",
                    "textAlign": "left"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
        style_table={"overflowX": "auto"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
    )


def kpi_card(value: str, label: str, sublabel: str | None = None, color: str = NAVY):
    children = [html.Div(value, style={"fontSize": 30, "fontWeight": 700, "color": color}),
                html.Div(label, style={"color": "#444", "fontWeight": 600, "marginTop": 4})]
    if sublabel:
        children.append(html.Div(sublabel, style={"color": "#888", "fontSize": 12,
                                                    "marginTop": 4}))
    return html.Div(children, style={"flex": 1, "minWidth": 160, "padding": "16px",
                                      "background": "#fafafa", "borderRadius": 10,
                                      "textAlign": "center", "border": "1px solid #eee"})


def kpi_row(cards: list):
    return html.Div(cards, style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                                   "margin": "16px 0"})


def section_header(title: str, subtitle: str | None = None):
    children = [html.H3(title, style={"marginBottom": 4})]
    if subtitle:
        children.append(html.P(subtitle, style={"color": "#666", "marginTop": 0}))
    return html.Div(children, style={"marginTop": 26})


def info_box(text: str, color: str = ACCENT, bg: str = "#e7f1ff", icon: str = "ℹ️"):
    return html.Div([html.Span(icon, style={"marginRight": 8}), text],
                     style={"background": bg, "borderLeft": f"4px solid {color}",
                            "padding": "12px 14px", "borderRadius": 6, "margin": "12px 0",
                            "lineHeight": 1.5})


def limit_card(color: str, bg: str, titulo: str, texto: str):
    return html.Div([html.B(titulo), html.P(texto, style={"margin": "8px 0 0"})],
                     style={"flex": 1, "minWidth": 220, "padding": "14px", "borderRadius": 8,
                            "background": bg, "borderLeft": f"4px solid {color}"})


def ref_tag(fuente: str = "referencia"):
    """Etiqueta pequeña indicando si un número viene del run o es de respaldo."""
    if fuente == "run":
        return html.Span(" (dato del run)", style={"color": "#999", "fontSize": 11})
    return html.Span(" (valor de referencia)", style={"color": "#c77700", "fontSize": 11})


# ---------------------------------------------------------------------------
# VISTA 1 · MODO PRESENTACIÓN (7 bloques)
# ---------------------------------------------------------------------------
BLOQUES = [
    "Pregunta inicial", "Hipótesis del equipo", "Datos sintéticos",
    "Evidencia real", "Discusión", "Implicancias", "Aprendizaje",
]


def bloque_progreso(n: int):
    dots = []
    for i in range(1, 8):
        active = i == n
        dots.append(html.Div(style={
            "width": 10, "height": 10, "borderRadius": "50%",
            "background": ACCENT if active else "#dbe4f0",
            "margin": "0 4px", "transition": "background .2s",
        }))
    return html.Div(dots, style={"display": "flex", "justifyContent": "center",
                                  "margin": "6px 0 18px"})


def bloque_1():
    scores = get_score_medio()
    return [
        html.H2("1 · Pregunta inicial"),
        kpi_row([
            kpi_card(f"{get_n_total():,}".replace(",", "."), "Comentarios analizados",
                      "Corpus real: Measuring Hate Speech"),
            kpi_card("3", "Plataformas comparadas", "YouTube · Reddit · Twitter"),
            kpi_card("¿Cuál es más tóxica?", "Pregunta de investigación", None,
                      color=ACCENT),
        ]),
        html.P("Cuando se habla de toxicidad en redes sociales es normal formarse una "
               "opinión rápida sobre qué plataforma es más problemática. El proyecto no se "
               "quedó en esa impresión: convirtió la pregunta en algo medible y la puso a "
               "prueba con datos.", style={"lineHeight": 1.6}),
        info_box("La pregunta parece sencilla, pero no puede responderse solo con una "
                 "percepción; necesita una medición.", icon="🎯"),
    ]


def bloque_2():
    return [
        html.H2("2 · Hipótesis del equipo"),
        html.Div([
            html.Div("🐦", style={"fontSize": 40}),
            html.Div([html.B(HIPOTESIS_INICIAL),
                      html.P("Punto de partida razonable, no una conclusión anticipada.",
                             style={"margin": "6px 0 0", "color": "#666"})]),
        ], style={"display": "flex", "gap": "16px", "alignItems": "center",
                  "padding": "18px", "background": "#fff4e5", "borderRadius": 10,
                  "borderLeft": "4px solid #ff7f0e"}),
        html.P("La intuición inicial del equipo apuntaba a Twitter como la red más "
               "tóxica. El objetivo del proyecto fue comprobar si esa percepción se "
               "sostenía al analizarla con datos reales, no descartarla de antemano ni "
               "confirmarla por default.", style={"lineHeight": 1.6, "marginTop": 16}),
    ]


def bloque_3():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Datos sintéticos", "Texto real"],
                          y=[F1["sintetico"], F1["real"]],
                          marker_color=["#8ea4c7", "#d62728"],
                          text=[f"{F1['sintetico']:.2f}", f"{F1['real']:.2f}"],
                          textposition="outside"))
    fig.update_layout(title="F1 del modelo de sentimiento (EV2): sintético vs. real",
                       yaxis_title="F1-score", yaxis_range=[0, 1.1], height=340,
                       margin=dict(t=50, b=20))
    return [
        html.H2("3 · Datos sintéticos"),
        html.P("Los datos sintéticos fueron útiles para desarrollar y probar el "
               "pipeline, los modelos y la API — pero no bastan para describir la "
               "toxicidad real.", style={"lineHeight": 1.6}),
        dcc.Graph(figure=fig),
        html.P(["La diferencia es considerable: el F1 es alto con datos sintéticos y "
                "cae fuerte con texto real.", ref_tag(F1["fuente"])],
               style={"color": "#666"}),
        info_box("Por eso el Dash separa con claridad el desarrollo técnico (sintético) "
                 "de la evidencia real (Measuring Hate Speech).", icon="🧪"),
    ]


def bloque_4():
    df_rank = toxicity_rank_df()
    plats = df_rank["plataforma"].tolist()
    fig = px.bar(df_rank, x="plataforma", y="score_medio", color="plataforma",
                 color_discrete_map=color_map(plats), text="score_original",
                 title="Score medio de toxicidad por plataforma (evidencia real)")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=360, xaxis_title="", margin=dict(t=50, b=10),
                       yaxis_title="hate_speech_score medio")
    kw = get_kw()
    return [
        html.H2("4 · Evidencia real"),
        dcc.Graph(figure=fig),
        info_box(f"Kruskal-Wallis: H = {kw['H']:.4f}, p ≈ {kw['p']:.2e} → las diferencias "
                  "entre plataformas son estadísticamente significativas.", icon="📊"),
        html.P("El corpus no confirmó la hipótesis inicial: dentro de esta muestra, "
               "YouTube aparece primero, seguido de Reddit y Twitter.",
               style={"lineHeight": 1.6}),
    ]


def bloque_5():
    df_falt = pd.DataFrame(FALTANTES).rename(columns={"tema": "Tema", "detalle": "Qué falta / qué se puede discutir"})
    return [
        html.H2("5 · Discusión"),
        html.P("Que una diferencia sea significativa no significa que se conozca su "
               "causa. Esta tabla separa lo que el corpus permite discutir de lo que "
               "todavía no puede probarse.", style={"lineHeight": 1.6}),
        table(df_falt),
    ]


def bloque_6():
    return [
        html.H2("6 · Implicancias"),
        kpi_row([
            limit_card(ACCENT, "#eef4ff", "🧑‍🤝‍🧑 Consumidores",
                       "El resultado no indica qué plataforma debería usar una persona en "
                       "particular."),
            limit_card("#ff7f0e", "#fff4e5", "🏷️ Marcas",
                       "Puede orientar el monitoreo de riesgo reputacional, sin medir "
                       "impacto publicitario directo."),
            limit_card("#2ca02c", "#eafaf1", "🛡️ Moderación",
                       "Ayuda a priorizar dónde enfocar recursos de moderación."),
            limit_card("#8e44ad", "#f5eefb", "🔬 Investigación",
                       "Abre preguntas sobre comunidad, fecha, idioma y tipo de "
                       "contenido para una siguiente versión."),
        ]),
    ]


def bloque_7():
    return [
        html.H2("7 · Aprendizaje"),
        info_box("Una conclusión responsable presenta la evidencia y, al mismo tiempo, "
                 "reconoce lo que todavía no puede afirmarse.", icon="🎓", color="#2ca02c",
                 bg="#eafaf1"),
        html.P("En este corpus, YouTube presenta la mayor toxicidad relativa, pero el "
               "resultado no demuestra causalidad ni puede generalizarse a todo internet. "
               "El aporte del proyecto es tanto técnico como analítico: integra datos, "
               "genera métricas, visualiza evidencia y ayuda a interpretar los resultados "
               "sin ir más allá de lo que permiten los datos.", style={"lineHeight": 1.6}),
    ]


BLOQUE_FN = {1: bloque_1, 2: bloque_2, 3: bloque_3, 4: bloque_4, 5: bloque_5,
             6: bloque_6, 7: bloque_7}


def view_presentacion(n: int) -> list:
    n = max(1, min(7, n))
    nav = html.Div([
        html.Button("← Anterior", id="btn-prev", n_clicks=0, disabled=(n == 1),
                    style={"padding": "8px 16px", "borderRadius": 6,
                           "border": "1px solid #ccc", "background": "#fff",
                           "cursor": "pointer" if n > 1 else "default"}),
        html.Div(f"Bloque {n} de 7 · {BLOQUES[n - 1]}",
                 style={"fontWeight": 600, "color": "#444"}),
        html.Button("Siguiente →", id="btn-next", n_clicks=0, disabled=(n == 7),
                    style={"padding": "8px 16px", "borderRadius": 6, "border": "none",
                           "background": ACCENT, "color": "#fff",
                           "cursor": "pointer" if n < 7 else "default"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
    return [nav, bloque_progreso(n), html.Div(BLOQUE_FN[n](), style={"minHeight": 360})]


# ---------------------------------------------------------------------------
# VISTA 2 · EJECUTIVA
# ---------------------------------------------------------------------------
def view_ejecutiva() -> list:
    df_rank = toxicity_rank_df(EX["ranking_y_test"] if EX else None)
    mas = df_rank.iloc[0]["plataforma"]
    menos = df_rank.iloc[-1]["plataforma"]
    kw = get_kw()
    muestra = get_muestra()
    thr = get_umbral_valor()

    children = [
        html.H2("Auditoría de toxicidad en redes sociales"),
        html.H4("¿Es una red social más tóxica que otra?"),
        html.P(
            f"Con datos reales etiquetados por personas, {mas.capitalize()} es la "
            f"plataforma con mayor toxicidad relativa y {menos.capitalize()} la menor, "
            f"dentro de este corpus. La diferencia es real, no casualidad de la muestra. "
            f"Importante: es una foto de estos datos, no una verdad permanente ni "
            f"generalizable a toda la plataforma.", style={"lineHeight": 1.6}),
    ]

    # KPIs clave (según guía: mayor toxicidad relativa, comentarios analizados,
    # Kruskal-Wallis p, n YouTube)
    children.append(kpi_row([
        kpi_card(mas.capitalize(), "Mayor toxicidad relativa",
                  "Solo dentro del corpus analizado", color="#d62728"),
        kpi_card(f"{get_n_total():,}".replace(",", "."), "Comentarios analizados",
                  "Muestra amplia, no representa todo internet"),
        kpi_card(f"{kw['p']:.2e}", "Kruskal-Wallis p",
                  "Diferencias estadísticamente significativas"),
        kpi_card(f"{muestra.get('youtube', 0):,}".replace(",", "."), "n YouTube",
                  "Muestra menor que Reddit y Twitter → interpretar con cautela"),
    ]))

    # KPIs de negocio (si hay CSV de campañas)
    if DF_CSV is not None:
        kpis = [("Publicaciones analizadas", f"{len(DF_CSV):,}")]
        if "platform" in DF_CSV.columns:
            kpis.append(("Plataformas", str(DF_CSV["platform"].nunique())))
        if "campaign_name" in DF_CSV.columns:
            kpis.append(("Campañas", str(DF_CSV["campaign_name"].nunique())))
        children.append(section_header("Contexto de negocio"))
        children.append(kpi_row([kpi_card(v, k) for k, v in kpis]))

    # % sobre umbral
    d2pct = get_umbral_pct()
    plats = ordered(d2pct)
    df_thr = pd.DataFrame({"plataforma": plats, "pct": [d2pct[p] for p in plats]})
    fig = px.bar(df_thr, x="plataforma", y="pct", color="plataforma",
                 color_discrete_map=color_map(plats), text="pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, height=360, xaxis_title="",
                       yaxis_title="% de comentarios tóxicos", margin=dict(t=30, b=10))
    children += [section_header("¿Por qué una plataforma sale peor?"), dcc.Graph(figure=fig)]

    # insight "la cola, no el promedio" (si hay datos de distribución en EX)
    if EX is not None:
        d1 = EX.get("opcion1_distribucion", {})
        pl1 = ordered(d1)
        if pl1:
            medianas = [d1[p]["median"] for p in pl1]
            p95 = [d1[p]["q95"] for p in pl1]
            if (max(p95) - min(p95)) > (max(medianas) - min(medianas)):
                children.append(info_box(
                    f"El comentario típico es parecido entre plataformas, pero "
                    f"{mas.capitalize()} concentra más comentarios extremos. La "
                    f"toxicidad la empuja una minoría agresiva, no el usuario promedio — "
                    f"conviene moderar focos, no audiencias enteras.", icon="📈"))

    # ¿de qué sirve saber esto?
    children += [
        section_header("¿De qué sirve saber esto?"),
        html.P("El resultado no indica qué plataforma debería usar una persona, pero sí "
               "puede orientar análisis de riesgo, moderación y seguridad de marca.",
               style={"lineHeight": 1.6}),
    ]

    # Matriz de interpretación
    df_matriz = pd.DataFrame([
        {"Elemento": "Hallazgo", "Contenido": f"{mas.capitalize()} presenta la mayor "
         "toxicidad relativa dentro del corpus analizado."},
        {"Elemento": "Evidencia", "Contenido": f"Score medio por plataforma + prueba de "
         f"Kruskal-Wallis (p ≈ {kw['p']:.2e})."},
        {"Elemento": "Posibles explicaciones", "Contenido": "Origen del corpus, tipo de "
         "contenido, dinámicas propias de cada plataforma (no confirmadas)."},
        {"Elemento": "Datos que faltan", "Contenido": "Comunidad, fecha, idioma, país, "
         "moderación, exposición de marcas."},
    ])
    children += [section_header("Matriz de interpretación",
                                 "Separa el hallazgo, la evidencia, las explicaciones "
                                 "posibles y lo que falta — para no presentar una "
                                 "interpretación como si fuera un hecho comprobado."),
                 table(df_matriz)]

    # Qué NO podemos concluir
    children += [
        section_header("Qué NO podemos concluir"),
        html.Ul([
            html.Li("No podemos afirmar causalidad: el resultado es asociación "
                    "estadística, no una explicación del porqué."),
            html.Li("No podemos generalizar a toda la plataforma ni a todo internet: "
                    "el corpus es una muestra de 2019."),
            html.Li("No podemos comparar entre comunidades, países o idiomas: esas "
                    "variables no están en los datos."),
            html.Li("No podemos afirmar que esta jerarquía se mantendrá en el tiempo."),
        ]),
    ]

    # Semáforo de decisión
    children += [
        section_header("Qué puede ofrecer la agencia con confianza"),
        kpi_row([
            limit_card("#2ca02c", "#eafaf1", "🟢 Monitoreo de reputación por sentimiento",
                       "Funciona. Acierta ~75-80% en el mundo real, no el 94% de "
                       "laboratorio."),
            limit_card("#d62728", "#fdecea", "🔴 Detección de toxicidad sobre datos del "
                       "cliente", "No vendible: en el dataset sintético la toxicidad es "
                       "ruido sin relación con el texto."),
            limit_card("#ff7f0e", "#fff4e5", "🟡 Detección de toxicidad con datos reales",
                       "Vendible si se invierte en datos etiquetados, como en esta "
                       "auditoría."),
        ]),
    ]
    return children


# ---------------------------------------------------------------------------
# VISTA 3 · TÉCNICA
# ---------------------------------------------------------------------------
def view_tecnica() -> list:
    thr = get_umbral_valor()
    kw = get_kw()

    children = [
        html.H2("Vista técnica · Investigación de toxicidad"),
        html.P(f"Fuente: {EX['_meta']['fuente'] if EX else 'Measuring Hate Speech'} · "
               f"n={get_n_total():,} · umbral tóxico = {thr}. Asociación, no causalidad.",
               style={"color": "#666"}),
    ]

    # 1 · ranking + Kruskal-Wallis
    rt = EX["ranking_y_test"] if EX is not None else None
    df_rank = toxicity_rank_df(rt)
    plats = df_rank["plataforma"].tolist()
    fig = px.scatter(
        df_rank, x="score_medio", y="plataforma", color="plataforma",
        size="toxicidad_relativa", size_max=24, color_discrete_map=color_map(plats),
        text="score_original", title="Score medio de toxicidad por plataforma",
    )
    fig.update_traces(textposition="middle right")
    fig.update_layout(
        showlegend=False, height=340,
        xaxis_title="hate_speech_score medio (más a la derecha = más tóxico)",
        yaxis_title="",
        xaxis={"range": [df_rank["score_medio"].min() - 0.08,
                          df_rank["score_medio"].max() + 0.14]},
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(plats))},
        margin=dict(t=55, b=10),
    )
    veredicto = "significativas" if kw["significativo"] else "NO significativas"
    children += [section_header("1 · Score medio de toxicidad y significancia"),
                 dcc.Graph(figure=fig),
                 html.P(f"Kruskal-Wallis: H = {kw['H']:.4f}, p = {kw['p']:.2e} → las "
                        f"diferencias son {veredicto}. Esto indica asociación "
                        f"estadística entre plataforma y toxicidad; no demuestra "
                        f"causalidad.", style={"color": "#666"})]

    if rt is not None and "comparaciones_por_pares" in rt:
        df_pares = pd.DataFrame([
            {"comparación": k.replace("_", " "), "más tóxico": v["mas_toxico"],
             "p-valor": f"{v['p_valor']:.2e}",
             "significativo": "sí" if v["significativo"] else "no"}
            for k, v in rt["comparaciones_por_pares"].items()
        ])
        children += [html.P("Comparaciones por pares (Mann-Whitney):"), table(df_pares)]

    # 2 · umbral y distribución
    d2pct = get_umbral_pct()
    pl2 = ordered(d2pct)
    df_thr = pd.DataFrame([{"plataforma": p, "% tóxicos": d2pct[p]} for p in pl2])
    fig_thr = px.bar(df_thr, x="plataforma", y="% tóxicos", color="plataforma",
                      color_discrete_map=color_map(pl2), text="% tóxicos",
                      title=f"Comentarios sobre el umbral tóxico (> {thr})")
    fig_thr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_thr.update_layout(showlegend=False, height=360, xaxis_title="",
                           yaxis_title="% de comentarios tóxicos", margin=dict(t=55, b=20))
    children += [html.Hr(), section_header(f"2 · Comentarios sobre el umbral tóxico "
                                            f"(> {thr})",
                 "Más clara para presentar que un histograma o boxplot: muestra qué "
                 "proporción de comentarios supera el límite definido como tóxico."),
                 dcc.Graph(figure=fig_thr), table(df_thr)]

    if EX is not None and "opcion1_distribucion" in EX:
        d1 = EX["opcion1_distribucion"]
        pl1 = ordered(d1)
        df_d1 = pd.DataFrame([
            {"plataforma": p, "mediana": d1[p]["median"], "media": d1[p]["mean"],
             "media sin cola": d1[p].get("mean_sin_cola"), "p95": d1[p]["q95"],
             "máx": d1[p]["max"], "n": d1[p]["n"]} for p in pl1
        ])
        children += [html.H5("Resumen de la distribución"), table(df_d1),
                     html.P("La mediana muestra el comentario típico; el p95 y el máximo "
                            "muestran la cola extrema. Si una plataforma tiene más "
                            "porcentaje sobre el umbral y una cola alta, concentra mayor "
                            "riesgo de moderación.", style={"color": "#666"})]
    if DIST_B64:
        children += [html.H5("Boxplot / distribución completa"),
                     html.Img(src=DIST_B64, style={"maxWidth": "100%", "borderRadius": 8})]

    # 3 · tamaño muestral
    muestra = get_muestra()
    df_n = pd.DataFrame([{"plataforma": p, "n comentarios": muestra[p]}
                          for p in ordered(muestra)])
    children += [html.Hr(), section_header("3 · Tamaño muestral por plataforma",
                 "Las muestras no tienen el mismo tamaño: interpretar el resultado de "
                 "YouTube con cautela dado su n menor."), table(df_n)]

    # 4 · benchmark sintético vs real
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Bar(x=["F1 sintético (EV2)", "F1 texto real"],
                             y=[F1["sintetico"], F1["real"]],
                             marker_color=["#8ea4c7", "#d62728"],
                             text=[f"{F1['sintetico']:.4f}", f"{F1['real']:.4f}"],
                             textposition="outside"))
    fig_f1.update_layout(title="Benchmark: rendimiento del modelo sintético vs. real",
                          yaxis_title="F1-score", yaxis_range=[0, 1.1], height=340,
                          margin=dict(t=50, b=20))
    children += [html.Hr(),
                 section_header("4 · Sintético vs. real (benchmark del modelo)"),
                 dcc.Graph(figure=fig_f1),
                 html.P(["La diferencia es considerable entre entornos.", ref_tag(F1["fuente"])],
                        style={"color": "#666"}),
                 info_box("Esto muestra por qué no debe usarse el modelo entrenado con "
                          "datos sintéticos para concluir sobre toxicidad real.",
                          icon="⚠️", color="#d62728", bg="#fdecea")]

    # 5 · términos distintivos
    if EX is not None and "opcion3_terminos_distintivos" in EX:
        d3 = EX["opcion3_terminos_distintivos"]
        children += [html.Hr(), section_header("5 · Qué se dice distinto en los "
                                                "comentarios tóxicos")]
        if "_aviso" in d3:
            children.append(html.P(d3["_aviso"]))
        else:
            pl3 = ordered(d3)
            cols = []
            for p in pl3:
                terms = pd.DataFrame(d3[p])
                block = [html.B(p)]
                if not terms.empty:
                    block.append(table(terms[["termino", "z", "frecuencia"]]))
                cols.append(html.Div(block, style={"flex": 1, "minWidth": 200}))
            children.append(html.Div(cols, style={"display": "flex", "gap": "16px",
                                                    "flexWrap": "wrap"}))
            children.append(html.P(
                "Log-odds con prior de Dirichlet (Monroe et al. 2008). z alto = más "
                "característico de esa plataforma. Es qué se dice distinto, no por qué "
                "— correlación, no causa.", style={"color": "#666"}))

    # honestidad metodológica
    children += [
        html.Hr(), section_header("Honestidad metodológica"),
        html.Ul([
            html.Li("Comparación entre plataformas medidas con la misma regla "
                    "(hate_speech_score), no entre datasets distintos."),
            html.Li("El código 1 = reference se excluye siempre (no es una red social)."),
            html.Li("Resultado contra-hipótesis (Twitter NO es el más tóxico) = hallazgo "
                    "válido, no fracaso."),
            html.Li("Todos los números vienen de código ejecutado, nunca estimados a "
                    "mano; los valores de referencia se marcan explícitamente."),
        ]),
    ]
    return children


# ---------------------------------------------------------------------------
# VISTA 4 · CONTEXTO / HIPÓTESIS
# ---------------------------------------------------------------------------
def view_contexto() -> list:
    children = [
        html.H2("Contexto / Hipótesis"),
        html.P("Esta vista distingue entre lo que los datos permiten discutir, lo que "
               "todavía no se puede comprobar y la información que haría falta para "
               "profundizar. Es la mejor vista para mostrar que el equipo reconoce los "
               "límites del análisis y no está forzando una conclusión.",
               style={"lineHeight": 1.6}),
        section_header("Lo que aún no se puede comprobar"),
        html.Div([
            limit_card("#8e44ad", "#f5eefb", item["tema"], item["detalle"])
            for item in FALTANTES
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
    ]

    if EX is not None and "opcion3_terminos_distintivos" in EX:
        d3 = EX["opcion3_terminos_distintivos"]
        children.append(section_header("Términos distintivos",
                         "Ayudan a explorar el lenguaje presente en los comentarios "
                         "tóxicos, pero no prueban por sí solos por qué ocurre la "
                         "toxicidad."))
        if "_aviso" not in d3:
            pl3 = ordered(d3)
            cols = []
            for p in pl3:
                terms = pd.DataFrame(d3[p])
                block = [html.B(p)]
                if not terms.empty:
                    block.append(table(terms[["termino", "z", "frecuencia"]].head(8)))
                cols.append(html.Div(block, style={"flex": 1, "minWidth": 200}))
            children.append(html.Div(cols, style={"display": "flex", "gap": "16px",
                                                    "flexWrap": "wrap"}))

    children += [
        section_header("Preguntas probables del profesor"),
        table(pd.DataFrame(PREGUNTAS_PROFESOR, columns=["Pregunta", "Respuesta recomendada"])),
    ]
    return children


# ---------------------------------------------------------------------------
# VISTA 5 · METODOLOGÍA
# ---------------------------------------------------------------------------
def view_metodologia() -> list:
    return [
        html.H2("Metodología"),
        html.P("El proyecto se organiza en capas de evidencia. Esta vista explica por "
               "qué las fuentes se analizan por separado y por qué no sería correcto "
               "mezclarlas artificialmente en una sola tabla.", style={"lineHeight": 1.6}),
        section_header("Capa 1 · Datos sintéticos"),
        html.P("Se usaron para desarrollar y probar el pipeline, los modelos y la API. "
               "No sirven para afirmar cómo se comporta la toxicidad en datos reales.",
               style={"lineHeight": 1.6}),
        section_header("Capa 2 · Texto real (Measuring Hate Speech)"),
        html.P("Permite analizar comentarios reales que ya cuentan con anotaciones de "
               "toxicidad hechas por personas.", style={"lineHeight": 1.6}),
        section_header("Capa 3 · Discusión"),
        html.P("Conecta la evidencia con las hipótesis y las limitaciones, sin presentar "
               "relaciones causales que los datos no pueden demostrar.",
               style={"lineHeight": 1.6}),
        info_box("Las fuentes no se mezclan fila por fila porque no comparten la misma "
                 "unidad de análisis.", icon="🧩"),
        section_header("Reproducibilidad"),
        html.P("El proceso es reproducible: las métricas se generan mediante ETL y el "
               "dashboard consume un JSON consolidado.", style={"lineHeight": 1.6}),
        html.Div([
            html.Code("python etl/generate_dashboard_metrics.py"),
            html.Br(),
            html.Code("python dashboards/app_dash.py"),
            html.Br(),
            html.Code("pytest -v"),
            html.Br(),
            html.Code("docker compose up --build"),
        ], style={"background": "#0f172a", "color": "#e5e9f0", "padding": "14px",
                   "borderRadius": 8, "fontFamily": "monospace", "lineHeight": 2,
                   "marginTop": 10}),
        section_header("Números clave para recordar"),
        table(pd.DataFrame([
            {"Dato": "Comentarios reales", "Valor": f"{get_n_total():,}".replace(",", ".")},
            {"Dato": "Ranking", "Valor": "YouTube > Reddit > Twitter"},
            {"Dato": "Kruskal-Wallis", "Valor": f"H = {get_kw()['H']:.4f}; "
                                                  f"p ≈ {get_kw()['p']:.2e}"},
            {"Dato": "F1 sintético (EV2)", "Valor": f"{F1['sintetico']:.4f}"},
            {"Dato": "F1 texto real", "Valor": f"{F1['real']:.4f}"},
        ])),
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, title="EV3 · Toxicidad multiplataforma",
           suppress_callback_exceptions=True)
server = app.server      # para despliegue con gunicorn

TABS = ["Presentación", "Ejecutiva", "Técnica", "Contexto / Hipótesis", "Metodología"]

app.layout = html.Div([
    html.Div([
        html.H3("EV3 · Toxicidad multiplataforma", style={"margin": 0, "color": "#fff"}),
        html.Span("¿Es una red social más tóxica que otra, y por qué?",
                   style={"color": "#c9d3e0", "fontStyle": "italic"}),
        dcc.RadioItems(
            id="audiencia",
            options=[{"label": f" {t}", "value": t} for t in TABS],
            value="Presentación",
            inline=True,
            inputStyle={"marginLeft": "18px", "marginRight": "5px"},
            style={"marginTop": "14px", "color": "#fff"},
        ),
    ], style={"background": NAVY, "padding": "18px 24px", "borderRadius": "0 0 10px 10px"}),
    dcc.Store(id="block-store", data=1),
    html.Div(id="content", style={"maxWidth": 1050, "margin": "20px auto", "padding": "0 16px"}),
], style={"maxWidth": 1150, "margin": "0 auto", "fontFamily": "system-ui",
          "background": "#fff"})


@app.callback(
    Output("block-store", "data"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("block-store", "data"),
    prevent_initial_call=True,
)
def move_block(_prev_clicks, _next_clicks, current):
    current = current or 1
    if ctx.triggered_id == "btn-prev":
        return max(1, current - 1)
    if ctx.triggered_id == "btn-next":
        return min(7, current + 1)
    return current


@app.callback(Output("content", "children"),
              Input("audiencia", "value"),
              Input("block-store", "data"))
def render(audiencia, block):
    if EX is None and audiencia != "Presentación":
        return html.Div([
            html.H4("Falta toxicity_explain.json"),
            html.P("Esta vista usa valores de referencia donde puede, pero para el "
                   "análisis completo genéralo primero:"),
            html.Code("python3 etl/explain_toxicity_platform.py"),
        ])
    if audiencia == "Presentación":
        return view_presentacion(block or 1)
    if audiencia == "Ejecutiva":
        return view_ejecutiva()
    if audiencia == "Técnica":
        return view_tecnica()
    if audiencia == "Contexto / Hipótesis":
        return view_contexto()
    return view_metodologia()


if __name__ == "__main__":
    # Dash >=2.16 / 3.x / 4.x usan app.run; en Dash 2.x antiguo usar app.run_server
    app.run(debug=False, host="0.0.0.0", port=8050)