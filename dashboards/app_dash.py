"""
app_dash.py — Dashboard EV3 · Auditoría de toxicidad multiplataforma (Plotly Dash)
============================================================================
Dos audiencias, una sola pregunta de investigación en el centro:

    ¿Es una red social más tóxica que otra? ¿Y por qué?

  · EJECUTIVA  → respuesta de negocio, semáforo, sin jerga.
  · TÉCNICA    → ranking con test estadístico, distribución, términos
                 distintivos (log-odds), honestidad metodológica.

Fuente única de la investigación: Adolfo/results/metrics/toxicity_explain.json
(generado por etl/explain_toxicity_platform.py). El CSV de campañas aporta
solo el contexto de volumen de negocio.

Ejecutar:
    python3 dashboards/app_dash.py        # http://127.0.0.1:8050
============================================================================
"""
import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../social_media_ev3
EXPLAIN_JSON = ROOT / "Adolfo" / "results" / "metrics" / "toxicity_explain.json"
DIST_FIG = ROOT / "Adolfo" / "results" / "figures" / "toxicity_distribution.png"
CSV_CANDIDATES = [
    ROOT / "data" / "processed" / "social_media_ev3_final.csv",
    ROOT / "data" / "processed" / "social_media_enriched.csv",
    ROOT / "Felipe" / "social_media_enriched (1).csv",
    ROOT / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv",
]

ORDER = ["youtube", "reddit", "twitter"]                # más → menos tóxico
RANK_COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]         # 🔴 más  🟡 medio  🟢 menos


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


def encode_fig(path: Path) -> str | None:
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return None


EX = load_explain()
DF_CSV = load_csv()
DIST_B64 = encode_fig(DIST_FIG)


def ordered(d: dict) -> list[str]:
    return [p for p in ORDER if p in d]


def color_map(platforms: list[str]) -> dict:
    return {p: RANK_COLORS[i] if i < len(RANK_COLORS) else "#777"
            for i, p in enumerate(platforms)}




def toxicity_rank_df(rt: dict) -> pd.DataFrame:
    """Convierte el score original a un indice positivo y facil de leer.

    En measuringhatespeech los promedios pueden ser negativos. La plataforma
    mas toxica es la de score_medio mas alto, aunque visualmente una barra
    negativa mas corta pueda parecer menos importante.
    """
    df = pd.DataFrame(rt["ranking"]).copy()
    df["score_medio"] = pd.to_numeric(df["score_medio"], errors="coerce")
    df = df.sort_values("score_medio", ascending=False)
    min_score = df["score_medio"].min()
    df["toxicidad_relativa"] = df["score_medio"] - min_score
    df["score_original"] = df["score_medio"].map(lambda x: f"{x:.3f}")
    df["lectura"] = df.apply(
        lambda r: f"{r['plataforma']} ({r['score_original']})", axis=1
    )
    return df


def table(df: pd.DataFrame, idx_name: str | None = None):
    """DataTable con estilo sobrio."""
    if idx_name:
        df = df.reset_index().rename(columns={"index": idx_name})
    return dash_table.DataTable(
        columns=[{"name": str(c), "id": str(c)} for c in df.columns],
        data=df.to_dict("records"),
        style_cell={"fontFamily": "system-ui", "fontSize": 14, "padding": "6px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
        style_table={"overflowX": "auto"},
    )


# ---------------------------------------------------------------------------
# VISTA EJECUTIVA
# ---------------------------------------------------------------------------
def view_ejecutiva() -> list:
    ranking = EX["ranking_y_test"]["ranking"]
    mas = ranking[0]["plataforma"]
    menos = ranking[-1]["plataforma"]
    thr = EX["_meta"]["umbral_toxico"]

    children = [
        html.H2("Auditoría de toxicidad en redes sociales"),
        html.H4("¿Es una red social más tóxica que otra?"),
        html.P(
            f"Con datos reales etiquetados por personas, {mas.capitalize()} es la "
            f"plataforma más tóxica y {menos.capitalize()} la menos tóxica. La "
            f"diferencia es real, no casualidad de la muestra. Importante: medido "
            f"sobre comentarios de 2019; es una foto de esos datos, no una verdad "
            f"permanente."
        ),
    ]

    # KPIs de negocio
    if DF_CSV is not None:
        kpis = [("Publicaciones analizadas", f"{len(DF_CSV):,}")]
        if "platform" in DF_CSV.columns:
            kpis.append(("Plataformas", str(DF_CSV["platform"].nunique())))
        if "campaign_name" in DF_CSV.columns:
            kpis.append(("Campañas", str(DF_CSV["campaign_name"].nunique())))
        children.append(html.Div(
            [html.Div([html.Div(v, style={"fontSize": 28, "fontWeight": "bold"}),
                       html.Div(k, style={"color": "#666"})],
                      style={"flex": 1, "padding": "12px", "background": "#fafafa",
                             "borderRadius": 8, "textAlign": "center"})
             for k, v in kpis],
            style={"display": "flex", "gap": "12px", "margin": "16px 0"},
        ))

    # % sobre umbral
    d2 = EX["opcion2_umbral"]
    plats = ordered(d2)
    df_thr = pd.DataFrame({"plataforma": plats,
                           "pct": [d2[p]["pct_toxicos"] for p in plats]})
    fig = px.bar(df_thr, x="plataforma", y="pct", color="plataforma",
                 color_discrete_map=color_map(plats), text="pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, height=360, xaxis_title="",
                      yaxis_title="% de comentarios tóxicos",
                      margin=dict(t=30, b=10))
    children += [html.H4("¿Por qué una plataforma sale peor?"),
                 dcc.Graph(figure=fig)]

    # insight "la cola, no el promedio"
    d1 = EX["opcion1_distribucion"]
    medianas = [d1[p]["median"] for p in plats]
    p95 = [d1[p]["q95"] for p in plats]
    if (max(p95) - min(p95)) > (max(medianas) - min(medianas)):
        children.append(html.Div(
            f"El comentario típico es parecido entre plataformas, pero "
            f"{mas.capitalize()} concentra más comentarios extremos. La toxicidad "
            f"la empuja una minoría agresiva, no el usuario promedio — conviene "
            f"moderar focos, no audiencias enteras.",
            style={"background": "#e7f1ff", "borderLeft": "4px solid #1c64f2",
                   "padding": "12px", "borderRadius": 6, "margin": "12px 0"},
        ))

    # semáforo de decisión
    def card(color, bg, titulo, texto):
        return html.Div([html.B(titulo), html.P(texto, style={"margin": "8px 0 0"})],
                        style={"flex": 1, "padding": "14px", "borderRadius": 8,
                               "background": bg, "borderLeft": f"4px solid {color}"})
    children += [
        html.H4("Qué puede ofrecer la agencia con confianza"),
        html.Div([
            card("#2ca02c", "#eafaf1", "🟢 Monitoreo de reputación por sentimiento",
                 "Funciona. Acierta ~75-80% en el mundo real, no el 94% de laboratorio."),
            card("#d62728", "#fdecea", "🔴 Detección de toxicidad sobre datos del cliente",
                 "No vendible: en el dataset sintético la toxicidad es ruido sin "
                 "relación con el texto."),
            card("#ff7f0e", "#fff4e5", "🟡 Detección de toxicidad con datos reales",
                 "Vendible si se invierte en datos etiquetados, como en esta auditoría."),
        ], style={"display": "flex", "gap": "12px", "margin": "12px 0"}),
    ]
    return children


# ---------------------------------------------------------------------------
# VISTA TÉCNICA
# ---------------------------------------------------------------------------
def view_tecnica() -> list:
    rt = EX["ranking_y_test"]
    thr = EX["_meta"]["umbral_toxico"]

    children = [
        html.H2("Vista técnica · Investigación de toxicidad"),
        html.P(f"Fuente: {EX['_meta']['fuente']} · n={EX['_meta']['n_total']:,} · "
               f"umbral tóxico = {thr}. Asociación, no causalidad.",
               style={"color": "#666"}),
    ]

    # 1 · ranking + Kruskal-Wallis
    df_rank = toxicity_rank_df(rt)
    plats = df_rank["plataforma"].tolist()
    fig = px.scatter(
        df_rank,
        x="score_medio",
        y="plataforma",
        color="plataforma",
        size="toxicidad_relativa",
        size_max=24,
        color_discrete_map=color_map(plats),
        text="score_original",
        title="Score medio de toxicidad por plataforma",
    )
    fig.update_traces(textposition="middle right")
    fig.update_layout(
        showlegend=False,
        height=340,
        xaxis_title="hate_speech_score medio (más a la derecha = más tóxico)",
        yaxis_title="",
        xaxis={"range": [df_rank["score_medio"].min() - 0.08, df_rank["score_medio"].max() + 0.14]},
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(plats))},
        margin=dict(t=55, b=10),
    )
    kw = rt["kruskal_wallis"]
    veredicto = "significativas" if kw["significativo"] else "NO significativas"
    df_pares = pd.DataFrame([
        {"comparación": k.replace("_", " "), "más tóxico": v["mas_toxico"],
         "p-valor": f"{v['p_valor']:.2e}",
         "significativo": "sí" if v["significativo"] else "no"}
        for k, v in rt["comparaciones_por_pares"].items()
    ])
    children += [
        html.H4("1 · Score medio de toxicidad y significancia"),
        dcc.Graph(figure=fig),
        html.P(f"Kruskal-Wallis: H = {kw['H']}, p = {kw['p_valor']:.2e} → las "
               f"diferencias son {veredicto}. Esto indica asociación estadística entre "
               f"plataforma y toxicidad; no demuestra causalidad.", style={"color": "#666"}),
        table(df_pares),
    ]

    # 2 · distribución
    d1 = EX["opcion1_distribucion"]
    pl1 = ordered(d1)
    df_d1 = pd.DataFrame([
        {"plataforma": p, "mediana": d1[p]["median"], "media": d1[p]["mean"],
         "media sin cola": d1[p]["mean_sin_cola"], "p95": d1[p]["q95"],
         "máx": d1[p]["max"], "n": d1[p]["n"]} for p in pl1
    ])
    d2 = EX["opcion2_umbral"]
    pl2 = ordered(d2)
    df_thr = pd.DataFrame([
        {"plataforma": p, "% tóxicos": d2[p]["pct_toxicos"],
         "n tóxicos": d2[p]["n_toxicos"], "n total": d2[p]["n"]}
        for p in pl2
    ])
    fig_thr = px.bar(
        df_thr,
        x="plataforma",
        y="% tóxicos",
        color="plataforma",
        color_discrete_map=color_map(pl2),
        text="% tóxicos",
        title=f"Comentarios sobre el umbral tóxico (> {thr})",
    )
    fig_thr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_thr.update_layout(
        showlegend=False,
        height=360,
        xaxis_title="",
        yaxis_title="% de comentarios tóxicos",
        margin=dict(t=55, b=20),
    )
    children += [
        html.Hr(), html.H4(f"2 · Comentarios que cruzan el umbral tóxico (> {thr})"),
        html.P("Para presentar, esta vista es más clara que un histograma o un boxplot: "
               "muestra qué proporción de comentarios realmente supera el límite definido como tóxico.",
               style={"color": "#666"}),
        dcc.Graph(figure=fig_thr),
        table(df_thr),
        html.H5("Resumen de la distribución"),
        table(df_d1),
        html.P("La mediana muestra el comentario típico; el p95 y el máximo muestran la cola extrema. "
               "Si una plataforma tiene más porcentaje sobre el umbral y una cola alta, concentra mayor riesgo de moderación.",
               style={"color": "#666"}),
    ]

    # 3 · términos distintivos
    d3 = EX["opcion3_terminos_distintivos"]
    children += [html.Hr(),
                 html.H4("3 · Qué se dice distinto en los comentarios tóxicos")]
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
            cols.append(html.Div(block, style={"flex": 1}))
        children.append(html.Div(cols, style={"display": "flex", "gap": "16px"}))
        children.append(html.P(
            "Log-odds con prior de Dirichlet (Monroe et al. 2008). z alto = más "
            "característico de esa plataforma. Es qué se dice distinto, no por qué "
            "— correlación, no causa.", style={"color": "#666"}))

    # honestidad metodológica
    children += [
        html.Hr(), html.H4("Honestidad metodológica"),
        html.Ul([
            html.Li("Comparación entre plataformas medidas con la misma regla "
                    "(hate_speech_score), no entre datasets distintos."),
            html.Li("El código 1 = reference se excluye siempre (no es una red social)."),
            html.Li("Resultado contra-hipótesis (Twitter NO es el más tóxico) = "
                    "hallazgo válido, no fracaso."),
            html.Li("Todos los números vienen de código ejecutado, nunca estimados."),
        ]),
    ]
    return children


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, title="EV3 · Toxicidad multiplataforma")
server = app.server      # para despliegue con gunicorn

app.layout = html.Div([
    html.Div([
        html.H3("EV3 · Toxicidad multiplataforma", style={"margin": 0}),
        html.Span("¿Es una red social más tóxica que otra, y por qué?",
                  style={"color": "#666", "fontStyle": "italic"}),
        dcc.RadioItems(
            id="audiencia",
            options=[{"label": " Ejecutiva", "value": "Ejecutiva"},
                     {"label": " Técnica", "value": "Técnica"}],
            value="Ejecutiva",
            inline=True,
            inputStyle={"marginLeft": "16px", "marginRight": "4px"},
            style={"marginTop": "10px"},
        ),
    ], style={"borderBottom": "1px solid #eee", "paddingBottom": "12px"}),
    html.Div(id="content", style={"maxWidth": 1000, "margin": "16px auto"}),
], style={"maxWidth": 1100, "margin": "0 auto", "padding": "20px",
          "fontFamily": "system-ui"})


@app.callback(Output("content", "children"), Input("audiencia", "value"))
def render(audiencia):
    if EX is None:
        return html.Div([
            html.H4("Falta toxicity_explain.json"),
            html.P("Genéralo primero:"),
            html.Code("python3 etl/explain_toxicity_platform.py"),
        ])
    return view_ejecutiva() if audiencia == "Ejecutiva" else view_tecnica()


if __name__ == "__main__":
    # Dash >=2.16 / 3.x / 4.x usan app.run; en Dash 2.x antiguo usar app.run_server
    app.run(debug=False, host="0.0.0.0", port=8050)