from pathlib import Path
import os

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Social Media EV3", layout="wide")
st.title("Dashboard Social Media EV3")
st.caption("Vista inicial para la defensa EV3: negocio, técnica y operación.")

ROOT = Path(os.environ.get("PROJECT_ROOT", "/app"))
CANDIDATES = [
    ROOT / "data" / "processed" / "social_media_ev3_final.csv",
    ROOT / "data" / "processed" / "social_media_enriched.csv",
    ROOT / "Felipe" / "social_media_enriched (1).csv",
    ROOT / "Adolfo" / "data" / "Social_Media_Engagement_Dataset.csv",
]

@st.cache_data
def load_data():
    for path in CANDIDATES:
        if path.exists():
            return pd.read_csv(path), path
    return pd.DataFrame(), None

df, source = load_data()

if df.empty:
    st.warning("No se encontró un dataset disponible. Revisa data/processed o las carpetas de integrantes.")
    st.stop()

st.success(f"Dataset cargado: {source}")

if "total_interactions" not in df.columns and {"likes_count", "shares_count", "comments_count"}.issubset(df.columns):
    df["total_interactions"] = df["likes_count"].fillna(0) + df["shares_count"].fillna(0) + df["comments_count"].fillna(0)

if "authentic_success" not in df.columns and {"engagement_rate", "sentiment_score", "toxicity_score"}.issubset(df.columns):
    engagement_threshold = df["engagement_rate"].quantile(0.75)
    toxicity_threshold = df["toxicity_score"].quantile(0.40)
    df["authentic_success"] = (
        (df["engagement_rate"] >= engagement_threshold)
        & (df["sentiment_score"] > 0)
        & (df["toxicity_score"] <= toxicity_threshold)
    ).astype(int)

view = st.sidebar.radio("Audiencia", ["Ejecutiva", "Técnica", "Operativa"])

if view == "Ejecutiva":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", f"{len(df):,}")
    if "platform" in df.columns:
        c2.metric("Plataformas", df["platform"].nunique())
    if "authentic_success" in df.columns:
        c3.metric("Éxito auténtico", f"{df['authentic_success'].mean() * 100:.1f}%")
    if "engagement_rate" in df.columns:
        c4.metric("Engagement medio", f"{df['engagement_rate'].mean():.3f}")

    if {"platform", "engagement_rate"}.issubset(df.columns):
        chart = df.groupby("platform", as_index=False)["engagement_rate"].mean()
        st.plotly_chart(px.bar(chart, x="platform", y="engagement_rate", title="Engagement promedio por plataforma"), use_container_width=True)

elif view == "Técnica":
    st.subheader("Calidad y columnas disponibles")
    st.dataframe(pd.DataFrame({"columna": df.columns, "nulos": df.isna().sum().values, "tipo": df.dtypes.astype(str).values}))
    if {"sentiment_score", "toxicity_score"}.issubset(df.columns):
        st.plotly_chart(px.scatter(df.sample(min(len(df), 1500), random_state=42), x="sentiment_score", y="toxicity_score", color="authentic_success" if "authentic_success" in df.columns else None, title="Sentimiento vs toxicidad"), use_container_width=True)

else:
    st.subheader("Exploración operativa")
    cols = [c for c in ["platform", "brand_name", "campaign_phase", "sentiment_label", "engagement_rate", "toxicity_score", "authentic_success"] if c in df.columns]
    st.dataframe(df[cols].head(200) if cols else df.head(200), use_container_width=True)
