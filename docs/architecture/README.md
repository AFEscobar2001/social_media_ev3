# Arquitectura EV3

```text
CSV sintético EV2 ─┐
                   ├─ ETL / validación ── data/processed/social_media_ev3_final.csv
MongoDB MHS ───────┤
                   ├─ métricas reales ─── Adolfo/results/metrics/*.json
API sentimiento ───┘

Métricas consolidadas ── etl/generate_dashboard_metrics.py
                         ↓
Dashboard Dash ───────── dashboards/app_dash.py ── Render / Docker / local
API REST ─────────────── api/main.py ───────────── /docs
```

## Capas

1. **Datos sintéticos EV2:** útiles para practicar pipeline, modelado y API, pero no suficientes para concluir toxicidad real.
2. **Measuring Hate Speech:** corpus real etiquetado por personas para auditar toxicidad por plataforma.
3. **Dashboard crítico:** interpreta evidencia, límites e hipótesis futuras sin afirmar causalidad.

## Decisión clave

No se fuerza un join fila a fila entre fuentes porque no comparten el mismo comentario ni la misma unidad de análisis. Se integran por capas y se documenta qué responde cada una.

## Despliegue

- Dashboard: `gunicorn --chdir /app -b 0.0.0.0:8050 dashboards.app_dash:server`
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Docker Compose: `docker compose up --build`