# Social Media EV3 - Solución End-to-End SCY1101

Proyecto grupal para la Evaluación Parcial N°3 de **Programación para la Ciencia de Datos (SCY1101)**. La solución toma el trabajo de modelado realizado en EV2 y lo transforma en una base profesional para EV3: pipeline ETL, dashboard, documentación, Docker y evidencia de colaboración.

## Objetivo del proyecto

Construir una solución reproducible para analizar publicaciones en redes sociales y apoyar decisiones de marketing. El foco no es solo entrenar modelos, sino demostrar un flujo end-to-end: integrar datos, procesarlos, validar resultados, visualizar hallazgos y ejecutar el proyecto en un entorno Docker.

## Preguntas de negocio

- ¿Qué tono tiene un post según su contenido textual?
- ¿Qué señales ayudan a detectar engagement atípico o no orgánico?
- ¿Qué hace que un post tenga potencial de éxito auténtico sin caer en data leakage?

## Estructura EV3

```text
.
├── etl/                  # Scripts y notebooks del pipeline de integración y transformación
├── dashboards/           # Aplicación Streamlit/Dash y visualizaciones interactivas
├── docs/                 # Manuales, arquitectura, API, despliegue y capturas
├── api/                  # Código de API REST si se expone el modelo o métricas
├── docker/               # Archivos auxiliares de Docker y configuración de entorno
├── tests/                # Pruebas automatizadas del pipeline y reglas de datos
├── data/                 # Datos originales, externos y procesados para EV3
├── repo/                 # Evidencia de Git: ramas, PRs, issues, capturas o bitácora
├── Adolfo/               # Desarrollo del insight de sentimiento/texto
├── Arelis/               # Desarrollo del insight de engagement orgánico/fake engagement
├── Felipe/               # Desarrollo del insight de éxito auténtico sin leakage
├── Resumen.ipynb         # Notebook de síntesis del proyecto
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Estado actual

Ya existe una base analítica heredada de EV2:

- Notebooks de modelado supervisado y no supervisado.
- Dataset principal de redes sociales.
- Dataset enriquecido con variables derivadas.
- Informe técnico previo.
- Docker básico para levantar Jupyter.

Para EV3, el proyecto se está adaptando a una entrega profesional con pipeline, dashboard, documentación, pruebas y despliegue.

## Fuentes de datos previstas

La pauta exige integrar al menos tres fuentes. La propuesta recomendada es:

1. **CSV principal:** dataset de publicaciones de redes sociales.
2. **CSV enriquecido/procesado:** variables derivadas de engagement, toxicidad, país, fecha y éxito auténtico.
3. **Fuente externa/API o base SQL:** archivo o servicio complementario para simular costos, objetivos de campaña, calendario comercial o metadata de marcas.

> Nota: si no se usa una API real, se debe documentar claramente la fuente simulada y cargarla mediante un script ETL reproducible.


## Ejecutar ETL

Antes de abrir el dashboard final, se recomienda generar el dataset procesado:

```bash
python etl/run_etl.py
```

Salida principal:

```text
data/processed/social_media_ev3_final.csv
```

Reportes generados:

```text
data/processed/reports/etl_quality_report.csv
data/processed/reports/etl_summary.json
```
## Ejecución con Docker

### Levantar Jupyter Lab

```bash
docker compose up --build jupyter
```

Luego abrir:

```text
http://127.0.0.1:8888/lab
```

### Levantar dashboard Streamlit

```bash
docker compose up --build dashboard
```

Luego abrir:

```text
http://127.0.0.1:8501
```

## Variables de entorno

El proyecto usa variables para que las rutas no queden amarradas a un computador específico.

```text
PROJECT_ROOT=/app
DATA_DIR=/app/data
RAW_DATA_DIR=/app/data/raw
PROCESSED_DATA_DIR=/app/data/processed
```

## Reproducibilidad

- Python 3.12 en Docker.
- Dependencias fijadas en `requirements.txt`.
- Semilla recomendada: `RANDOM_STATE = 42`.
- Separación entre datos originales, datos procesados, notebooks, scripts y dashboard.

## Criterios técnicos EV3 cubiertos por esta estructura

- Pipeline ETL modular.
- Documentación técnica y guía de despliegue.
- Dashboard interactivo por audiencia.
- Docker y docker-compose.
- Base para testing automatizado.
- Carpeta para evidenciar colaboración en Git.

## Pendientes principales

- Implementar scripts reales en `etl/`.
- Crear dashboard definitivo en `dashboards/`.
- Agregar tests en `tests/`.
- Documentar arquitectura en `docs/`.
- Agregar evidencia de ramas, commits, pull requests e issues en `repo/`.
- Corregir o reemplazar archivos mal nombrados, por ejemplo CSV que en realidad sea imagen.
## Dashboard crítico de toxicidad

El dashboard principal es `dashboards/app_dash.py`. La pregunta original se mantiene como punto de partida:

> ¿Es Twitter más tóxico que Reddit y YouTube usando datos reales?

El resultado refuta esa hipótesis: en el corpus Measuring Hate Speech, YouTube aparece con mayor toxicidad relativa, seguido de Reddit y luego Twitter. La conclusión final no es que YouTube sea universalmente más tóxico, sino que el ranking abre una discusión sobre corpus, comunidades, diseño de interacción, moderación y posibles riesgos para marcas.

Antes de ejecutar el dashboard, generar las métricas consolidadas:

```bash
python etl/generate_dashboard_metrics.py
```

Ejecutar localmente:

```bash
python dashboards/app_dash.py
```

Abrir:

```text
http://127.0.0.1:8050
```

Documentación específica:

- `docs/dashboard.md`
- `docs/architecture/README.md`

Comandos de verificación recomendados:

```bash
python etl/generate_dashboard_metrics.py
python dashboards/app_dash.py
pytest -v
docker compose up --build
```

<!-- storytelling-presentacion -->

## Presentación final: storytelling del dashboard

El dashboard principal (`dashboards/app_dash.py`) ahora incluye **Modo Presentación**, una vista diseñada para usar el dashboard como guion oral de la exposición final.

La narrativa sigue este hilo:

1. **Intuición:** el equipo esperaba que Twitter fuera la plataforma más tóxica.
2. **Conflicto:** el dataset sintético servía para construir el sistema, pero no bastaba para concluir toxicidad real.
3. **Giro:** en el corpus Measuring Hate Speech, YouTube aparece con mayor toxicidad relativa.
4. **Discusión:** el ranking no prueba causalidad ni una verdad universal; abre hipótesis sobre corpus, comunidades, moderación y diseño de interacción.
5. **Aprendizaje:** una conclusión responsable reconoce límites y datos faltantes.

Documentos de apoyo:

- `docs/storytelling_presentacion_final.md`
- `docs/storytelling_presentacion_final.docx`
- `docs/presentation_storytelling.md`
- `docs/dashboard.md`

Comandos recomendados para la demo:

```bash
python etl/generate_dashboard_metrics.py
python dashboards/app_dash.py
pytest -v
docker compose up --build
```
