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
