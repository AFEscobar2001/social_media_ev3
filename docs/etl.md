# Diseño del Pipeline ETL

## Objetivo

Unificar los datos de redes sociales en un dataset final para dashboard, análisis técnico y presentación EV3.

## Fuentes actuales

1. `Adolfo/data/Social_Media_Engagement_Dataset.csv`: dataset original.
2. `Felipe/social_media_enriched (1).csv`: dataset enriquecido con variables derivadas.
3. `data/external/campaign_metadata.csv`: metadata externa de campañas. Actualmente puede ser provisional hasta que se integre API/base de datos.

## Flujo

```text
CSV original + CSV enriquecido + metadata campañas
        ↓
validación de columnas obligatorias
        ↓
transformaciones y variables EV3
        ↓
integración por campaign_name
        ↓
data/processed/social_media_ev3_final.csv
        ↓
dashboard / análisis / presentación
```

## Validaciones aplicadas

- Existencia de archivos fuente.
- Detección de archivos mal nombrados como CSV pero que son imágenes.
- Revisión de columnas obligatorias.
- Reporte de nulos, tipos de datos y columnas marcadas como leakage.

## Criterio anti-leakage

El ETL conserva variables de resultado porque sirven para análisis y visualización, pero las marca como columnas con riesgo de leakage en `etl_quality_report.csv`. Para entrenar modelos predictivos, no deben usarse como features si forman parte del target.