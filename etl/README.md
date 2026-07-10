# ETL EV3

Esta carpeta contiene el pipeline ETL del proyecto.

## Qué hace `run_etl.py`

1. Carga el CSV original de Adolfo.
2. Carga el CSV enriquecido de Felipe.
3. Carga una fuente externa de metadata de campañas desde `data/external/campaign_metadata.csv`.
4. Si la metadata externa todavía no existe, crea una versión provisional para que el pipeline sea ejecutable.
5. Valida columnas obligatorias.
6. Genera variables derivadas para EV3:
   - `total_interactions`
   - `er_recalc_ev3`
   - `log_engagement`
   - `fake_engagement_flag`
   - `authentic_success_ev3`
   - features simples del texto
7. Integra la metadata de campaña.
8. Exporta el dataset final a `data/processed/social_media_ev3_final.csv`.
9. Exporta reportes de calidad a `data/processed/reports/`.

## Cómo ejecutarlo

Desde la raíz del proyecto:

```bash
python etl/run_etl.py
```

Con Docker:

```bash
docker compose run --rm jupyter python etl/run_etl.py
```

## Nota sobre API/base de datos

El script ya tiene puntos de entrada para:

- `data/external/api_campaign_metadata.json`
- `data/external/campaign_metadata.sqlite`

Cuando se integre la API o base de datos del compañero, se puede adaptar la función `load_external_campaign_metadata()` sin cambiar el dashboard ni el resto del flujo.