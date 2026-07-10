# Fuentes externas

Esta carpeta guarda fuentes complementarias para cumplir la integración de datos de EV3.

Por ahora `campaign_metadata.csv` puede ser generado automáticamente por el ETL como fuente provisional. Cuando esté lista la API o base de datos del grupo, esta carpeta puede recibir:

- `api_campaign_metadata.json`: export desde API.
- `campaign_metadata.sqlite`: base SQLite con tabla `campaign_metadata`.

La llave de integración esperada es `campaign_name`.