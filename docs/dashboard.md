# Dashboard EV3: auditoría de toxicidad multiplataforma

## Objetivo

El dashboard ya no busca responder solo "qué plataforma es más tóxica". La lectura final es más crítica:

> En este corpus, YouTube presenta mayor toxicidad relativa que Reddit y Twitter. El resultado es estadísticamente significativo, pero su valor no está en afirmar que YouTube sea universalmente más tóxico. Su valor está en abrir una pregunta: qué características del corpus, de las comunidades o del diseño de interacción pueden explicar esta diferencia.

## Archivo principal

- `dashboards/app_dash.py`
- Métricas consolidadas: `Adolfo/results/metrics/dashboard_discussion_summary.json`
- Generador: `etl/generate_dashboard_metrics.py`

## Vistas del dashboard

### 1. Ejecutiva

Explica el hallazgo central y responde "¿de qué sirve saber esto?".

Incluye:

- plataforma con mayor toxicidad relativa;
- total de comentarios;
- p-valor del test;
- tamaño muestral de YouTube;
- limitación principal;
- matriz de interpretación;
- bloque de lo que no se puede concluir.

### 2. Técnica

Presenta evidencia estadística:

- score medio por plataforma;
- Kruskal-Wallis y comparaciones por pares;
- porcentaje de comentarios sobre el umbral tóxico;
- media, mediana y p95;
- boxplot histórico generado por el ETL anterior;
- tamaño muestral;
- brecha de generalización entre datos sintéticos y texto real.

### 3. Contexto / Hipótesis

Separa hipótesis comprobables de discusión futura:

- comunidades/canales;
- año o fecha de captura;
- país/idioma;
- origen del corpus;
- diseño de interacción;
- implicancias para marcas y anunciantes.

### 4. Metodología

Explica por qué el proyecto usa capas de evidencia y no mezcla todos los datasets por fila.

## Cómo interpretar el resultado de YouTube

Significa:

- dentro de este corpus, YouTube aparece con mayor toxicidad relativa;
- el resultado tiene evidencia estadística;
- el ranking original refuta la hipótesis inicial de que Twitter sería la plataforma más tóxica.

No significa:

- que YouTube sea universalmente más tóxico;
- que toda la plataforma o todos sus usuarios sean tóxicos;
- que exista causalidad probada;
- que un consumidor individual tenga una acción directa;
- que se pueda inferir país, comunidad, idioma o año si esas columnas no existen.

## Implicancia para marcas y anunciantes

Si una marca aparece junto a contenido o conversaciones de alta toxicidad, podría existir riesgo reputacional. Este proyecto no mide anuncios ni presencia de marcas, por lo tanto no afirma impacto publicitario directo. La línea futura correcta es cruzar toxicidad con categorías de contenido, presencia de marca, campañas y reglas de moderación.

## Comandos

```bash
python etl/generate_dashboard_metrics.py
python dashboards/app_dash.py
pytest -v
docker compose up --build
```

<!-- storytelling-presentacion -->

## Modo Presentación

El dashboard incluye una vista llamada **Modo Presentación**. Esta vista está pensada para exponer el proyecto en orden, sin saltar entre gráficos ni depender de diapositivas externas.

Secuencia:

1. Pregunta inicial.
2. Hipótesis del equipo.
3. Primer problema: el dato sintético no bastaba.
4. Evidencia real: ranking y test.
5. Discusión: por qué podría pasar.
6. Implicancia: consumidor, marcas, moderación e investigación.
7. Aprendizaje final.

La vista no agrega nuevas métricas ni cambia el análisis. Solo reorganiza la evidencia existente para presentar una historia defendible: intuición, conflicto, hallazgo, discusión y aprendizaje.

Frase de cierre recomendada:

> Un ranking aislado no sirve. Un ranking con evidencia, límites e hipótesis abre una discusión útil.
