# Storytelling presentación final - EV3 SCY1101

**Proyecto:** Social Media EV3 - Auditoría de toxicidad multiplataforma  
**Asignatura:** SCY1101 - Programación para la Ciencia de Datos  
**Equipo:** Felipe, Adolfo y Arelis (según carpetas del repositorio)  
**Fecha:** Julio 2026  
**Repositorio local:** `social_media_ev3-main`

## 1. Objetivo del documento

Este documento resume qué usamos, por qué lo usamos, cómo funciona el proyecto, cómo se presenta el dashboard y cuál es el storytelling recomendado para la exposición final. La idea no es adornar el dashboard, sino ordenar la evidencia para que la presentación tenga inicio, conflicto, descubrimiento, discusión y aprendizaje.

La frase central de la presentación es:

> Un ranking aislado no sirve. Un ranking con evidencia, límites e hipótesis abre una discusión útil.

## 2. Qué usamos en el proyecto

| Elemento | Para qué se usó | Evidencia en el repo |
|---|---|---|
| Python | Lenguaje principal para ETL, API, dashboard y pruebas. | `etl/*.py, api/*.py, dashboards/app_dash.py` |
| Pandas | Lectura, transformacion y armado de tablas de metricas. | `etl/generate_dashboard_metrics.py, dashboards/app_dash.py` |
| NumPy | Dependencia cientifica listada; no es el centro del dashboard final. | `requirements.txt` |
| Scikit-learn | Modelos y metricas del modelo EV2/API. | `api/model_service.py, api/main.py, Adolfo/src/*.py` |
| Plotly Dash | Dashboard interactivo por vistas y modo presentacion. | `dashboards/app_dash.py` |
| FastAPI | API REST para prediccion, comparacion y benchmark. | `api/main.py` |
| Docker / docker-compose | Ejecucion reproducible de Jupyter, dashboard y API. | `Dockerfile, docker-compose.yml, docker/Dockerfile.api` |
| MongoDB Atlas / pymongo | Fuente prevista para Measuring Hate Speech; si no hay credenciales, se usan JSON existentes. | `etl/generate_dashboard_metrics.py, etl/load_mhs_to_mongo.py` |
| JSON/CSV de metricas | Evidencia consumida por el dashboard sin escribir valores a mano. | `Adolfo/results/metrics/*.json, data/processed/*.csv` |
| pytest | Pruebas automatizadas de metricas/API. | `tests/test_dashboard_metrics.py, tests/test_api.py` |


Nota: la copia local revisada no está dentro de un repositorio Git inicializado (`git status` no detecta `.git`). Por eso Git/GitHub se trata como requisito académico/documental, no como evidencia técnica local validada.

## 3. Por qué usamos eso y no otra alternativa

- **Dash y no solo notebook:** el notebook sirve para explorar, pero el dashboard permite exponer resultados de forma ordenada, interactiva y reutilizable.
- **JSON de métricas y no valores escritos a mano:** el dashboard consume `dashboard_discussion_summary.json`, lo que reduce errores y permite regenerar evidencia desde ETL.
- **API si aplica:** FastAPI permite mostrar una solución end-to-end, no solo análisis estático. En este repo aparece en `api/main.py`.
- **Docker si aplica:** Docker reduce diferencias entre computadores y facilita una demo reproducible con dashboard, API y Jupyter.
- **Vista Ejecutiva, Técnica y Storytelling:** cada audiencia necesita una lectura distinta. La vista técnica defiende el método; la ejecutiva resume valor; el modo presentación guía la exposición.
- **Datos reales para toxicidad:** el dataset sintético ayudó a construir el sistema, pero no bastaba para concluir toxicidad real.
- **No mezclar fuentes distintas por fila:** no comparten la misma unidad de análisis ni una llave común de comentario. Mezclarlas artificialmente habría creado una conclusión débil.

## 4. Funcionamiento del proyecto

```text
Datos EV2 / CSV sintético -> ETL -> JSON/CSV procesados -> Dashboard Dash
                                          |-> API FastAPI
                                          |-> Docker
                                          |-> Tests
```

Comandos principales:

```bash
python etl/generate_dashboard_metrics.py
python dashboards/app_dash.py
pytest -v
docker compose up --build
```

## 5. Storytelling de la presentación

### Acto 1: La intuición

El equipo partió con una hipótesis intuitiva: Twitter podía ser la plataforma más tóxica. Parecía razonable por su dinámica pública, discusiones rápidas y alta exposición de opiniones. Pero en ciencia de datos una intuición no se defiende con opinión; se contrasta con evidencia.

### Acto 2: El obstáculo

El primer dataset servía para construir pipeline, modelos y API, pero no necesariamente para sostener una conclusión real sobre toxicidad. El benchmark mostró una brecha entre rendimiento sintético y texto real, por lo que fue necesario separar evidencia de laboratorio y evidencia real.

### Acto 3: El giro

Dentro del corpus Measuring Hate Speech, Youtube aparece con mayor toxicidad relativa, seguido de Reddit y luego Twitter. El test Kruskal-Wallis tiene p-value `4.473689713359982e-37`, por lo que las diferencias observadas son estadísticamente significativas en este corpus.

### Acto 4: La pregunta importante

La pregunta útil no es solo "qué plataforma ganó el ranking". La pregunta útil es por qué aparece así y qué condiciones del dato podrían explicarlo: comunidades, origen del corpus, año de captura, diseño de comentarios, reglas de moderación y exposición de marcas.

### Acto 5: El aprendizaje

El valor del proyecto está en convertir una intuición en una discusión basada en evidencia. La hipótesis inicial fue refutada, y eso es válido. La conclusión defendible es que el ranking abre una línea de análisis, no una verdad universal ni una solución cerrada.

## 6. Guion de presentación para 3 personas

### Persona 1: Contexto, hipótesis y problema

**Duración aproximada:** 2 a 3 minutos.  
**Parte del dashboard:** Modo Presentación, bloques 1 a 3.

Texto listo para decir:

> Nuestro punto de partida fue una intuición: pensamos que Twitter podía ser la plataforma más tóxica. Pero en vez de presentarlo como opinión, lo tratamos como hipótesis. El primer problema fue que el dataset sintético nos servía para construir el pipeline, pero no para afirmar toxicidad real. Por eso separamos el sistema técnico de la evidencia real.

Pregunta probable: ¿Por qué no bastó el dataset sintético?  
Respuesta corta: porque era útil para probar el sistema, pero no tenía señal suficiente para concluir toxicidad real.

### Persona 2: Evidencia, dashboard y análisis

**Duración aproximada:** 3 a 4 minutos.  
**Parte del dashboard:** Modo Presentación bloque 4, Vista Técnica y Vista Ejecutiva.

Texto listo para decir:

> Al usar el corpus Measuring Hate Speech, el resultado cambió la intuición inicial. YouTube aparece con mayor toxicidad relativa, luego Reddit y finalmente Twitter. El test estadístico indica diferencias significativas, pero lo importante es el alcance: esto aplica al corpus analizado, no a toda la realidad.

Pregunta probable: ¿Pueden decir que YouTube es siempre más tóxico?  
Respuesta corta: no; solo podemos decir que aparece así dentro de este corpus.

### Persona 3: Discusión, sistema y cierre

**Duración aproximada:** 3 minutos.  
**Parte del dashboard:** Modo Presentación bloques 5 a 7, Metodología, API/Docker si se muestra demo.

Texto listo para decir:

> El hallazgo no entrega una solución directa para un consumidor, pero sí abre una discusión útil para moderación, marcas e investigación. Para explicar la causa faltan variables como comunidad, fecha, idioma, país, categoría de contenido y exposición publicitaria. Técnicamente, el proyecto deja un flujo reproducible con ETL, dashboard, API, Docker, JSON de métricas y tests.

Pregunta probable: ¿Por qué no proponen una solución directa?  
Respuesta corta: porque el dato no prueba causalidad; sería más honesto proponer líneas de análisis y monitoreo.

## 7. Guion tipo storytelling para presentar el dashboard

> Quiero partir con la intuición que teníamos como equipo. Si alguien nos preguntaba qué red podía ser más tóxica, probablemente pensábamos en Twitter. Pero el objetivo del proyecto no era confirmar lo que creíamos, sino probarlo con datos.
>
> El primer obstáculo fue metodológico. Teníamos datos sintéticos que nos servían para construir el pipeline y probar modelos, pero no bastaban para sostener una conclusión real sobre toxicidad. Ahí cambió el proyecto: dejamos de mirar solo el modelo y empezamos a mirar la calidad de la evidencia.
>
> Cuando analizamos el corpus real, la hipótesis inicial no se confirmó. Dentro de estos datos, YouTube aparece con mayor toxicidad relativa, luego Reddit y después Twitter. Ese resultado es significativo, pero no significa que YouTube sea siempre más tóxico.
>
> Entonces viene la pregunta más importante: ¿de qué sirve este hallazgo? No sirve para decirle a una persona que abandone una plataforma. Sirve para abrir preguntas mejores: qué comunidades están representadas, cómo se capturó el corpus, qué tipo de contenido hay, qué reglas de moderación existen y si una marca podría quedar expuesta a conversaciones de mayor riesgo.
>
> Por eso nuestra conclusión no es una solución cerrada. Nuestra conclusión es que un ranking aislado no sirve, pero un ranking con evidencia, límites e hipótesis sí permite una discusión útil.

## 8. Preguntas probables del profesor y respuestas

- **¿De que sirve saber que YouTube aparece mas toxico?** Sirve como senal para investigar contexto, comunidades, moderacion y riesgo reputacional. No sirve como accion directa para un consumidor individual.
- **¿Pueden afirmar que YouTube es siempre mas toxico?** No. Solo se afirma que aparece con mayor toxicidad relativa dentro de este corpus analizado.
- **¿Por que no basta el dataset sintetico?** Porque permitio construir el sistema, pero no tenia senal suficiente para concluir toxicidad real.
- **¿Por que no proponen una solucion directa?** Porque el dato no prueba causa. Proponer una solucion cerrada seria sobreinterpretar la evidencia.
- **¿Que datos faltan para explicar la causa?** Comunidad/canal, fecha, idioma, pais, categoria de contenido, reglas de moderacion, anuncios y presencia de marca.
- **¿Que relacion tiene esto con marcas o anunciantes?** Puede orientar monitoreo de brand safety, pero no mide exposicion publicitaria ni impacto de campana.
- **¿Que aprendieron del proyecto?** Que ciencia de datos tambien consiste en reconocer limites y no forzar una conclusion cuando el dato solo permite abrir discusion.
- **¿Que cambiarian con mas tiempo?** Recolectar datos reales con mas contexto, entrenar modelos con texto real y validar el dashboard con usuarios del negocio.


## 9. Limitaciones

### Limitaciones de datos

- No hay comunidad/canal/subreddit por comentario en los datos disponibles localmente.
- No hay fecha/año por comentario disponible localmente.
- No hay país, región, idioma ni cultura de origen.
- La muestra está desbalanceada por plataforma.
- En la ejecución local sin Mongo, algunas métricas se consolidan desde JSON existentes.

### Limitaciones del modelo

- El buen rendimiento en datos sintéticos no garantiza generalización a texto real.
- La toxicidad sintética no mostró una señal suficientemente fuerte para conclusiones reales.
- Falta entrenar/validar con datos reales etiquetados del mismo dominio.

### Limitaciones del dashboard

- El dashboard explica evidencia y límites, pero no permite explorar comentarios individuales si el dato crudo no está disponible.
- Depende de que los JSON de métricas estén generados.
- Algunas hipótesis quedan documentadas, pero no verificadas.

### Limitaciones de interpretación

- El resultado no prueba causalidad.
- No permite afirmar que YouTube sea universalmente más tóxico.
- No permite inferir comunidades, países o periodos si esas columnas no existen.

### Limitaciones de negocio

- No mide anuncios, campañas ni presencia real de marcas.
- No entrega una acción directa para consumidores individuales.
- Solo sugiere líneas de monitoreo y análisis de brand safety.

## 10. Checklist final de presentación

- [ ] Dashboard probado localmente.
- [ ] Docker probado con `docker compose up --build`.
- [ ] API probada si se mostrará en demo.
- [ ] JSON de métricas generado con `python etl/generate_dashboard_metrics.py`.
- [ ] README actualizado.
- [ ] Guion revisado por los tres integrantes.
- [ ] Roles claros: contexto, evidencia, discusión/cierre.
- [ ] Respuestas preparadas para preguntas sobre causalidad, límites y datos faltantes.

## 11. Referencias de storytelling y Dash

- Santander Open Academy: referencia validada. Se usó como inspiración para conectar datos con audiencia y ordenar argumento, conflicto y contexto.
- Venngage: referencia validada. Se usó como inspiración estructural del viaje: inicio, obstáculo, giro y aprendizaje.
- Tutorial oficial de Dash: referencia validada. Se mantuvo el uso de layout claro, `dcc.Graph`, `RadioItems` y callback simple para cambiar vistas.
- Video de YouTube entregado por el usuario: recurso entregado como referencia, pero no se pudo validar directamente desde el entorno.
- Universidad de Palermo: recurso entregado como referencia, pero no se pudo validar directamente desde el entorno.
