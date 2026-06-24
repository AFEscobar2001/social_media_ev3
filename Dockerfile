FROM python:3.12-slim

# Evita que pip y python generen archivos basura / mejora logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalamos primero requirements para aprovechar el cache de Docker:
# si solo cambia el código, no se reinstalan las librerías de nuevo.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Resumen.ipynb busca el CSV junto a sí mismo (raíz) antes que en cualquier
# subcarpeta. Creamos un enlace simbólico al CSV real (que vive en
# Adolfo/data/) para que lo encuentre al instante, sin duplicar el archivo
# ni tener que tocar el código del notebook.
RUN ln -sf Adolfo/data/Social_Media_Engagement_Dataset.csv \
    Social_Media_Engagement_Dataset.csv

# Variable que usa Adolfo/config.py para saber dónde está la raíz del proyecto
# (reemplaza la ruta de Colab /content/proyecto_modelado)
ENV PROJECT_ROOT=/app/Adolfo

EXPOSE 8888

# Levanta Jupyter Lab accesible desde fuera del contenedor, sin token
# (entorno de desarrollo local, no producción).
#
# Resumen.ipynb es el notebook "ejecutable" del proyecto: es autocontenido
# (no depende de src/, modelos .joblib previos, ni de correr otros notebooks
# antes) y reúne las conclusiones de los insights de Adolfo y Arelis en un
# solo archivo. Por eso Jupyter lo abre directo al levantar el contenedor,
# en vez de mostrar solo el listado de carpetas.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--LabApp.default_url=/lab/tree/Resumen.ipynb"]