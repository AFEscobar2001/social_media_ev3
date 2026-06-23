FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    DATA_DIR=/app/data \
    RAW_DATA_DIR=/app/data/raw \
    PROCESSED_DATA_DIR=/app/data/processed

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/raw /app/data/processed /app/data/external

EXPOSE 8888 8501

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--LabApp.default_url=/lab/tree/Resumen.ipynb"]
