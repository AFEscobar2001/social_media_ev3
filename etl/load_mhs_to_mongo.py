from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

HF_DATASET = "ucberkeley-dlab/measuring-hate-speech"
DEFAULT_COLLECTION = "measuringhatespeech"

# Columnas que nos interesan (se quedan solo las que existan en el dataset).
KEEP_COLS = [
    "comment_id", "text", "platform", "hate_speech_score",
    # dimensiones del "por qué es tóxico" (sirven para el análisis posterior):
    "sentiment", "respect", "insult", "humiliate", "status", "dehumanize",
    "violence", "genocide", "attack_defend", "hatespeech",
]


def download_mhs(dedup: bool) -> pd.DataFrame:
    """Descarga el dataset desde Hugging Face y devuelve un DataFrame limpio."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Falta la librería 'datasets'. Instala con: "
                 'python3 -m pip install datasets')

    print(f"Descargando {HF_DATASET} desde Hugging Face ...")
    # Si pide token, corre antes: huggingface-cli login
    ds = load_dataset(HF_DATASET, "default")

    # Chequeo del mapeo de plataforma (autoritativo si es ClassLabel)
    feat = ds["train"].features.get("platform")
    if feat is not None and hasattr(feat, "names"):
        print("  Mapeo oficial de platform (codigo -> nombre):")
        for code, name in enumerate(feat.names):
            print(f"    {code} -> {name}")
    else:
        print("  AVISO: 'platform' viene como entero sin nombres. "
              "Verifica el mapeo en el D-Lab antes de analizar por plataforma.")

    df = ds["train"].to_pandas()
    print(f"  Filas crudas (nivel anotacion): {len(df):,}")

    # Nos quedamos solo con las columnas que existan
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Una fila por comentario: el hate_speech_score es por comentario y se
    # repite entre anotadores, asi que deduplicar no pierde informacion
    # y baja mucho el tamano (clave para el free tier de 512 MB).
    if dedup and "comment_id" in df.columns:
        antes = len(df)
        df = df.drop_duplicates(subset="comment_id", keep="first").reset_index(drop=True)
        print(f"  Deduplicado por comment_id: {antes:,} -> {len(df):,} comentarios unicos")

    if "platform" in df.columns:
        print("\n  Distribucion por codigo de plataforma:")
        print(df["platform"].value_counts().sort_index().to_string())
    print()
    return df


def upload_to_mongo(df: pd.DataFrame, db_name: str, collection: str, drop: bool) -> None:
    """Sube el DataFrame a MongoDB Atlas en batches."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI no esta definida. Revisa tu archivo .env "
                 "(necesita pymongo[srv] para URIs mongodb+srv://).")

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        sys.exit(f"No se pudo conectar a MongoDB Atlas: {exc}")

    coll = client[db_name][collection]
    if drop:
        coll.drop()
        print(f"Coleccion '{collection}' vaciada antes de cargar.")

    records = df.to_dict("records")
    batch = 5000
    for i in range(0, len(records), batch):
        coll.insert_many(records[i:i + batch])
        print(f"  Insertados {min(i + batch, len(records)):,} / {len(records):,}")

    total = coll.count_documents({})
    print(f"\nListo. La coleccion '{db_name}.{collection}' tiene {total:,} documentos.")
    print("Documento de ejemplo:")
    print(coll.find_one({}, {"_id": 0}))


def main() -> None:
    p = argparse.ArgumentParser(description="Carga Measuring Hate Speech en MongoDB Atlas.")
    p.add_argument("--db", default=os.environ.get("MONGODB_DB_NAME", "Ev3"),
                   help="Base de datos en Atlas (default: Ev3 o MONGODB_DB_NAME)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"Coleccion destino (default: {DEFAULT_COLLECTION})")
    p.add_argument("--dedup", action="store_true", default=True,
                   help="Deja una fila por comentario (recomendado)")
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--drop", action="store_true",
                   help="Vacia la coleccion antes de cargar (evita duplicados al re-correr)")
    args = p.parse_args()

    df = download_mhs(dedup=args.dedup)
    upload_to_mongo(df, db_name=args.db, collection=args.collection, drop=args.drop)


if __name__ == "__main__":
    main()