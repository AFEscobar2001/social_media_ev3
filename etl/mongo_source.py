"""Conexión a MongoDB Atlas para la fuente externa de metadata de campañas.

Reemplaza la fuente provisional `campaign_metadata.csv` por la colección real
en MongoDB Atlas. Si la conexión falla (red, credenciales, etc.), el llamador
puede decidir hacer fallback al CSV provisional para no romper el ETL.

Variables de entorno esperadas (definidas en .env, NUNCA en el código):
    MONGODB_URI        - connection string completo (mongodb+srv://...)
    MONGODB_DB_NAME     - nombre de la base de datos (default: "ev3")
    MONGODB_COLLECTION  - nombre de la colección (default: "campaign_metadata")
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

DEFAULT_DB_NAME = "Ev3"
DEFAULT_COLLECTION = "Ev3"


def get_mongo_client(timeout_ms: int = 8000) -> MongoClient:
    """Crea un cliente de MongoDB a partir de MONGODB_URI en el entorno.

    Lanza ValueError si la variable no está definida, para fallar rápido
    y con un mensaje claro en lugar de un error críptico de pymongo.
    """
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError(
            "MONGODB_URI no está definida. Crea un archivo .env (ver .env.example) "
            "con la variable MONGODB_URI antes de usar esta fuente."
        )
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def test_connection() -> dict:
    """Prueba la conexión y devuelve un diagnóstico legible.

    Pensado para correr una vez en VS Code y confirmar qué hay realmente
    en el cluster (nombres de base, colecciones, conteo de documentos)
    antes de integrarlo al ETL.
    """
    report: dict = {"ok": False}
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        report["ok"] = True
        report["databases"] = client.list_database_names()

        db_name = os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME)
        db = client[db_name]
        report["db_name"] = db_name
        report["collections"] = db.list_collection_names()

        coll_name = os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION)
        if coll_name in report["collections"]:
            coll = db[coll_name]
            report["collection_name"] = coll_name
            report["document_count"] = coll.count_documents({})
            report["sample_document"] = coll.find_one({}, {"_id": 0})
        else:
            report["collection_name"] = coll_name
            report["warning"] = (
                f"La colección '{coll_name}' no existe en la base '{db_name}'. "
                f"Colecciones disponibles: {report['collections']}"
            )
    except PyMongoError as exc:
        report["ok"] = False
        report["error"] = str(exc)
    except ValueError as exc:
        report["ok"] = False
        report["error"] = str(exc)
    return report


def load_campaign_metadata_from_mongo(
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> pd.DataFrame:
    """Carga la metadata de campañas desde MongoDB Atlas como DataFrame.

    Excluye el campo _id (no aporta valor analítico y no es serializable
    de forma trivial a CSV). Lanza la excepción original si falla la
    conexión o la colección está vacía; el llamador decide si hace fallback.
    """
    client = get_mongo_client()
    db = client[db_name or os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME)]
    coll = db[collection_name or os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION)]

    documents = list(coll.find({}, {"_id": 0}))
    if not documents:
        raise ValueError(
            f"La colección '{coll.name}' en la base '{db.name}' está vacía. "
            "Revisa que los datos estén cargados en Atlas."
        )

    df = pd.DataFrame(documents)
    df["metadata_source"] = "mongodb_atlas"
    logging.info(
        "Cargados %s documentos desde MongoDB Atlas (db=%s, collection=%s)",
        len(df), db.name, coll.name,
    )
    return df


if __name__ == "__main__":
    import json

    result = test_connection()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))