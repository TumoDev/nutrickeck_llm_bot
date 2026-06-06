"""Carga el CSV de productos y prepara los índices Chroma (semántico) y TF-IDF (léxico)."""

import os
from dataclasses import dataclass
from typing import List

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

SELLOS = [
    "sello_alto_calorias",
    "sello_alto_azucares",
    "sello_alto_grasas_sat",
    "sello_alto_sodio",
]


@dataclass
class BaseDatos:
    """Contenedor de los artefactos necesarios para la recuperación híbrida."""

    collection: "chromadb.Collection"
    vectorizer: TfidfVectorizer
    tfidf_matrix: "any"
    documentos: List[dict]


def _fila_a_ficha(row: pd.Series) -> str:
    sellos_activos = [
        s.replace("sello_", "").replace("_", " ").upper()
        for s in SELLOS
        if str(row[s]).upper() == "SI"
    ]
    sellos_txt = ", ".join(sellos_activos) if sellos_activos else "LIBRE DE SELLOS"

    return (
        f"PRODUCTO: {row['nombre']} | MARCA: {row['marca']} | CATEGORÍA: {row['categoria']}\n"
        f"SELLOS: {sellos_txt}\n"
        f"INGREDIENTES: {row['ingredientes']}\n"
        f"CONDICIONES: {row['condiciones_alimentarias']}\n"
        f"NUTRIENTES (100g): {row['energia_kcal']} kcal, Proteína {row['proteinas_g']}g, "
        f"Grasas Totales {row['grasas_totales_g']}g, Sat {row['grasas_saturadas_g']}g, "
        f"Trans {row['grasas_trans_g']}g, Azúcar {row['azucares_g']}g, "
        f"Sodio {row['sodio_mg']}mg, Fibra {row['fibra_dietetica_g']}g."
    ).lower()


def cargar_base_datos(ruta_csv: str, embedder: SentenceTransformer) -> BaseDatos:
    """Lee el CSV y devuelve la colección Chroma + el índice TF-IDF listos para consultar."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el CSV en {ruta_csv}")

    df = pd.read_csv(ruta_csv).fillna("No disponible")
    documentos = [
        {"id": f"doc_{i}", "texto": _fila_a_ficha(row), "metadata": row.to_dict()}
        for i, row in df.iterrows()
    ]
    textos = [d["texto"] for d in documentos]

    # Índice semántico (Chroma + embeddings).
    db_client = chromadb.Client()
    try:
        db_client.delete_collection("nutricheck_adv")
    except Exception:
        pass
    collection = db_client.create_collection(name="nutricheck_adv")
    collection.add(
        ids=[d["id"] for d in documentos],
        documents=textos,
        embeddings=embedder.encode(textos).tolist(),
    )

    # Índice léxico (TF-IDF con uni y bi-gramas).
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(textos)

    return BaseDatos(collection, vectorizer, tfidf_matrix, documentos)
