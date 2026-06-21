"""RAG local de NutriCheck — recuperación híbrida sobre el CSV de productos.

Porta la lógica del notebook de la Tarea (Hybrid Search TF-IDF + ChromaDB + MMR)
al bot. Se usa como PRIMERA fuente: si encuentra el producto, evita el scraping;
si no, el pipeline cae al MCP de Jumbo (scraping en vivo).

Es un singleton perezoso: la ingesta (embeddings de ~1.3k fichas) ocurre una sola
vez, en la primera consulta.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Ruta del CSV dentro de la imagen (ver bot/data/). Configurable por entorno.
RUTA_CSV = os.getenv(
    "NUTRICHECK_CSV",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "productos.csv"),
)
MODELO_EMBED = os.getenv("NUTRICHECK_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Expansión de consulta por patología (Multi-Query Retrieval) — del notebook.
DICCIONARIO_TECNICO = {
    "hipertensión": ["sodio", "sal", "presión", "cloruro", "grasas trans"],
    "hipertension": ["sodio", "sal", "presión", "cloruro", "grasas trans"],
    "diabetes":     ["azúcar", "glucosa", "carbohidratos", "fibra", "maltodextrina"],
    "cáncer":       ["nitritos", "procesados", "conservantes", "grasas saturadas"],
    "cancer":       ["nitritos", "procesados", "conservantes", "grasas saturadas"],
    "respiratorias": ["sulfitos", "alérgenos", "inflamatorio"],
    "depresión":     ["magnesio", "omega 3", "triptófano", "proteína", "vitaminas"],
}

_LIQUIDO_KW = ("bebida", "jugo", "agua", "leche", "néctar", "nectar", "gaseosa",
               "refresco", "yogur", "yoghurt", "bebestible", "ml", "litro", "lt")


def _es_liquido(nombre: str, categoria: str, tipo: str) -> bool:
    txt = f"{nombre} {categoria} {tipo}".lower()
    return any(k in txt for k in _LIQUIDO_KW)


class RAGLocal:
    """Índice híbrido (léxico + semántico) sobre el CSV de productos."""

    def __init__(self, ruta_csv: str = RUTA_CSV) -> None:
        # Imports pesados aquí para que el módulo se pueda importar sin ellos.
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No se encontró el CSV de productos en {ruta_csv}")

        logger.info("RAGLocal: cargando modelo de embeddings '%s'…", MODELO_EMBED)
        self.embedder = SentenceTransformer(MODELO_EMBED)

        df = pd.read_csv(ruta_csv).fillna("No disponible")
        self.docs: list[dict] = []
        for i, row in df.iterrows():
            sellos = [
                s.replace("sello_alto_", "ALTO EN ").replace("_", " ").upper()
                for s in ("sello_alto_calorias", "sello_alto_azucares",
                          "sello_alto_grasas_sat", "sello_alto_sodio")
                if str(row[s]).upper() == "SI"
            ]
            sellos_txt = ", ".join(sellos) if sellos else "LIBRE DE SELLOS"
            ficha = (
                f"PRODUCTO: {row['nombre']} | MARCA: {row['marca']} | CATEGORÍA: {row['categoria']}\n"
                f"SELLOS: {sellos_txt}\n"
                f"INGREDIENTES: {row['ingredientes']}\n"
                f"CONDICIONES: {row['condiciones_alimentarias']}\n"
                f"NUTRIENTES (100g): {row['energia_kcal']} kcal, Azúcar {row['azucares_g']}g, "
                f"Sodio {row['sodio_mg']}mg, Grasas Sat {row['grasas_saturadas_g']}g."
            ).lower()
            self.docs.append({
                "id": f"doc_{i}",
                "texto": ficha,
                "sellos": sellos,
                "metadata": row.to_dict(),
            })

        textos = [d["texto"] for d in self.docs]

        logger.info("RAGLocal: embebiendo %d fichas en ChromaDB…", len(textos))
        self._client = chromadb.Client()
        try:
            self._client.delete_collection("nutricheck_adv")
        except Exception:
            pass
        self.collection = self._client.create_collection(name="nutricheck_adv")
        self.collection.add(
            ids=[d["id"] for d in self.docs],
            documents=textos,
            embeddings=self.embedder.encode(textos).tolist(),
        )

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_mtx = self.tfidf.fit_transform(textos)
        logger.info("RAGLocal: índice híbrido listo (%d productos).", len(self.docs))

    # ── Recuperación (del notebook) ─────────────────────────────────────────────

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.4) -> list[dict]:
        q_tfidf = self.tfidf.transform([query.lower()])
        l_scores = cosine_similarity(q_tfidf, self.tfidf_mtx).flatten()

        q_emb = self.embedder.encode([query.lower()]).tolist()
        s_res = self.collection.query(query_embeddings=q_emb, n_results=len(self.docs))
        s_scores = np.zeros(len(self.docs))
        for i, id_doc in enumerate(s_res["ids"][0]):
            idx = int(id_doc.split("_")[1])
            s_scores[idx] = 1 - s_res["distances"][0][i]

        combined = (alpha * l_scores) + ((1 - alpha) * s_scores)
        top = np.argsort(combined)[::-1][:k]
        return [{**self.docs[i], "_score": float(combined[i])} for i in top]

    def _mmr_rerank(self, query_emb, candidates: list[dict], k: int = 3,
                    lam: float = 0.5) -> list[dict]:
        if not candidates:
            return []
        doc_embs = self.embedder.encode([c["texto"] for c in candidates])
        query_emb = np.array(query_emb).reshape(1, -1)
        selected, unselected = [0], list(range(1, len(candidates)))
        while len(selected) < k and unselected:
            best_mmr, best_idx = -np.inf, -1
            for i in unselected:
                rel = cosine_similarity(doc_embs[i].reshape(1, -1), query_emb)[0][0]
                div = max(cosine_similarity(doc_embs[i].reshape(1, -1),
                                            doc_embs[j].reshape(1, -1))[0][0] for j in selected)
                score = lam * rel - (1 - lam) * div
                if score > best_mmr:
                    best_mmr, best_idx = score, i
            selected.append(best_idx)
            unselected.remove(best_idx)
        return [candidates[i] for i in selected]

    def recuperar(self, pregunta: str, condicion: Optional[str] = None,
                  k: int = 3) -> list[dict]:
        """MQR + Hybrid + MMR → lista de fichas (la primera es la mejor)."""
        queries = [pregunta]
        if condicion and condicion.lower() in DICCIONARIO_TECNICO:
            queries += [f"{pregunta} {t}" for t in DICCIONARIO_TECNICO[condicion.lower()][:2]]

        candidatos, vistos = [], set()
        for q in queries:
            for c in self.hybrid_search(q, k=5):
                if c["id"] not in vistos:
                    candidatos.append(c)
                    vistos.add(c["id"])

        q_emb = self.embedder.encode([pregunta])
        return self._mmr_rerank(q_emb, candidatos, k=k)

    def buscar_producto(self, query: str, condicion: Optional[str] = None) -> Optional[dict]:
        """Devuelve el mejor producto en el MISMO formato que el MCP de Jumbo, o None."""
        fichas = self.recuperar(query, condicion, k=3)
        if not fichas:
            return None
        m = fichas[0]["metadata"]

        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        return {
            "nombre":       str(m.get("nombre", "")),
            "marca":        str(m.get("marca", "")),
            "categoria":    str(m.get("categoria", "")),
            "ingredientes": str(m.get("ingredientes", "")),
            "es_liquido":   _es_liquido(str(m.get("nombre", "")), str(m.get("categoria", "")),
                                        str(m.get("tipo_alimento", ""))),
            "nutricion": {
                "calorias_kcal": _num(m.get("energia_kcal")),
                "sodio_mg":      _num(m.get("sodio_mg")),
                "azucares_g":    _num(m.get("azucares_g")),
                "grasas_sat_g":  _num(m.get("grasas_saturadas_g")),
            },
            "sellos": fichas[0]["sellos"],
            "_score": fichas[0]["_score"],
            "_fuente": "rag_local_csv",
        }


# ── Singleton perezoso ─────────────────────────────────────────────────────────

_INSTANCIA: Optional[RAGLocal] = None
_FALLIDO = False


def obtener_rag_local() -> Optional[RAGLocal]:
    """Devuelve el índice (lo construye en la 1ª llamada). None si no está disponible."""
    global _INSTANCIA, _FALLIDO
    if _INSTANCIA is not None:
        return _INSTANCIA
    if _FALLIDO:
        return None
    try:
        _INSTANCIA = RAGLocal()
        return _INSTANCIA
    except Exception as e:
        logger.warning("RAGLocal no disponible (%s); el bot usará solo el MCP de Jumbo.", e)
        _FALLIDO = True
        return None
