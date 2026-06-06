"""Recuperación híbrida (TF-IDF + embeddings) y re-ranking con MMR."""

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import BaseDatos


def hybrid_search(
    query: str,
    base: BaseDatos,
    embedder,
    k: int = 10,
    alpha: float = 0.4,
) -> List[dict]:
    """Combina similitud léxica (TF-IDF) y semántica (embeddings) en una sola score."""
    n_docs = len(base.documentos)
    query_norm = query.lower()

    # Componente léxico.
    q_tfidf = base.vectorizer.transform([query_norm])
    lex_scores = cosine_similarity(q_tfidf, base.tfidf_matrix).flatten()

    # Componente semántico.
    q_emb = embedder.encode([query_norm]).tolist()
    sem_res = base.collection.query(query_embeddings=q_emb, n_results=n_docs)
    sem_scores = np.zeros(n_docs)
    for i, doc_id in enumerate(sem_res["ids"][0]):
        idx = int(doc_id.split("_")[1])
        sem_scores[idx] = 1 - sem_res["distances"][0][i]

    combined = alpha * lex_scores + (1 - alpha) * sem_scores
    top_idx = np.argsort(combined)[::-1][:k]
    return [base.documentos[i] for i in top_idx]


def mmr_rerank(
    query_emb: np.ndarray,
    candidates: List[dict],
    embedder,
    k: int = 3,
    lam: float = 0.5,
) -> List[dict]:
    """Re-rankea candidatos balanceando relevancia (lam) y diversidad (1-lam)."""
    if not candidates:
        return []

    doc_embs = embedder.encode([c["texto"] for c in candidates])
    query_emb = np.array(query_emb).reshape(1, -1)

    selected = [0]
    unselected = list(range(1, len(candidates)))

    while len(selected) < k and unselected:
        best_score, best_idx = -np.inf, -1
        for i in unselected:
            rel = cosine_similarity(doc_embs[i].reshape(1, -1), query_emb)[0][0]
            div = max(
                cosine_similarity(
                    doc_embs[i].reshape(1, -1), doc_embs[j].reshape(1, -1)
                )[0][0]
                for j in selected
            )
            score = lam * rel - (1 - lam) * div
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        unselected.remove(best_idx)

    return [candidates[i] for i in selected]
