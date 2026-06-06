"""Pipeline RAG de NutriCheck: MQR + Hybrid Search + MMR + generación con Mistral."""

from typing import Optional, Tuple

from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from . import config
from .data_loader import cargar_base_datos
from .retrieval import hybrid_search, mmr_rerank


def _system_prompt(condicion: Optional[str]) -> str:
    cond_name = condicion.upper() if condicion else "SALUD GENERAL"
    return f"""Eres NutriCheck, un Asistente Senior de Nutrición Clínica especializado en la población chilena y la Ley de Etiquetado 20.606.

INICIO OBLIGATORIO: Debes comenzar SIEMPRE tu respuesta con el siguiente párrafo:
"{config.DISCLAIMER}"

TU MISIÓN:
Analizar la aptitud de un producto alimentario para un usuario con una condición específica, basándote ÚNICAMENTE en la FICHA TÉCNICA proporcionada en el contexto.

REGLAS DE RAZONAMIENTO:
1. VERIFICACIÓN DE IDENTIDAD: Si el nombre del producto en la PREGUNTA no coincide razonablemente con el PRODUCTO del CONTEXTO, declara que no posees la información oficial y no inventes datos.
2. ANÁLISIS DE SELLOS: Evalúa el impacto de los sellos negros (Alto en Sodio, Azúcares, etc.) según la patología del usuario.
3. RIGOR TÉCNICO: Cita valores exactos (mg/g) por cada 100g para respaldar tu veredicto.
4. EVALUACIÓN DE INGREDIENTES: Busca ingredientes ocultos como Maltodextrina, Nitritos o Jarabes que representen un riesgo bioquímico para la condición informada.

FORMATO DE RESPUESTA (ESTRICTO):
---
🏷️ *PRODUCTO IDENTIFICADO*
Nombre: [Nombre comercial] | Marca: [Marca] | Categoría: [Categoría]

🛡️ *ANÁLISIS DE SELLOS LEY 20.606*
[Lista de sellos detectados] -> [Breve explicación del riesgo para la salud del usuario].

🩺 *VERDICTO PARA {cond_name}*
[APTO / NO APTO / MODERADO]

📊 *ANÁLISIS NUTRICIONAL (Valores por 100g)*
- Sodio: [mg] | Azúcares: [g] | Grasas Sat: [g] | Fibra: [g]
- Nota sobre alérgenos: [si aplica]

🔍 *JUSTIFICACIÓN TÉCNICA*
[Explicación breve: por qué los ingredientes son compatibles o peligrosos para la patología].

💬 *CONSEJO PARA EL USUARIO*
[Mensaje empático y directo. Sugiere alternativas si el producto es 'No Apto'].
---
"""


class NutriCheckRAG:
    """Encapsula los modelos y la base de datos; expone `.ask()` como única API pública."""

    def __init__(
        self,
        csv_path: str = config.CSV_PATH,
        mistral_api_key: Optional[str] = None,
        embedding_model: str = config.EMBEDDING_MODEL,
        mistral_model: str = config.MISTRAL_MODEL,
    ):
        api_key = mistral_api_key or config.MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError("Falta MISTRAL_API_KEY en el entorno.")

        self.mistral_model = mistral_model
        self.client = Mistral(api_key=api_key)
        self.embedder = SentenceTransformer(embedding_model)
        self.base = cargar_base_datos(csv_path, self.embedder)

    def ask(
        self, pregunta: str, condicion: Optional[str] = None
    ) -> Tuple[str, str]:
        """Ejecuta el pipeline completo y devuelve (respuesta, contexto_usado)."""
        # 1. Multi-Query Retrieval: expandimos la pregunta con términos técnicos.
        queries = [pregunta]
        cond_key = condicion.lower() if condicion else None
        if cond_key in config.DICCIONARIO_TECNICO:
            queries.extend(
                f"{pregunta} {term}"
                for term in config.DICCIONARIO_TECNICO[cond_key][:2]
            )

        # 2. Búsqueda híbrida sobre todas las queries y deduplicación.
        seen, candidatos = set(), []
        for q in queries:
            for doc in hybrid_search(q, self.base, self.embedder, k=5):
                if doc["id"] not in seen:
                    seen.add(doc["id"])
                    candidatos.append(doc)

        # 3. Re-ranking con MMR para diversificar el contexto.
        q_emb = self.embedder.encode([pregunta])
        contexto_docs = mmr_rerank(q_emb, candidatos, self.embedder, k=3)
        contexto = "\n---\n".join(d["texto"] for d in contexto_docs)

        # 4. Generación con Mistral usando el contexto como única fuente de verdad.
        response = self.client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": _system_prompt(condicion)},
                {
                    "role": "user",
                    "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}",
                },
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content, contexto
