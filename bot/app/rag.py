"""Pipeline NutriCheck — Plan-and-Execute con tool calling de Mistral.

Flujo secuencial forzado:
  1. Planner : Mistral decide llamar a buscar_producto_jumbo
  2. Executor: Ejecutamos la llamada al MCP de Jumbo
  3. Planner : Mistral decide llamar a calcular_sellos_ley20606
  4. Executor: Ejecutamos la llamada al MCP de Sellos
  5. Generator: Mistral produce el análisis clínico final
"""

import json
import logging
from typing import Optional, Tuple

import requests
from mistralai import Mistral

from . import config

logger = logging.getLogger(__name__)

MAX_ITER = 8  # techo de seguridad para el loop de tool calls

# ── Definición de herramientas ────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_producto_jumbo",
            "description": (
                "Busca un producto alimentario en Jumbo.cl y retorna su ficha completa: "
                "nombre, marca, categoría, valores nutricionales por 100g/ml e ingredientes. "
                "DEBE ser el primer paso de cualquier análisis nutricional."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre o descripción del producto a buscar",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_sellos_ley20606",
            "description": (
                "Calcula los sellos negros Ley 20.606 (Chile) a partir de los valores "
                "nutricionales por 100g/ml. DEBE llamarse después de buscar_producto_jumbo, "
                "usando los valores nutricionales obtenidos de ese resultado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calorias_kcal": {
                        "type": "number",
                        "description": "Calorías por 100g/ml",
                    },
                    "sodio_mg": {
                        "type": "number",
                        "description": "Sodio en mg por 100g/ml",
                    },
                    "azucares_g": {
                        "type": "number",
                        "description": "Azúcares totales en g por 100g/ml",
                    },
                    "grasas_sat_g": {
                        "type": "number",
                        "description": "Grasas saturadas en g por 100g/ml",
                    },
                    "es_liquido": {
                        "type": "boolean",
                        "description": "True si el producto es bebida o líquido",
                    },
                },
                "required": ["calorias_kcal", "sodio_mg", "azucares_g", "grasas_sat_g"],
            },
        },
    },
]

# ── Ejecutores de herramientas ────────────────────────────────────────────────

def _exec_buscar_jumbo(query: str) -> str:
    try:
        r = requests.get(config.JUMBO_MCP_URL, params={"q": query}, timeout=120)
        if r.status_code == 404:
            return json.dumps(
                {"error": f"Producto '{query}' no encontrado en Jumbo.cl"},
                ensure_ascii=False,
            )
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error("Error Jumbo MCP: %s", e)
        return json.dumps({"error": str(e)})


def _exec_calcular_sellos(args: dict) -> str:
    try:
        r = requests.post(config.SELLOS_MCP_URL, json=args, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error("Error Sellos MCP: %s", e)
        return json.dumps({"error": str(e)})


def _execute_tool(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Argumentos JSON inválidos"})

    if name == "buscar_producto_jumbo":
        return _exec_buscar_jumbo(args.get("query", ""))
    if name == "calcular_sellos_ley20606":
        return _exec_calcular_sellos(args)
    return json.dumps({"error": f"Herramienta desconocida: {name}"})


# ── System prompt ─────────────────────────────────────────────────────────────

def _system_prompt(condicion: Optional[str]) -> str:
    cond_name = condicion.upper() if condicion else "SALUD GENERAL"
    return f"""Eres NutriCheck, un Asistente Senior de Nutrición Clínica especializado en la población chilena y la Ley de Etiquetado 20.606.

INICIO OBLIGATORIO: Comienza SIEMPRE tu respuesta final con:
"{config.DISCLAIMER}"

FLUJO OBLIGATORIO — Plan-and-Execute:
  Paso 1: Llama a `buscar_producto_jumbo` para obtener los datos reales del producto.
  Paso 2: Llama a `calcular_sellos_ley20606` con los valores nutricionales obtenidos en el paso 1.
  Paso 3: Con ambos resultados, genera el análisis clínico final.
Nunca omitas ningún paso ni inventes datos nutricionales.

REGLAS DE RAZONAMIENTO:
1. IDENTIDAD: Si el producto encontrado no coincide con lo preguntado, decláralo explícitamente.
2. SELLOS: Evalúa el impacto de cada sello según la condición del usuario: {cond_name}.
3. RIGOR: Cita valores exactos (mg/g por 100g) para respaldar tu veredicto.
4. INGREDIENTES: Detecta riesgos ocultos: Maltodextrina, Nitritos, Jarabe de alta fructosa.

FORMATO DE RESPUESTA FINAL (ESTRICTO):
---
🏷️ *PRODUCTO IDENTIFICADO*
Nombre: [Nombre comercial] | Marca: [Marca]

🛡️ *SELLOS LEY 20.606*
[Lista de sellos o "Ninguno"] → [Riesgo para {cond_name}]

🩺 *VEREDICTO PARA {cond_name}*
[APTO / NO APTO / MODERADO]

📊 *ANÁLISIS NUTRICIONAL (por 100g/ml)*
- Sodio: [mg] | Azúcares: [g] | Grasas Sat: [g] | Calorías: [kcal]

🔍 *JUSTIFICACIÓN TÉCNICA*
[Por qué los valores son compatibles o peligrosos para {cond_name}]

💬 *CONSEJO PARA EL USUARIO*
[Mensaje empático. Sugiere alternativas si es No Apto]
---
"""


# ── Agente Plan-and-Execute ───────────────────────────────────────────────────

class NutriCheckRAG:
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        mistral_model: str = config.MISTRAL_MODEL,
    ):
        api_key = mistral_api_key or config.MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError("Falta MISTRAL_API_KEY en el entorno.")
        self.mistral_model = mistral_model
        self.client = Mistral(api_key=api_key)

    def ask(self, pregunta: str, condicion: Optional[str] = None) -> Tuple[str, list]:
        messages: list = [
            {"role": "system", "content": _system_prompt(condicion)},
            {"role": "user",   "content": pregunta},
        ]

        for iteration in range(MAX_ITER):
            response = self.client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            choice = response.choices[0]

            # Sin tool calls → respuesta final del modelo
            if choice.finish_reason != "tool_calls":
                logger.info("Plan-and-Execute completado en %d iteración(es)", iteration + 1)
                return choice.message.content, messages

            # Agregar el turno del assistant con sus tool_calls
            messages.append(choice.message)

            # Ejecutar cada tool call y devolver resultados
            for tc in choice.message.tool_calls:
                result = _execute_tool(tc.function.name, tc.function.arguments)
                logger.info(
                    "[iter %d] Tool '%s' args=%s → %s…",
                    iteration + 1,
                    tc.function.name,
                    tc.function.arguments[:80],
                    result[:100],
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        # Límite de iteraciones alcanzado — forzar respuesta final sin tools
        logger.warning("MAX_ITER (%d) alcanzado para: %s", MAX_ITER, pregunta)
        final = self.client.chat.complete(
            model=self.mistral_model,
            messages=messages,
            temperature=0.1,
        )
        return final.choices[0].message.content, messages
