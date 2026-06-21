"""Pipeline NutriCheck — function calling + ciclo de autocorrección.

Estado compartido que va mutando, con trazabilidad en terminal (ver trazas.py):

  1. nodo_analizador_llm  → muta perfil_clinico  (patología del usuario)
  2. buscar_producto      → PASO FIJO (no function call): RAG local (CSV) y, si no
                            hay match confiable, scraping_jumbo en vivo (NO es RAG)
  3. calcular_sellos_ley20606 → FUNCTION CALLING OPCIONAL: el modelo decide. Si los
                            sellos ya vienen de la fuente los usa; si faltan, los calcula
  4. agente_clinico  (iter N)  → muta reporte           (borrador clínico, Mistral)
  5. agente_validador          → muta siguiente_accion  ("corregir" | "aprobar")
     └─ [CICLO] si "corregir": el router redirige a agente_clinico con el feedback
"""

import json
import logging
from typing import Optional, Tuple

import requests
from mistralai import Mistral

from . import config
from .trazas import Traza

logger = logging.getLogger(__name__)

MAX_CICLOS = 3  # techo de seguridad para el ciclo clínico↔validador

# ── Herramienta para FUNCTION CALLING ──────────────────────────────────────────
# buscar_producto NO es function call: se ejecuta siempre desde el código.
# Solo calcular_sellos_ley20606 es function call (el modelo decide llamarla).

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calcular_sellos_ley20606",
            "description": (
                "Calcula los sellos negros de la Ley 20.606 (Chile) a partir de los valores "
                "nutricionales por 100g/ml del producto entregado. Llámala con esos valores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calorias_kcal": {"type": "number", "description": "Calorías por 100g/ml"},
                    "sodio_mg":      {"type": "number", "description": "Sodio en mg por 100g/ml"},
                    "azucares_g":    {"type": "number", "description": "Azúcares en g por 100g/ml"},
                    "grasas_sat_g":  {"type": "number", "description": "Grasas saturadas en g por 100g/ml"},
                    "es_liquido":    {"type": "boolean", "description": "True si es bebida/líquido"},
                },
                "required": ["calorias_kcal", "sodio_mg", "azucares_g", "grasas_sat_g"],
            },
        },
    },
]

MAX_TOOL_ITER = 5  # techo del loop de function calling

# ── Ejecutores de herramientas (REST a los MCP) ────────────────────────────────

def _exec_buscar_jumbo(query: str) -> dict:
    try:
        r = requests.get(config.JUMBO_MCP_URL, params={"q": query}, timeout=120)
        if r.status_code == 404:
            return {"error": f"Producto '{query}' no encontrado en Jumbo.cl"}
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Error Jumbo MCP: %s", e)
        return {"error": str(e)}


def _exec_calcular_sellos(args: dict) -> dict:
    try:
        r = requests.post(config.SELLOS_MCP_URL, json=args, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Error Sellos MCP: %s", e)
        return {"error": str(e)}


# ── Prompts ────────────────────────────────────────────────────────────────────

def _prompt_clinico(perfil: dict, feedback: Optional[str]) -> str:
    cond = perfil.get("patologia", "salud general").upper()
    correccion = ""
    if feedback:
        correccion = (
            "\n\n⚠️ CORRECCIÓN OBLIGATORIA DEL VALIDADOR — tu borrador anterior fue "
            f"rechazado. Debes incorporar esto sí o sí:\n«{feedback}»"
        )
    return f"""Eres NutriCheck, Asistente Senior de Nutrición Clínica (Chile, Ley 20.606).
Genera el reporte para un usuario con condición: {cond}.

Comienza SIEMPRE con:
"{config.DISCLAIMER}"

Usa EXCLUSIVAMENTE los datos del producto y los sellos que se te entregan; no inventes.
Evalúa el impacto de cada sello para {cond}, incluyendo riesgos INDIRECTOS
(p. ej. cómo el azúcar afecta la presión arterial en hipertensión).{correccion}

FORMATO ESTRICTO:
---
🏷️ *PRODUCTO* — Nombre | Marca
🛡️ *SELLOS LEY 20.606* — [lista o "Ninguno"]
🩺 *VEREDICTO PARA {cond}* — [APTO / NO APTO / MODERADO]
📊 *NUTRICIÓN (100g/ml)* — Sodio | Azúcares | Grasas Sat | Calorías
🔍 *JUSTIFICACIÓN TÉCNICA* — incluye riesgos indirectos relevantes para {cond}
💬 *CONSEJO* — empático; sugiere alternativas si es No Apto
---"""


def _prompt_validador(perfil: dict) -> str:
    cond = perfil.get("patologia", "salud general").upper()
    return f"""Eres el Auditor Clínico de NutriCheck. Revisas el reporte de otro agente
para un usuario con condición: {cond}.

Tu única tarea: verificar que el reporte explique correctamente TODOS los riesgos
relevantes para {cond}, incluidos los INDIRECTOS. Ejemplo crítico: si hay sello
"ALTO EN AZÚCARES" y la condición es HIPERTENSION, el reporte DEBE explicar el
vínculo del azúcar con la presión arterial / riesgo cardiovascular; si lo omite,
es un error grave.

Responde SOLO con un JSON válido, sin texto adicional:
{{"veredicto": "aprobar" | "corregir", "motivo": "<qué falta o por qué aprueba>"}}"""


# ── Pipeline ───────────────────────────────────────────────────────────────────

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

    # ── Nodos ──────────────────────────────────────────────────────────────────

    def _nodo_analizador_llm(self, pregunta: str, condicion: Optional[str], t: Traza) -> dict:
        t.nodo("🧠 nodo_analizador_llm")
        if condicion:
            perfil = {"patologia": condicion}
        else:
            perfil = {"patologia": self._inferir_patologia(pregunta)}
        t.muta("perfil_clinico", perfil)
        t.fin_paso()
        return perfil

    def _inferir_patologia(self, pregunta: str) -> str:
        try:
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content":
                        "Identifica la condición de salud relevante en la pregunta. "
                        "Responde SOLO una palabra: "
                        + ", ".join(sorted(config.CONDICIONES_VALIDAS))
                        + ", o 'general'."},
                    {"role": "user", "content": pregunta},
                ],
                temperature=0,
            )
            cond = r.choices[0].message.content.strip().lower()
            return cond if cond in config.CONDICIONES_VALIDAS else "general"
        except Exception as e:
            logger.warning("Fallo inferencia de patología: %s", e)
            return "general"

    def _resumen_producto(self, producto: dict) -> dict:
        """Resumen compacto para la traza (sin volcar todos los ingredientes)."""
        nutri = producto.get("nutricion", {})
        return {
            "nombre": producto.get("nombre", "?"),
            "marca": producto.get("marca", "?"),
            "azucares_g": nutri.get("azucares_g"),
            "sodio_mg": nutri.get("sodio_mg"),
            "es_liquido": producto.get("es_liquido"),
        }

    def _herramienta_rag_local(self, query: str, condicion: Optional[str],
                               t: Traza) -> Optional[dict]:
        """RAG local (CSV + ChromaDB). Devuelve el producto si hay match confiable, o None."""
        t.nodo("📚 herramienta_rag_local  (CSV + ChromaDB)")
        try:
            from .rag_local import obtener_rag_local
            rag = obtener_rag_local()
        except Exception as e:
            logger.warning("RAG local no importable (%s).", e)
            rag = None

        if rag is None:
            t.info("RAG local no disponible → se intentará el MCP de Jumbo")
            t.fin_paso()
            return None

        producto = rag.buscar_producto(query, condicion)
        if producto is None:
            t.info("sin candidatos en el CSV → fallback al MCP de Jumbo")
            t.fin_paso()
            return None

        candidato = producto.get("nombre", "?")
        t.info(f"mejor candidato CSV: '{candidato}' (score={producto.get('_score'):.3f})")

        # El bot DECIDE: ¿el candidato corresponde a lo que pidió el usuario?
        if not self._verifica_identidad(query, candidato):
            t.muta("siguiente_accion", "buscar_en_mcp")
            t.info(f"verificación de identidad: '{candidato}' NO coincide → fallback MCP Jumbo")
            t.fin_paso()
            return None

        t.muta("producto_datos", {**self._resumen_producto(producto), "fuente": "rag_local_csv"})
        t.info("✅ encontrado en RAG local — se evita el scraping")
        t.fin_paso()
        return producto

    def _verifica_identidad(self, query: str, candidato: str) -> bool:
        """LLM decide si el producto recuperado corresponde al pedido."""
        try:
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content":
                        "Decide si el PRODUCTO RECUPERADO es el mismo que pide el USUARIO "
                        "(misma identidad comercial, tolerando variaciones de formato/tamaño). "
                        'Responde SOLO JSON: {"coincide": true|false}.'},
                    {"role": "user", "content": f"USUARIO: {query}\nRECUPERADO: {candidato}"},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return bool(json.loads(r.choices[0].message.content).get("coincide", False))
        except Exception as e:
            logger.warning("Verificación de identidad falló (%s); se acepta el candidato.", e)
            return True

    def _scraping_jumbo(self, query: str, t: Traza) -> dict:
        """Scraping de Jumbo.cl en vivo (NO es RAG): fallback cuando el RAG local no acierta."""
        t.nodo("🌐 scraping_jumbo  (web en vivo — fallback, no es RAG)")
        t.info(f"REST GET {config.JUMBO_MCP_URL}  q='{query}'")
        producto = _exec_buscar_jumbo(query)
        if not producto.get("error"):
            t.muta("producto_datos", {**self._resumen_producto(producto), "fuente": "scraping_jumbo"})
        t.fin_paso()
        return producto

    def _extraer_producto(self, pregunta: str) -> str:
        """Extrae el nombre del producto desde el texto (llamada normal a Mistral)."""
        try:
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content":
                        "Extrae SOLO el nombre del producto alimentario de la pregunta. "
                        "Responde únicamente con el nombre, sin ninguna otra palabra."},
                    {"role": "user", "content": pregunta},
                ],
                temperature=0,
            )
            return (r.choices[0].message.content or "").strip() or pregunta
        except Exception as e:
            logger.warning("Extracción de producto falló (%s); uso la pregunta cruda.", e)
            return pregunta

    def extraer_producto_foto(self, imagen_b64: str) -> dict:
        """Visión de Mistral: LEE la etiqueta de la foto y devuelve el producto estructurado
        (nombre, marca, nutrición y sellos visibles). Se analiza con ESTOS datos, sin buscar
        en Jumbo. Devuelve {} si no se pudo identificar."""
        try:
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content":
                        "Eres un lector de etiquetas de alimentos chilenos. Mira la foto y "
                        "extrae SOLO lo que se ve. Responde un JSON con: nombre, marca, "
                        "es_liquido (bool), nutricion (objeto con calorias_kcal, azucares_g, "
                        "sodio_mg, grasas_sat_g por 100 g/ml; usa null si la tabla no se ve), "
                        "ingredientes (texto o ''), y sellos (lista de sellos negros "
                        "'ALTO EN ...' impresos en el envase, o null si no se ven). "
                        "No inventes valores que no aparezcan en la imagen."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extrae los datos del producto de esta foto."},
                        {"type": "image_url",
                         "image_url": f"data:image/jpeg;base64,{imagen_b64}"},
                    ]},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
        except Exception as e:
            logger.warning("Lectura de foto falló: %s", e)
            return {}

        if not str(data.get("nombre") or "").strip():
            return {}  # no se pudo identificar

        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        nutri = data.get("nutricion") or {}
        return {
            "nombre":       str(data.get("nombre", "")),
            "marca":        str(data.get("marca", "") or ""),
            "categoria":    "",
            "ingredientes": str(data.get("ingredientes", "") or ""),
            "es_liquido":   bool(data.get("es_liquido", False)),
            "nutricion": {
                "calorias_kcal": _num(nutri.get("calorias_kcal")),
                "azucares_g":    _num(nutri.get("azucares_g")),
                "sodio_mg":      _num(nutri.get("sodio_mg")),
                "grasas_sat_g":  _num(nutri.get("grasas_sat_g")),
            },
            "sellos":  data.get("sellos"),  # lista visible en el envase, o None
            "_fuente": "foto",
        }

    def _buscar_producto(self, pregunta: str, patologia: Optional[str], t: Traza,
                         producto_directo: Optional[dict] = None) -> dict:
        """Paso FIJO. Si viene un producto leído de una foto, se usa TAL CUAL (sin buscar en
        fuentes externas). Si no, se busca en el RAG local y, si falla, en el scraping Jumbo."""
        t.nodo("📦 buscar_producto  (paso fijo — siempre se ejecuta)")
        if producto_directo:
            t.info(f"producto leído de la foto: '{producto_directo.get('nombre', '?')}' "
                   "→ se analiza con los datos de la imagen (NO se busca en Jumbo)")
            t.muta("producto_datos", {**self._resumen_producto(producto_directo), "fuente": "foto"})
            t.fin_paso()
            return producto_directo

        query = self._extraer_producto(pregunta)
        t.info(f"producto extraído: '{query}'")
        t.fin_paso()

        producto = self._herramienta_rag_local(query, patologia, t)
        if producto is None:
            producto = self._scraping_jumbo(query, t)
        return producto

    def _recolectar_sellos(self, producto: dict, t: Traza) -> list:
        """Sellos del producto. Si la fuente (web/CSV) ya los trae, se usan directo.
        Solo si faltan (null) el modelo DECIDE llamar calcular_sellos (function calling)."""
        t.nodo("🔎 decisión de sellos")
        sellos_fuente = producto.get("sellos")  # del RAG local (CSV) o del scraping Jumbo

        # La fuente ya los trae (incluso lista vacía = "sin sellos") → no se llama nada.
        if sellos_fuente is not None:
            t.info(f"la fuente ya trae sellos → no se llama la herramienta")
            t.muta("sellos_identificados", sellos_fuente)
            t.fin_paso()
            return sellos_fuente

        t.info("la fuente NO trae sellos → el modelo decide calcular (function calling)")
        t.fin_paso()

        nutri = producto.get("nutricion", {})
        messages: list = [
            {"role": "system", "content":
                "No hay sellos calculados para este producto. Llama a la herramienta "
                "calcular_sellos_ley20606 con sus valores nutricionales para obtenerlos. "
                "Llámala UNA sola vez."},
            {"role": "user", "content": json.dumps({
                "nutricion": nutri,
                "es_liquido": producto.get("es_liquido", False),
            }, ensure_ascii=False)},
        ]

        sellos: Optional[list] = None
        for _ in range(MAX_TOOL_ITER):
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            choice = r.choices[0]
            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                break

            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                t.function_call(tc.function.name, args)
                t.nodo("⚖️  calcular_sellos_ley20606  (Ley 20.606)")
                t.info(f"REST POST {config.SELLOS_MCP_URL}  args={json.dumps(args, ensure_ascii=False)}")
                resultado = _exec_calcular_sellos(args)
                sellos = resultado.get("sellos", [])
                t.muta("sellos_identificados", sellos)
                t.fin_paso()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(resultado, ensure_ascii=False),
                })

            # Ya tenemos los sellos → cortamos el loop (evita llamadas repetidas).
            if sellos is not None:
                break

        return sellos if sellos is not None else []

    def _agente_clinico(self, perfil: dict, producto: dict, sellos: list,
                        feedback: Optional[str], iteracion: int, t: Traza) -> str:
        t.nodo("🩺 agente_clinico", iteracion=iteracion)
        contexto = {
            "producto": {k: producto.get(k) for k in ("nombre", "marca", "categoria")},
            "nutricion": producto.get("nutricion", {}),
            "sellos": sellos,
            "ingredientes": producto.get("ingredientes", ""),
        }
        r = self.client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": _prompt_clinico(perfil, feedback)},
                {"role": "user", "content": json.dumps(contexto, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        reporte = r.choices[0].message.content
        t.muta("reporte", f"<{len(reporte)} chars> " +
               ("(corrige el borrador anterior)" if feedback else "(borrador inicial)"))
        t.fin_paso()
        return reporte

    def _agente_validador(self, perfil: dict, reporte: str, t: Traza) -> Tuple[str, str]:
        t.nodo("🔎 agente_validador")
        try:
            r = self.client.chat.complete(
                model=self.mistral_model,
                messages=[
                    {"role": "system", "content": _prompt_validador(perfil)},
                    {"role": "user", "content": reporte},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            accion = data.get("veredicto", "aprobar")
            motivo = data.get("motivo", "")
        except Exception as e:
            logger.warning("Validador falló (%s); se aprueba por defecto.", e)
            accion, motivo = "aprobar", "validador no disponible"

        accion = "corregir" if accion == "corregir" else "aprobar"
        t.muta("siguiente_accion", accion)
        if motivo:
            t.info(f"notas_validador: {motivo}")
        destino = "vuelve a agente_clinico" if accion == "corregir" else "END → Telegram"
        t.router(accion, destino)
        return accion, motivo

    # ── Orquestación (el "router" + ciclo) ───────────────────────────────────────

    def ask(self, pregunta: str, condicion: Optional[str] = None,
            producto_directo: Optional[dict] = None) -> Tuple[str, list]:
        t = Traza()
        t.inicio(pregunta, condicion)

        perfil = self._nodo_analizador_llm(pregunta, condicion, t)

        # Paso FIJO: buscar_producto siempre se ejecuta (no es function call).
        producto = self._buscar_producto(pregunta, perfil.get("patologia"), t, producto_directo)

        if producto.get("error"):
            reporte = f"{config.DISCLAIMER}\n\n❌ {producto['error']}"
            t.telegram(reporte)
            return reporte, []

        # FUNCTION CALLING: el modelo decide llamar calcular_sellos_ley20606.
        sellos = self._recolectar_sellos(producto, t)

        feedback: Optional[str] = None
        reporte = ""
        for iteracion in range(1, MAX_CICLOS + 1):
            reporte = self._agente_clinico(perfil, producto, sellos, feedback, iteracion, t)
            accion, motivo = self._agente_validador(perfil, reporte, t)

            if accion == "aprobar":
                break
            if iteracion < MAX_CICLOS:
                t.ciclo(motivo)               # router redirige en caliente al clínico
                feedback = motivo
            else:
                t.info("Techo de ciclos alcanzado; se entrega el último borrador.")

        t.telegram(reporte)
        return reporte, []
