"""Baseline 1: LLM + RAG simple (single-shot).

Una sola llamada que recibe el contexto recuperado y genera la respuesta sin verificación.
"""

import time
import requests
import json
from typing import Tuple, List, Optional
from mistralai import Mistral

from ..config import MISTRAL_API_KEY, MISTRAL_MODEL, JUMBO_MCP_URL, DISCLAIMER
from ..metricas.metricas import MetricasRespuesta


class BaselineSimpleRAG:
    """LLM + RAG simple: una sola llamada, sin ciclos de validación."""

    def __init__(self, mistral_api_key: Optional[str] = None):
        api_key = mistral_api_key or MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY no configurada")
        self.client = Mistral(api_key=api_key)
        self.mistral_model = MISTRAL_MODEL

    def procesar(
        self,
        pregunta: str,
        patologia: str,
        producto_esperado: str,
    ) -> Tuple[str, MetricasRespuesta]:
        """Procesa una consulta con baseline simple.

        Retorna:
            (respuesta_texto, metricas)
        """
        metrica = MetricasRespuesta(
            baseline_name="LLM + RAG Simple",
            pregunta=pregunta,
            patologia=patologia,
            producto_esperado=producto_esperado,
        )

        inicio = time.time()
        llamadas_llm = 0

        try:
            # Paso 1: Buscar producto (RAG simple)
            producto = self._buscar_producto(producto_esperado)
            metrica.num_llamadas_api += 1

            if not producto or producto.get("error"):
                respuesta = f"{DISCLAIMER}\n\n❌ Producto no encontrado: {producto_esperado}"
                metrica.respuesta_texto = respuesta
                metrica.errores.append(f"Producto no encontrado: {producto_esperado}")
                metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000
                return respuesta, metrica

            # Paso 2: Una sola llamada LLM (single-shot)
            respuesta = self._llamada_llm_unica(pregunta, patologia, producto)
            llamadas_llm += 1

            metrica.respuesta_texto = respuesta
            metrica.num_llamadas_llm = llamadas_llm
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000

            # Intentar extraer veredicto del texto
            metrica.respuesta_veredicto = self._extraer_veredicto(respuesta)

            # Intentar extraer sellos del texto
            metrica.sellos_obtenidos = self._extraer_sellos(respuesta)

            # Intentar detectar riesgos clínicos mencionados
            metrica.riesgos_mencionados = self._extraer_riesgos(respuesta, patologia)

            return respuesta, metrica

        except Exception as e:
            metrica.errores.append(str(e))
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000
            respuesta = f"{DISCLAIMER}\n\n❌ Error: {str(e)}"
            return respuesta, metrica

    def _buscar_producto(self, nombre_producto: str) -> Optional[dict]:
        """Busca un producto en Jumbo MCP."""
        try:
            r = requests.get(JUMBO_MCP_URL, params={"q": nombre_producto}, timeout=30)
            if r.status_code == 404:
                return {"error": f"Producto '{nombre_producto}' no encontrado"}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _llamada_llm_unica(self, pregunta: str, patologia: str, producto: dict) -> str:
        """Una sola llamada LLM con contexto del producto."""
        contexto = self._formatear_contexto(producto)
        prompt = f"""Eres NutriCheck, asistente de nutrición clínica (Chile, Ley 20.606).
Usuario pregunta: {pregunta}
Condición: {patologia}

Contexto del producto:
{contexto}

{DISCLAIMER}

Responde de forma clara indicando si el producto es APTO/NO APTO/MODERADO para su condición.
Usa el formato:
🏷️ *PRODUCTO*: [nombre]
🛡️ *SELLOS LEY 20.606*: [sellos o Ninguno]
🩺 *VEREDICTO PARA {patologia}*: [APTO/NO APTO/MODERADO]
📊 *NUTRICIÓN*: [datos]
🔍 *JUSTIFICACIÓN*: [explicación]
💬 *CONSEJO*: [consejo]"""

        response = self.client.messages.create(
            model=self.mistral_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        return response.content[0].text

    def _formatear_contexto(self, producto: dict) -> str:
        """Formatea los datos del producto como contexto."""
        if producto.get("error"):
            return f"Error: {producto['error']}"

        nutricion = producto.get("nutricion", {})
        sellos = producto.get("sellos", [])

        return f"""
Nombre: {producto.get('nombre', 'N/A')}
Marca: {producto.get('marca', 'N/A')}
Sellos identificados: {', '.join(sellos) if sellos else 'Ninguno'}
Nutrición (por 100g/ml):
  - Calorías: {nutricion.get('calorias_kcal', 'N/A')} kcal
  - Azúcares: {nutricion.get('azucares_g', 'N/A')} g
  - Sodio: {nutricion.get('sodio_mg', 'N/A')} mg
  - Grasas saturadas: {nutricion.get('grasas_sat_g', 'N/A')} g
"""

    def _extraer_veredicto(self, respuesta: str) -> str:
        """Intenta extraer APTO/NO APTO/MODERADO del texto."""
        respuesta_upper = respuesta.upper()
        if "NO APTO" in respuesta_upper:
            return "NO APTO"
        elif "APTO" in respuesta_upper:
            return "APTO"
        elif "MODERADO" in respuesta_upper:
            return "MODERADO"
        return "DESCONOCIDO"

    def _extraer_sellos(self, respuesta: str) -> List[str]:
        """Extrae sellos mencionados en la respuesta."""
        sellos_posibles = [
            "ALTO EN CALORÍAS",
            "ALTO EN AZÚCARES",
            "ALTO EN SODIO",
            "ALTO EN GRASAS SATURADAS",
        ]
        encontrados = [s for s in sellos_posibles if s in respuesta.upper()]
        return encontrados

    def _extraer_riesgos(self, respuesta: str, patologia: str) -> List[str]:
        """Extrae riesgos clínicos mencionados."""
        riesgos_palabras = [
            "glucemia", "diabetes", "presión", "hipertensión",
            "colesterol", "cardiovascular", "alérgico", "alergia",
            "gástrico", "anemia", "hierro", "descompensación",
            "sobrecarga", "ateroesclerosis"
        ]
        encontrados = [r for r in riesgos_palabras if r.lower() in respuesta.lower()]
        return encontrados
