"""Baseline 2: Agente único ReAct.

Un solo agente que decide y ejecuta herramientas en bucle, sin separación de roles
ni nodo validador dedicado.
"""

import time
import requests
import json
from typing import Tuple, List, Optional
from mistralai import Mistral

from ..config import MISTRAL_API_KEY, MISTRAL_MODEL, JUMBO_MCP_URL, SELLOS_MCP_URL, DISCLAIMER
from ..metricas.metricas import MetricasRespuesta


class BaselineReActUnico:
    """Agente único ReAct: loop de decisión-acción sin roles separados."""

    def __init__(self, mistral_api_key: Optional[str] = None):
        api_key = mistral_api_key or MISTRAL_API_KEY
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY no configurada")
        self.client = Mistral(api_key=api_key)
        self.mistral_model = MISTRAL_MODEL
        self.max_iteraciones = 5

    def procesar(
        self,
        pregunta: str,
        patologia: str,
        producto_esperado: str,
    ) -> Tuple[str, MetricasRespuesta]:
        """Procesa una consulta con ReAct único.

        Retorna:
            (respuesta_texto, metricas)
        """
        metrica = MetricasRespuesta(
            baseline_name="Agente ReAct Único",
            pregunta=pregunta,
            patologia=patologia,
            producto_esperado=producto_esperado,
        )

        inicio = time.time()
        llamadas_llm = 0

        try:
            # Estado del agente
            estado = {
                "pregunta": pregunta,
                "patologia": patologia,
                "producto": None,
                "sellos": [],
                "reporte": "",
                "terminado": False,
            }

            # Loop ReAct
            for iteracion in range(self.max_iteraciones):
                if estado["terminado"]:
                    break

                # Llamada LLM para decidir acción
                accion, args = self._decidir_accion(estado, iteracion)
                llamadas_llm += 1

                if accion == "buscar_producto":
                    estado["producto"] = self._buscar_producto(args.get("nombre"))
                    metrica.num_llamadas_api += 1

                elif accion == "calcular_sellos":
                    resultado = self._calcular_sellos(args)
                    estado["sellos"] = resultado.get("sellos", [])
                    metrica.num_llamadas_api += 1

                elif accion == "generar_reporte":
                    estado["reporte"] = self._generar_reporte(estado)
                    llamadas_llm += 1

                elif accion == "finalizar":
                    estado["terminado"] = True

            respuesta = estado.get("reporte", f"{DISCLAIMER}\n\n❌ No se pudo generar reporte")
            metrica.respuesta_texto = respuesta
            metrica.num_llamadas_llm = llamadas_llm
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000

            # Extraer datos de la respuesta
            metrica.respuesta_veredicto = self._extraer_veredicto(respuesta)
            metrica.sellos_obtenidos = self._extraer_sellos(respuesta)
            metrica.riesgos_mencionados = self._extraer_riesgos(respuesta, patologia)

            return respuesta, metrica

        except Exception as e:
            metrica.errores.append(str(e))
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000
            respuesta = f"{DISCLAIMER}\n\n❌ Error: {str(e)}"
            return respuesta, metrica

    def _decidir_accion(self, estado: dict, iteracion: int) -> Tuple[str, dict]:
        """LLM decide la próxima acción (ReAct)."""
        historial = f"Iteración {iteracion + 1}/5"
        if estado["producto"]:
            historial += f"\nProducto encontrado: {estado['producto'].get('nombre')}"
        if estado["sellos"]:
            historial += f"\nSellos calculados: {', '.join(estado['sellos'])}"

        prompt = f"""Eres un agente ReAct de análisis de nutrición.
Tarea: Responder: {estado['pregunta']}
Patología: {estado['patologia']}
Estado actual:
{historial}

Disponible: buscar_producto, calcular_sellos, generar_reporte, finalizar.

Responde SOLO con JSON:
{{"accion": "buscar_producto|calcular_sellos|generar_reporte|finalizar", "args": {{...}}}}

Ejemplo: {{"accion": "buscar_producto", "args": {{"nombre": "producto_name"}}}}"""

        response = self.client.messages.create(
            model=self.mistral_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )

        try:
            resultado = json.loads(response.content[0].text)
            return resultado.get("accion", "finalizar"), resultado.get("args", {})
        except json.JSONDecodeError:
            return "finalizar", {}

    def _buscar_producto(self, nombre: str) -> Optional[dict]:
        """Busca producto en Jumbo."""
        try:
            r = requests.get(JUMBO_MCP_URL, params={"q": nombre}, timeout=30)
            if r.status_code == 404:
                return {"error": "No encontrado"}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _calcular_sellos(self, args: dict) -> dict:
        """Calcula sellos usando MCP."""
        try:
            r = requests.post(SELLOS_MCP_URL, json=args, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e), "sellos": []}

    def _generar_reporte(self, estado: dict) -> str:
        """Genera el reporte clínico final."""
        producto = estado.get("producto", {})
        sellos = estado.get("sellos", [])
        patologia = estado.get("patologia", "SALUD_GENERAL")

        prompt = f"""Eres experto en nutrición clínica. Genera reporte conciso.
Producto: {producto.get('nombre', 'N/A')}
Sellos: {', '.join(sellos) if sellos else 'Ninguno'}
Paciente: {patologia}

Responde en formato:
🏷️ *PRODUCTO*: [nombre]
🛡️ *SELLOS*: [lista]
🩺 *VEREDICTO*: [APTO/NO APTO/MODERADO]
🔍 *JUSTIFICACIÓN*: [breve]
💬 *CONSEJO*: [alternativa]"""

        response = self.client.messages.create(
            model=self.mistral_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        return f"{DISCLAIMER}\n\n{response.content[0].text}"

    def _extraer_veredicto(self, respuesta: str) -> str:
        """Extrae veredicto de la respuesta."""
        respuesta_upper = respuesta.upper()
        if "NO APTO" in respuesta_upper:
            return "NO APTO"
        elif "APTO" in respuesta_upper:
            return "APTO"
        elif "MODERADO" in respuesta_upper:
            return "MODERADO"
        return "DESCONOCIDO"

    def _extraer_sellos(self, respuesta: str) -> List[str]:
        """Extrae sellos mencionados."""
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
        ]
        encontrados = [r for r in riesgos_palabras if r.lower() in respuesta.lower()]
        return encontrados
