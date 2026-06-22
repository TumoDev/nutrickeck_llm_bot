"""Baseline 3: NutriCheck RAG completo.

Pipeline multiagente con planificación, RAG, cálculo determinista y validación clínica.
Reutiliza la arquitectura existente del proyecto.
"""

import sys
import time
from typing import Tuple, List, Optional
from pathlib import Path

from ..config import DISCLAIMER
from ..metricas.metricas import MetricasRespuesta

# Agregar el directorio del bot al path
bot_path = Path(__file__).parent.parent.parent / "bot"
if str(bot_path) not in sys.path:
    sys.path.insert(0, str(bot_path))

try:
    from app.graph import NutriCheckGraph
except ImportError:
    NutriCheckGraph = None


class BaselineNutriCheckCompleto:
    """NutriCheck RAG: pipeline multiagente con validación clínica y ciclos."""

    def __init__(self):
        if NutriCheckGraph is None:
            raise RuntimeError("No se pudo importar NutriCheckGraph. Verifica la instalación.")
        self.graph = NutriCheckGraph()
        self.num_llamadas_llm = 0
        self.num_llamadas_api = 0

    def procesar(
        self,
        pregunta: str,
        patologia: str,
        producto_esperado: str,
    ) -> Tuple[str, MetricasRespuesta]:
        """Procesa una consulta con NutriCheck completo.

        Retorna:
            (respuesta_texto, metricas)
        """
        metrica = MetricasRespuesta(
            baseline_name="NutriCheck RAG Completo",
            pregunta=pregunta,
            patologia=patologia,
            producto_esperado=producto_esperado,
        )

        inicio = time.time()
        self.num_llamadas_llm = 0
        self.num_llamadas_api = 0

        try:
            # Ejecutar el grafo completo
            respuesta, _ = self.graph.ask(pregunta, condicion=patologia)

            metrica.respuesta_texto = respuesta
            metrica.num_llamadas_llm = self.num_llamadas_llm
            metrica.num_llamadas_api = self.num_llamadas_api
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000

            # Extraer datos de la respuesta
            metrica.respuesta_veredicto = self._extraer_veredicto(respuesta)
            metrica.sellos_obtenidos = self._extraer_sellos(respuesta)
            metrica.riesgos_mencionados = self._extraer_riesgos(respuesta, patologia)

            # Si hay ciclos de corrección activos
            if "CICLO" in respuesta or "ciclo" in respuesta.lower():
                metrica.ciclos_activados += 1
                metrica.tasa_correccion_efectiva = 1.0 if respuesta else 0.0

            return respuesta, metrica

        except Exception as e:
            metrica.errores.append(str(e))
            metrica.tiempo_respuesta_ms = (time.time() - inicio) * 1000
            respuesta = f"{DISCLAIMER}\n\n❌ Error en NutriCheck: {str(e)}"
            return respuesta, metrica

    def _extraer_veredicto(self, respuesta: str) -> str:
        """Extrae APTO/NO APTO/MODERADO del texto."""
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
            "sobrecarga", "ateroesclerosis", "inflamación",
        ]
        encontrados = [r for r in riesgos_palabras if r.lower() in respuesta.lower()]
        return encontrados
