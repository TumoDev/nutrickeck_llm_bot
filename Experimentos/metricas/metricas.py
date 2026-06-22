"""Sistema de recolección y cálculo de métricas."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class SelloVeredicto(Enum):
    """Veredictos posibles para un sello."""
    CORRECTO = "correcto"
    INCORRECTO = "incorrecto"
    OMITIDO = "omitido"


@dataclass
class MetricasRespuesta:
    """Métricas de una única respuesta."""
    baseline_name: str
    pregunta: str
    patologia: str
    producto_esperado: str

    # Latencia
    tiempo_respuesta_ms: float = 0.0
    num_llamadas_llm: int = 0
    num_llamadas_api: int = 0

    # Exactitud de sellos (Ley 20.606)
    sellos_esperados: List[str] = field(default_factory=list)
    sellos_obtenidos: List[str] = field(default_factory=list)
    exactitud_sellos: float = 0.0  # 0.0 a 1.0

    # Riesgos clínicos omitidos
    riesgos_esperados: List[str] = field(default_factory=list)
    riesgos_mencionados: List[str] = field(default_factory=list)
    tasa_omision_riesgo: float = 0.0  # % de riesgos no mencionados

    # Respuesta del LLM
    respuesta_texto: str = ""
    respuesta_veredicto: str = ""  # APTO / NO APTO / MODERADO

    # Activación del router de corrección (solo para NutriCheck)
    ciclos_activados: int = 0
    tasa_correccion_efectiva: float = 0.0  # % de ciclos que llegaron a reporte completo

    # Trazas
    traza_llm: List[str] = field(default_factory=list)
    errores: List[str] = field(default_factory=list)

    def calcular_exactitud_sellos(self) -> float:
        """Calcula exactitud de sellos: match entre esperados y obtenidos."""
        if not self.sellos_esperados:
            return 1.0 if not self.sellos_obtenidos else 0.0

        esperados_set = set(self.sellos_esperados)
        obtenidos_set = set(self.sellos_obtenidos)

        # Intersection / Union (Jaccard)
        intersecion = len(esperados_set & obtenidos_set)
        union = len(esperados_set | obtenidos_set)

        return intersecion / union if union > 0 else 0.0

    def calcular_tasa_omision_riesgo(self) -> float:
        """% de riesgos esperados que no fueron mencionados."""
        if not self.riesgos_esperados:
            return 0.0

        mencionados_set = set(self.riesgos_mencionados)
        omitidos = [r for r in self.riesgos_esperados if r not in mencionados_set]

        return len(omitidos) / len(self.riesgos_esperados)

    def calcular_costo(self) -> float:
        """Score de costo: tiempo + llamadas LLM."""
        # Normalizar: (time_ms / 1000) + (llamadas * peso)
        return (self.tiempo_respuesta_ms / 1000.0) + (self.num_llamadas_llm * 0.05)


@dataclass
class MetricasAgregadas:
    """Agregación de métricas para un baseline completo."""
    baseline_name: str
    num_casos: int = 0

    # Exactitud de sellos
    exactitud_sellos_promedio: float = 0.0
    exactitud_sellos_std: float = 0.0

    # Tasa de omisión de riesgo clínico
    tasa_omision_promedio: float = 0.0
    tasa_omision_std: float = 0.0

    # Tasa de corrección efectiva
    tasa_correccion_efectiva_promedio: float = 0.0
    tasa_correccion_efectiva_std: float = 0.0

    # Latencia / Costo
    latencia_promedio_ms: float = 0.0
    latencia_std_ms: float = 0.0
    num_llamadas_llm_promedio: int = 0
    num_llamadas_api_promedio: int = 0
    costo_promedio: float = 0.0

    # Errores
    tasa_errores: float = 0.0
    errores_registrados: List[str] = field(default_factory=list)

    respuestas: List[MetricasRespuesta] = field(default_factory=list)

    def agregar_respuesta(self, metrica: MetricasRespuesta):
        """Agrega una respuesta y recalcula agregados."""
        self.respuestas.append(metrica)
        self._recalcular()

    def _recalcular(self):
        """Recalcula todas las métricas agregadas."""
        if not self.respuestas:
            return

        import statistics

        self.num_casos = len(self.respuestas)

        # Exactitud de sellos
        exactitudes = [r.calcular_exactitud_sellos() for r in self.respuestas]
        self.exactitud_sellos_promedio = statistics.mean(exactitudes)
        self.exactitud_sellos_std = statistics.stdev(exactitudes) if len(exactitudes) > 1 else 0.0

        # Tasa de omisión
        omisiones = [r.calcular_tasa_omision_riesgo() for r in self.respuestas]
        self.tasa_omision_promedio = statistics.mean(omisiones)
        self.tasa_omision_std = statistics.stdev(omisiones) if len(omisiones) > 1 else 0.0

        # Tasa de corrección
        correcciones = [r.tasa_correccion_efectiva for r in self.respuestas if r.ciclos_activados > 0]
        self.tasa_correccion_efectiva_promedio = statistics.mean(correcciones) if correcciones else 0.0
        self.tasa_correccion_efectiva_std = statistics.stdev(correcciones) if len(correcciones) > 1 else 0.0

        # Latencia
        latencias = [r.tiempo_respuesta_ms for r in self.respuestas]
        self.latencia_promedio_ms = statistics.mean(latencias)
        self.latencia_std_ms = statistics.stdev(latencias) if len(latencias) > 1 else 0.0

        # Llamadas LLM
        self.num_llamadas_llm_promedio = int(statistics.mean([r.num_llamadas_llm for r in self.respuestas]))
        self.num_llamadas_api_promedio = int(statistics.mean([r.num_llamadas_api for r in self.respuestas]))

        # Costo
        costos = [r.calcular_costo() for r in self.respuestas]
        self.costo_promedio = statistics.mean(costos)

        # Errores
        con_error = [r for r in self.respuestas if r.errores]
        self.tasa_errores = len(con_error) / len(self.respuestas) if self.respuestas else 0.0
        self.errores_registrados = [e for r in con_error for e in r.errores]

    def resumen_tabla(self) -> str:
        """Genera una tabla de resumen para printear."""
        return f"""
╔════════════════════════════════════════════════════════════════════════╗
║ BASELINE: {self.baseline_name:25} | CASOS: {self.num_casos:3}                    ║
╠════════════════════════════════════════════════════════════════════════╣
║ Exactitud Sellos (Ley 20.606):      {self.exactitud_sellos_promedio*100:6.2f}% ± {self.exactitud_sellos_std*100:5.2f}%  ║
║ Tasa Omisión Riesgo Clínico:        {self.tasa_omision_promedio*100:6.2f}% ± {self.tasa_omision_std*100:5.2f}%  ║
║ Tasa Corrección Efectiva:           {self.tasa_correccion_efectiva_promedio*100:6.2f}% ± {self.tasa_correccion_efectiva_std*100:5.2f}%  ║
╠════════════════════════════════════════════════════════════════════════╣
║ Latencia Promedio:                  {self.latencia_promedio_ms:7.0f} ms ± {self.latencia_std_ms:6.0f}    ║
║ Llamadas LLM (promedio):            {self.num_llamadas_llm_promedio:3} llamadas                    ║
║ Llamadas API (promedio):            {self.num_llamadas_api_promedio:3} llamadas                    ║
║ Costo Combinado:                    {self.costo_promedio:7.3f}                       ║
║ Tasa de Errores:                    {self.tasa_errores*100:6.2f}%                      ║
╚════════════════════════════════════════════════════════════════════════╝
"""
