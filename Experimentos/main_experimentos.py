"""Script principal para ejecutar experimentos y comparar baselines.

Uso:
    python main_experimentos.py [--num-casos N] [--baseline baseline_name]
"""

# 1. IMPORTS DEL SISTEMA
import sys
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

# 2. INYECTAR LA RAÍZ AL PATH
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

# 3. IMPORTS DE TU PROYECTO
from Experimentos.baselines import BaselineSimpleRAG, BaselineReActUnico, BaselineNutriCheckCompleto
from Experimentos.datos.dataset import CasoPrueba, obtener_dataset  # Dataset real sincronizado con Jumbo
from Experimentos.metricas.metricas import MetricasRespuesta, MetricasAgregadas

# --- SOLUCIÓN PARA LA RESOLUCIÓN DE PUERTOS MCP ---
# Resuelve dinámicamente las variables que contienen expresiones tipo ${VAR}
for key, value in os.environ.items():
    if value and "${" in value:
        for sub_key, sub_value in os.environ.items():
            if f"${{{sub_key}}}" in value:
                os.environ[key] = value.replace(f"${{{sub_key}}}", sub_value)


def ejecutar_baseline(
    baseline_class,
    baseline_name: str,
    dataset: List[CasoPrueba],
    verbose: bool = False,
) -> MetricasAgregadas:
    """Ejecuta un baseline contra todos los casos de prueba del dataset."""
    print(f"\n{'='*80}")
    print(f"🚀 Ejecutando: {baseline_name}")
    print(f"{'='*80}\n")

    metricas_agregadas = MetricasAgregadas(baseline_name=baseline_name)

    try:
        baseline = baseline_class()
    except Exception as e:
        print(f"❌ Error inicializando baseline: {e}")
        return metricas_agregadas

    for idx, caso in enumerate(dataset, 1):
        print(f"  [{idx}/{len(dataset)}] {caso.id}: {caso.pregunta[:60]}...")

        try:
            respuesta, metrica = baseline.procesar(
                pregunta=caso.pregunta,
                patologia=caso.patologia,
                producto_esperado=caso.producto_esperado,
            )

            # Sincronizar valores esperados definidos en el nuevo dataset
            metrica.sellos_esperados = caso.sellos_esperados
            metrica.riesgos_esperados = caso.riesgos_clinicos_esperados
            
            # Forzar persistencia del veredicto si el pipeline lo retorna estructurado
            if isinstance(respuesta, dict) and "veredicto" in respuesta:
                metrica.respuesta_veredicto = respuesta["veredicto"]

            # Agregar al agregador (calcula de forma interna desviaciones estándar y promedios)
            metricas_agregadas.agregar_respuesta(metrica)

            if verbose:
                print(f"      ✓ Veredicto Obtenido: {metrica.respuesta_veredicto} (Esperado: {caso.veredicto_esperado})")
                print(f"      ✓ Sellos Detectados:  {metrica.sellos_obtenidos}")
                print(f"      ✓ Latencia:           {metrica.tiempo_respuesta_ms:.0f}ms")
                print()

        except Exception as e:
            print(f"      ❌ Error ejecutando {caso.id}: {e}")
            continue

    return metricas_agregadas


def generar_reporte_comparativo(
    resultados: Dict[str, MetricasAgregadas],
    output_file: Optional[str] = None,
) -> str:
    """Genera un reporte comparativo estético de todos los baselines evaluados."""
    reporte = "\n" + "="*80 + "\n"
    reporte += "📊 REPORTE COMPARATIVO DE BASELINES\n"
    reporte += "="*80 + "\n"

    reporte += "\n" + "RESUMEN EJECUTIVO".center(80) + "\n"
    reporte += "-"*80 + "\n\n"

    for baseline_name, metricas in resultados.items():
        reporte += metricas.resumen_tabla()

    reporte += "\n" + "="*80 + "\n"
    reporte += "COMPARACIÓN DE MÉTRICAS CLAVE\n"
    reporte += "="*80 + "\n\n"

    headers = " | ".join([f"{name:<15}" for name in resultados.keys()])
    reporte += f"{'Métrica':<40} | {headers}\n"
    reporte += "-" * (43 + 18 * len(resultados)) + "\n"

    # Exactitud de sellos
    valores = {name: f"{m.exactitud_sellos_promedio*100:.1f}%" for name, m in resultados.items()}
    reporte += f"{'Exactitud Sellos (Ley 20.606)':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    # Tasa de omisión de riesgo
    valores = {name: f"{m.tasa_omision_promedio*100:.1f}%" for name, m in resultados.items()}
    reporte += f"{'Tasa Omisión Riesgo Clínico':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    # Tasa de corrección
    valores = {name: f"{m.tasa_correccion_efectiva_promedio*100:.1f}%" for name, m in resultados.items()}
    reporte += f"{'Tasa Corrección Efectiva':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    # Latencia
    valores = {name: f"{m.latencia_promedio_ms:.0f}ms" for name, m in resultados.items()}
    reporte += f"{'Latencia Promedio':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    # Llamadas LLM
    valores = {name: f"{m.num_llamadas_llm_promedio}" for name, m in resultados.items()}
    reporte += f"{'Llamadas LLM (promedio)':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    # Llamadas API
    valores = {name: f"{m.num_llamadas_api_promedio}" for name, m in resultados.items()}
    reporte += f"{'Llamadas API (promedio)':<40} | "
    reporte += " | ".join([f"{val:<15}" for val in valores.values()]) + "\n"

    reporte += "\n" + "="*80 + "\n"
    reporte += "ANÁLISIS DE TRADE-OFF: PRECISIÓN vs COSTO\n"
    reporte += "="*80 + "\n\n"

    for baseline_name, metricas in resultados.items():
        score_precision = (metricas.exactitud_sellos_promedio + (1 - metricas.tasa_omision_promedio)) / 2
        score_precision = score_precision * 100

        reporte += f"{baseline_name}:\n"
        reporte += f"  Precisión (score):        {score_precision:.1f}%\n"
        reporte += f"  Costo (time + calls):     {metricas.costo_promedio:.3f}\n"
        reporte += f"  Ratio Precisión/Costo:   {score_precision / max(metricas.costo_promedio, 0.01):.2f}\n"
        reporte += f"  Tasa de Errores:          {metricas.tasa_errores*100:.1f}%\n\n"

    reporte += "="*80 + "\n"
    reporte += "CONCLUSIONES\n"
    reporte += "="*80 + "\n\n"

    if resultados:
        mejor_exactitud = max(resultados.items(), key=lambda x: x[1].exactitud_sellos_promedio)
        mejor_latencia = min(resultados.items(), key=lambda x: x[1].latencia_promedio_ms)
        mejor_precision = max(resultados.items(), key=lambda x: (x[1].exactitud_sellos_promedio + (1 - x[1].tasa_omision_promedio)) / 2)

        reporte += f"✓ Mejor Exactitud de Sellos:    {mejor_exactitud[0]} ({mejor_exactitud[1].exactitud_sellos_promedio*100:.1f}%)\n"
        reporte += f"✓ Mejor Latencia:               {mejor_latencia[0]} ({mejor_latencia[1].latencia_promedio_ms:.0f}ms)\n"
        reporte += f"✓ Mejor Precisión General:      {mejor_precision[0]}\n"
    else:
        reporte += "No se ejecutaron baselines.\n"

    reporte += "\n" + "="*80 + "\n"

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(reporte, encoding="utf-8")
        print(f"\n✓ Reporte guardado en: {output_file}\n")

    return reporte


def guardar_resultados_json(
    resultados: Dict[str, MetricasAgregadas],
    output_file: str = "resultados/experimentos.json",
) -> None:
    """Guarda las métricas agregadas en un archivo estructurado JSON."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    datos = {}
    for baseline_name, metricas in resultados.items():
        datos[baseline_name] = {
            "num_casos": metricas.num_casos,
            "exactitud_sellos_promedio": metricas.exactitud_sellos_promedio,
            "exactitud_sellos_std": metricas.exactitud_sellos_std,
            "tasa_omision_promedio": metricas.tasa_omision_promedio,
            "tasa_omision_std": metricas.tasa_omision_std,
            "tasa_correccion_efectiva_promedio": metricas.tasa_correccion_efectiva_promedio,
            "latencia_promedio_ms": metricas.latencia_promedio_ms,
            "latencia_std_ms": metricas.latencia_std_ms,
            "num_llamadas_llm_promedio": metricas.num_llamadas_llm_promedio,
            "num_llamadas_api_promedio": metricas.num_llamadas_api_promedio,
            "costo_promedio": metricas.costo_promedio,
            "tasa_errores": metricas.tasa_errores,
            "errores": metricas.errores_registrados,
        }

    Path(output_file).write_text(json.dumps(datos, indent=2), encoding="utf-8")
    print(f"✓ Resultados JSON guardados en: {output_file}\n")


def main():
    """Función de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Ejecutar experimentos comparativos de baselines NutriCheck",
    )
    parser.add_argument(
        "--num-casos",
        type=int,
        default=None,
        help="Número de casos de prueba a ejecutar (default: todos)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["simple", "react", "nutricheck", "todos"],
        default="todos",
        help="Qué baseline ejecutar",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles de cada caso",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resultados/reporte_comparativo.txt",
        help="Archivo de salida del reporte",
    )

    args = parser.parse_args()

    # Cargar el set completo de 10 casos reales sincronizados
    dataset = obtener_dataset(args.num_casos)
    print(f"\n📋 Dataset cargado con éxito: {len(dataset)} casos de prueba configurados.\n")

    baselines = {
        "simple": BaselineSimpleRAG,
        "react": BaselineReActUnico,
        "nutricheck": BaselineNutriCheckCompleto,
    }

    baselines_a_ejecutar = list(baselines.keys()) if args.baseline == "todos" else [args.baseline]

    resultados = {}
    for baseline_key in baselines_a_ejecutar:
        resultados[baseline_key] = ejecutar_baseline(
            baselines[baseline_key],
            baseline_key.upper(),
            dataset,
            verbose=args.verbose,
        )

    print("\n" + "="*80)
    print("📊 GENERANDO REPORTES COMPARATIVOS...")
    print("="*80 + "\n")

    reporte = generar_reporte_comparativo(resultados, args.output)
    print(reporte)

    guardar_resultados_json(resultados)


if __name__ == "__main__":
    main()