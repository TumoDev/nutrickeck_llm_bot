"""Dataset de casos de prueba para experimentos de NutriCheck.

Estructurado con los nombres de productos exactos extraídos del dataset de Jumbo
para evaluar Modo General y Modo Específico con las 5 patologías objetivo.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CasoPrueba:
    """Un caso de prueba para los baselines."""
    id: str
    pregunta: str
    patologia: str  # GENERAL o una de las 5 específicas
    producto_esperado: str
    sellos_esperados: List[str]  # Sello Ley 20.606
    riesgos_clinicos_esperados: List[str]
    veredicto_esperado: str  # "APTO", "NO APTO", "MODERADO"
    notas: Optional[str] = None


# Dataset oficial emparejado con las filas exactas del catálogo de Jumbo
DATASET_EXPERIMENTOS = [
    # --- 1. DIABETES ---
    CasoPrueba(
        id="caso_001",
        pregunta="Tengo diabetes tipo 2, ¿me recomendas el producto Leche Condensada 397 g?",
        patologia="DIABETES",
        producto_esperado="Leche Condensada 397 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN AZÚCARES"],
        riesgos_clinicos_esperados=[
            "aumento de glucemia",
            "riesgo de descompensación aguda",
            "impacto negativo en el control glucémico a largo plazo"
        ],
        veredicto_esperado="NO APTO",
        notas="Modo Específico: Alto en azúcares directo por ingredientes (Azúcar, Leche)"
    ),
    CasoPrueba(
        id="caso_002",
        pregunta="Tengo resistencia a la insulina y diabetes. ¿Qué tal la Leche Colun Descremada 1 L?",
        patologia="DIABETES",
        producto_esperado="Leche Colun Descremada 1 L",
        sellos_esperados=[],
        riesgos_clinicos_esperados=[],
        veredicto_esperado="APTO",
        notas="Modo Específico: Producto libre de sellos, seguro para diabéticos"
    ),

    # --- 2. HIPERTENSIÓN ---
    CasoPrueba(
        id="caso_003",
        pregunta="Sufro de hipertensión arterial. ¿Puedo comer el producto Cheezels Sabor Queso 100 g?",
        patologia="HIPERTENSION",
        producto_esperado="Cheezels Sabor Queso 100 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN SODIO"],
        riesgos_clinicos_esperados=[
            "aumento de presión arterial",
            "retención de líquidos",
            "sobrecarga del sistema cardiovascular"
        ],
        veredicto_esperado="NO APTO",
        notas="Modo Específico: Contiene 706 mg de sodio por cada 100g"
    ),

    # --- 3. CÁNCER ---
    CasoPrueba(
        id="caso_004",
        pregunta="Estoy en tratamiento contra el cáncer. ¿Es seguro comer Hamburguesa de Vacuno Montina 150 g?",
        patologia="CANCER",
        producto_esperado="Hamburguesa de Vacuno Montina 150 g",
        sellos_esperados=["ALTO EN GRASAS SATURADAS", "ALTO EN SODIO"],
        riesgos_clinicos_esperados=[
            "consumo de compuestos proinflamatorios",
            "empeoramiento del estado inflamatorio sistémico",
            "ingesta de aditivos cárnicos no recomendados"
        ],
        veredicto_esperado="NO APTO",
        notas="Modo Específico: Ultraprocesado con grasas saturadas, sodio y aditivos inflamatorios"
    ),

    # --- 4. ENFERMEDADES RESPIRATORIAS ---
    CasoPrueba(
        id="caso_005",
        pregunta="Tengo asma crónica y problemas respiratorios. ¿Me afecta la Crema Maggi Mariscos 45 g?",
        patologia="ENFERMEDADES_RESPIRATORIAS",
        producto_esperado="Crema Maggi Mariscos 45 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN AZÚCARES", "ALTO EN SODIO"],
        riesgos_clinicos_esperados=[
            "potencial broncoespasmo inducido por aditivos/sulfitos",
            "inflamación sistémica por alto contenido de sodio"
        ],
        veredicto_esperado="NO APTO",
        notas="Modo Específico: Alérgeno crítico detectado en ingredientes: contiene Sulfitos"
    ),

    # --- 5. DEPRESIÓN ---
    CasoPrueba(
        id="caso_006",
        pregunta="Me diagnosticaron depresión severa. ¿Influye si como Galletas Gretel Yogurt Frutilla 85 g?",
        patologia="DEPRESION",
        producto_esperado="Galletas Gretel Yogurt Frutilla 85 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN AZÚCARES", "ALTO EN GRASAS SATURADAS"],
        riesgos_clinicos_esperados=[
            "alteración del eje intestino-cerebro por dieta proinflamatoria",
            "picos y caídas drásticas de energía que alteran el estado de ánimo",
            "empeoramiento de marcadores inflamatorios asociados a la depresión"
        ],
        veredicto_esperado="MODERADO",
        notas="Modo Específico: Ultraprocesado con 3 sellos negros (alto azúcar/grasas sat) perjudicial"
    ),

    # --- 6. CASOS DE COMORBILIDADES ---
    CasoPrueba(
        id="caso_007",
        pregunta="Tengo hipertensión y también diabetes. ¿El producto Paté de Cerdo Llanquihue 125 g me sirve?",
        patologia="HIPERTENSION_DIABETES",
        producto_esperado="Paté de Cerdo Llanquihue 125 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN GRASAS SATURADAS", "ALTO EN SODIO"],
        riesgos_clinicos_esperados=[
            "elevación simultánea de la presión arterial y glucemia",
            "daño endotelial acelerado por comorbilidad",
            "alto riesgo metabólico combinado"
        ],
        veredicto_esperado="NO APTO",
        notas="Modo Combinado: Alto en Sodio (962 mg/100g) y Grasas Saturadas perjudicial para ambas condiciones"
    ),

    # --- 7. MODO GENERAL ---
    CasoPrueba(
        id="caso_008",
        pregunta="No tengo ninguna enfermedad, ¿es saludable consumir Chocolate de Leche Trencito 80 g?",
        patologia="GENERAL",
        producto_esperado="Chocolate de Leche Trencito 80 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN AZÚCARES", "ALTO EN GRASAS SATURADAS"],
        riesgos_clinicos_esperados=[
            "ingesta excesiva de azúcares libres",
            "alto aporte de calorías densas y grasas saturadas"
        ],
        veredicto_esperado="MODERADO",
        notas="Modo General: Alimento con 3 sellos, se advierte consumo regulado y controlado"
    ),
    CasoPrueba(
        id="caso_009",
        pregunta="Busco comer sano para prevención general. ¿Qué tal el Yogurt Loncoleche Proteína Natural 140 g?",
        patologia="GENERAL",
        producto_esperado="Yogurt Loncoleche Proteína Natural 140 g",
        sellos_esperados=[],
        riesgos_clinicos_esperados=[],
        veredicto_esperado="APTO",
        notas="Modo General: Opción limpia, alta en proteínas y sin sellos de advertencia"
    ),
    CasoPrueba(
        id="caso_010",
        pregunta="Para cuidar mi salud en general, ¿puedo comer el producto Mantequilla Loncoleche 200 g?",
        patologia="GENERAL",
        producto_esperado="Mantequilla Loncoleche 200 g",
        sellos_esperados=["ALTO EN CALORÍAS", "ALTO EN GRASAS SATURADAS", "ALTO EN SODIO"],
        riesgos_clinicos_esperados=[
            "aporte excesivo de calorías densas",
            "alto consumo de grasas saturadas y sodio"
        ],
        veredicto_esperado="MODERADO",
        notas="Modo General: Alimento graso con sellos que requiere porcionamiento estricto"
    ),
]


def obtener_dataset(num_casos: Optional[int] = None) -> List[CasoPrueba]:
    """Retorna el dataset de test cases ajustado para NutriCheck.

    Args:
        num_casos: Si se especifica, retorna solo los primeros N casos.

    Returns:
        Lista de casos de prueba.
    """
    if num_casos is None:
        return DATASET_EXPERIMENTOS
    return DATASET_EXPERIMENTOS[:num_casos]


def obtener_caso(caso_id: str) -> Optional[CasoPrueba]:
    """Busca un caso por ID dentro del dataset."""
    for caso in DATASET_EXPERIMENTOS:
        if caso.id == caso_id:
            return caso
    return None