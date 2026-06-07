"""Calcula sellos nutricionales Ley 20.606 (Chile) a partir de valores por 100g/100ml."""

_SOLIDOS = {
    "calorias_kcal": (350.0, "Alto en Calorías"),
    "azucares_g":    (22.5,  "Alto en Azúcares"),
    "grasas_sat_g":  (6.0,   "Alto en Grasas Saturadas"),
    "sodio_mg":      (800.0, "Alto en Sodio"),
}

_LIQUIDOS = {
    "calorias_kcal": (70.0, "Alto en Calorías"),
    "azucares_g":    (6.0,  "Alto en Azúcares"),
    "grasas_sat_g":  (3.0,  "Alto en Grasas Saturadas"),
    "sodio_mg":      (100.0,"Alto en Sodio"),
}


def calcular_sellos(nutricion: dict, es_liquido: bool = False) -> list[str]:
    tabla = _LIQUIDOS if es_liquido else _SOLIDOS
    return [
        etiqueta
        for campo, (umbral, etiqueta) in tabla.items()
        if nutricion.get(campo) is not None and float(nutricion[campo]) >= umbral
    ]
