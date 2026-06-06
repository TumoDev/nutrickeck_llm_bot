"""Configuración estática y variables de entorno del bot NutriCheck."""

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PORT = int(os.getenv("PORT", 5000))

CSV_PATH = os.getenv("CSV_PATH", "data/productos.csv")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

DISCLAIMER = (
    "⚠️ ADVERTENCIA DE SALUD: NutriCheck es un prototipo informativo basado en IA. "
    "La información puede contener errores. Consulte SIEMPRE a un profesional de la salud."
)

# Mapeo de condición de salud → términos técnicos para expansión de query (MQR).
DICCIONARIO_TECNICO = {
    "hipertensión": ["sodio", "sal", "presión", "cloruro", "grasas trans"],
    "diabetes": ["azúcar", "glucosa", "carbohidratos", "fibra", "maltodextrina"],
    "cáncer": ["nitritos", "procesados", "conservantes", "grasas saturadas"],
    "respiratorias": ["sulfitos", "alérgenos", "inflamatorio"],
    "salud mental": ["magnesio", "omega 3", "triptófano", "proteína", "vitaminas"],
}

CONDICIONES_VALIDAS = set(DICCIONARIO_TECNICO.keys())
