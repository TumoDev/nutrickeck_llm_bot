"""Configuración del bot NutriCheck."""

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY    = os.getenv("MISTRAL_API_KEY")
JUMBO_MCP_URL      = os.getenv("JUMBO_MCP_URL",   "http://mcp_server_jumbo:3090/buscar")
SELLOS_MCP_URL     = os.getenv("SELLOS_MCP_URL",  "http://mcp_server_sellos:3091/calcular")

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

DISCLAIMER = (
    "⚠️ ADVERTENCIA DE SALUD: NutriCheck es un prototipo informativo basado en IA. "
    "La información puede contener errores. Consulte SIEMPRE a un profesional de la salud."
)

DICCIONARIO_TECNICO = {
    "hipertensión": ["sodio", "sal", "presión", "cloruro", "grasas trans"],
    "diabetes":     ["azúcar", "glucosa", "carbohidratos", "fibra", "maltodextrina"],
    "cáncer":       ["nitritos", "procesados", "conservantes", "grasas saturadas"],
    "respiratorias":["sulfitos", "alérgenos", "inflamatorio"],
    "depresión":    ["magnesio", "omega 3", "triptófano", "proteína", "vitaminas"],
}

CONDICIONES_VALIDAS = set(DICCIONARIO_TECNICO.keys())
