"""Configuración compartida para experimentos."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# APIs de MCP — los experimentos corren en el HOST, así que apuntan a localhost con los
# puertos mapeados por docker-compose (NO a los hostnames internos mcp_server_*).
_JUMBO_PORT = os.getenv("JUMBO_MCP_PORT", "3090")
_SELLOS_PORT = os.getenv("SELLOS_MCP_PORT", "3091")
JUMBO_MCP_URL = os.getenv("JUMBO_MCP_URL_HOST", f"http://localhost:{_JUMBO_PORT}/buscar")
SELLOS_MCP_URL = os.getenv("SELLOS_MCP_URL_HOST", f"http://localhost:{_SELLOS_PORT}/calcular")

# LLM
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Disclaimers y umbrales
DISCLAIMER = """⚠️ *Aviso Legal* — Este análisis es informativo. No reemplaza consulta médica profesional."""
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ExperimentConfig:
    """Configuración de un experimento."""
    baseline_name: str
    num_test_cases: int = 10
    timeout_seconds: int = 60
    store_traces: bool = True
    verbose: bool = False
