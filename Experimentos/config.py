"""Configuración compartida para experimentos."""

import os
from dataclasses import dataclass
from typing import Optional

# APIs de MCP
JUMBO_MCP_URL = os.getenv("JUMBO_MCP_URL", "http://localhost:8001/search")
SELLOS_MCP_URL = os.getenv("SELLOS_MCP_URL", "http://localhost:8002/calcular")

# LLM
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

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
