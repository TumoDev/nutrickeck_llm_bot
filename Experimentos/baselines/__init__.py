"""Baselines para comparación de sistemas."""

from .baseline_1_simple_rag import BaselineSimpleRAG
from .baseline_2_react import BaselineReActUnico
from .baseline_3_nutricheck import BaselineNutriCheckCompleto

__all__ = [
    "BaselineSimpleRAG",
    "BaselineReActUnico",
    "BaselineNutriCheckCompleto",
]
