"""Trazabilidad bonita en terminal para el pipeline NutriCheck.

Imprime, paso a paso, qué nodo se ejecutó y qué campo del estado mutó —
incluyendo el ciclo de autocorrección (validador → router → agente_clinico).

Estilo: árbol con emojis y colores ANSI (se desactivan solos si no hay TTY).
"""

from __future__ import annotations

import json
import sys
from typing import Any

# ── Colores ANSI ───────────────────────────────────────────────────────────────
_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAG = "\033[35m"


def _fmt(valor: Any) -> str:
    if isinstance(valor, (dict, list)):
        return json.dumps(valor, ensure_ascii=False)
    return str(valor)


class Traza:
    """Acumula y dibuja la trazabilidad del estado de una consulta."""

    def __init__(self, color: bool | None = None) -> None:
        self.n = 0
        self.ciclos = 0
        self.color = sys.stdout.isatty() if color is None else color

    def _c(self, code: str, texto: str) -> str:
        return f"{code}{texto}{_RST}" if self.color else texto

    # ── Secciones ──────────────────────────────────────────────────────────────

    def inicio(self, pregunta: str, condicion: str | None) -> None:
        linea = "═" * 72
        print("\n" + self._c(_BOLD, linea))
        print(self._c(_BOLD, "  TRAZABILIDAD DEL ESTADO — NutriCheck  (pipeline real)"))
        print(self._c(_BOLD, linea))
        print(f"  ❓ Pregunta : {pregunta}")
        print(f"  🩺 Condición: {condicion or 'salud general'}\n")

    def nodo(self, etiqueta: str, iteracion: int | None = None) -> None:
        self.n += 1
        it = f"  {self._c(_DIM, f'(Iteración {iteracion})')}" if iteracion else ""
        print(f"{self._c(_BOLD, f'[{self.n}]')} {etiqueta}{it}")

    def muta(self, campo: str, valor: Any) -> None:
        print(f"      └─ muta {self._c(_CYAN, campo)} → {_fmt(valor)}")

    def function_call(self, nombre: str, args: Any) -> None:
        """Una llamada a herramienta DECIDIDA por el modelo (function calling)."""
        self.n += 1
        etiqueta = self._c(_GREEN, f"🔧 function_call → {nombre}()")
        print(f"{self._c(_BOLD, f'[{self.n}]')} {etiqueta}  {self._c(_DIM, '(el modelo lo decidió)')}")
        print(f"      └─ args {self._c(_CYAN, _fmt(args))}")

    def info(self, texto: str) -> None:
        print(f"      {self._c(_DIM, '· ' + texto)}")

    def router(self, accion: str, destino: str) -> None:
        col = _YELLOW if accion == "corregir" else _GREEN
        flecha = "↩️" if accion == "corregir" else "✅"
        print(f"      ⮑  router: siguiente_accion="
              f"{self._c(col, repr(accion))}  ⇒  {flecha} {destino}")
        print()

    def ciclo(self, motivo: str) -> None:
        self.ciclos += 1
        print(self._c(_MAG, "  ┄┄┄ ♻️  [CICLO ACTIVADO] router redirige en caliente a "
                            "agente_clinico ┄┄┄"))
        print(self._c(_DIM, f"        motivo: {motivo}\n"))

    def fin_paso(self) -> None:
        print()

    def telegram(self, reporte: str) -> None:
        print(self._c(_BOLD, "📨 MARKDOWN FINAL ENVIADO A TELEGRAM"))
        print("┌" + "─" * 70)
        for linea in reporte.splitlines():
            print("│ " + linea)
        print("└" + "─" * 70)
        print(self._c(_DIM, f"  Resumen: {self.n} mutaciones de estado · "
                            f"{self.ciclos} ciclo(s) de autocorrección.\n"))
