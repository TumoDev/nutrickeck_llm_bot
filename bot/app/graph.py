"""Workflow de NutriCheck como grafo LangGraph (patrón de reflexión / self-correction).

NO cambia el workflow actual: cada nodo delega en la lógica ya existente de
NutriCheckRAG (RAG local → Jumbo, function calling de sellos, ciclo clínico↔auditor).
Solo lo reorganiza en un StateGraph con los nodos de producción.

Topología:

  planificador_rag → mcp_jumbo_scraper ─┬─(sin producto)──────────────→ formateador_telegram
                                        └─(con producto)→ parser_de_datos
                                                              ↓
                                                     mcp_calculadora_sellos
                                                              ↓
                                                        juez_clinico ◀────┐
                                                              ↓           │ "corregir"
                                                        auditor_clinico ──┘
                                                              │ "aprobar"
                                                              ▼
                                                     formateador_telegram → END
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import config
from .rag import MAX_CICLOS, NutriCheckRAG
from .trazas import Traza


class NutriCheckProductionState(TypedDict, total=False):
    """Estado que fluye y muta a lo largo del grafo."""
    pregunta: str
    condicion: Optional[str]
    producto_directo: Optional[dict]
    perfil_clinico: dict
    producto_datos: dict
    product_found: bool
    nutricion: dict
    sellos_identificados: list
    reporte: str
    siguiente_accion: str
    feedback: Optional[str]
    iteracion: int
    mensaje_final: str


class NutriCheckGraph:
    """Compila y ejecuta el workflow. Reutiliza una instancia de NutriCheckRAG."""

    def __init__(self, rag: Optional[NutriCheckRAG] = None) -> None:
        self.rag = rag or NutriCheckRAG()
        self._t: Optional[Traza] = None   # Traza por corrida (bot single-thread)
        self.app = self._build()

    # ── Nodos ────────────────────────────────────────────────────────────────────

    def _planificador_rag(self, state: NutriCheckProductionState) -> dict:
        self._t = Traza()
        self._t.inicio(state["pregunta"], state.get("condicion"))
        perfil = self.rag._nodo_analizador_llm(
            state["pregunta"], state.get("condicion"), self._t)
        return {"perfil_clinico": perfil, "iteracion": 0}

    def _mcp_jumbo_scraper(self, state: NutriCheckProductionState) -> dict:
        producto = self.rag._buscar_producto(
            state["pregunta"], state["perfil_clinico"].get("patologia"),
            self._t, state.get("producto_directo"))
        return {"producto_datos": producto, "product_found": not producto.get("error")}

    def _parser_de_datos(self, state: NutriCheckProductionState) -> dict:
        nutri = state["producto_datos"].get("nutricion", {})
        self._t.nodo("🔧 parser_de_datos")
        self._t.muta("nutricion", {k: nutri.get(k)
                     for k in ("calorias_kcal", "azucares_g", "sodio_mg")})
        self._t.fin_paso()
        return {"nutricion": nutri}

    def _mcp_calculadora_sellos(self, state: NutriCheckProductionState) -> dict:
        sellos = self.rag._recolectar_sellos(state["producto_datos"], self._t)
        return {"sellos_identificados": sellos}

    def _juez_clinico(self, state: NutriCheckProductionState) -> dict:
        iteracion = state.get("iteracion", 0) + 1
        reporte = self.rag._agente_clinico(
            state["perfil_clinico"], state["producto_datos"],
            state.get("sellos_identificados", []), state.get("feedback"),
            iteracion, self._t)
        return {"reporte": reporte, "iteracion": iteracion}

    def _auditor_clinico(self, state: NutriCheckProductionState) -> dict:
        accion, motivo = self.rag._agente_validador(
            state["perfil_clinico"], state["reporte"], self._t)
        return {"siguiente_accion": accion, "feedback": motivo}

    def _formateador_telegram(self, state: NutriCheckProductionState) -> dict:
        producto = state.get("producto_datos", {})
        if producto.get("error"):
            mensaje = f"{config.DISCLAIMER}\n\n❌ {producto['error']}"
        else:
            mensaje = state.get("reporte", "")
        if self._t is not None:
            self._t.telegram(mensaje)
        return {"mensaje_final": mensaje}

    # ── Aristas condicionales ─────────────────────────────────────────────────────

    def _ruta_producto(self, state: NutriCheckProductionState) -> str:
        return "parser_de_datos" if state.get("product_found") else "formateador_telegram"

    def _router_reflexion(self, state: NutriCheckProductionState) -> str:
        if state.get("siguiente_accion") == "corregir" and state.get("iteracion", 0) < MAX_CICLOS:
            self._t.ciclo(state.get("feedback", ""))   # [CICLO ACTIVADO]
            return "juez_clinico"
        return "formateador_telegram"

    # ── Construcción ──────────────────────────────────────────────────────────────

    def _build(self):
        g = StateGraph(NutriCheckProductionState)

        g.add_node("planificador_rag", self._planificador_rag)
        g.add_node("mcp_jumbo_scraper", self._mcp_jumbo_scraper)
        g.add_node("parser_de_datos", self._parser_de_datos)
        g.add_node("mcp_calculadora_sellos", self._mcp_calculadora_sellos)
        g.add_node("juez_clinico", self._juez_clinico)
        g.add_node("auditor_clinico", self._auditor_clinico)
        g.add_node("formateador_telegram", self._formateador_telegram)

        g.set_entry_point("planificador_rag")
        g.add_edge("planificador_rag", "mcp_jumbo_scraper")
        g.add_conditional_edges("mcp_jumbo_scraper", self._ruta_producto, {
            "parser_de_datos": "parser_de_datos",
            "formateador_telegram": "formateador_telegram",
        })
        g.add_edge("parser_de_datos", "mcp_calculadora_sellos")
        g.add_edge("mcp_calculadora_sellos", "juez_clinico")
        g.add_edge("juez_clinico", "auditor_clinico")
        g.add_conditional_edges("auditor_clinico", self._router_reflexion, {
            "juez_clinico": "juez_clinico",
            "formateador_telegram": "formateador_telegram",
        })
        g.add_edge("formateador_telegram", END)
        return g.compile()

    # ── API compatible con el bot (drop-in de NutriCheckRAG) ──────────────────────

    def ask(self, pregunta: str, condicion: Optional[str] = None,
            producto_directo: Optional[dict] = None):
        estado: NutriCheckProductionState = {
            "pregunta": pregunta, "condicion": condicion, "producto_directo": producto_directo,
        }
        final = self.app.invoke(estado)
        return final.get("mensaje_final", ""), []

    def extraer_producto_foto(self, imagen_b64: str) -> dict:
        return self.rag.extraer_producto_foto(imagen_b64)
