"""MCP Server NutriCheck — expone herramienta buscar_producto (Jumbo.cl).

Endpoints:
  GET  /buscar?q=NOMBRE   → REST simple para el bot
  POST /mcp               → MCP Streamable HTTP (para clientes MCP/LLM)
"""

import json
import logging

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from jumbo_scraper import buscar_producto as _scrape
from sellos import calcular_sellos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("jumbo-nutricheck")


@mcp.tool()
async def buscar_producto(nombre: str) -> str:
    """Busca un producto en Jumbo.cl y retorna su información nutricional
    completa junto con los sellos de la Ley 20.606 calculados.

    Args:
        nombre: Nombre o descripción del producto a buscar

    Returns:
        JSON con nombre, marca, categoría, nutrición, sellos e ingredientes
    """
    datos = await _scrape(nombre)
    if datos is None:
        return json.dumps({"error": f"No se encontró '{nombre}' en Jumbo.cl"}, ensure_ascii=False)

    datos["sellos"] = calcular_sellos(datos["nutricion"], datos.get("es_liquido", False))
    return json.dumps(datos, ensure_ascii=False)


# ── REST endpoint para el bot ─────────────────────────────────────────────────

async def rest_buscar(request: Request) -> JSONResponse:
    q = request.query_params.get("q", "").strip()
    if not q:
        return JSONResponse({"error": "Parámetro 'q' requerido"}, status_code=400)

    logger.info("REST /buscar q=%s", q)
    datos = await _scrape(q)

    if datos is None:
        return JSONResponse({"error": f"No se encontró '{q}' en Jumbo.cl"}, status_code=404)

    datos["sellos"] = calcular_sellos(datos["nutricion"], datos.get("es_liquido", False))
    return JSONResponse(datos)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


# ── App compuesta ─────────────────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/buscar", rest_buscar),
        Route("/health", health),
        Mount("/", app=mcp.streamable_http_app()),
    ]
)

if __name__ == "__main__":
    import os
    port = int(os.getenv("JUMBO_MCP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
