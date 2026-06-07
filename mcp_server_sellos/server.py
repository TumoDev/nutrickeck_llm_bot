"""MCP Server Sellos — calcula sellos Ley 20.606 a partir de valores nutricionales.

Endpoints:
  GET  /calcular?calorias_kcal=&sodio_mg=&azucares_g=&grasas_sat_g=&es_liquido=
  POST /calcular   body JSON con los mismos campos
  GET  /umbrales   devuelve los umbrales vigentes de la ley
"""

import json
import logging
import os

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from sellos import calcular_sellos, _UMBRALES_DOC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("sellos-nutricheck")


@mcp.tool()
def calcular_sellos_producto(
    calorias_kcal: float = 0,
    sodio_mg: float = 0,
    azucares_g: float = 0,
    grasas_sat_g: float = 0,
    es_liquido: bool = False,
) -> str:
    """Calcula los sellos negros Ley 20.606 para un producto chileno.

    Recibe los valores nutricionales por 100g o 100ml y devuelve
    la lista de sellos que corresponden según la ley.

    Args:
        calorias_kcal: Calorías por 100g/ml
        sodio_mg:      Sodio en mg por 100g/ml
        azucares_g:    Azúcares totales en g por 100g/ml
        grasas_sat_g:  Grasas saturadas en g por 100g/ml
        es_liquido:    True si el producto es bebida/líquido

    Returns:
        JSON con lista de sellos y los valores evaluados
    """
    nutricion = {
        "calorias_kcal": calorias_kcal,
        "sodio_mg":      sodio_mg,
        "azucares_g":    azucares_g,
        "grasas_sat_g":  grasas_sat_g,
    }
    sellos = calcular_sellos(nutricion, es_liquido)
    return json.dumps({
        "sellos":    sellos,
        "cantidad":  len(sellos),
        "tipo":      "líquido" if es_liquido else "sólido",
        "evaluado":  nutricion,
    }, ensure_ascii=False)


# ── REST ──────────────────────────────────────────────────────────────────────

def _nutricion_desde_params(params) -> tuple[dict, bool]:
    nutricion = {
        "calorias_kcal": float(params.get("calorias_kcal", 0) or 0),
        "sodio_mg":      float(params.get("sodio_mg",      0) or 0),
        "azucares_g":    float(params.get("azucares_g",    0) or 0),
        "grasas_sat_g":  float(params.get("grasas_sat_g",  0) or 0),
    }
    es_liquido = str(params.get("es_liquido", "false")).lower() in ("true", "1", "yes")
    return nutricion, es_liquido


async def rest_calcular(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        nutricion, es_liquido = _nutricion_desde_params(body)
    except Exception as e:
        return JSONResponse({"error": f"Parámetros inválidos: {e}"}, status_code=400)

    sellos = calcular_sellos(nutricion, es_liquido)
    logger.info("calcular sellos tipo=%s sellos=%s", "líquido" if es_liquido else "sólido", sellos)
    return JSONResponse({
        "sellos":   sellos,
        "cantidad": len(sellos),
        "tipo":     "líquido" if es_liquido else "sólido",
        "evaluado": nutricion,
    })


async def rest_umbrales(_: Request) -> JSONResponse:
    return JSONResponse(_UMBRALES_DOC)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


app = Starlette(
    routes=[
        Route("/calcular", rest_calcular, methods=["POST"]),
        Route("/umbrales", rest_umbrales),
        Route("/health",   health),
        Mount("/", app=mcp.streamable_http_app()),
    ]
)

if __name__ == "__main__":
    port = int(os.getenv("SELLOS_MCP_PORT", 3091))
    uvicorn.run(app, host="0.0.0.0", port=port)
