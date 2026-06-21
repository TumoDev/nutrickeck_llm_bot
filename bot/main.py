"""Entry point: long polling de Telegram — no requiere URL pública."""

import json
import logging
import time

import requests

from app import NutriCheckRAG, config
from app.graph import NutriCheckGraph
from app.telegram_bot import TELEGRAM_API, handle_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _delete_webhook() -> None:
    try:
        r = requests.post(
            f"{TELEGRAM_API}/deleteWebhook",
            json={"drop_pending_updates": True},
            timeout=10,
        )
        logger.info("Webhook eliminado: %s", r.json())
    except Exception as e:
        logger.warning("No se pudo eliminar el webhook: %s", e)


def _get_updates(offset=None, timeout=30) -> list:
    # allowed_updates DEBE ir como JSON string; si se pasa como lista, requests la
    # serializa con claves repetidas y Telegram NO entrega los callback_query (botones).
    params = {
        "timeout": timeout,
        "allowed_updates": json.dumps(["message", "callback_query"]),
    }
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params=params,
            timeout=timeout + 5,
        )
        return r.json().get("result", [])
    except Exception as e:
        logger.warning("Error obteniendo updates: %s", e)
        time.sleep(3)
        return []


def main() -> None:
    _delete_webhook()

    logger.info("Inicializando NutriCheck (grafo LangGraph)…")
    agente = NutriCheckGraph(NutriCheckRAG())
    logger.info("NutriCheck listo. Esperando mensajes de Telegram…")

    offset = None
    while True:
        updates = _get_updates(offset=offset)
        for update in updates:
            handle_update(update, agente)
            offset = update["update_id"] + 1


if __name__ == "__main__":
    main()
