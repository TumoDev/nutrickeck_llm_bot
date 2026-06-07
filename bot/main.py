"""Entry point: long polling de Telegram — no requiere URL pública."""

import logging
import time

import requests

from app import NutriCheckRAG, config
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
    params = {"timeout": timeout, "allowed_updates": ["message"]}
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

    logger.info("Inicializando NutriCheckRAG…")
    rag = NutriCheckRAG()
    logger.info("NutriCheckRAG listo. Esperando mensajes de Telegram…")

    offset = None
    while True:
        updates = _get_updates(offset=offset)
        for update in updates:
            handle_update(update, rag)
            offset = update["update_id"] + 1


if __name__ == "__main__":
    main()
