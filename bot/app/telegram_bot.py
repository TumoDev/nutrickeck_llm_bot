"""Capa de Telegram: parsea updates, mantiene el estado por chat y envía respuestas."""

import logging
from typing import Dict, Optional

import requests

from . import config
from .rag import NutriCheckRAG

logger = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
MAX_MESSAGE_LEN = 4000  # Telegram permite 4096; dejamos margen de seguridad.

WELCOME = (
    "👋 Hola, soy *NutriCheck*.\n\n"
    "Analizo productos del mercado chileno según la Ley 20.606.\n\n"
    "Escríbeme una pregunta como:\n"
    "_¿La Mantequilla Calo con Sal es alta en azúcar?_\n\n"
    "Comandos disponibles: /help"
)

HELP = (
    "*Cómo usarme:*\n\n"
    "• Envíame una pregunta sobre cualquier producto.\n"
    "• `/condicion <nombre>` fija tu condición de salud para las próximas preguntas.\n"
    "• `/clear` borra la condición.\n"
    "• `/start` o `/help` muestran este mensaje.\n\n"
    "*Condiciones soportadas:*\n"
    + "\n".join(f"• `{c}`" for c in sorted(config.CONDICIONES_VALIDAS))
)


# Estado por chat: chat_id -> condición activa. En memoria a propósito (prototipo).
_condiciones: Dict[int, str] = {}


def _send_message(chat_id: int, text: str) -> None:
    """Envía un mensaje a Telegram, fragmentándolo si excede el límite."""
    for chunk in _split_message(text):
        try:
            requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except requests.RequestException as e:
            logger.warning("Fallo enviando mensaje a %s: %s", chat_id, e)


def _split_message(text: str) -> list:
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > MAX_MESSAGE_LEN:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def _handle_command(chat_id: int, text: str) -> Optional[str]:
    """Devuelve la respuesta para un comando, o None si no es un comando conocido."""
    parts = text.strip().split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # quita @BotName en grupos
    args = parts[1] if len(parts) > 1 else ""

    if command in ("/start", "/help"):
        return WELCOME if command == "/start" else HELP

    if command == "/condicion":
        cond = args.strip().lower()
        if not cond:
            return (
                "Uso: `/condicion <nombre>`\n"
                "Condiciones válidas: "
                + ", ".join(sorted(config.CONDICIONES_VALIDAS))
            )
        if cond not in config.CONDICIONES_VALIDAS:
            return (
                f"❌ Condición no reconocida: `{cond}`\n"
                "Válidas: " + ", ".join(sorted(config.CONDICIONES_VALIDAS))
            )
        _condiciones[chat_id] = cond
        return f"✅ Condición fijada: *{cond}*. Tus próximas preguntas la usarán."

    if command == "/clear":
        _condiciones.pop(chat_id, None)
        return "🧹 Condición borrada. Volvemos a análisis general."

    return None


def handle_update(update: dict, rag: NutriCheckRAG) -> None:
    """Punto de entrada para cada update entrante de Telegram."""
    message = update.get("message") or {}
    text = message.get("text")
    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not text or chat_id is None:
        return

    if text.startswith("/"):
        respuesta = _handle_command(chat_id, text)
        if respuesta is not None:
            _send_message(chat_id, respuesta)
            return
        # comando desconocido → cae al flujo normal

    condicion = _condiciones.get(chat_id)
    _send_message(chat_id, "🔎 Analizando producto…")

    try:
        respuesta, _ = rag.ask(text, condicion=condicion)
    except Exception as e:
        logger.exception("Error en RAG para chat %s", chat_id)
        _send_message(chat_id, f"❌ Error al procesar tu consulta: {e}")
        return

    _send_message(chat_id, respuesta)
