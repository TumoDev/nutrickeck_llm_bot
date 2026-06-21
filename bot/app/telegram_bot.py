"""Capa de Telegram: menús con botones, estado por chat, fotos y respuestas."""

import base64
import logging
import time
from typing import Dict, Optional

import requests

from . import config
from .rag import NutriCheckRAG

logger = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
MAX_MESSAGE_LEN = 4000  # Telegram permite 4096; dejamos margen de seguridad.
IDLE_SEGUNDOS = 60      # tras este tiempo sin interactuar, se reenvía la bienvenida.

WELCOME = (
    "👋 ¡Hola! Soy *NutriCheck*.\n\n"
    "Analizo productos del mercado chileno según la *Ley 20.606* y te digo si son "
    "aptos para ti.\n\n"
    "Elige una modalidad para empezar 👇"
)

HELP = (
    "*Cómo usarme:*\n\n"
    "• /menu — abre el menú de modalidades.\n"
    "• *Modalidad General* → consulta por *Texto* o por *Foto* (análisis general).\n"
    "• *Modalidad Especial* → elige tu condición de salud y luego pregunta.\n"
    "• /clear — borra tu condición y vuelve a análisis general.\n\n"
    "También puedes escribir directamente una pregunta como:\n"
    "_¿La Mantequilla Calo con Sal es alta en sodio?_"
)

# Etiqueta visible → valor interno de la condición.
CONDICIONES_MENU = [
    ("🫀 Hipertensión",          "hipertensión"),
    ("🍬 Diabetes",              "diabetes"),
    ("🎗️ Cáncer",                "cáncer"),
    ("🫁 Enf. respiratorias",    "respiratorias"),
    ("🧠 Depresión",             "depresión"),
]

# Estado por chat: chat_id -> condición activa. En memoria a propósito (prototipo).
_condiciones: Dict[int, str] = {}
# chat_id -> timestamp de la última interacción (para la bienvenida por inactividad).
_ultima_interaccion: Dict[int, float] = {}
# chat_id -> True cuando el usuario YA eligió una opción del menú. Sin esto NO se analiza.
_modo_activo: Dict[int, bool] = {}
# chat_id -> producto identificado en una foto, a la espera de la pregunta del usuario.
_producto_foto: Dict[int, str] = {}


def _reset_chat(chat_id: int) -> None:
    """Vuelve al estado de menú: sin modo activo ni producto-foto pendiente."""
    _modo_activo.pop(chat_id, None)
    _producto_foto.pop(chat_id, None)


def _es_inactivo(chat_id: int) -> bool:
    anterior = _ultima_interaccion.get(chat_id)
    return anterior is None or (time.time() - anterior) > IDLE_SEGUNDOS


# Saludos / chitchat que deben mostrar el menú en vez de analizarse como producto.
_SALUDOS = {
    "hola", "holaa", "holi", "ola", "alo", "aló", "buenas", "buenos dias",
    "buenos días", "buenas tardes", "buenas noches", "hey", "hi", "hello",
    "menu", "menú", "inicio", "empezar", "que tal", "qué tal", "gracias", "ok",
}
_SALUDO_INICIAL = {"hola", "holaa", "ola", "buenas", "buenos", "hey", "hi", "hello", "menu", "menú"}


def _es_saludo(text: str) -> bool:
    t = text.strip().lower().strip("¡!¿?.,")
    if not t:
        return False
    return t in _SALUDOS or t.split()[0] in _SALUDO_INICIAL


# ── Teclados (inline keyboards) ─────────────────────────────────────────────────

def _kb_principal() -> dict:
    return {"inline_keyboard": [
        [{"text": "🥗 Modalidad General",  "callback_data": "menu:general"}],
        [{"text": "🩺 Modalidad Especial", "callback_data": "menu:especial"}],
    ]}


def _kb_general() -> dict:
    return {"inline_keyboard": [
        [{"text": "📷 Foto", "callback_data": "gen:foto"},
         {"text": "✍️ Texto", "callback_data": "gen:texto"}],
        [{"text": "⬅️ Volver", "callback_data": "menu:main"}],
    ]}


def _kb_especial() -> dict:
    filas = [[{"text": lbl, "callback_data": f"cond:{val}"}] for lbl, val in CONDICIONES_MENU]
    filas.append([{"text": "⬅️ Volver", "callback_data": "menu:main"}])
    return {"inline_keyboard": filas}


# ── Envío a Telegram ────────────────────────────────────────────────────────────

def _send_message(chat_id: int, text: str, keyboard: Optional[dict] = None) -> None:
    """Envía un mensaje, fragmentándolo si excede el límite. El teclado va en el último trozo."""
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks):
        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}
        if keyboard is not None and i == len(chunks) - 1:
            payload["reply_markup"] = keyboard
        try:
            r = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
            if not r.ok:
                # Casi siempre es un error de parseo de Markdown → reintenta en texto plano.
                logger.warning("Telegram %s al enviar (%s); reintento sin Markdown.",
                               r.status_code, r.text[:150])
                payload.pop("parse_mode", None)
                requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
        except requests.RequestException as e:
            logger.warning("Fallo enviando mensaje a %s: %s", chat_id, e)


def _edit_message(chat_id: int, message_id: int, text: str, keyboard: Optional[dict]) -> None:
    """Edita el mensaje del menú en sitio (para navegar entre submenús)."""
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text,
               "parse_mode": "Markdown"}
    if keyboard is not None:
        payload["reply_markup"] = keyboard
    try:
        requests.post(f"{TELEGRAM_API}/editMessageText", json=payload, timeout=10)
    except requests.RequestException as e:
        logger.warning("Fallo editando mensaje en %s: %s", chat_id, e)


def _answer_callback(callback_id: str) -> None:
    """Quita el spinner de carga del botón presionado."""
    try:
        requests.post(f"{TELEGRAM_API}/answerCallbackQuery",
                      json={"callback_query_id": callback_id}, timeout=10)
    except requests.RequestException:
        pass


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


# ── Comandos de texto ───────────────────────────────────────────────────────────

def _handle_command(chat_id: int, text: str) -> bool:
    """Maneja comandos. Devuelve True si era un comando (y ya respondió)."""
    command = text.strip().split(maxsplit=1)[0].lower().split("@")[0]

    if command in ("/start", "/menu"):
        _reset_chat(chat_id)  # vuelve al menú: exige re-elegir
        _send_message(chat_id, WELCOME, _kb_principal())
        return True
    if command == "/help":
        _send_message(chat_id, HELP)
        return True
    if command == "/clear":
        _condiciones.pop(chat_id, None)
        _send_message(chat_id, "🧹 Condición borrada. Volvemos a análisis general.")
        return True
    return False


# ── Callbacks de los botones ────────────────────────────────────────────────────

def _handle_callback(callback: dict) -> None:
    data = callback.get("data", "")
    msg = callback.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")
    _answer_callback(callback.get("id"))
    if chat_id is None:
        return

    # Navegar entre menús NO activa el análisis.
    if data == "menu:main":
        _reset_chat(chat_id)
        _edit_message(chat_id, message_id, WELCOME, _kb_principal())
    elif data == "menu:general":
        _reset_chat(chat_id)
        _edit_message(chat_id, message_id,
                      "🥗 *Modalidad General*\nElige cómo quieres consultar:", _kb_general())
    elif data == "menu:especial":
        _reset_chat(chat_id)
        _edit_message(chat_id, message_id,
                      "🩺 *Modalidad Especial*\nElige tu condición de salud:", _kb_especial())
    # Elegir Texto / Foto / condición SÍ activa el análisis.
    elif data == "gen:texto":
        _condiciones.pop(chat_id, None)
        _producto_foto.pop(chat_id, None)
        _modo_activo[chat_id] = True
        _send_message(chat_id,
                      "✍️ *Modo texto* (análisis general).\n"
                      "Escríbeme tu pregunta. Ej: _¿La Coca-Cola es alta en azúcar?_")
    elif data == "gen:foto":
        _condiciones.pop(chat_id, None)
        _producto_foto.pop(chat_id, None)
        _modo_activo[chat_id] = True
        _send_message(chat_id,
                      "📷 *Modo foto*.\n"
                      "Envíame una *foto del producto*. La identificaré y luego podrás "
                      "preguntarme lo que quieras sobre ella.")
    elif data.startswith("cond:"):
        cond = data.split(":", 1)[1]
        if cond in config.CONDICIONES_VALIDAS:
            _condiciones[chat_id] = cond
            _producto_foto.pop(chat_id, None)
            _modo_activo[chat_id] = True
            _send_message(chat_id,
                          f"✅ Condición fijada: *{cond}*.\n"
                          "Ahora escríbeme tu pregunta sobre un producto, o envíame una "
                          "*foto* con tu pregunta en el pie de foto.")


# ── Fotos ───────────────────────────────────────────────────────────────────────

def _descargar_foto_b64(file_id: str) -> str:
    r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=10)
    r.raise_for_status()
    file_path = r.json()["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{config.TELEGRAM_BOT_TOKEN}/{file_path}"
    img = requests.get(url, timeout=30)
    img.raise_for_status()
    return base64.b64encode(img.content).decode()


def _handle_photo(chat_id: int, message: dict, rag: NutriCheckRAG) -> None:
    """Identifica el producto de la foto y lo deja en espera de la pregunta del usuario.
    NO analiza todavía: solo describe lo que ve y pregunta si quiere consultar."""
    file_id = message["photo"][-1]["file_id"]  # la última es la de mayor resolución

    _send_message(chat_id, "🔎 Mirando la foto…")
    try:
        imagen_b64 = _descargar_foto_b64(file_id)
    except Exception as e:
        logger.exception("Error descargando foto de %s", chat_id)
        _send_message(chat_id, f"❌ No pude descargar la foto: {e}")
        return

    descripcion = rag.describir_producto_foto(imagen_b64)
    if not descripcion:
        _send_message(chat_id,
                      "🤔 No pude identificar el producto en la foto. "
                      "Prueba con otra imagen más clara, o usa el modo *Texto*.")
        return

    _producto_foto[chat_id] = descripcion  # queda a la espera de la pregunta
    _send_message(chat_id,
                  f"📷 En la foto identifiqué:\n\n*{descripcion}*\n\n"
                  "¿Qué quieres saber de este producto? Escríbeme tu pregunta "
                  "(ej: _¿es alto en azúcar?_).")


# ── Punto de entrada ────────────────────────────────────────────────────────────

def handle_update(update: dict, rag: NutriCheckRAG) -> None:
    """Punto de entrada para cada update entrante de Telegram."""
    if "callback_query" in update:
        cb = update["callback_query"]
        cid = ((cb.get("message") or {}).get("chat") or {}).get("id")
        if cid is not None:
            _ultima_interaccion[cid] = time.time()  # pulsar un botón cuenta como interacción
        _handle_callback(cb)
        return

    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    inactivo = _es_inactivo(chat_id)
    _ultima_interaccion[chat_id] = time.time()
    text = message.get("text") or ""

    # Comandos primero (/start, /menu, /help, /clear).
    if text.startswith("/") and _handle_command(chat_id, text):
        return

    # Tras inactividad (≥1 min) o ante un saludo → de vuelta al menú, estado reseteado.
    if inactivo or (text and _es_saludo(text)):
        _reset_chat(chat_id)
        _send_message(chat_id, WELCOME, _kb_principal())
        return

    # NO se analiza NADA hasta que el usuario elija una opción del menú.
    if not _modo_activo.get(chat_id):
        _send_message(chat_id, WELCOME, _kb_principal())
        return

    # Ya hay un modo activo (eligió Texto / Foto / condición).
    if message.get("photo"):
        _handle_photo(chat_id, message, rag)
        return

    if not text:
        return

    # Si hay un producto identificado por foto, la pregunta se refiere a ÉL.
    producto_foto = _producto_foto.get(chat_id)
    condicion = _condiciones.get(chat_id)
    _send_message(chat_id, "🔎 Analizando producto…")
    try:
        respuesta, _ = rag.ask(text, condicion=condicion, producto_foto=producto_foto)
    except Exception as e:
        logger.exception("Error en RAG para chat %s", chat_id)
        _send_message(chat_id, f"❌ Error al procesar tu consulta: {e}")
        return
    _send_message(chat_id, respuesta)
