import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", 5000))
TELEGRAM_API = f"https://api.telegram.org/bot{TOKEN}"


def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={
        "chat_id": chat_id,
        "text": text,
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"]["text"]
        send_message(chat_id, text)
    return jsonify({"ok": True})


@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Pasa ?url=TU_URL_PUBLICA"}), 400
    resp = requests.post(f"{TELEGRAM_API}/setWebhook", json={"url": f"{url}/webhook"})
    return jsonify(resp.json())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
