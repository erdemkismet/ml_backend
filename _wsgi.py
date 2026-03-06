"""Flask WSGI app implementing the Label Studio ML backend protocol."""

import logging
import logging.config
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

from branch_catalog import BRANCHES
from model import MODEL_VERSION, load_model, predict_tasks
from ner_model import (
    NER_MODEL_VERSION,
    load_ner_model,
    predict_tasks as ner_predict_tasks,
)


logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "handlers": ["console"],
        },
    }
)

logger = logging.getLogger(__name__)

UI_LABEL_CONFIG = """<View>
  <Labels name="label" toName="text">
    <Label value="TERM" background="#c49b2c"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>"""

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config["JSON_AS_ASCII"] = False
if hasattr(app, "json"):
    app.json.ensure_ascii = False

# Load models once at startup (shared with workers when preloaded by gunicorn)
load_model()
load_ner_model()


def _get_json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP", "model_version": MODEL_VERSION, "ner_model_version": NER_MODEL_VERSION})


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_version=MODEL_VERSION,
        branches=BRANCHES,
        label_config=UI_LABEL_CONFIG,
    )


@app.route("/branches", methods=["GET"])
def branches():
    return jsonify({"branches": BRANCHES})


@app.route("/setup", methods=["POST"])
def setup():
    data = _get_json_body()
    logger.info("Setup called — project=%s", data.get("project"))
    return jsonify({"model_version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    data = _get_json_body()
    tasks = data.get("tasks")
    label_config = data.get("label_config")
    score_threshold = data.get("score_threshold")

    if not isinstance(tasks, list) or not tasks:
        return jsonify({"error": "No tasks provided"}), 400

    predictions = predict_tasks(
        tasks,
        label_config,
        score_threshold=score_threshold,
    )
    return jsonify({"results": predictions, "model_version": MODEL_VERSION})


@app.route("/webhook", methods=["POST"])
def webhook():
    data = _get_json_body()
    event = data.get("action", "unknown")
    logger.info("Webhook event: %s", event)
    return jsonify({"status": "ok"}), 201


@app.route("/versions", methods=["POST"])
def versions():
    return jsonify({"versions": [MODEL_VERSION]})


# ── NER (PER / LOC / ORG) endpoints ──────────────────────────────


@app.route("/ner/setup", methods=["POST"])
def ner_setup():
    data = _get_json_body()
    logger.info("NER setup called — project=%s", data.get("project"))
    return jsonify({"model_version": NER_MODEL_VERSION})


@app.route("/ner/predict", methods=["POST"])
def ner_predict():
    data = _get_json_body()
    tasks = data.get("tasks")
    label_config = data.get("label_config")
    score_threshold = data.get("score_threshold")

    if not isinstance(tasks, list) or not tasks:
        return jsonify({"error": "No tasks provided"}), 400

    predictions = ner_predict_tasks(
        tasks,
        label_config,
        score_threshold=score_threshold,
    )
    return jsonify({"results": predictions, "model_version": NER_MODEL_VERSION})


@app.route("/ner/webhook", methods=["POST"])
def ner_webhook():
    data = _get_json_body()
    event = data.get("action", "unknown")
    logger.info("NER webhook event: %s", event)
    return jsonify({"status": "ok"}), 201


@app.route("/ner/versions", methods=["POST"])
def ner_versions():
    return jsonify({"versions": [NER_MODEL_VERSION]})


@app.before_request
def _log_request():
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s %s", request.method, request.path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 9090)), debug=True)
