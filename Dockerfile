FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    MODEL_DIR=/app/model

# Flask + gunicorn (no git needed — SDK removed)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# torch (CPU) + transformers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model weights (~420 MB) — cached Docker layer
COPY model/ /app/model/

# App code
COPY model.py _wsgi.py branch_catalog.py dual_head_model.py ./
COPY templates/ /app/templates/
COPY static/ /app/static/

EXPOSE 9090

CMD gunicorn --preload --bind :${PORT} --workers ${WORKERS} --threads ${THREADS} --timeout 0 _wsgi:app
