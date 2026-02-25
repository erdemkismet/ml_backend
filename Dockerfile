FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    MODEL_DIR=/app/model \
    HF_REPO_ID=ekismet/TerimTespitModeli

# Flask + gunicorn
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# torch (CPU) + transformers + huggingface_hub
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY model.py _wsgi.py branch_catalog.py dual_head_model.py model_loader.py ./
COPY templates/ /app/templates/
COPY static/ /app/static/

EXPOSE 9090

CMD gunicorn --preload --bind :${PORT} --workers ${WORKERS} --threads ${THREADS} --timeout 0 _wsgi:app
