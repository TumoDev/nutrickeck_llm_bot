FROM python:3.12-slim

# Sin esto, los print() de la traza quedan bufferizados y no aparecen en los logs de Docker.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descarga el modelo de embeddings para que la 1ª consulta no lo baje en runtime.
ENV NUTRICHECK_EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

CMD ["python", "main.py"]
