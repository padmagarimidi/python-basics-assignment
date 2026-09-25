FROM python:3.11-slim

WORKDIR /app

# System deps for building some ML wheels (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Graded baseline runs fully offline: MOCK_LLM defaults to "1" if not set.
ENV MOCK_LLM=1
ENV PORT=7860

EXPOSE 7860

# Pre-ingest the corpus at build time so the container starts up ready to
# serve retrieval requests without a first-request delay.
RUN python ingest.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
