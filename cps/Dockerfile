# Образ для Hugging Face Space (sdk: docker). Локально не нужен — там start.command.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cloud.txt
COPY --chown=user . .
RUN mkdir -p scans data && chown -R user /app
USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
