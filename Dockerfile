FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch + torchvision FIRST (from PyTorch CPU index)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

# Then install your runtime deps
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY app ./app
COPY scripts ./scripts
COPY docs ./docs
COPY gold ./gold

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
