FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and local artifacts
COPY serving/ ./serving/
COPY data/processed/dv.pkl ./data/processed/dv.pkl
COPY models/model.pkl ./models/model.pkl

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
