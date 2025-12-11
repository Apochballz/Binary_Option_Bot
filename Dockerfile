# Dockerfile
FROM python:3.11-slim

# No TA-Lib system dependencies needed!
# Just update apt-get for security
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "app:app"]
