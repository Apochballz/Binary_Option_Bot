# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use shell form to expand $PORT environment variable
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 app:app
