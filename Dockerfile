FROM python:3.11-slim

# EXPOSE helps Railway identify this as a Dockerfile
EXPOSE 8080

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use port 8080 (Railway's Docker default)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
