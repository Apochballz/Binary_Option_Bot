# Dockerfile
FROM python:3.11-slim

# Install TA-Lib system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && rm -rf /var/lib/apt/lists/* ta-lib*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
