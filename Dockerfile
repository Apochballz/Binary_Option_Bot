# Dockerfile
FROM python:3.11-slim

# ========== STEP 1: INSTALL SYSTEM DEPENDENCIES ==========
# Install TA-Lib C library and build tools
RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ \
    make \
    && wget -O ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/ \
    && rm -rf /var/lib/apt/lists/*

# ========== STEP 2: SET UP WORKING DIRECTORY ==========
WORKDIR /app

# ========== STEP 3: INSTALL PYTHON DEPENDENCIES ==========
# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python packages with compatible numpy version
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ========== STEP 4: COPY APPLICATION CODE ==========
COPY . .

# ========== STEP 5: RUN APPLICATION ==========
# Use $PORT environment variable (Railway provides this)
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "app:app"]
