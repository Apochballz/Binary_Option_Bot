# Dockerfile
FROM python:3.11-slim

# ========== STEP 1: INSTALL SYSTEM DEPENDENCIES ==========
# Install TA-Lib C library version 0.4.0 (matches Python wrapper)
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
# Copy requirements first
COPY requirements.txt .

# Install from TA-Lib's official PyPI mirror FIRST
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url=https://pypi.ta-lib.org/simple \
        TA-Lib==0.4.28 \
    && pip install --no-cache-dir -r requirements.txt

# ========== STEP 4: COPY APPLICATION CODE ==========
COPY . .

# ========== STEP 5: RUN APPLICATION ==========
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "app:app"]
