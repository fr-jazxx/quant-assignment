FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create output directories
RUN mkdir -p outputs reports data_cache

CMD ["python", "scripts/run_backtest.py"]
