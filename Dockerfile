# Rossmann Sales Forecasting - Production Docker Container
# Machine Learning Zoomcamp 2025 - Midterm Project
# Enhanced with Prophet and ARIMA time series models

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/predict.py
ENV FLASK_ENV=production

# Install system dependencies for time series libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create models directory
RUN mkdir -p src/models

# Expose port
EXPOSE 5000

# Health check (Prophet models may take longer to load)
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "src/predict.py"]