# Base image: Python 3.11 slim
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies including TensorFlow
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Copy trained models
COPY Trained_models /app/Trained_models

# Ensure upload directory exists
RUN mkdir -p /app/static/uploads

# Expose Flask port
EXPOSE 8080

# Run Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
