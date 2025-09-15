# Use official Python runtime as a parent image
FROM python:3.13.3-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Cloud Run will provide PORT via environment variable)
EXPOSE 8080

# Use PORT environment variable provided by Cloud Run, defaulting to 8080
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
