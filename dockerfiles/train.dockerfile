# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Install Python dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install -e .

# Copy project files
COPY configs/ configs/
COPY src/ src/

# Create necessary directories
RUN mkdir -p models/checkpoints data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TORCH_HOME=/app/models

# Set the entrypoint
ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]