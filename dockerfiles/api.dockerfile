# Use the official Python image as the base
FROM python:3.11-slim AS base

# Update and install necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy application source files and configurations into the container
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/  
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir --verbose && \
    pip install . --no-deps --no-cache-dir --verbose

# Set the Python path to include the src directory
ENV PYTHONPATH="/app/src"

# Expose the port the app runs on
EXPOSE 8000

# Define the entrypoint to run the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]



