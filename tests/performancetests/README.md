# Application Load Testing

This describes how to perform load testing on our FastAPI-based image classification service using Locust.

## Prerequisites

- Python 3.8+
- FastAPI application running (either PyTorch or ONNX version)
- Install locust package:
  ```bash
  pip install locust
  ```

## Setup

1. Start the FastAPI Application
   ```bash
   # From project root
   uvicorn src.mlops_project.api:app --host 0.0.0.0 --port 8000
   ```
   Or for the ONNX version:
   ```bash
   uvicorn src.mlops_project.api_onnx:app --host 0.0.0.0 --port 8000
   ```

2. Start Locust
   ```bash
   # From project root
   locust -f tests/performancetests/locustfile.py --web-host=127.0.0.1 --web-port=8888
   ```

## Running Load Tests

### Using Web Interface

1. Open http://127.0.0.1:8888 in your browser
2. Configure test parameters:
   - Number of users: Total number of concurrent users to simulate
   - Spawn rate: Number of users to add per second
   - Host: URL where your API is running (e.g., http://localhost:8000)
3. Click "Start swarming" to begin the test

### Using Command Line
```bash
locust -f tests/performancetests/locustfile.py \
    --headless \
    --users 10 \
    --spawn-rate 1 \
    --run-time 1m \
    --host http://localhost:8000
```