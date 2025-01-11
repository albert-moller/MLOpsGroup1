# Docker Instructions

Build image:
```bash
docker build -f dockerfiles/train.dockerfile . -t mlops-train:latest
```

Run training:
```bash
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/configs:/app/configs" \
    --env-file .env \
    mlops-train:latest
```