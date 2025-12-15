FROM python:3.12-slim

WORKDIR /app

# Install system deps then python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose default port if needed (not required for file monitor)
# EXPOSE 8080

# Default command: run sentinel in production mode
ENTRYPOINT ["python", "sentinel.py"]
CMD ["--config", "config.yaml"]
