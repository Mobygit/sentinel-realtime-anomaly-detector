# Sentinel - Real-time Log Anomaly Detector

Sentinel is a lightweight real-time log anomaly detector that tails a log file, extracts features, scores lines with an unsupervised model (IsolationForest), and sends alerts (Slack or console) for high-confidence anomalies.

This README provides cross-platform, step-by-step instructions for running the project on Windows, macOS, or Linux.

## Requirements

* Python 3.10+ (3.12 used for development)
* Docker (optional)

## Setup (Cross-Platform)

1. Clone the repository:

```bash
git clone https://github.com/Mobygit/sentinel-realtime-anomaly-detector.git
cd sentinel-realtime-anomaly-detector
```

2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Demo (Recommended)

```bash
python scripts/demo_end_to_end.py --disable-alerting
```

## Development Run (Fallback Mode)

```bash
python sentinel.py --log-file demo/demo.log --allow-fallback --disable-alerting --config config.yaml
```

## Production Run (Requires Model Artifacts)

```bash
python sentinel.py --log-file /path/to/app.log --model models/model_latest.pkl --scaler models/scaler_latest.pkl --config config.yaml --production
```

Notes:

* Do NOT use `--allow-fallback` in production.
* Use `SLACK_WEBHOOK_URL` environment variable for Slack alerts.

## Docker (Optional)

**Build Image**

```bash
docker build -t sentinel-anomaly .
```

**Run Container**

Windows (PowerShell):

```powershell
docker run -v "${PWD}\demo:/logs" -v "${PWD}\config.yaml:/app/config.yaml" sentinel-anomaly --config /app/config.yaml --production
```

Bash / macOS / WSL:

```bash
docker run -v "$(pwd)/demo:/logs" -v "$(pwd)/config.yaml:/app/config.yaml" sentinel-anomaly --config /app/config.yaml --production
```

## Tests

```bash
python -m unittest discover -v
```

## Config & Secrets

**PowerShell**

```powershell
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

## Architecture

* File watcher (`watchdog`) tails logs
* `LogParser` extracts fields via regex
* `FeatureExtractor` converts fields to numeric vectors
* `StandardScaler` scales features
* `IsolationForest` scores anomalies
* `Alerting` module sends Slack or console alerts

## Useful Files

* `sentinel.py` — main monitor and CLI
* `train_advanced.py` — training pipeline
* `scripts/demo_end_to_end.py` — demo harness
* `alerting.py` — Slack and console alerts
* `metrics.py` — runtime metrics collector
