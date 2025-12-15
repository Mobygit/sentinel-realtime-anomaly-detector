# Sentinel - Real-time Anomaly Detector

This repository contains a real-time log anomaly detection system (Sentinel). This README covers quick setup, running in dev and production modes, and testing.

## Quick setup (Windows PowerShell)

Activate the existing virtualenv in the repo:

```powershell
& "C:/mobython/projects/Zaalima Projects/sentinel-realtime-anomaly-detector/sentinel_env/Scripts/Activate.ps1"
```

Install dependencies (if needed):

```powershell
pip install -r requirements.txt
```

## Running the monitor (development)

You can run the monitor without a trained model by allowing fallback (development only):

```powershell
& "C:/mobython/projects/Zaalima Projects/sentinel-realtime-anomaly-detector/sentinel_env/Scripts/python.exe" sentinel.py --log-file path\to\file.log --allow-fallback --config config.yaml
```

This will run using a lightweight dummy model/scaler so you can exercise parsing, metrics, and alerting (console or Slack).

## Running in production

Train a model using `train_advanced.py` and use the produced model and scaler files. After training, run:

```powershell
& "C:/mobython/projects/Zaalima Projects/sentinel-realtime-anomaly-detector/sentinel_env/Scripts/python.exe" sentinel.py --log-file path\to\file.log --model models/model_latest.pkl --scaler models/scaler_latest.pkl --config config.yaml --production
```

- Do NOT use `--allow-fallback` in production.
- You can set `SLACK_WEBHOOK_URL` in the environment instead of placing it in `config.yaml`.

### Docker

Build the container image locally:

```powershell
docker build -t sentinel-anomaly .
```

Run (mount logs and config into the container):

```powershell
docker run -v C:/path/to/logs:/logs -v C:/path/to/config.yaml:/app/config.yaml sentinel-anomaly --config /app/config.yaml --production
```

## Tests

Run unit tests with the built-in unittest runner:

```powershell
& "C:/mobython/projects/Zaalima Projects/sentinel-realtime-anomaly-detector/sentinel_env/Scripts/python.exe" -m unittest discover -v
```

## Notes

- Use environment variable `SLACK_WEBHOOK_URL` to override Slack webhook secrets (keeps secrets out of config files).
- For a production rollout consider adding CI, linting, packaging, and type checking.
