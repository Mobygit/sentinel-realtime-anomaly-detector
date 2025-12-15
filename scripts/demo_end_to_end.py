"""Demo: train a small model on synthetic logs and run sentinel to monitor appended lines.
This script will:
 - Generate a synthetic training log
 - Train a small IsolationForest model using train_advanced.ModelTrainer
 - Start sentinel.py in a subprocess monitoring the demo log
 - Append lines to the demo log to simulate real-time traffic
 - Terminate sentinel after a short demo

Designed for local demo/testing only.
"""
import sys
import time
import logging
import subprocess
from pathlib import Path
import importlib

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import project modules after ensuring repo root is on sys.path
_load_test = importlib.import_module("load_test")
_train_advanced = importlib.import_module("train_advanced")
LoadTester = _load_test.LoadTester
ModelTrainer = _train_advanced.ModelTrainer


def generate_training_log(path: str, lines: int = 2000, anomaly_rate: float = 0.05):
    tester = LoadTester(log_file=path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(lines):
            anomaly = (i % int(1 / anomaly_rate) == 0) and (i != 0)
            f.write(tester.generate_log_line(anomaly=anomaly))


def append_stream(log_path: str, duration: float = 8.0, rate: float = 50.0):
    """Append lines to the log file at approximately `rate` lines/sec for `duration` seconds."""
    tester = LoadTester(log_file=log_path)
    end_time = time.time() + duration
    interval = 1.0 / rate
    with open(log_path, 'a', encoding='utf-8') as f:
        while time.time() < end_time:
            anomaly = (time.time() % 7) < 0.35  # occasional anomalies
            f.write(tester.generate_log_line(anomaly=anomaly))
            f.flush()
            time.sleep(interval)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    demo_log = str(ROOT / 'demo' / 'demo.log')
    models_dir = str(ROOT / 'models')

    logging.info('Generating synthetic training data...')
    generate_training_log(demo_log, lines=1500, anomaly_rate=0.05)

    logging.info('Training small model...')
    trainer = ModelTrainer(model_dir=models_dir)
    version = trainer.train_from_scratch(
        log_files=[demo_log], contamination=0.05, sample_ratio=1.0
    )
    logging.info('Model trained: %s', version)

    # Copy latest artifacts to stable names so sentinel can use them by default
    import shutil
    latest_model_src = Path(models_dir) / f"model_{version}.pkl"
    latest_scaler_src = Path(models_dir) / f"scaler_{version}.pkl"
    latest_model_dst = Path(models_dir) / "model_latest.pkl"
    latest_scaler_dst = Path(models_dir) / "scaler_latest.pkl"
    try:
        shutil.copy2(latest_model_src, latest_model_dst)
        shutil.copy2(latest_scaler_src, latest_scaler_dst)
        logging.info('Copied model_latest and scaler_latest to models/')
    except Exception as exc:
        logging.warning('Could not copy latest model artifacts: %s', exc)

    # Start sentinel as a subprocess so we can kill it later
    python_exe = sys.executable
    sentinel_script = str(ROOT / 'sentinel.py')
    cmd = [
        python_exe,
        sentinel_script,
        '--log-file',
        demo_log,
        '--model',
        str(Path(models_dir) / 'model_latest.pkl'),
        '--scaler',
        str(Path(models_dir) / 'scaler_latest.pkl'),
        '--config',
        str(ROOT / 'config.yaml'),
    ]

    logging.info('Starting sentinel subprocess...')
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1
    )

    # Start a reader thread to stream sentinel output without blocking
    import threading

    def _reader_thread(p):
        try:
            for line in p.stdout:
                if line:
                    print('[sentinel] ' + line.rstrip())
        except Exception:
            pass

    reader = threading.Thread(target=_reader_thread, args=(proc,), daemon=True)
    reader.start()

    try:
        # Let sentinel warm up
        time.sleep(1.0)
        logging.info('Appending live stream to demo log...')
        append_stream(demo_log, duration=8.0, rate=120.0)

        # Allow a short grace period for sentinel to process
        time.sleep(2.0)

    finally:
        logging.info('Stopping sentinel subprocess...')
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        # Ensure reader thread has finished
        reader.join(timeout=1.0)

    logging.info('Demo completed.')


if __name__ == '__main__':
    main()
