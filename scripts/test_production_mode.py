#!/usr/bin/env python3
"""Quick production-mode smoke test."""
import subprocess
import time
import sys
from pathlib import Path

# Test file
test_log = Path("test_prod_mode.log")
test_log.write_text("")

# Start sentinel in production mode
proc = subprocess.Popen([
    sys.executable, "sentinel.py",
    "--log-file", str(test_log),
    "--model", "models/model_latest.pkl",
    "--scaler", "models/scaler_latest.pkl",
    "--production",
    "--disable-alerting"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# Give sentinel time to start
time.sleep(2)

# Append a test log line
test_log.write_text('127.0.0.1 - - [15/Dec/2025:17:30:00 +0000] "GET /api/health HTTP/1.1" 200 42\n')

# Wait for sentinel to process
time.sleep(3)

# Stop sentinel
proc.terminate()
try:
    stdout, _ = proc.communicate(timeout=5)
    print(stdout)
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, _ = proc.communicate()
    print(stdout)

# Verify no errors (check for the message with or without emoji encoding issues)
if "Model and scaler loaded successfully" in stdout:
    print("\n✅ Production mode test PASSED: Model loaded successfully")
    test_log.unlink()
    sys.exit(0)
else:
    print("\n❌ Production mode test FAILED")
    print(f"Output:\n{stdout}")
    test_log.unlink()
    sys.exit(1)
