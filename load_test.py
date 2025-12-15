import time
import random
from datetime import datetime
import logging
import argparse


class LoadTester:
    """Generate high-volume log data for performance testing."""

    def __init__(self, log_file="load_test.log", lines_per_second=1000, duration=60):
        self.log_file = log_file
        self.lines_per_second = lines_per_second
        self.duration = duration
        self.generated_count = 0
        self.running = False

        # Common log components for realistic data
        self.ips = [f"192.168.1.{i}" for i in range(1, 255)]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "curl/7.68.0",
            "python-requests/2.25.1",
        ]
        self.endpoints = [
            "/",
            "/api/v1/users",
            "/api/v1/products",
            "/admin",
            "/login",
            "/static/css/main.css",
            "/static/js/app.js",
            "/images/logo.png",
        ]
        self.methods = ["GET", "POST", "PUT", "DELETE"]
        self.status_codes = [200, 200, 200, 200, 304, 404, 500]  # Weighted

    def generate_log_line(self, anomaly=False):
        """Generate a single log line."""
        timestamp = datetime.now().strftime("%d/%b/%Y:%H:%M:%S %z")
        ip = random.choice(self.ips)
        method = random.choice(self.methods)

        if anomaly:
            # Generate anomalous patterns
            endpoint = random.choice(
                [
                    "/../../../etc/passwd",
                    "/wp-admin.php",
                    f"/api/delete/{random.randint(1000, 9999)}",
                    "/shell?cmd=rm+-rf+%2F",
                ]
            )
            status = random.choice([500, 403, 401])
            size = random.choice([0, 9999999])
            user_agent = (
                "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"  # Old IE
            )
        else:
            endpoint = random.choice(self.endpoints)
            status = random.choice(self.status_codes)
            size = random.randint(100, 5000)
            user_agent = random.choice(self.user_agents)

        return f"{ip} - - [{timestamp}] \"{method} {endpoint} HTTP/1.1\" {status} {size} \"{user_agent}\"\n"

    def run_test(self):
        """Run the load test."""
        self.running = True
        self.generated_count = 0
        start_time = time.time()
        end_time = start_time + self.duration

        logging.info(
            "🚀 Starting load test: %s lines/sec for %s seconds",
            self.lines_per_second,
            self.duration,
        )
        logging.info("📁 Output file: %s", self.log_file)

        # Clear the file first
        with open(self.log_file, "w") as f:
            f.write("# Load test started\n")

        try:
            while time.time() < end_time and self.running:
                batch_start = time.time()
                lines_in_batch = 0

                # Write a batch of lines
                with open(self.log_file, "a") as f:
                    while lines_in_batch < self.lines_per_second and time.time() < end_time:
                        # 5% anomalies for testing
                        anomaly = random.random() < 0.05
                        line = self.generate_log_line(anomaly=anomaly)
                        f.write(line)
                        lines_in_batch += 1
                        self.generated_count += 1

                # Sleep to maintain target rate
                batch_time = time.time() - batch_start
                sleep_time = 1.0 - batch_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Progress reporting
                if int(time.time() - start_time) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = self.generated_count / elapsed
                    logging.info("📈 Progress: %s lines, %.1f lines/sec", self.generated_count, rate)

        except KeyboardInterrupt:
            logging.info("🛑 Load test interrupted by user")

        finally:
            self.running = False
            total_time = time.time() - start_time
            final_rate = self.generated_count / total_time if total_time > 0 else 0

            logging.info("✅ Load test completed:")
            logging.info(" Total lines: %s", self.generated_count)
            logging.info(" Total time: %.2f seconds", total_time)
            logging.info(" Average rate: %.1f lines/sec", final_rate)

    def stop(self):
        """Stop the load test."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Log Load Tester")
    parser.add_argument("--log-file", default="load_test.log", help="Output log file")
    parser.add_argument("--rate", type=int, default=1000, help="Lines per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    tester = LoadTester(log_file=args.log_file, lines_per_second=args.rate, duration=args.duration)

    try:
        tester.run_test()
    except KeyboardInterrupt:
        tester.stop()


if __name__ == "__main__":
    main()