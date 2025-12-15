import re
import time
import pickle
import argparse
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from alerting import SlackAlerter, ConsoleAlerter
from metrics import PerformanceMetrics





class LogParser:
    def __init__(self):
        # A common log format pattern (adjust as needed for your logs)
        # Example line: '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 1024'
        self.pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)'
        self.regex = re.compile(self.pattern)

    def parse_line(self, line):
        """Parses a single log line and returns a dictionary of components."""
        match = self.regex.search(line)
        if match:
            return {
                'ip': match.group(1),
                'timestamp': match.group(2),
                'request': match.group(3),
                'status_code': int(match.group(4)),
                'response_size': int(match.group(5)),
                'raw_message': line.strip()
            }
        else:
            return None




class FeatureExtractor:
    def __init__(self):
        pass

    def featurize(self, parsed_log_dict):
        """Converts a parsed log dictionary into a numerical feature vector."""
        if parsed_log_dict is None:
            return [0, 0, 0, 0]

        features = []
        # Feature 1: Status Code
        features.append(parsed_log_dict['status_code'])
        # Feature 2: Length of the request string
        features.append(len(parsed_log_dict['request']))
        # Feature 3: Response size
        features.append(parsed_log_dict['response_size'])
        # Feature 4: Numerical representation of the hour
        try:
            dt = datetime.strptime(parsed_log_dict['timestamp'], '%d/%b/%Y:%H:%M:%S %z')
            features.append(dt.hour)
        except ValueError:
            features.append(0)

        return features




class SentinelConfig:
    """Enhanced configuration management for Sentinel."""

    def __init__(self):
        # Core settings
        self.log_file = None
        self.model_path = 'models/model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.slack_webhook_url = None
        self.slack_channel = None
        self.alert_threshold = -0.1
        self.log_level = 'INFO'
        # Allow using a lightweight fallback model when real artifacts are not present
        self.allow_fallback = False
        # Allow disabling alerts (useful for demos)
        self.disable_alerting = False

        # Performance settings
        self.batch_size = 1
        self.processing_delay = 0
        self.max_line_length = 10000

        # Monitoring settings
        self.metrics_enabled = True
        self.stats_interval = 300
        self.performance_alert_threshold = 1000

        # Model settings
        self.confidence_threshold = 0.6
        self.enable_explanations = True




    @classmethod
    def from_args(cls):
        """Create config from command line arguments with enhanced options."""
        config = cls()


        parser = argparse.ArgumentParser(
            description='Real-time Log Anomaly Detector - Production Grade',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Required arguments
        parser.add_argument('--log-file', required=True, help='Path to the log file to monitor')


        # Core optional arguments
        parser.add_argument(
            '--model', default='model.pkl', help='Path to the trained model'
        )
        parser.add_argument(
            '--scaler', default='scaler.pkl', help='Path to the scaler'
        )
        parser.add_argument('--config', help='Path to YAML config file')
        parser.add_argument('--slack-webhook', help='Slack webhook URL for alerts')
        parser.add_argument(
            '--allow-fallback',
            action='store_true',
            help='Allow running without trained model/scaler (development only)',
        )
        parser.add_argument(
            '--disable-alerting',
            action='store_true',
            help='Disable sending alerts (use console only)',
        )
        parser.add_argument('--slack-channel', help='Slack channel for alerts')
        parser.add_argument(
            '--alert-threshold', type=float, default=-0.1,
            help='Anomaly score threshold for alerts',
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level',
        )


        # Performance arguments
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help='Number of log lines to process in batch',
        )
        parser.add_argument(
            '--processing-delay',
            type=float,
            default=0,
            help='Seconds to wait between processing batches',
        )
        parser.add_argument(
            '--max-line-length',
            type=int,
            default=10000,
            help='Maximum log line length to process',
        )


        # Monitoring arguments
        parser.add_argument('--no-metrics', action='store_true', help='Disable performance metrics collection')
        parser.add_argument('--stats-interval', type=int, default=300, help='Seconds between statistics reports')

        args = parser.parse_args()


        # Load config file if specified
        if args.config:
            config.load_from_file(args.config)   


        # Apply command line arguments (override config file)
        config.log_file = args.log_file
        config.model_path = args.model
        config.scaler_path = args.scaler
        config.slack_webhook_url = args.slack_webhook or config.slack_webhook_url
        config.slack_channel = args.slack_channel or config.slack_channel
        config.allow_fallback = args.allow_fallback or getattr(config, 'allow_fallback', False)
        config.disable_alerting = args.disable_alerting or getattr(config, 'disable_alerting', False)
        config.alert_threshold = args.alert_threshold
        config.log_level = args.log_level
        config.batch_size = args.batch_size
        config.processing_delay = args.processing_delay
        config.max_line_length = args.max_line_length
        config.metrics_enabled = not args.no_metrics
        config.stats_interval = args.stats_interval
        
        return config

    def load_from_file(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            if file_config:
                for key, value in file_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            # Respect environment variables for secrets (e.g., Slack webhook)
            import os
            env_webhook = os.environ.get('SLACK_WEBHOOK_URL') or os.environ.get('SLACK_WEBHOOK')
            if env_webhook:
                self.slack_webhook_url = env_webhook
            logging.info(f"✅ Loaded configuration from {config_path}")

        except FileNotFoundError:
            logging.warning(f"⚠️ Config file not found: {config_path}")
        except yaml.YAMLError as e:
            logging.error(f"❌ Error parsing config file: {e}")


    def validate(self):
        """Enhanced configuration validation."""
        errors = []
        
        if not Path(self.log_file).exists():
            errors.append(f"Log file does not exist: {self.log_file}")
        # Model/scaler are required in production, but may be optional for development
        if not getattr(self, 'allow_fallback', False):
            if not Path(self.model_path).exists():
                errors.append(f"Model file does not exist: {self.model_path}")
            if not Path(self.scaler_path).exists():
                errors.append(f"Scaler file does not exist: {self.scaler_path}")
        
        if self.batch_size < 1:
            errors.append("Batch size must be at least 1")
            
        if self.processing_delay < 0:
            errors.append("Processing delay cannot be negative")
        
        if errors:
            for error in errors:
                logging.error(error)
            return False
        
        return True


class LogFileHandler(FileSystemEventHandler):
    def __init__(self, file_path, model, scaler, parser, featurizer, callback, config, metrics):
        self.file_path = file_path
        self.model = model
        self.scaler = scaler
        self.parser = parser
        self.featurizer = featurizer
        self.callback = callback
        self.config = config
        self.metrics = metrics
        self._last_position = 0
        self._line_buffer = []
        self._last_process_time = time.time()
        self._last_position = self.get_current_file_size()

    def get_current_file_size(self):
        """Get the current size of the file."""
        return Path(self.file_path).stat().st_size
    
    def on_modified(self, event):
        """Called when the file is modified."""
        if event.src_path == str(self.file_path):
            self.process_new_lines()
    
    def process_new_lines(self):
        """Read and process new lines with batching support."""
        current_size = self.get_current_file_size()
        
        if current_size < self._last_position:
            self._last_position = 0
        
        if current_size > self._last_position:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self._last_position)
                new_lines = f.readlines()
                self._last_position = f.tell()
                
                # Filter lines that are too long
                filtered_lines = [
                    line.strip() for line in new_lines 
                    if line.strip() and len(line) <= self.config.max_line_length
                ]
                
                self._line_buffer.extend(filtered_lines)


                # Process in batches if configured
                if len(self._line_buffer) >= self.config.batch_size or (time.time() - self._last_process_time >= 1.0):
                    self.process_batch()

    def process_batch(self):
        """Process a batch of log lines."""
        if not self._line_buffer:
            return
            
        batch_lines = self._line_buffer[:self.config.batch_size]
        self._line_buffer = self._line_buffer[self.config.batch_size:]
        
        for line in batch_lines:
            start_time = time.time()
            try:
                self.process_single_line(line)
                processing_time = time.time() - start_time
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_line_processed(
                        line_length=len(line),
                        processing_time=processing_time
                    )
                    
            except Exception as e:
                logging.error(f"❌ Error processing line: {e}")
                if self.metrics:
                    self.metrics.record_error("processing")
        
        self._last_process_time = time.time()
        
        # Optional delay between batches
        if self.config.processing_delay > 0:
            time.sleep(self.config.processing_delay)


    def process_single_line(self, line):
        """Process a single log line through the ML pipeline."""
        # Parse the log line
        parsed = self.parser.parse_line(line)
        
        if parsed is None:
            logging.warning(f"⚠️ Failed to parse line: {line[:100]}...")
            if self.metrics:
                self.metrics.record_error("parse")
            return
        
        # Extract features
        features = self.featurizer.featurize(parsed)
        feature_vector = [features]
        
        # Scale the features
        features_scaled = self.scaler.transform(feature_vector)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.decision_function(features_scaled)[0]
        
        # Call the callback with results
        self.callback({
            'line': line,
            'parsed': parsed,
            'prediction': prediction,
            'anomaly_score': anomaly_score,
            'features': features
        })


def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_alerter(config):
    """Create the appropriate alerter based on configuration."""
    if getattr(config, 'disable_alerting', False):
        logging.info("🔕 Alerting disabled by configuration. Using console alerter.")
        return ConsoleAlerter()

    if config.slack_webhook_url:
        logging.info("✅ Using Slack alerter")
        return SlackAlerter(webhook_url=config.slack_webhook_url, channel=config.slack_channel)

    logging.info("ℹ️ No Slack webhook configured. Using console alerter.")
    return ConsoleAlerter()


def anomaly_callback_factory(alerter, threshold, metrics):
    """Create a callback function that uses the alerter and threshold."""
    def callback(anomaly_data):
        # Only alert if the anomaly score is below the threshold
        if anomaly_data['prediction'] == -1 and anomaly_data['anomaly_score'] < threshold:
            logging.warning(
                f"🚨 Anomaly detected! Score: {anomaly_data['anomaly_score']:.3f} "
                f"(threshold: {threshold})"
            )
            alert_sent = alerter.send_alert(anomaly_data)

            # Update metrics if alert was sent
            if metrics and alert_sent:
                metrics.record_line_processed(
                    line_length=len(anomaly_data['line']),
                    processing_time=0,  # We don't have processing time here
                    is_anomaly=True,
                    alert_sent=True,
                )
        else:
            # Log normal or below-threshold anomalies for debugging
            if anomaly_data['prediction'] == -1:
                logging.debug(
                    f"⚠️ Anomaly below threshold: {anomaly_data['anomaly_score']:.3f} "
                    f"(threshold: {threshold})"
                )
            else:
                logging.debug("✅ Normal log line")
    
    return callback
                                        

def start_monitoring(config):
    """Start real-time log monitoring with the given configuration."""
    
    # Validate configuration
    if not config.validate():
        logging.error("❌ Configuration validation failed. Exiting.")
        return
    
    # Set up logging
    setup_logging(config.log_level)
    
    # Load the trained model and scaler
    logging.info("Loading trained model and scaler...")
    try:
        with open(config.model_path, 'rb') as f:
            model = pickle.load(f)
        with open(config.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("✅ Model and scaler loaded successfully!")
    except Exception as e:
        # If model/scaler cannot be loaded, create lightweight fallbacks so
        # the monitor can run for development/testing without trained artifacts.
        logging.warning(
            f"⚠️ Failed to load model or scaler: {e} - using fallback dummy model/scaler for testing"
        )

        # Minimal scaler that returns inputs as-is (expects lists/arrays)
        class DummyScaler:
            def transform(self, X):
                return X

        # Minimal model that treats everything as normal and returns neutral scores
        class DummyModel:
            def predict(self, X):
                # Return 1 (normal) for every sample
                return [1 for _ in range(len(X))]

            def decision_function(self, X):
                # Return zero score for every sample
                return [0.0 for _ in range(len(X))]

        scaler = DummyScaler()
        model = DummyModel()
    
    # Initialize components
    parser = LogParser()
    featurizer = FeatureExtractor()
    alerter = create_alerter(config)
    
    # Initialize metrics if enabled
    metrics = None
    if config.metrics_enabled:
        metrics = PerformanceMetrics(stats_interval=config.stats_interval)
        logging.info("✅ Performance metrics enabled")
    
    # Create the anomaly callback
    anomaly_callback = anomaly_callback_factory(alerter, config.alert_threshold, metrics)
    
    logging.info(f"🔍 Starting to monitor: {config.log_file}")
    logging.info(f"📊 Alert threshold: {config.alert_threshold}")
    logging.info(f"⚡ Batch size: {config.batch_size}")
    logging.info("Press Ctrl+C to stop monitoring...")
    logging.info("-" * 80)
    
    # Set up file monitoring
    event_handler = LogFileHandler(
        file_path=config.log_file,
        model=model,
        scaler=scaler,
        parser=parser,
        featurizer=featurizer,
        callback=anomaly_callback,
        config=config,
        metrics=metrics
    )
    
    observer = Observer()
    observer.schedule(event_handler, path=str(Path(config.log_file).parent), recursive=False)
    observer.start()
    
    try:
        # Process any existing content first
        event_handler.process_new_lines()
        
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\n🛑 Stopping monitoring...")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        observer.stop()
        observer.join()
        if metrics:
            metrics.stop()
        logging.info("✅ Sentinel stopped gracefully.")




if __name__ == "__main__":
    try:
        config = SentinelConfig.from_args()
        start_monitoring(config)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        sys.exit(1)
