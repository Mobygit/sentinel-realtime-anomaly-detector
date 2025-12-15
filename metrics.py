import time
import threading
from datetime import datetime, timedelta
import logging



class PerformanceMetrics:
    """Track performance metrics for the Sentinel system."""
    
    def __init__(self, stats_interval=300):
        self.stats_interval = stats_interval
        self.lock = threading.Lock()
        
        # Counters
        self.total_lines_processed = 0
        self.anomalies_detected = 0
        self.alerts_sent = 0
        self.parse_errors = 0
        self.processing_errors = 0
        
        # Timing
        self.start_time = time.time()
        self.last_stats_time = self.start_time
        
        # Performance tracking
        self.processing_times = []
        self.line_lengths = []
        
        # Start background stats thread
        self._running = True
        self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        self.stats_thread.start()
    
    
    def record_line_processed(self, line_length, processing_time, is_anomaly=False, alert_sent=False):
        """Record metrics for a processed log line."""
        with self.lock:
            self.total_lines_processed += 1
            self.line_lengths.append(line_length)
            self.processing_times.append(processing_time)
            
            if is_anomaly:
                self.anomalies_detected += 1
            if alert_sent:
                self.alerts_sent += 1
    

    def record_error(self, error_type="processing"):
        """Record different types of errors."""
        with self.lock:
            if error_type == "parse":
                self.parse_errors += 1
            else:
                self.processing_errors += 1
    

    def get_stats(self):
        """Get current performance statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            lines_per_second = self.total_lines_processed / uptime if uptime > 0 else 0
            
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0
            )
            
            avg_line_length = (
                sum(self.line_lengths) / len(self.line_lengths) 
                if self.line_lengths else 0
            )
            
            anomaly_rate = (
                (self.anomalies_detected / self.total_lines_processed * 100) 
                if self.total_lines_processed > 0 else 0
            )
            
            return {
                'uptime_seconds': uptime,
                'total_lines_processed': self.total_lines_processed,
                'lines_per_second': lines_per_second,
                'anomalies_detected': self.anomalies_detected,
                'alerts_sent': self.alerts_sent,
                'anomaly_rate_percent': anomaly_rate,
                'parse_errors': self.parse_errors,
                'processing_errors': self.processing_errors,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'avg_line_length': avg_line_length,
                'current_time': datetime.now().isoformat()
            }
    

    def _stats_loop(self):
        """Background thread to print statistics periodically."""
        while self._running:
            time.sleep(self.stats_interval)
            stats = self.get_stats()
            self._log_stats(stats)
    

    def _log_stats(self, stats):
        """Log performance statistics."""
        logging.info("📊 Performance Statistics:")
        logging.info(f" Uptime: {timedelta(seconds=int(stats['uptime_seconds']))}")
        logging.info(f" Lines Processed: {stats['total_lines_processed']}")
        logging.info(f" Processing Rate: {stats['lines_per_second']:.2f} lines/sec")
        logging.info(f" Anomalies: {stats['anomalies_detected']} ({stats['anomaly_rate_percent']:.2f}%)")
        logging.info(f" Alerts Sent: {stats['alerts_sent']}")
        logging.info(f" Parse Errors: {stats['parse_errors']}")
        logging.info(f" Avg Processing Time: {stats['avg_processing_time_ms']:.2f} ms")
        logging.info("-" * 50)
    

    def stop(self):
        """Stop the metrics collector."""
        self._running = False
        if self.stats_thread.is_alive():
            self.stats_thread.join(timeout=1.0)