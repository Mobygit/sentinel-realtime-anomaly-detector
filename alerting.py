import requests
import json
import logging
from datetime import datetime


class SlackAlerter:
    def __init__(self, webhook_url, channel=None, username="Sentinel Bot"):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.session = requests.Session()

    def send_alert(self, anomaly_data):
        """Send an alert to Slack about a detected anomaly."""
        if not self.webhook_url:
            logging.warning("No Slack webhook URL configured")
            return False

        try:
            # Create a formatted message for Slack
            message = self._create_slack_message(anomaly_data)

            response = self.session.post(
                self.webhook_url,
                data=json.dumps(message),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logging.info("✅ Alert sent to Slack successfully")
                return True

            logging.error(
                f"❌ Failed to send alert: {response.status_code} - {response.text}"
            )
            return False

        except Exception as exc:  # pragma: no cover - best-effort alerting
            logging.error(f"❌ Error sending Slack alert: {exc}")
            return False

    def _create_slack_message(self, anomaly_data):
        """Create a formatted Slack message with anomaly details."""
        parsed = anomaly_data.get("parsed", {})

        # Determine severity based on anomaly score
        score = anomaly_data.get("anomaly_score", 0)
        if score < -0.2:
            color = "#FF0000"  # Red for high severity
            severity = "HIGH"
        elif score < -0.1:
            color = "#FFA500"  # Orange for medium severity
            severity = "MEDIUM"
        else:
            color = "#FFFF00"  # Yellow for low severity
            severity = "LOW"

        message = {
            "channel": self.channel,
            "username": self.username,
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 Log Anomaly Detected ({severity})",
                    "fields": [
                        {"title": "Anomaly Score", "value": f"`{score:.3f}`", "short": True},
                        {"title": "Timestamp", "value": parsed.get("timestamp", "Unknown"), "short": True},
                        {"title": "IP Address", "value": f"`{parsed.get('ip', 'Unknown')}`", "short": True},
                        {"title": "Status Code", "value": f"`{parsed.get('status_code', 'Unknown')}`", "short": True},
                        {"title": "Request", "value": f"`{parsed.get('request', 'Unknown')}`", "short": False},
                        {
                            "title": "Raw Log Line",
                            "value": f"```{anomaly_data.get('line', '')[:100]}...```",
                            "short": False,
                        },
                    ],
                    "footer": "Sentinel Anomaly Detector",
                    "ts": datetime.now().timestamp(),
                }
            ],
        }

        return message


class ConsoleAlerter:
    """Fallback alerter that prints to console."""

    def send_alert(self, anomaly_data):
        """Print anomaly to console."""
        parsed = anomaly_data.get("parsed", {})
        score = anomaly_data.get("anomaly_score", 0)

        print("🚨 ALERT - Score: {:.3f}".format(score))
        print(" Time: {}".format(parsed.get("timestamp", "Unknown")))
        print(" IP: {}".format(parsed.get("ip", "Unknown")))
        print(" Status: {}".format(parsed.get("status_code", "Unknown")))
        print(" Request: {}".format(parsed.get("request", "Unknown")))
        print(" Full Line: {}".format(anomaly_data.get("line", "")))
        print("-" * 80)
        return True