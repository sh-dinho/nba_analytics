"""
Prometheus metrics exporter for NBA Analytics v3.
"""

from prometheus_client import start_http_server, Gauge
from loguru import logger
import time
import random  # Example: simulate metric values

# -------------------------------------------------------------
# Metrics definitions
# -------------------------------------------------------------
games_scheduled_gauge = Gauge(
    "nba_games_scheduled", "Number of games scheduled for today"
)
games_completed_gauge = Gauge("nba_games_completed", "Number of games completed today")
home_win_rate_gauge = Gauge(
    "nba_home_win_rate", "Average home team win rate in recent games"
)


# -------------------------------------------------------------
# Start metrics server
# -------------------------------------------------------------
def start_metrics_server(port: int = 8000):
    """
    Start Prometheus metrics server on given port.
    """
    logger.info(f"Starting Prometheus metrics server on port {port}")
    start_http_server(port)


# -------------------------------------------------------------
# Example function to update metrics
# -------------------------------------------------------------
def update_metrics():
    """
    Update metrics values periodically.
    Replace with real computation from your feature store / predictions.
    """
    # Example: random numbers for demonstration
    scheduled = random.randint(0, 10)
    completed = random.randint(0, 10)
    home_win_rate = random.uniform(0, 1)

    games_scheduled_gauge.set(scheduled)
    games_completed_gauge.set(completed)
    home_win_rate_gauge.set(home_win_rate)

    logger.info(
        f"Updated metrics: scheduled={scheduled}, completed={completed}, home_win_rate={home_win_rate:.2f}"
    )


# -------------------------------------------------------------
# Run exporter loop
# -------------------------------------------------------------
if __name__ == "__main__":
    start_metrics_server(port=8000)
    while True:
        update_metrics()
        time.sleep(60)  # Update every 60 seconds
