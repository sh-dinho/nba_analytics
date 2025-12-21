# ============================================================
# File: src/monitoring/metrics_exporter.py
# Purpose: Start Prometheus metrics HTTP server
# Version: 3.0
# Author: Your Team
# Date: December 2025
# ============================================================

from loguru import logger
from prometheus_client import start_http_server


def start_metrics_server(port: int = 8000):
    logger.info(f"Starting Prometheus metrics server on port {port}")
    start_http_server(port)
