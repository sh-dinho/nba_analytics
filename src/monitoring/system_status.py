from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Unified system status aggregator combining
#              health, data quality, model monitor, and
#              dashboard metrics into a single snapshot.
# ============================================================


from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, Any, List, Optional

from loguru import logger

from src.monitoring.health_check import HealthChecker, HealthReport
from src.monitoring.data_quality import DataQualityChecker, DataQualityReport
from src.monitoring.model_monitor import ModelMonitor, ModelMonitorReport
from src.monitoring.dashboard_data import DashboardDataBuilder, DashboardBundle


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------


@dataclass
class SystemStatus:
    timestamp: str
    target_date: str
    overall_ok: bool
    severity_score: int
    health: Dict[str, Any]
    data_quality: Dict[str, Any]
    model_monitor: Dict[str, Any]
    dashboard: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# Severity scoring helper
# ------------------------------------------------------------


def _severity_from_issues(issues: List[Dict[str, Any]]) -> int:
    """
    Compute a simple severity score:
      0 = no issues
      1 = info only
      2 = warnings present
      3 = errors present
    """
    if not issues:
        return 0

    levels = {i.get("level", "") for i in issues}
    if "error" in levels:
        return 3
    if "warning" in levels:
        return 2
    return 1


# ------------------------------------------------------------
# System status aggregator
# ------------------------------------------------------------


class SystemStatusAggregator:
    """
    Aggregates all monitoring signals into a single system status snapshot:

      - HealthChecker
      - DataQualityChecker
      - ModelMonitor
      - DashboardDataBuilder

    Intended to be:
      - Called by the orchestrator after runs
      - Exposed via API / CLI / dashboard
      - Fed into AlertManager for severity-based alerts
    """

    def __init__(
        self,
        health_checker: Optional[HealthChecker] = None,
        dq_checker: Optional[DataQualityChecker] = None,
        model_monitor: Optional[ModelMonitor] = None,
        dashboard_builder: Optional[DashboardDataBuilder] = None,
    ):
        self.health_checker = health_checker or HealthChecker()
        self.dq_checker = dq_checker or DataQualityChecker()
        self.model_monitor = model_monitor or ModelMonitor()
        self.dashboard_builder = dashboard_builder or DashboardDataBuilder()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def collect(self, target_date: Optional[date] = None) -> SystemStatus:
        if target_date is None:
            target_date = date.today()

        target_date_str = target_date.isoformat()
        timestamp = datetime.utcnow().isoformat()

        logger.info(
            f"SystemStatusAggregator: collecting status for target_date={target_date_str}"
        )

        # 1. Health report
        health_report: HealthReport = self.health_checker.run(target_date)

        # 2. Data quality
        dq_report: DataQualityReport = self.dq_checker.run_all()

        # 3. Model monitor
        model_report: ModelMonitorReport = self.model_monitor.run()

        # 4. Dashboard metrics (non-critical, best-effort)
        try:
            dashboard_bundle: DashboardBundle = self.dashboard_builder.build()
            dashboard_dict = dashboard_bundle.to_dict()
        except Exception as e:
            logger.exception(
                f"SystemStatusAggregator: failed to build dashboard bundle: {e}"
            )
            dashboard_dict = {}

        # Severity scoring
        health_dict = health_report.to_dict()
        dq_dict = dq_report.to_dict()
        model_dict = model_report.to_dict()

        health_sev = _severity_from_issues(health_dict.get("issues", []))
        dq_sev = _severity_from_issues(dq_dict.get("issues", []))
        model_sev = _severity_from_issues(model_dict.get("issues", []))

        severity_score = max(health_sev, dq_sev, model_sev)
        overall_ok = (
            (severity_score < 3)
            and health_dict.get("ok", False)
            and dq_dict.get("ok", False)
        )

        status = SystemStatus(
            timestamp=timestamp,
            target_date=target_date_str,
            overall_ok=overall_ok,
            severity_score=severity_score,
            health=health_dict,
            data_quality=dq_dict,
            model_monitor=model_dict,
            dashboard=dashboard_dict,
        )

        logger.info(
            f"SystemStatusAggregator: collection complete "
            f"(overall_ok={overall_ok}, severity_score={severity_score})."
        )

        return status


if __name__ == "__main__":
    aggregator = SystemStatusAggregator()
    status = aggregator.collect()
    print(status.to_dict())
