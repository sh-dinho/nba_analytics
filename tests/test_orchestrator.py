from src.pipeline.orchestrator import Orchestrator
from datetime import date


def test_orchestrator_runs(monkeypatch):
    orch = Orchestrator(notify=False)

    # Patch steps to avoid real work
    monkeypatch.setattr(orch, "_step_ingestion", lambda d: {})
    monkeypatch.setattr(orch, "_step_predict_moneyline", lambda d: {})
    monkeypatch.setattr(orch, "_step_predict_totals", lambda d: {})
    monkeypatch.setattr(orch, "_step_predict_spread", lambda d: {})
    monkeypatch.setattr(orch, "_step_betting_pipeline", lambda d: {})
    monkeypatch.setattr(
        orch, "_step_recommendations", lambda d: {"num_recommendations": 0}
    )

    orch.run_daily(target_date=date(2024, 1, 1))
