from datetime import date
from src.pipeline.orchestrator import Orchestrator


def test_orchestrator_end_to_end(monkeypatch):
    orch = Orchestrator(notify=False)

    # Patch all steps to avoid real work
    monkeypatch.setattr(orch, "_step_ingestion", lambda d: {"ok": True})
    monkeypatch.setattr(
        orch,
        "_step_predict_moneyline",
        lambda d: {"model_name": "ml", "model_version": "v1", "feature_version": "v1"},
    )
    monkeypatch.setattr(orch, "_step_predict_totals", lambda d: {"ok": True})
    monkeypatch.setattr(orch, "_step_predict_spread", lambda d: {"ok": True})
    monkeypatch.setattr(orch, "_step_betting_pipeline", lambda d: {"bets": 1})
    monkeypatch.setattr(
        orch, "_step_recommendations", lambda d: {"num_recommendations": 3}
    )

    orch.run_daily(target_date=date(2024, 1, 1))
