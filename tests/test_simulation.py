# tests/test_simulation.py
import pandas as pd
from nba_analytics_core.simulate_ai_bankroll import simulate_ai_strategy

def test_simulation_runs_and_kpis():
    df = simulate_ai_strategy(initial_bankroll=1000, strategy="flat")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"game_id", "team", "decimal_odds", "prob", "stake", "pnl", "realized", "bankroll"}
    assert expected_cols.issubset(set(df.columns))
    assert pd.api.types.is_numeric_dtype(df["bankroll"])
    kpis = getattr(df, "kpis", None)
    assert kpis is not None and "roi" in kpis and "win_rate" in kpis and "max_drawdown" in kpis