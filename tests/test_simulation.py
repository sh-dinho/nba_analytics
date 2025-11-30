import pandas as pd
from app.simulate_ai_bankroll import simulate_ai_strategy


def test_simulation_runs():
    df = simulate_ai_strategy(initial_bankroll=1000, strategy="flat")
    assert isinstance(df, pd.DataFrame)