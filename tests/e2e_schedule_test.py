# ============================================================
# File: tests/e2e_schedule_test.py
# Purpose: End-to-end test for schedule pipeline (mocked)
# ============================================================

import unittest
import pandas as pd
from pathlib import Path
from src.schedule.generate import generate_today, generate_historical
from src.schedule.enrich import enrich_schedule
from src.schedule.refresh import refresh_incremental
from src.analytics.rankings import RankingsManager


class TestE2EPipeline(unittest.TestCase):
    def setUp(self):
        # Temporary directories for test outputs
        self.tmp_cache = Path("tests/tmp_cache")
        self.tmp_cache.mkdir(parents=True, exist_ok=True)
        self.tmp_history = Path("tests/tmp_history")
        self.tmp_history.mkdir(parents=True, exist_ok=True)
        self.tmp_analytics = Path("tests/tmp_analytics")
        self.tmp_analytics.mkdir(parents=True, exist_ok=True)

        # Mock historical data
        self.hist_df = pd.DataFrame(
            {
                "GAME_ID": [1, 2],
                "TEAM_ID": [100, 101],
                "PTS": [102, 95],
                "WL": ["W", "L"],
                "GAME_DATE": pd.to_datetime(["2025-12-10", "2025-12-11"]),
            }
        )
        self.hist_file = self.tmp_history / "historical_schedule.csv"
        self.hist_df.to_csv(self.hist_file, index=False)

        # Mock today's games
        self.today_df = pd.DataFrame(
            {
                "GAME_ID": [3],
                "TEAM_ID": [100],
                "PTS": [110],
                "WL": ["W"],
                "GAME_DATE": pd.to_datetime(["2025-12-15"]),
            }
        )

    def test_pipeline_end_to_end(self):
        # Enrich historical schedule
        enriched_file = self.tmp_cache / "master_enriched.csv"
        enriched_df = enrich_schedule(str(self.hist_file), str(enriched_file))
        self.assertFalse(enriched_df.empty, "Enriched schedule should not be empty")

        # Incremental refresh with today's games
        refreshed_file = self.tmp_cache / "master_refreshed.csv"
        refreshed_df = refresh_incremental(
            str(enriched_file), self.today_df, str(refreshed_file)
        )
        self.assertIn(
            3,
            refreshed_df["GAME_ID"].values,
            "Today's game should be in refreshed schedule",
        )

        # Generate rankings
        manager = RankingsManager()
        rankings = manager.generate_rankings(
            refreshed_df.assign(predicted_win=[1, 0, 1])
        )
        self.assertFalse(rankings.empty, "Rankings should not be empty")
        self.assertIn(
            "win_pct", rankings.columns, "Rankings should have win_pct column"
        )

        # Betting recommendations
        recs = manager.betting_recommendations(rankings, win_thr=0.0, acc_thr=0.0)
        self.assertIn("bet_on", recs)
        self.assertIn("avoid", recs)

    def tearDown(self):
        # Clean up temporary directories
        for d in [self.tmp_cache, self.tmp_history, self.tmp_analytics]:
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()


if __name__ == "__main__":
    unittest.main()
