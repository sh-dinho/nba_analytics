# ============================================================
# Path: src/analytics/rankings.py
# Purpose: Track top teams/players, betting recommendations,
#          and winning streaks from prediction history + box scores
# Version: 4.0 (config-driven RankingsManager + MLflow logging + visualization)
# ============================================================

import os
import logging
import pandas as pd
import yaml
import mlflow
import matplotlib.pyplot as plt

from mlflow_setup import start_run_with_metadata, configure_mlflow


class RankingsManager:
    def __init__(self, config_file="config.yaml"):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        self.paths = cfg["paths"]
        self.nba_cfg = cfg["nba"]
        self.output_cfg = cfg["output"]
        self.logging_cfg = cfg["logging"]
        self.mlflow_cfg = cfg.get("mlflow", {})

        os.makedirs(self.paths.get("history", "data/history"), exist_ok=True)
        os.makedirs(self.paths.get("raw", "data/raw"), exist_ok=True)
        os.makedirs(self.paths.get("analytics", "data/analytics"), exist_ok=True)

        logging.basicConfig(
            filename=self.logging_cfg["file"],
            level=getattr(logging, self.logging_cfg["level"]),
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        self.conf_map_file = os.path.join(self.paths["raw"], "team_conference_map.csv")
        self.analytics_dir = self.paths.get("analytics", "data/analytics")

    def _save_outputs(self, df: pd.DataFrame, name: str):
        df.to_parquet(os.path.join(self.analytics_dir, f"{name}.parquet"), index=False)
        df.to_csv(os.path.join(self.analytics_dir, f"{name}.csv"), index=False)

    def _log_to_mlflow(self, df: pd.DataFrame, name: str, metrics: dict = None, plot_func=None):
        if not self.mlflow_cfg.get("enabled", False) or df.empty:
            return
        configure_mlflow(experiment_name=self.mlflow_cfg.get("experiment", "nba_analytics"))
        run_name = f"{self.mlflow_cfg.get('run_prefix','analytics_')}{name}"
        with start_run_with_metadata(run_name):
            csv_path = os.path.join(self.analytics_dir, f"{name}.csv")
            mlflow.log_artifact(csv_path, artifact_path="analytics")
            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(f"{name}_{k}", v)
            if plot_func:
                fig = plot_func(df)
                plot_path = os.path.join(self.analytics_dir, f"{name}.png")
                fig.savefig(plot_path, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(plot_path, artifact_path="analytics")

    # -----------------------------
    # TEAM RANKINGS
    # -----------------------------
    def compute_team_rankings(self, history_file, top_n=6):
        if not os.path.exists(history_file):
            logging.error(f"History file not found: {history_file}")
            return pd.DataFrame()

        df = pd.read_parquet(history_file)
        df["correct_pred"] = (df["pred_label"] == df["win"]).astype(int)
        team_stats = df.groupby("TEAM_ID").agg(
            games_played=("GAME_ID", "count"),
            wins=("win", "sum"),
            avg_pred_proba=("pred_proba", "mean"),
            pred_accuracy=("correct_pred", "mean"),
        ).reset_index()
        team_stats["win_pct"] = team_stats["wins"] / team_stats["games_played"]

        if os.path.exists(self.conf_map_file):
            conf_map = pd.read_csv(self.conf_map_file)
            team_stats = team_stats.merge(conf_map, on="TEAM_ID", how="left")
        team_stats["Conference"] = team_stats.get("Conference", "Unknown")

        top_teams = team_stats.sort_values(["Conference", "win_pct"], ascending=[True, False])
        top_teams = top_teams.groupby("Conference").head(top_n)

        self._save_outputs(top_teams, "team_rankings")

        def plot_team_rankings(df):
            fig, ax = plt.subplots(figsize=(8, 5))
            df.groupby("Conference").apply(
                lambda g: g.plot.bar(x="TEAM_ID", y="win_pct", ax=ax, legend=False)
            )
            ax.set_title("Top Teams by Win %")
            ax.set_ylabel("Win Percentage")
            return fig

        self._log_to_mlflow(top_teams, "team_rankings",
                            metrics={"avg_win_pct": top_teams["win_pct"].mean()},
                            plot_func=plot_team_rankings)
        return top_teams

    # -----------------------------
    # PLAYER RANKINGS
    # -----------------------------
    def compute_player_rankings(self, player_box_file, top_n=6):
        if not os.path.exists(player_box_file):
            logging.error(f"Player box file not found: {player_box_file}")
            return pd.DataFrame()

        df = pd.read_parquet(player_box_file)
        player_stats = df.groupby("PLAYER_ID").agg(
            avg_points=("PTS", "mean"),
            games_played=("GAME_ID", "count"),
            prop_hit_rate=("PTS", lambda x: (x >= self.nba_cfg["players_min_points"]).mean()),
            points_std=("PTS", "std"),
        ).reset_index()

        if os.path.exists(self.conf_map_file):
            team_map = pd.read_csv(self.conf_map_file)
            player_team_map = df[["PLAYER_ID", "TEAM_ID"]].drop_duplicates()
            player_stats = player_stats.merge(player_team_map, on="PLAYER_ID", how="left")
            player_stats = player_stats.merge(team_map, on="TEAM_ID", how="left")
        player_stats["Conference"] = player_stats.get("Conference", "Unknown")

        top_players = player_stats.sort_values(["Conference", "avg_points"], ascending=[True, False])
        top_players = top_players.groupby("Conference").head(top_n)

        self._save_outputs(top_players, "player_rankings")

        def plot_player_rankings(df):
            fig, ax = plt.subplots(figsize=(8, 5))
            df.plot.bar(x="PLAYER_ID", y="avg_points", ax=ax, legend=False)
            ax.set_title("Top Players by Avg Points")
            ax.set_ylabel("Points per Game")
            return fig

        self._log_to_mlflow(top_players, "player_rankings",
                            metrics={"avg_points": top_players["avg_points"].mean()},
                            plot_func=plot_player_rankings)
        return top_players

    # -----------------------------
    # BETTING RECOMMENDATIONS
    # -----------------------------
    def betting_recommendations(self, team_rankings, threshold=0.55):
        bet_on = team_rankings[
            (team_rankings["win_pct"] > threshold) & (team_rankings["pred_accuracy"] > threshold)
        ]
        avoid = team_rankings[
            (team_rankings["win_pct"] < threshold) | (team_rankings["pred_accuracy"] < threshold)
        ]

        self._log_to_mlflow(bet_on, "bet_on", metrics={"count": len(bet_on)})
        self._log_to_mlflow(avoid, "avoid", metrics={"count": len(avoid)})
        return {"bet_on": bet_on, "avoid": avoid}

    # -----------------------------
    # WINNING STREAKS
    # -----------------------------
    def track_winning_streaks(self, history_file):
        if not os.path.exists(history_file):
            logging.error(f"History file not found: {history_file}")
            return pd.DataFrame()

        df = pd.read_parquet(history_file)
        streaks = []
        for team_id, group in df.groupby("TEAM_ID"):
            group = group.sort_values("prediction_date")
            current_streak = 0
            max_streak = 0
            for win in group["win"]:
                if win == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            streaks.append({
                "TEAM_ID": team_id,
                "max_streak": max_streak,
                "current_streak": current_streak,
            })

        streaks_df = pd.DataFrame(streaks)
        self._save_outputs(streaks_df, "winning_streaks")

        def plot_streaks(df):
            fig, ax = plt.subplots(figsize=(8, 5))
            df.plot.bar(x="TEAM_ID", y="current_streak", ax=ax, legend=False)
            ax.set_title("Current Winning Streaks")
            ax.set_ylabel("Games")
            return fig

        self._log_to_mlflow(streaks_df, "winning_streaks",
                            metrics={"avg_max_streak": streaks_df["max_streak"].mean()},
                            plot_func=plot_streaks)
        return streaks_df
