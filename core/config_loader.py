# ============================================================
# File: core/config_loader.py
# Purpose: Load and validate TOML configuration for NBA pipeline
# ============================================================

import toml
import re
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    REQUIRED_SEASON_KEYS = {
        "start_date", "end_date",
        "start_year", "end_year",
        "season_label"
    }

    def __init__(self, config_path: str = "config.toml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            self.config = toml.load(self.config_path)
        except Exception as e:
            raise ValueError(f"Failed to parse TOML config: {e}")

    # ---------------------------------------------------------
    # Section helpers
    # ---------------------------------------------------------

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a section (e.g., 'get-data', 'get-odds-data', 'create-games')."""
        return self.config.get(section, {})

    def get_season(self, section: str, season: str) -> Dict[str, Any]:
        """Return a specific season block under a given section."""
        block = self.get_section(section).get(season)
        if block is None:
            raise KeyError(f"Season '{season}' not found under section '{section}'")
        return block

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_season(self, season_data: Dict[str, Any]) -> bool:
        """Validate a season block for consistency and correctness."""
        missing = self.REQUIRED_SEASON_KEYS - season_data.keys()
        if missing:
            raise ValueError(f"Season block missing required fields: {missing}")

        try:
            start_date = season_data["start_date"]
            end_date = season_data["end_date"]
            start_year = int(season_data["start_year"])
            end_year = int(season_data["end_year"])
            season_label = season_data["season_label"]

            # Validate YYYY-MM-DD format
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_date):
                raise ValueError(f"Invalid start_date format '{start_date}'")
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", end_date):
                raise ValueError(f"Invalid end_date format '{end_date}'")

            # Validate season label YYYY-YY
            if not re.match(r"^\d{4}-\d{2}$", season_label):
                raise ValueError(f"Invalid season_label format '{season_label}'")

            # Chronological consistency
            if start_year > end_year:
                raise ValueError(f"start_year > end_year for '{season_label}'")
            if start_date >= end_date:
                raise ValueError(f"start_date >= end_date for '{season_label}'")

            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    # ---------------------------------------------------------
    # URL Builder
    # ---------------------------------------------------------

    def build_data_url(self, season_data: Dict[str, Any]) -> str:
        """
        Construct an NBA API URL using named placeholders from config.toml.

        Example in config.toml:
        data_url = "https://api.nba.com/data?season={season_label}&start={start_date}&end={end_date}"
        """

        url_template = self.config.get("data_url")
        if not url_template:
            raise ValueError("Missing 'data_url' in config.toml")

        allowed_keys = {
            "season_label",
            "start_date", "end_date",
            "start_year", "end_year",
        }

        missing = [k for k in allowed_keys if k not in season_data]
        if missing:
            raise ValueError(f"URL cannot be built; missing keys in season block: {missing}")

        try:
            return url_template.format(
                season_label=season_data["season_label"],
                start_date=season_data["start_date"],
                end_date=season_data["end_date"],
                start_year=season_data["start_year"],
                end_year=season_data["end_year"],
            )
        except KeyError as e:
            raise ValueError(
                f"URL template references missing placeholder: {e}. "
                f"Available placeholders: {list(season_data.keys())}"
            )
        except Exception as e:
            raise ValueError(f"Failed to build URL: {e}")


# ---------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------

if __name__ == "__main__":
    loader = ConfigLoader("config.toml")
    season = loader.get_season("get-data", "2023-24")

    if loader.validate_season(season):
        url = loader.build_data_url(season)
        print(f"✅ URL for {season['season_label']}: {url}")
