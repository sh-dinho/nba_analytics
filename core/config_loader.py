# ============================================================
# File: core/config_loader.py
# Purpose: Load and validate TOML configuration for NBA pipeline
# ============================================================

import toml
import re
from pathlib import Path
from typing import Dict, Any
from datetime import date, datetime


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
        return self.config.get(section, {})

    def get_season(self, section: str, season: str) -> Dict[str, Any]:
        block = self.get_section(section).get(season)
        if block is None:
            raise KeyError(f"Season '{season}' not found under section '{section}'")
        return block

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_season(self, season_data: Dict[str, Any]) -> bool:
        missing = self.REQUIRED_SEASON_KEYS - season_data.keys()
        if missing:
            raise ValueError(f"Season block missing required fields: {missing}")

        try:
            start_date = season_data["start_date"]
            end_date = season_data["end_date"]
            start_year = int(season_data["start_year"])
            end_year = int(season_data["end_year"])
            season_label = season_data["season_label"]

            if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_date):
                raise ValueError(f"Invalid start_date format '{start_date}'")
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", end_date):
                raise ValueError(f"Invalid end_date format '{end_date}'")
            if not re.match(r"^\d{4}-\d{2}$", season_label):
                raise ValueError(f"Invalid season_label format '{season_label}'")

            if start_year > end_year:
                raise ValueError(f"start_year > end_year for '{season_label}'")
            if start_date >= end_date:
                raise ValueError(f"start_date >= end_date for '{season_label}'")

            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    # ---------------------------------------------------------
    # Auto‑generate current season blocks
    # ---------------------------------------------------------

    def ensure_current_season_blocks(self) -> str:
        """
        Ensure the current season blocks exist in config.toml.
        Auto‑generate [get-data], [get-odds-data], [create-games] if missing.
        Logs events to pipeline.log.
        """
        today = date.today()
        year = today.year
        month = today.month

        if month >= 10:
            start_year = year
            end_year = year + 1
        else:
            start_year = year - 1
            end_year = year

        season_label = f"{start_year}-{str(end_year)[-2:]}"
        start_date = f"{start_year}-10-21"
        end_date   = f"{end_year}-06-15"

        generated_sections = []

        for section in ["get-data", "get-odds-data", "create-games"]:
            section_data = self.config.setdefault(section, {})
            if season_label not in section_data:
                section_data[season_label] = {
                    "season_label": season_label,
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_year": str(start_year),
                    "end_year": str(end_year),
                }
                generated_sections.append(section)

        if generated_sections:
            with open(self.config_path, "w") as f:
                toml.dump(self.config, f)

            log_entry = (
                f"[{datetime.now().isoformat()}] "
                f"Auto‑generated season block '{season_label}' in sections: {', '.join(generated_sections)}\n"
            )
            with open("pipeline.log", "a") as log_file:
                log_file.write(log_entry)

            print(f"⚠️ Auto‑generated season block '{season_label}' under {', '.join(generated_sections)}")

        return season_label

    # ---------------------------------------------------------
    # URL Builder
    # ---------------------------------------------------------

    def build_data_url(self, season_data: Dict[str, Any]) -> str:
        url_template = self.config.get("data_url")
        if not url_template:
            raise ValueError("Missing 'data_url' in config.toml")

        from datetime import datetime as dt
        try:
            start_dt = dt.strptime(season_data["start_date"], "%Y-%m-%d")
            end_dt = dt.strptime(season_data["end_date"], "%Y-%m-%d")

            start_month = start_dt.strftime("%m")
            start_day   = start_dt.strftime("%d")
            end_month   = end_dt.strftime("%m")
            end_day     = end_dt.strftime("%d")
        except Exception as e:
            raise ValueError(f"❌ Could not parse dates for month/day: {e}")

        return url_template.format(
            season_label=season_data["season_label"],
            start_date=season_data["start_date"],
            end_date=season_data["end_date"],
            start_year=season_data["start_year"],
            end_year=season_data["end_year"],
            start_month=start_month,
            start_day=start_day,
            end_month=end_month,
            end_day=end_day
        )