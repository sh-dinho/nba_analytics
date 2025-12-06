# ============================================================
# File: core/config_loader.py
# Purpose: Load and validate TOML configuration for NBA pipeline
#          - Handles reading and parsing of the TOML config file
#          - Validates configuration blocks, including season data
#          - Ensures current season configuration is available
#          - Provides utility to build a data URL for fetching external data
# ============================================================

import toml
import re
from pathlib import Path
from typing import Dict, Any
from datetime import date, datetime


class ConfigLoader:
    """
    ConfigLoader handles the loading, validation, and auto-generation of configuration data
    from the TOML file for the NBA pipeline.

    Key functionality includes:
    - Loading the config file and parsing the TOML format.
    - Validating season data to ensure all required fields are present and correct.
    - Ensuring the current season block is available in the config and auto-generating it if missing.
    - Constructing data URLs for API calls based on season information.
    """
    
    REQUIRED_SEASON_KEYS = {
        "start_date", "end_date",
        "start_year", "end_year",
        "season_label"
    }

    def __init__(self, config_path: str = "config.toml"):
        """
        Initializes the ConfigLoader and loads the configuration from a TOML file.

        Args:
        - config_path (str): Path to the TOML configuration file. Defaults to 'config.toml'.

        Raises:
        - FileNotFoundError: If the config file does not exist.
        - ValueError: If the config file cannot be parsed.
        """
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
        """
        Retrieves the data for a given section from the config.

        Args:
        - section (str): The name of the section to retrieve.

        Returns:
        - dict: The section data or an empty dictionary if not found.
        """
        return self.config.get(section, {})

    def get_season(self, section: str, season: str) -> Dict[str, Any]:
        """
        Retrieves the season block for a specific season under a section.

        Args:
        - section (str): The section (e.g., 'get-data', 'create-games') containing the season data.
        - season (str): The season label (e.g., '2025-2026').

        Returns:
        - dict: The season data.

        Raises:
        - KeyError: If the season is not found under the specified section.
        """
        block = self.get_section(section).get(season)
        if block is None:
            raise KeyError(f"Season '{season}' not found under section '{section}'")
        return block

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_season(self, season_data: Dict[str, Any]) -> bool:
        """
        Validates the provided season data to ensure required fields are correct.

        Args:
        - season_data (dict): The season data to validate.

        Returns:
        - bool: True if the season data is valid, False otherwise.
        
        Raises:
        - ValueError: If required keys are missing or data is invalid.
        """
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
        Ensures the current season blocks exist in config.toml. 
        If missing, auto-generates blocks under [get-data], [get-odds-data], [create-games].

        Args:
        - None
        
        Returns:
        - str: The current season label in 'YYYY-YY' format (e.g., '2025-26').

        Logs events to 'pipeline.log' and prints a warning when blocks are auto-generated.
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

        # Auto-generate missing sections for the current season
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

        # Write the updated config back to the file
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
        """
        Builds a URL for fetching data based on the season's start and end dates.

        Args:
        - season_data (dict): The season data containing start_date, end_date, and other details.

        Returns:
        - str: The constructed URL based on the provided template in config.toml.
        
        Raises:
        - ValueError: If the URL template is missing or dates are invalid.
        """
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
