import toml
import re
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self.config = toml.load(self.config_path)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a section (e.g., get-data, get-odds-data, create-games)."""
        return self.config.get(section, {})

    def get_season(self, section: str, season: str) -> Dict[str, Any]:
        """Return a specific season block."""
        return self.get_section(section).get(season, {})

    def validate_season(self, season_data: Dict[str, Any]) -> bool:
        """Validate a season block for consistency."""
        try:
            start_date = season_data["start_date"]
            end_date = season_data["end_date"]
            start_year = int(season_data["start_year"])
            end_year = int(season_data["end_year"])
            season_label = season_data["season_label"]

            # Ensure season_label matches YYYY-YY format
            if not re.match(r"^\d{4}-\d{2}$", season_label):
                raise ValueError(f"Invalid season_label: {season_label}")

            # Ensure chronological consistency
            if start_year > end_year:
                raise ValueError(f"start_year > end_year in {season_label}")
            if start_date >= end_date:
                raise ValueError(f"start_date >= end_date in {season_label}")

            return True
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    def build_data_url(self, season_data: Dict[str, Any]) -> str:
        """Construct NBA stats API URL using season data."""
        url_template = self.config.get("data_url")
        if not url_template:
            raise ValueError("Missing data_url in config.toml")

        # Example: substitute placeholders with season values
        return url_template.format(
            season_data["start_date"].split("-")[1],  # month
            season_data["end_date"].split("-")[2],    # day
            season_data["start_date"].split("-")[0],  # year
            season_data["end_date"].split("-")[0],    # year
            season_data["season_label"]
        )

if __name__ == "__main__":
    loader = ConfigLoader("config.toml")
    season = loader.get_season("get-data", "2023-24")
    if loader.validate_season(season):
        url = loader.build_data_url(season)
        print(f"✅ URL for {season['season_label']}: {url}")
