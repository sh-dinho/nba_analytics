import pandas as pd
import pytest

from src.ingestion.schedule_scraper import (
    _parse_day_html,
    _infer_season_window,
    _date_range,
)


def test_infer_season_window():
    start, end = _infer_season_window(2024)
    assert start.year == 2024
    assert start.month == 10
    assert start.day == 1

    assert end.year == 2025
    assert end.month == 6
    assert end.day == 30


def test_date_range_inclusive():
    start = pd.Timestamp("2024-10-01")
    end = pd.Timestamp("2024-10-03")
    dates = _date_range(start, end)
    assert len(dates) == 3
    assert dates[0].isoformat() == "2024-10-01"
    assert dates[-1].isoformat() == "2024-10-03"


def test_parse_day_html_basic():
    # Minimal synthetic HTML similar to ESPN structure
    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>Matchup</th><th>Matchup</th><th>Time</th>
          </tr>
          <tr>
            <td>Boston Celtics</td>
            <td>Los Angeles Lakers</td>
            <td>7:30 PM</td>
          </tr>
          <tr>
            <td>Miami Heat</td>
            <td>New York Knicks</td>
            <td>8:00 PM</td>
          </tr>
        </table>
      </body>
    </html>
    """
    df = _parse_day_html(html, "20241015")
    assert not df.empty
    assert list(df.columns) == ["date", "away_team", "home_team", "game_time"]
    assert len(df) == 2
    assert (df["away_team"] == ["Boston Celtics", "Miami Heat"]).all()
    assert (df["home_team"] == ["Los Angeles Lakers", "New York Knicks"]).all()
    assert (df["game_time"] == ["7:30 PM", "8:00 PM"]).all()


def test_parse_day_html_no_tables():
    html = "<html><body><div>No schedule</div></body></html>"
    df = _parse_day_html(html, "20241015")
    assert df.empty
