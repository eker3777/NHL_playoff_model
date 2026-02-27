"""
Minimal 2026-focused pipeline entrypoint.

Refactors the existing load/combine process into staged outputs:
- data/raw
- data/processed
- data/final
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import config
from core.data_loader import NHLDataLoader
from main import combine_team_data


def run_v2_data_pipeline(season: Optional[int] = None) -> Path:
    """Run extraction + cleaning flow and return final dataset path."""
    target_season = season or config.current_season
    season_str = f"{target_season}{target_season + 1}"

    raw_dir = Path(config.RAW_DATA_DIR)
    processed_dir = Path(config.PROCESSED_DATA_DIR)
    final_dir = Path(config.FINAL_DATA_DIR)

    loader = NHLDataLoader(str(raw_dir), target_season)

    standings_raw = loader.get_standings_data()
    stats_raw = loader.get_team_stats_data(target_season)
    advanced_df = loader.get_advanced_stats_data(target_season)

    standings_df = loader.process_standings_data(standings_raw, season_str) if standings_raw else None
    stats_df = loader.process_team_stats_data(stats_raw) if stats_raw else None

    if standings_df is not None:
        standings_df.to_csv(processed_dir / f"standings_processed_{season_str}.csv", index=False)
    if stats_df is not None:
        stats_df.to_csv(processed_dir / f"stats_processed_{season_str}.csv", index=False)
    if advanced_df is not None and not advanced_df.empty:
        advanced_df.to_csv(processed_dir / f"advanced_processed_{season_str}.csv", index=False)

    team_data = combine_team_data(standings_df, stats_df, advanced_df, season_str)
    final_path = final_dir / f"team_data_modeling_{season_str}.csv"
    team_data.to_csv(final_path, index=False)
    return final_path


if __name__ == "__main__":
    output = run_v2_data_pipeline()
    print(f"Saved final v2 dataset: {output}")
