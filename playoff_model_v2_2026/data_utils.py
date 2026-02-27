"""Refactored data combination utilities for playoff_model_v2_2026."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def combine_team_data(
    standings_df: pd.DataFrame | None,
    stats_df: pd.DataFrame | None,
    advanced_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine standings, team stats, and advanced stats into one table."""
    if standings_df is None or standings_df.empty:
        logger.warning("No standings data available")
        return pd.DataFrame()

    team_data = standings_df.copy()

    if stats_df is not None and not stats_df.empty:
        if "season" in stats_df.columns:
            stats_df["season"] = stats_df["season"].astype(str)
        team_data["season"] = team_data["season"].astype(str)
        team_data = pd.merge(
            team_data,
            stats_df,
            on=["season", "teamName"],
            how="left",
            suffixes=("", "_stats"),
        )

    if advanced_df is not None and not advanced_df.empty:
        advanced_df["season"] = advanced_df["season"].astype(str)
        team_data = pd.merge(
            team_data,
            advanced_df,
            on=["season", "teamName"],
            how="left",
            suffixes=("", "_advanced"),
        )

    return team_data
