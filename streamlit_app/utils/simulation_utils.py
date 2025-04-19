"""
Utility functions for NHL playoff simulations
"""

import numpy as np
import pandas as pd
import streamlit as st

# Import constants from central config
from streamlit_app.config import (
    HOME_ICE_ADVANTAGE, 
    SERIES_LENGTH_DISTRIBUTION
)

def get_series_schedule(series_length):
    """Get the home/away schedule for a playoff series of a given length.
    
    In the NHL playoff format:
    - Higher seed gets home games 1, 2, 5, 7
    - Lower seed gets home games 3, 4, 6
    
    Args:
        series_length: Number of games in the series (4-7)
        
    Returns:
        list: Schedule of home/away games from higher seed perspective
             (1 = home game, 0 = away game)
    """
    # NHL playoff format: 2-2-1-1-1
    # Higher seed (home): Games 1, 2, 5, 7
    # Lower seed (away): Games 3, 4, 6
    base_schedule = [1, 1, 0, 0, 1, 0, 1]
    
    # Return only the games that are played
    return base_schedule[:series_length]

def simulate_series_length():
    """Simulate a series length based on historical NHL data.
    
    Returns:
        int: Number of games in the series (4, 5, 6, or 7)
    """
    return np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)

def determine_top_seed(team1, team2, team_data):
    """Determine which team is the higher seed based on NHL rules.
    
    Args:
        team1: First team dictionary
        team2: Second team dictionary
        team_data: DataFrame with team stats
        
    Returns:
        tuple: (top_seed, bottom_seed)
    """
    # Extract division ranks (or wildcard ranks)
    div_rank1 = team1.get("division_rank", 99)
    div_rank2 = team2.get("division_rank", 99)
    
    # Convert wildcard ranks to effective division ranks (worse than division ranks 1-3)
    if "wildcard_rank" in team1:
        div_rank1 = 3 + team1["wildcard_rank"]
    if "wildcard_rank" in team2:
        div_rank2 = 3 + team2["wildcard_rank"]
    
    # Get points for tiebreaker
    team1_data = team_data[team_data["teamAbbrev"] == team1["teamAbbrev"]]
    team2_data = team_data[team_data["teamAbbrev"] == team2["teamAbbrev"]]
    
    points1 = team1_data["points"].iloc[0] if not team1_data.empty and "points" in team1_data.columns else 0
    points2 = team2_data["points"].iloc[0] if not team2_data.empty and "points" in team2_data.columns else 0
    
    # Determine top/bottom seeds
    if div_rank1 < div_rank2:
        return team1, team2
    elif div_rank1 > div_rank2:
        return team2, team1
    else:
        # If ranks are equal, use points as tiebreaker
        return (team1, team2) if points1 >= points2 else (team2, team1)

def generate_series_outcome(win_probability, n_simulations=1000):
    """Generate series outcome distribution based on win probability."""
    # Create outcome distribution dictionary for all possible outcomes
    outcome_distribution = {
        # Higher seed wins
        '4-0': SERIES_LENGTH_DISTRIBUTION[0] * win_probability,
        '4-1': SERIES_LENGTH_DISTRIBUTION[1] * win_probability,
        '4-2': SERIES_LENGTH_DISTRIBUTION[2] * win_probability, 
        '4-3': SERIES_LENGTH_DISTRIBUTION[3] * win_probability,
        
        # Lower seed wins
        '0-4': SERIES_LENGTH_DISTRIBUTION[0] * (1 - win_probability),
        '1-4': SERIES_LENGTH_DISTRIBUTION[1] * (1 - win_probability),
        '2-4': SERIES_LENGTH_DISTRIBUTION[2] * (1 - win_probability),
        '3-4': SERIES_LENGTH_DISTRIBUTION[3] * (1 - win_probability)
    }
    
    return outcome_distribution

def predict_series(matchup_df, models, n_simulations=1000):
    """Predict the outcome of a playoff series."""
    # If there's any hardcoded home ice advantage, replace it
    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
    
    # ...existing code...
