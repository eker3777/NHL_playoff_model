"""
Utility functions for creating and managing team matchups.
"""

import pandas as pd
import numpy as np
import streamlit as st
from itertools import combinations
import streamlit_app.utils.model_utils as model_utils
import time

def create_matchup_data(top_seed, bottom_seed, team_data):
    """Create matchup data for model input
    
    Args:
        top_seed: Dictionary with top seed team information
        bottom_seed: Dictionary with bottom seed team information
        team_data: DataFrame with team statistics
        
    Returns:
        DataFrame: Matchup data for prediction
    """
    # Create a single row DataFrame for this matchup
    matchup_data = {}
    
    # Base matchup information
    matchup_data['top_seed_abbrev'] = top_seed.get('teamAbbrev', '')
    matchup_data['bottom_seed_abbrev'] = bottom_seed.get('teamAbbrev', '')
    
    # Get team data for each team
    if 'teamAbbrev' in team_data.columns:
        top_team_filter = team_data['teamAbbrev'] == matchup_data['top_seed_abbrev']
        bottom_team_filter = team_data['teamAbbrev'] == matchup_data['bottom_seed_abbrev']
        
        if sum(top_team_filter) > 0 and sum(bottom_team_filter) > 0:
            top_seed_data = team_data[top_team_filter].iloc[0]
            bottom_seed_data = team_data[bottom_team_filter].iloc[0]
            
            # Get the points difference for basic prediction
            if 'points' in top_seed_data and 'points' in bottom_seed_data:
                matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
            
            # Feature columns to use for prediction
            from streamlit_app.config import CRITICAL_FEATURES
            
            # Log which features are available
            available_features = [col for col in CRITICAL_FEATURES if col in top_seed_data and col in bottom_seed_data]
            print(f"Available features for matchup: {len(available_features)}/{len(CRITICAL_FEATURES)}")
            
            # Add features for each team if available
            for col in CRITICAL_FEATURES:
                if col in top_seed_data and col in bottom_seed_data:
                    # Only add if both values are not NaN
                    if pd.notna(top_seed_data[col]) and pd.notna(bottom_seed_data[col]):
                        matchup_data[f"{col}_top"] = top_seed_data[col]
                        matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                        matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
    
    matchup_df = pd.DataFrame([matchup_data])
    
    # Validate matchup data before returning
    from streamlit_app.utils.data_validation import validate_and_fix
    
    if isinstance(matchup_df, pd.DataFrame) and not matchup_df.empty:
        matchup_df, validation_report = validate_and_fix(
            matchup_df, 'pre-model', 
            ['top_seed_abbrev', 'bottom_seed_abbrev']
        )
    
    return matchup_df

def generate_all_matchup_combinations(team_data):
    """Generate all possible matchup combinations between teams
    
    Args:
        team_data: DataFrame with team data
        
    Returns:
        dict: Dictionary of all possible matchups
    """
    # Verify team_data has required columns
    if 'teamAbbrev' not in team_data.columns:
        print("Error: team_data missing teamAbbrev column")
        return {}
    
    all_teams = team_data['teamAbbrev'].unique()
    matchup_dict = {}
    
    # Generate all combinations of teams
    team_pairs = list(combinations(all_teams, 2))
    
    for team1, team2 in team_pairs:
        # Create both directions of matchup
        team1_data = team_data[team_data['teamAbbrev'] == team1]
        team2_data = team_data[team_data['teamAbbrev'] == team2]
        
        if team1_data.empty or team2_data.empty:
            continue
            
        # Create team dictionaries
        team1_dict = team1_data.iloc[0].to_dict()
        team2_dict = team2_data.iloc[0].to_dict()
        
        # Create matchup with team1 as top seed
        key1 = f"{team1}_{team2}"
        matchup_dict[key1] = create_matchup_data(team1_dict, team2_dict, team_data)
        
        # Create matchup with team2 as top seed
        key2 = f"{team2}_{team1}"
        matchup_dict[key2] = create_matchup_data(team2_dict, team1_dict, team_data)
    
    return matchup_dict

def get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data=None):
    """Get matchup data from pre-calculated dictionary or create it
    
    Args:
        top_seed: Dictionary or string with top seed team information
        bottom_seed: Dictionary or string with bottom seed team information
        matchup_dict: Dictionary of pre-calculated matchups
        team_data: DataFrame with team data (optional if matchup_dict has the matchup)
        
    Returns:
        DataFrame: Matchup data for prediction
    """
    # Extract team abbreviations
    if isinstance(top_seed, dict):
        top_abbrev = top_seed.get('teamAbbrev', '')
    else:
        top_abbrev = str(top_seed)
        
    if isinstance(bottom_seed, dict):
        bottom_abbrev = bottom_seed.get('teamAbbrev', '')
    else:
        bottom_abbrev = str(bottom_seed)
    
    # Check if matchup exists in pre-calculated dictionary
    key = f"{top_abbrev}_{bottom_abbrev}"
    if key in matchup_dict and not matchup_dict[key].empty:
        return matchup_dict[key]
    
    # If not found and team_data is provided, create the matchup
    if team_data is not None:
        if isinstance(top_seed, str) and isinstance(bottom_seed, str):
            # Convert strings to team dictionaries
            top_team_data = team_data[team_data['teamAbbrev'] == top_abbrev]
            bottom_team_data = team_data[team_data['teamAbbrev'] == bottom_abbrev]
            
            if not top_team_data.empty and not bottom_team_data.empty:
                top_seed = top_team_data.iloc[0].to_dict()
                bottom_seed = bottom_team_data.iloc[0].to_dict()
            else:
                return pd.DataFrame()  # Teams not found
        
        return create_matchup_data(top_seed, bottom_seed, team_data)
    
    # If matchup not found and no team_data provided
    return pd.DataFrame()

def analyze_matchup_data(matchup_dict):
    """Analyze matchup data quality and availability
    
    Args:
        matchup_dict: Dictionary of matchups
        
    Returns:
        dict: Analysis of matchup data
    """
    analysis = {
        "total_matchups": len(matchup_dict),
        "feature_counts": {},
        "nan_counts": {},
        "teams": set()
    }
    
    # Collect all unique features across matchups
    all_features = set()
    for key, matchup_df in matchup_dict.items():
        if matchup_df is not None and not matchup_df.empty:
            diff_features = [col for col in matchup_df.columns if '_diff' in col]
            all_features.update(diff_features)
            
            # Extract team abbreviations
            if 'top_seed_abbrev' in matchup_df.columns:
                analysis["teams"].add(matchup_df['top_seed_abbrev'].iloc[0])
            if 'bottom_seed_abbrev' in matchup_df.columns:
                analysis["teams"].add(matchup_df['bottom_seed_abbrev'].iloc[0])
    
    # Initialize counters for each feature
    for feature in all_features:
        analysis["feature_counts"][feature] = 0
        analysis["nan_counts"][feature] = 0
    
    # Count features and NaN values
    for key, matchup_df in matchup_dict.items():
        if matchup_df is None or matchup_df.empty:
            continue
            
        for feature in all_features:
            if feature in matchup_df.columns:
                analysis["feature_counts"][feature] += 1
                
                # Count NaN values
                if matchup_df[feature].isna().any():
                    analysis["nan_counts"][feature] += 1
    
    # Convert teams to count
    analysis["team_count"] = len(analysis["teams"])
    analysis["teams"] = list(analysis["teams"])
    
    return analysis
