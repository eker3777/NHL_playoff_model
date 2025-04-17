import pandas as pd
import numpy as np
from itertools import combinations
import model_utils
import time

def generate_all_matchup_combinations(team_data):
    """
    Pre-calculate all possible matchup combinations between teams
    
    Args:
        team_data (DataFrame): DataFrame containing team statistics
        
    Returns:
        dict: Dictionary with matchup data for all team combinations
    """
    if team_data is None or team_data.empty:
        print("No team data available to generate matchup combinations")
        return {}

    # Get list of teams
    team_abbrevs = team_data['teamAbbrev'].unique().tolist()
    team_count = len(team_abbrevs)
    
    print(f"Generating matchup combinations for {team_count} teams...")
    
    # Dictionary to store all matchups
    matchup_dict = {}
    
    # Track progress
    total_combinations = team_count * (team_count - 1) // 2
    start_time = time.time()
    combination_count = 0
    
    # Generate all combinations
    for i, team1 in enumerate(team_abbrevs):
        for team2 in team_abbrevs[i+1:]:
            # Create key for both directions
            key1 = f"{team1}_vs_{team2}"
            key2 = f"{team2}_vs_{team1}"
            
            # Get team data
            team1_data = team_data[team_data['teamAbbrev'] == team1]
            team2_data = team_data[team_data['teamAbbrev'] == team2]
            
            if len(team1_data) > 0 and len(team2_data) > 0:
                # Create matchup data for both directions
                # Team 1 as home team
                team1_dict = {
                    'teamAbbrev': team1,
                    'teamName': team1_data['teamName'].iloc[0],
                    'division_rank': 1  # Placeholder for rank
                }
                
                team2_dict = {
                    'teamAbbrev': team2,
                    'teamName': team2_data['teamName'].iloc[0],
                    'division_rank': 2  # Placeholder for rank
                }
                
                # Generate matchup data both ways
                matchup_df1 = model_utils.create_matchup_data(team1_dict, team2_dict, team_data)
                matchup_df2 = model_utils.create_matchup_data(team2_dict, team1_dict, team_data)
                
                # Store in dictionary
                matchup_dict[key1] = matchup_df1
                matchup_dict[key2] = matchup_df2
            
            # Update progress counter
            combination_count += 1
            if combination_count % 50 == 0 or combination_count == total_combinations:
                elapsed = time.time() - start_time
                print(f"Generated {combination_count}/{total_combinations} matchups ({(combination_count/total_combinations*100):.1f}%) in {elapsed:.1f}s")
    
    print(f"Successfully created {len(matchup_dict)} matchup combinations")
    return matchup_dict

def get_matchup_data(top_seed, bottom_seed, matchup_dict=None, team_data=None):
    """
    Get matchup data for a specific matchup, either from pre-calculated dictionary or by creating it
    
    Args:
        top_seed (dict): Higher seed team information
        bottom_seed (dict): Lower seed team information
        matchup_dict (dict): Pre-calculated matchup dictionary
        team_data (DataFrame): Team data for creating matchup if not in dictionary
        
    Returns:
        DataFrame: Matchup data
    """
    top_abbrev = top_seed['teamAbbrev']
    bottom_abbrev = bottom_seed['teamAbbrev']
    key = f"{top_abbrev}_vs_{bottom_abbrev}"
    
    # First try to get from dictionary
    if matchup_dict is not None and key in matchup_dict:
        return matchup_dict[key]
    
    # If not found in dictionary, create it
    if team_data is not None:
        # Use the model_utils.create_matchup_data function
        return model_utils.create_matchup_data(top_seed, bottom_seed, team_data)
    
    # If we reach here, we can't create the matchup
    print(f"WARNING: Unable to create matchup data for {top_abbrev} vs {bottom_abbrev}")
    return None

def analyze_matchup_data(matchup_dict):
    """
    Analyze the quality of matchup data
    
    Args:
        matchup_dict (dict): Dictionary with matchup data
        
    Returns:
        dict: Analysis results
    """
    if not matchup_dict:
        return {"error": "No matchup data provided"}
    
    results = {
        "total_matchups": len(matchup_dict),
        "feature_counts": {},
        "nan_counts": {},
        "feature_stats": {}
    }
    
    # Sample a matchup for feature list
    sample_df = next(iter(matchup_dict.values()))
    diff_features = [col for col in sample_df.columns if col.endswith('_diff')]
    
    # Count features available in each matchup
    for feature in diff_features:
        results["feature_counts"][feature] = 0
        results["nan_counts"][feature] = 0
    
    # Analyze each matchup
    for key, matchup_df in matchup_dict.items():
        # Check which features exist
        for feature in diff_features:
            if feature in matchup_df.columns:
                results["feature_counts"][feature] += 1
                # Check for NaN values
                if matchup_df[feature].isna().any():
                    results["nan_counts"][feature] += 1
    
    # Calculate statistics for each feature
    for feature in diff_features:
        values = []
        for matchup_df in matchup_dict.values():
            if feature in matchup_df.columns and not matchup_df[feature].isna().any():
                values.append(matchup_df[feature].values[0])
        
        if values:
            results["feature_stats"][feature] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
    
    return results
