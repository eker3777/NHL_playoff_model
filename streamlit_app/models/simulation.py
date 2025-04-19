"""
NHL Playoff Model Simulation Module

This module contains functions for simulating playoff series and brackets.
"""

# Import constants from config instead of defining them here
from streamlit_app.config import (
    HOME_ICE_ADVANTAGE,
    SERIES_LENGTH_DISTRIBUTION
)

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import streamlit as st

# Import utility functions
from streamlit_app.utils.model_utils import predict_series_winner, create_matchup_data, load_models
from streamlit_app.utils.simulation_utils import generate_series_outcome
from streamlit_app.utils.cache_manager import get_eastern_time, cache_simulation_results, load_cached_simulation_results
from streamlit_app.utils.data_handlers import load_team_data
from streamlit_app.utils.matchup_utils import load_current_playoff_matchups

# Remove these hardcoded constants:
# HOME_ICE_ADVANTAGE = 0.039  # Standard home ice advantage (3.9%)
# SERIES_LENGTH_DISTRIBUTION = [0.140, 0.243, 0.336, 0.281]  # 4, 5, 6, 7 games

def run_playoff_simulations(model_folder, data_folder, force_refresh=False):
    """Run playoff simulations and cache the results
    
    Args:
        model_folder: Path to folder containing models
        data_folder: Path to folder containing data
        force_refresh: Whether to force a refresh regardless of cache status
        
    Returns:
        dict: Simulation results or None if simulation failed
    """
    # Check if we need to run simulations (only if data was refreshed or forced)
    refresh_needed = force_refresh
    
    # Check if we have cached results
    cached_results = load_cached_simulation_results(data_folder)
    
    if cached_results is not None and not refresh_needed:
        # Check if we've refreshed data since the last simulation
        if 'last_data_refresh' in st.session_state and 'last_simulation' in st.session_state:
            # Convert to same timezone for comparison if needed
            last_data_refresh = st.session_state.last_data_refresh
            last_simulation = st.session_state.last_simulation
            
            if last_data_refresh > last_simulation:
                refresh_needed = True
                print(f"Data refreshed at {last_data_refresh} after last simulation at {last_simulation}. Running new simulations.")
    else:
        # No cached results or refresh forced
        refresh_needed = True
    
    # If no refresh needed, return cached results
    if not refresh_needed and cached_results is not None:
        print("Using cached simulation results")
        return cached_results
    
    # Otherwise, run new simulations
    print("Running new playoff simulations...")
    
    # Load team data
    team_data = load_team_data(data_folder)
    if team_data.empty:
        print("Error: No team data available")
        return None
    
    # Load playoff matchups
    playoff_matchups = load_current_playoff_matchups(data_folder)
    if not playoff_matchups:
        print("Error: No playoff matchups available")
        return None
    
    # Load models
    models = load_models(model_folder)
    if not models:
        print("Error: No models available")
        return None
    
    # Run simulations
    with st.spinner("Running playoff simulations..."):
        try:
            # Run bracket simulation
            results = simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=1000)
            
            # Cache results
            cache_success = cache_simulation_results(results, data_folder)
            if cache_success:
                print("Successfully cached simulation results")
            else:
                print("Warning: Failed to cache simulation results")
            
            return results
        except Exception as e:
            print(f"Error running playoff simulations: {str(e)}")
            # Log the error
            error_log_path = os.path.join(data_folder, "error_log.txt")
            with open(error_log_path, 'a') as f:
                f.write(f"{get_eastern_time().isoformat()}: Simulation error: {str(e)}\n")
            return None

def simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=1000, detailed_tracking=True):
    """Simulate the entire playoff bracket to predict playoff outcomes
    
    Args:
        playoff_matchups: Dictionary of playoff matchups by conference
        team_data: DataFrame with team statistics
        models: Dictionary with loaded models
        n_simulations: Number of simulations to run
        detailed_tracking: Whether to track detailed matchup statistics
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Track advancement for each team
    team_advancement = {}
    
    # Track all possible matchups for each round
    round_matchups = {
        'round_2': {},  # Division Finals
        'conf_final': {},  # Conference Finals
        'final': {}  # Stanley Cup Final
    }
    
    # Initialize all playoff teams
    for conference, matchups in playoff_matchups.items():
        for series_id, matchup in matchups.items():
            top_seed = matchup['top_seed']['teamAbbrev']
            bottom_seed = matchup['bottom_seed']['teamAbbrev']
            
            # Initialize tracking for top seed
            if top_seed not in team_advancement:
                team_advancement[top_seed] = {
                    'round_1': 0,
                    'round_2': 0,
                    'conf_final': 0,
                    'final': 0,
                    'champion': 0,
                    'teamName': matchup['top_seed']['teamName'],
                    'total_games': 0
                }
            
            # Initialize tracking for bottom seed
            if bottom_seed not in team_advancement:
                team_advancement[bottom_seed] = {
                    'round_1': 0,
                    'round_2': 0,
                    'conf_final': 0,
                    'final': 0,
                    'champion': 0,
                    'teamName': matchup['bottom_seed']['teamName'],
                    'total_games': 0
                }
    
    # Store bracket results for tracking most common outcome
    bracket_results_count = {}
    
    # Run simulations
    for sim in range(n_simulations):
        if sim % 100 == 0 and sim > 0:
            print(f"Completed {sim} simulations...")
            
        # Track winners for each round
        round_1_winners = {}
        round_2_winners = {}
        conf_winners = {}
        stanley_cup_winner = None
        
        # Track this simulation's bracket
        current_bracket = [] if detailed_tracking else None
        
        # First round
        for conference, matchups in playoff_matchups.items():
            round_1_winners[conference] = {}
            
            for series_id, matchup in matchups.items():
                # Create matchup data
                matchup_df = create_matchup_data(matchup['top_seed'], matchup['bottom_seed'], team_data)
                
                # Get prediction for the series
                if not matchup_df.empty:
                    # Get raw win probability (no home ice advantage)
                    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                    
                    # Apply home ice advantage
                    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                    win_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Simulate series outcome
                    higher_seed_wins = np.random.random() < win_prob
                    
                    # Determine winner and loser
                    if higher_seed_wins:
                        winner = matchup['top_seed']
                        loser = matchup['bottom_seed']
                    else:
                        winner = matchup['bottom_seed']
                        loser = matchup['top_seed']
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
                    
                    # Track games played
                    team_advancement[winner['teamAbbrev']]['total_games'] += series_length
                    team_advancement[loser['teamAbbrev']]['total_games'] += series_length
                    
                    # Store winner for next round
                    round_1_winners[conference][series_id] = {
                        'team': winner,
                        'original_matchup': series_id
                    }
                    
                    # Record advancement
                    team_advancement[winner['teamAbbrev']]['round_1'] += 1
                    
                    # Add to current bracket if tracking is enabled
                    if detailed_tracking:
                        current_bracket.append(f"{winner['teamAbbrev']} over {loser['teamAbbrev']}")
        
        # Process Second round (Division Finals)
        for conference, r1_winners in round_1_winners.items():
            round_2_winners[conference] = {}
            
            # Extract division information from series_ids
            divisions = set()
            for key in r1_winners.keys():
                if '_' in key:  # Handle different series_id formats
                    div = key.split('_')[0]
                    divisions.add(div)
                elif len(key) >= 1:
                    divisions.add(key[0])
            
            # Process each division's matchup
            for division in divisions:
                # Find matchup winners from this division's first round
                div_winners = []
                
                for k, v in r1_winners.items():
                    if ('_' in k and k.split('_')[0] == division) or (k.startswith(division)):
                        div_winners.append(v)
                
                if len(div_winners) >= 2:  # Need at least two teams for a matchup
                    # Identify teams
                    team1 = div_winners[0]['team']
                    team2 = div_winners[1]['team']
                    
                    # Determine seeding based on division or wildcard rank
                    div_rank1 = team1.get('division_rank', 99)
                    div_rank2 = team2.get('division_rank', 99)
                    
                    if 'wildcard_rank' in team1:
                        div_rank1 = 3 + team1['wildcard_rank']
                    
                    if 'wildcard_rank' in team2:
                        div_rank2 = 3 + team2['wildcard_rank']
                    
                    # Determine seeds based on ranks
                    if div_rank1 <= div_rank2:  # Lower number is better rank
                        top_seed = team1
                        bottom_seed = team2
                    else:
                        top_seed = team2
                        bottom_seed = team1
                    
                    # Track this potential matchup if detailed tracking is enabled
                    if detailed_tracking:
                        matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                        if matchup_key not in round_matchups['round_2']:
                            round_matchups['round_2'][matchup_key] = {
                                'conference': conference,
                                'division': division,
                                'top_seed': top_seed['teamAbbrev'],
                                'bottom_seed': bottom_seed['teamAbbrev'],
                                'top_seed_name': top_seed['teamName'],
                                'bottom_seed_name': bottom_seed['teamName'],
                                'count': 0,
                                'top_seed_wins': 0
                            }
                        round_matchups['round_2'][matchup_key]['count'] += 1
                    
                    # Create matchup data
                    matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                    
                    # Get raw win probability (no home ice advantage)
                    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                    
                    # Apply home ice advantage
                    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                    win_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Simulate series outcome
                    higher_seed_wins = np.random.random() < win_prob
                    winner = top_seed if higher_seed_wins else bottom_seed
                    loser = bottom_seed if higher_seed_wins else top_seed
                    
                    # Record the win if detailed tracking is enabled
                    if detailed_tracking and higher_seed_wins:
                        matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                        round_matchups['round_2'][matchup_key]['top_seed_wins'] += 1
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
                    
                    # Track games played
                    team_advancement[winner['teamAbbrev']]['total_games'] += series_length
                    team_advancement[loser['teamAbbrev']]['total_games'] += series_length
                    
                    # Store winner for next round
                    round_2_winners[conference][f"{division}_final"] = {
                        'team': winner,
                        'division': division
                    }
                    
                    # Record advancement
                    team_advancement[winner['teamAbbrev']]['round_2'] += 1
                    
                    # Add to current bracket if detailed tracking is enabled
                    if detailed_tracking:
                        current_bracket.append(f"{winner['teamAbbrev']} over {loser['teamAbbrev']}")
        
        # Conference Finals
        for conference, r2_winners in round_2_winners.items():
            if len(r2_winners) >= 2:  # Make sure we have both divisional final results
                # Get the two division winners
                div_winners = list(r2_winners.values())
                
                if len(div_winners) == 2:
                    team1 = div_winners[0]['team']
                    team2 = div_winners[1]['team']
                    
                    # Determine seeding (similar to above)
                    div_rank1 = team1.get('division_rank', 99)
                    div_rank2 = team2.get('division_rank', 99)
                    
                    if 'wildcard_rank' in team1:
                        div_rank1 = 3 + team1['wildcard_rank']
                    
                    if 'wildcard_rank' in team2:
                        div_rank2 = 3 + team2['wildcard_rank']
                    
                    # Use points as tiebreaker if ranks are equal
                    if div_rank1 == div_rank2:
                        team1_data = team_data[team_data['teamAbbrev'] == team1['teamAbbrev']]
                        team2_data = team_data[team_data['teamAbbrev'] == team2['teamAbbrev']]
                        
                        points1 = team1_data['points'].iloc[0] if not team1_data.empty and 'points' in team1_data.columns else 0
                        points2 = team2_data['points'].iloc[0] if not team2_data.empty and 'points' in team2_data.columns else 0
                        
                        if points1 > points2:
                            top_seed = team1
                            bottom_seed = team2
                        else:
                            top_seed = team2
                            bottom_seed = team1
                    else:
                        top_seed = team1 if div_rank1 < div_rank2 else team2
                        bottom_seed = team2 if div_rank1 < div_rank2 else team1
                    
                    # Track this potential matchup if detailed tracking is enabled
                    if detailed_tracking:
                        matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                        if matchup_key not in round_matchups['conf_final']:
                            round_matchups['conf_final'][matchup_key] = {
                                'conference': conference,
                                'top_seed': top_seed['teamAbbrev'],
                                'bottom_seed': bottom_seed['teamAbbrev'],
                                'top_seed_name': top_seed['teamName'],
                                'bottom_seed_name': bottom_seed['teamName'],
                                'count': 0,
                                'top_seed_wins': 0
                            }
                        round_matchups['conf_final'][matchup_key]['count'] += 1
                    
                    # Create matchup data
                    matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                    
                    # Get raw win probability (no home ice advantage)
                    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                    
                    # Apply home ice advantage
                    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                    win_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Simulate series outcome
                    higher_seed_wins = np.random.random() < win_prob
                    winner = top_seed if higher_seed_wins else bottom_seed
                    loser = bottom_seed if higher_seed_wins else top_seed
                    
                    # Record the win if detailed tracking is enabled
                    if detailed_tracking and higher_seed_wins:
                        matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                        round_matchups['conf_final'][matchup_key]['top_seed_wins'] += 1
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
                    
                    # Track games played
                    team_advancement[winner['teamAbbrev']]['total_games'] += series_length
                    team_advancement[loser['teamAbbrev']]['total_games'] += series_length
                    
                    # Store winner for final
                    conf_winners[conference] = {
                        'team': winner,
                        'conference': conference
                    }
                    
                    # Record advancement
                    team_advancement[winner['teamAbbrev']]['conf_final'] += 1
                    
                    # Add to current bracket if detailed tracking is enabled
                    if detailed_tracking:
                        current_bracket.append(f"{winner['teamAbbrev']} over {loser['teamAbbrev']}")
        
        # Stanley Cup Final
        if len(conf_winners) == 2:
            # Get the two conference champions
            conf_champs = list(conf_winners.values())
            
            if len(conf_champs) == 2:
                team1 = conf_champs[0]['team']
                team2 = conf_champs[1]['team']
                
                # Determine home ice advantage based on regular season points
                team1_data = team_data[team_data['teamAbbrev'] == team1['teamAbbrev']]
                team2_data = team_data[team_data['teamAbbrev'] == team2['teamAbbrev']]
                
                points1 = team1_data['points'].iloc[0] if not team1_data.empty and 'points' in team1_data.columns else 0
                points2 = team2_data['points'].iloc[0] if not team2_data.empty and 'points' in team2_data.columns else 0
                
                if points1 >= points2:
                    top_seed = team1
                    bottom_seed = team2
                else:
                    top_seed = team2
                    bottom_seed = team1
                
                # Track this potential matchup if detailed tracking is enabled
                if detailed_tracking:
                    matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                    if matchup_key not in round_matchups['final']:
                        round_matchups['final'][matchup_key] = {
                            'top_seed': top_seed['teamAbbrev'],
                            'bottom_seed': bottom_seed['teamAbbrev'],
                            'top_seed_name': top_seed['teamName'],
                            'bottom_seed_name': bottom_seed['teamName'],
                            'count': 0,
                            'top_seed_wins': 0
                        }
                    round_matchups['final'][matchup_key]['count'] += 1
                
                # Create matchup data
                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                
                # Get raw win probability (no home ice advantage)
                ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                
                # Apply home ice advantage
                home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                win_prob = min(1.0, ensemble_prob + home_ice_boost)
                
                # Simulate series outcome
                higher_seed_wins = np.random.random() < win_prob
                winner = top_seed if higher_seed_wins else bottom_seed
                loser = bottom_seed if higher_seed_wins else top_seed
                
                # Record the win if detailed tracking is enabled
                if detailed_tracking and higher_seed_wins:
                    matchup_key = f"{top_seed['teamAbbrev']}_vs_{bottom_seed['teamAbbrev']}"
                    round_matchups['final'][matchup_key]['top_seed_wins'] += 1
                
                # Simulate series length based on historical distribution
                series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
                
                # Track games played
                team_advancement[winner['teamAbbrev']]['total_games'] += series_length
                team_advancement[loser['teamAbbrev']]['total_games'] += series_length
                
                # Record champion and finalist
                team_advancement[winner['teamAbbrev']]['champion'] += 1
                team_advancement[winner['teamAbbrev']]['final'] += 1
                team_advancement[loser['teamAbbrev']]['final'] += 1
                
                # Add to current bracket if detailed tracking is enabled
                if detailed_tracking:
                    current_bracket.append(f"{winner['teamAbbrev']} over {loser['teamAbbrev']}")
                    stanley_cup_winner = winner['teamAbbrev']
        
        # Store this bracket result if detailed tracking is enabled
        if detailed_tracking and current_bracket:
            bracket_key = "|".join(current_bracket)
            if bracket_key not in bracket_results_count:
                bracket_results_count[bracket_key] = 0
            bracket_results_count[bracket_key] += 1
    
    # Calculate advancement percentages and average games played
    results_df = pd.DataFrame()
    for team, rounds in team_advancement.items():
        team_row = {'teamAbbrev': team, 'teamName': rounds.get('teamName', team)}
        for round_name in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
            team_row[round_name] = rounds[round_name] / n_simulations
        
        # Calculate average games played when team makes the playoffs
        team_row['avg_games_played'] = rounds['total_games'] / n_simulations
        
        results_df = pd.concat([results_df, pd.DataFrame([team_row])], ignore_index=True)
    
    # Sort by championship probability
    results_df = results_df.sort_values('champion', ascending=False)
    
    # Prepare return values
    results = {
        'team_results': results_df,
        'n_simulations': n_simulations,
        'timestamp': get_eastern_time().isoformat()
    }
    
    # Add detailed tracking results if enabled
    if detailed_tracking:
        # Find the most common bracket result
        most_common_bracket = max(bracket_results_count.items(), key=lambda x: x[1]) if bracket_results_count else ("", 0)
        
        # Convert matchup counts to DataFrames
        round2_df = pd.DataFrame([
            {
                'conference': data['conference'],
                'division': data.get('division', 'N/A'),
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0,
                'top_seed': data['top_seed'],
                'bottom_seed': data['bottom_seed']
            }
            for data in round_matchups['round_2'].values()
        ]).sort_values('count', ascending=False) if round_matchups['round_2'] else pd.DataFrame()
        
        conf_final_df = pd.DataFrame([
            {
                'conference': data['conference'],
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0,
                'top_seed': data['top_seed'],
                'bottom_seed': data['bottom_seed']
            }
            for data in round_matchups['conf_final'].values()
        ]).sort_values('count', ascending=False) if round_matchups['conf_final'] else pd.DataFrame()
        
        final_df = pd.DataFrame([
            {
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0,
                'top_seed': data['top_seed'],
                'bottom_seed': data['bottom_seed']
            }
            for data in round_matchups['final'].values()
        ]).sort_values('count', ascending=False) if round_matchups['final'] else pd.DataFrame()
        
        results.update({
            'most_common_bracket': {
                'bracket': most_common_bracket[0].split('|') if most_common_bracket[0] else [],
                'count': most_common_bracket[1],
                'probability': most_common_bracket[1]/n_simulations*100
            },
            'round2_matchups': round2_df,
            'conf_final_matchups': conf_final_df,
            'final_matchups': final_df
        })
    
    return results

def predict_series_probability(top_seed, bottom_seed, team_data, models):
    """Predict the probability of a playoff series outcome
    
    Args:
        top_seed: Dictionary with top seed team information
        bottom_seed: Dictionary with bottom seed team information
        team_data: DataFrame with team statistics
        models: Dictionary with loaded models
        
    Returns:
        dict: Series prediction results
    """
    # Create matchup data
    matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
    
    # Get win probability without home ice advantage
    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
    
    # Apply home ice advantage
    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
    win_prob = min(1.0, ensemble_prob + home_ice_boost)
    
    # Generate series outcome distribution
    win_distribution = generate_series_outcome(win_prob)
    
    # Format results
    results = {
        'top_seed': top_seed.get('teamAbbrev', 'Unknown'),
        'bottom_seed': bottom_seed.get('teamAbbrev', 'Unknown'),
        'top_seed_name': top_seed.get('teamName', top_seed.get('teamAbbrev', 'Unknown')),
        'bottom_seed_name': bottom_seed.get('teamName', bottom_seed.get('teamAbbrev', 'Unknown')),
        'win_probability': win_prob,
        'win_distribution': win_distribution,
        'raw_probability': ensemble_prob,
        'home_ice_boost': home_ice_boost,
        'lr_probability': lr_prob,
        'xgb_probability': xgb_prob
    }
    
    return results

def simulate_single_bracket(playoff_matchups, team_data, models):
    """Simulate a single playoff bracket following NHL rules precisely.
    
    Args:
        playoff_matchups: Dictionary of first round matchups by conference
        team_data: DataFrame with team stats data
        models: Dictionary of trained models for predictions
        
    Returns:
        Dictionary containing simulation results
    """
    # Initialize result dictionary
    result = {
        "champion": None,
        "finalist": None,
        "conference_champions": {},
        "rounds": {
            "1": {},  # First round matchups and winners
            "2": {},  # Second round (Division Finals)
            "3": {},  # Conference Finals
            "4": {}   # Stanley Cup Final
        },
        "total_games": 0  # Track total number of playoff games
    }
    
    # Track round winners
    round_1_winners = {}
    round_2_winners = {}
    conference_winners = {}
    
    # First round
    for conference, matchups in playoff_matchups.items():
        round_1_winners[conference] = {}
        result["rounds"]["1"][conference] = {}
        
        for series_id, matchup in matchups.items():
            top_seed = matchup["top_seed"]
            bottom_seed = matchup["bottom_seed"]
            
            # Create matchup data - using directly imported function
            matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
            
            # Get raw prediction (without home ice advantage)
            ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
            
            # Apply home ice advantage boost explicitly
            home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
            win_prob = min(1.0, ensemble_prob + home_ice_boost)
            
            # Simulate the outcome using the boosted probability
            higher_seed_wins = np.random.random() < win_prob
            
            # Determine winner and loser
            winner = top_seed if higher_seed_wins else bottom_seed
            loser = bottom_seed if higher_seed_wins else top_seed
            
            # Simulate series length based on historical distribution
            series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
            result["total_games"] += series_length
            
            # Store result
            result["rounds"]["1"][conference][series_id] = {
                "winner": winner["teamAbbrev"],
                "loser": loser["teamAbbrev"],
                "games": series_length,
                "higher_seed_won": higher_seed_wins
            }
            
            # Store winner for next round
            round_1_winners[conference][series_id] = {
                "team": winner,
                "original_matchup": series_id,
                "division": series_id.split("_")[0] if "_" in series_id else None
            }
    
    # Second round - Division Finals
    # Process each conference separately
    for conference, r1_winners in round_1_winners.items():
        round_2_winners[conference] = {}
        result["rounds"]["2"][conference] = {}
        
        # NHL rule: Division winners play wildcard winners, 2nd and 3rd place teams play each other
        # We need to match winners based on the original bracket structure
        division_matchups = {}
        
        # Group the winners by division
        for series_id, winner_data in r1_winners.items():
            division = winner_data.get("division")
            if division:
                if division not in division_matchups:
                    division_matchups[division] = []
                division_matchups[division].append(winner_data)
        
        # Create the second round matchups - strictly following NHL bracket format
        for division, winners in division_matchups.items():
            if len(winners) >= 2:  # Need at least two teams for a matchup
                # Sort based on original seeding to maintain bracket integrity
                sorted_winners = sorted(winners, key=lambda x: x["original_matchup"])
                
                # In each division: 
                # - Division winner plays winner of 2-3 matchup from that division OR
                # - If division winner lost, the wildcard winner plays winner of 2-3 matchup
                team1 = sorted_winners[0]["team"]
                team2 = sorted_winners[1]["team"]
                
                # Determine seeding based on regular season record
                # Higher seed is determined by regular season points if ranks are equal
                team1_data = team_data[team_data["teamAbbrev"] == team1["teamAbbrev"]]
                team2_data = team_data[team_data["teamAbbrev"] == team2["teamAbbrev"]]
                
                points1 = team1_data["points"].iloc[0] if not team1_data.empty and "points" in team1_data.columns else 0
                points2 = team2_data["points"].iloc[0] if not team2_data.empty and "points" in team2_data.columns else 0
                
                # Higher div_rank means lower seed (e.g., div_rank 1 is better than div_rank 2)
                div_rank1 = team1.get("division_rank", 99)
                div_rank2 = team2.get("division_rank", 99)
                
                # Convert wildcard ranks to effective division ranks (worse than division ranks 1-3)
                if "wildcard_rank" in team1:
                    div_rank1 = 3 + team1["wildcard_rank"]
                if "wildcard_rank" in team2:
                    div_rank2 = 3 + team2["wildcard_rank"]
                
                # Determine top/bottom seeds
                if div_rank1 < div_rank2:
                    top_seed = team1
                    bottom_seed = team2
                elif div_rank1 > div_rank2:
                    top_seed = team2
                    bottom_seed = team1
                else:
                    # If ranks are equal, use points as tiebreaker
                    top_seed = team1 if points1 >= points2 else team2
                    bottom_seed = team2 if points1 >= points2 else team1
                
                # Create matchup data - using directly imported function
                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                
                # Get raw prediction (without home ice advantage)
                ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                
                # Apply home ice advantage
                home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                win_prob = min(1.0, ensemble_prob + home_ice_boost)
                
                # Simulate outcome
                higher_seed_wins = np.random.random() < win_prob
                winner = top_seed if higher_seed_wins else bottom_seed
                loser = bottom_seed if higher_seed_wins else top_seed
                
                # Simulate series length
                series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
                result["total_games"] += series_length
                
                # Store result
                series_id = f"{division}_final"
                result["rounds"]["2"][conference][series_id] = {
                    "winner": winner["teamAbbrev"],
                    "loser": loser["teamAbbrev"],
                    "games": series_length,
                    "higher_seed_won": higher_seed_wins
                }
                
                # Store winner for conference finals
                round_2_winners[conference][series_id] = {
                    "team": winner,
                    "division": division
                }
    
    # Conference Finals
    for conference, r2_winners in round_2_winners.items():
        if len(r2_winners) == 2:  # We need exactly two division winners
            result["rounds"]["3"][conference] = {}
            
            # Get the two division winners
            winners = list(r2_winners.values())
            team1 = winners[0]["team"]
            team2 = winners[1]["team"]
            
            # Determine seeding based on regular season record (same logic as round 2)
            team1_data = team_data[team_data["teamAbbrev"] == team1["teamAbbrev"]]
            team2_data = team_data[team_data["teamAbbrev"] == team2["teamAbbrev"]]
            
            points1 = team1_data["points"].iloc[0] if not team1_data.empty and "points" in team1_data.columns else 0
            points2 = team2_data["points"].iloc[0] if not team2_data.empty and "points" in team2_data.columns else 0
            
            # Preserve original seeding
            div_rank1 = team1.get("division_rank", 99)
            div_rank2 = team2.get("division_rank", 99)
            
            if "wildcard_rank" in team1:
                div_rank1 = 3 + team1["wildcard_rank"]
            if "wildcard_rank" in team2:
                div_rank2 = 3 + team2["wildcard_rank"]
            
            # Determine top/bottom seeds
            if div_rank1 < div_rank2:
                top_seed = team1
                bottom_seed = team2
            elif div_rank1 > div_rank2:
                top_seed = team2
                bottom_seed = team1
            else:
                # If ranks are equal, use points as tiebreaker
                top_seed = team1 if points1 >= points2 else team2
                bottom_seed = team2 if points1 >= points2 else team1
            
            # Create matchup data - using directly imported function
            matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
            
            # Get raw prediction (without home ice advantage)
            ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
            
            # Apply home ice advantage
            home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
            win_prob = min(1.0, ensemble_prob + home_ice_boost)
            
            # Simulate outcome
            higher_seed_wins = np.random.random() < win_prob
            winner = top_seed if higher_seed_wins else bottom_seed
            loser = bottom_seed if higher_seed_wins else top_seed
            
            # Simulate series length
            series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
            result["total_games"] += series_length
            
            # Store result
            series_id = f"{conference}_final"
            result["rounds"]["3"][conference] = {
                "winner": winner["teamAbbrev"],
                "loser": loser["teamAbbrev"],
                "games": series_length,
                "higher_seed_won": higher_seed_wins
            }
            
            # Store conference champion
            conference_winners[conference] = winner
            result["conference_champions"][conference] = winner["teamAbbrev"]
    
    # Stanley Cup Final
    if len(conference_winners) == 2:
        result["rounds"]["4"] = {}
        
        # Get teams from each conference
        conferences = list(conference_winners.keys())
        team1 = conference_winners[conferences[0]]
        team2 = conference_winners[conferences[1]]
        
        # NHL rule: Team with better regular season record gets home ice advantage
        team1_data = team_data[team_data["teamAbbrev"] == team1["teamAbbrev"]]
        team2_data = team_data[team_data["teamAbbrev"] == team2["teamAbbrev"]]
        
        points1 = team1_data["points"].iloc[0] if not team1_data.empty and "points" in team1_data.columns else 0
        points2 = team2_data["points"].iloc[0] if not team2_data.empty and "points" in team2_data.columns else 0
        
        # Higher points gets home ice advantage
        if points1 >= points2:
            top_seed = team1
            bottom_seed = team2
        else:
            top_seed = team2
            bottom_seed = team1
        
        # Create matchup data - using directly imported function
        matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
        
        # Get raw prediction (without home ice advantage)
        ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
        
        # Apply home ice advantage
        home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
        win_prob = min(1.0, ensemble_prob + home_ice_boost)
        
        # Simulate outcome
        higher_seed_wins = np.random.random() < win_prob
        winner = top_seed if higher_seed_wins else bottom_seed
        loser = bottom_seed if higher_seed_wins else top_seed
        
        # Simulate series length
        series_length = np.random.choice([4, 5, 6, 7], p=SERIES_LENGTH_DISTRIBUTION)
        result["total_games"] += series_length
        
        # Store result
        series_id = "stanley_cup_final"
        result["rounds"]["4"][series_id] = {
            "winner": winner["teamAbbrev"],
            "loser": loser["teamAbbrev"],
            "games": series_length,
            "higher_seed_won": higher_seed_wins
        }
        
        # Store champion and finalist
        result["champion"] = winner["teamAbbrev"]
        result["finalist"] = loser["teamAbbrev"]
    
    return result

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

