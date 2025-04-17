import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Any
import model_utils
import os
import data_handlers

# Seed random number generator for reproducibility but with a time-based seed
np.random.seed(int(time.time()))

def should_refresh_simulations():
    """Check if we should refresh the daily simulations."""
    if 'last_simulation_refresh' not in st.session_state:
        return True
    
    # Get current time
    now = datetime.utcnow()
    
    # Check if it's been 24 hours since last refresh
    if st.session_state.last_simulation_refresh is None:
        return True
        
    if now - st.session_state.last_simulation_refresh > timedelta(hours=24):
        return True
    
    # If it's a new day and before 9 AM UTC (early morning), refresh the data
    if now.date() > st.session_state.last_simulation_refresh.date() and now.hour < 9:
        return True
    
    return False

def update_daily_simulations(n_simulations=10000, force=False):
    """Run daily simulations if needed or if force=True"""
    # Check if we need to update simulations
    if not force:
        # Check if we've already refreshed simulations today
        if 'last_simulation_refresh' in st.session_state and st.session_state.last_simulation_refresh:
            last_refresh = st.session_state.last_simulation_refresh
            now = datetime.now()
            # If simulations were run today, skip the update
            if (now.date() == last_refresh.date() and 
                not (now.hour >= 4 and last_refresh.hour < 4)):  # Unless it's after 4am and last update was before 4am
                return False
    
    # Get the current season
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    season_str = f"{current_season}{current_season+1}"
    
    # Get data folder path
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Make sure data is updated first
    try:
        data_handlers.update_daily_data(data_folder, current_season, season_str, force=True)
    except Exception as e:
        st.error(f"Error updating data before running simulations: {str(e)}")
        return False
    
    # Load required data
    try:
        # Load standings data for playoff matchups
        standings_df = data_handlers.load_data(f"standings_{season_str}.csv", data_folder)
        if standings_df is None or standings_df.empty:
            st.error("Standings data not available for simulations")
            return False
        
        # Load team data for predictions
        team_data = data_handlers.load_data(f"team_data_{season_str}.csv", data_folder)
        if team_data is None or team_data.empty:
            st.error("Team data not available for simulations")
            return False
        
        # Determine playoff matchups
        playoff_matchups = data_handlers.determine_playoff_teams(standings_df)
        if not playoff_matchups:
            st.error("Could not determine playoff matchups")
            return False
            
        # Load models
        models = model_utils.load_models(model_folder)
        if not models or 'models' not in models or not models['models']:
            st.error("No models available for simulations")
            return False
        
        # Run full bracket simulation
        bracket_results = simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations)
        
        if bracket_results is None or 'team_advancement' not in bracket_results:
            st.error("Simulation failed to produce results")
            return False
        
        # Store the results in session state
        st.session_state.daily_simulations = bracket_results
        st.session_state.last_simulation_refresh = datetime.now()
        
        # Save results to disk
        if 'team_advancement' in bracket_results:
            try:
                results_df = bracket_results['team_advancement']
                data_handlers.save_data(results_df, f"playoff_sim_results_{season_str}.csv", data_folder)
                
                # Save detailed results if available
                if 'round2_matchups' in bracket_results:
                    data_handlers.save_data(bracket_results['round2_matchups'], f"playoff_round2_matchups_{season_str}.csv", data_folder)
                    data_handlers.save_data(bracket_results['conf_final_matchups'], f"playoff_conf_finals_matchups_{season_str}.csv", data_folder)
                    data_handlers.save_data(bracket_results['final_matchups'], f"playoff_stanley_cup_matchups_{season_str}.csv", data_folder)
            except Exception as save_error:
                st.error(f"Error saving simulation results: {str(save_error)}")
        
        return True
        
    except Exception as e:
        st.error(f"Error running simulations: {str(e)}")
        return False

def get_outcome_distributions(win_prob):
    """Calculate outcome distributions based on win probability and historical NHL data.
    
    Using the updated series length distribution from notebook:
    - 4 games: 14.0%
    - 5 games: 24.3%
    - 6 games: 33.6%
    - 7 games: 28.1%
    """
    # Base distribution by series length
    total_percent = 14.0 + 24.3 + 33.6 + 28.1
    
    # Normalization to ensure sum to 1.0
    base_sweep = 14.0/total_percent
    base_five = 24.3/total_percent
    base_six = 33.6/total_percent
    base_seven = 28.1/total_percent
    
    # Distribution for higher seed wins (maintain relative proportions)
    higher_seed_outcome_dist = {
        '4-0': base_sweep,
        '4-1': base_five, 
        '4-2': base_six, 
        '4-3': base_seven
    }
    
    # Same distribution for when lower seed wins
    lower_seed_outcome_dist = {
        '0-4': base_sweep,
        '1-4': base_five, 
        '2-4': base_six, 
        '3-4': base_seven
    }
    
    return higher_seed_outcome_dist, lower_seed_outcome_dist

def simulate_playoff_series(matchup_df, models, n_simulations=1000):
    """Predict series outcome using the pre-trained models with updated historical series length distribution."""
    # Extract team abbreviations for display
    top_seed = matchup_df['top_seed_abbrev'].iloc[0]
    bottom_seed = matchup_df['bottom_seed_abbrev'].iloc[0]
    
    # Get win probability using models
    ensemble_prob = 0.5 # default
    lr_prob = 0.5
    xgb_prob = 0.5
    
    # Check which models we have available and make predictions
    if 'models' in models and 'lr' in models['models'] and hasattr(models['models']['lr'], 'predict_proba'):
        try:
            # Extract model and features
            lr_model = models['models']['lr']
            
            # Get features if available in model
            if hasattr(lr_model, 'feature_names_'):
                lr_features = [f for f in lr_model.feature_names_ if f in matchup_df.columns]
            else:
                # Default features - columns ending with '_diff'
                lr_features = [col for col in matchup_df.columns if '_diff' in col]
            
            # Make prediction
            if lr_features and len(lr_features) > 0:
                lr_prob = lr_model.predict_proba(matchup_df[lr_features])[:, 1][0]
        except Exception as e:
            st.warning(f"Error in LR prediction: {str(e)}")
    
    if 'models' in models and 'xgb' in models['models'] and hasattr(models['models']['xgb'], 'predict_proba'):
        try:
            # Extract model and features
            xgb_model = models['models']['xgb']
            
            # Get features if available in model
            if hasattr(xgb_model, 'feature_names_'):
                xgb_features = [f for f in xgb_model.feature_names_ if f in matchup_df.columns]
            else:
                # Default features - columns ending with '_diff'
                xgb_features = [col for col in matchup_df.columns if '_diff' in col]
            
            # Make prediction
            if xgb_features and len(xgb_features) > 0:
                xgb_prob = xgb_model.predict_proba(matchup_df[xgb_features])[:, 1][0]
        except Exception as e:
            st.warning(f"Error in XGB prediction: {str(e)}")
    
    # Calculate ensemble probability if both models available
    if lr_prob != 0.5 and xgb_prob != 0.5:
        ensemble_prob = (lr_prob + xgb_prob) / 2
    # Otherwise use the non-default model if available
    elif lr_prob != 0.5:
        ensemble_prob = lr_prob
    elif xgb_prob != 0.5:
        ensemble_prob = xgb_prob
    
    # Apply home ice advantage boost to the ensemble probability
    home_ice_boost = models.get('home_ice_boost', 0.039)
    win_prob = min(1.0, ensemble_prob + home_ice_boost)
    
    # Track results
    higher_seed_wins = 0
    win_distribution = {
        '4-0': 0, '4-1': 0, '4-2': 0, '4-3': 0,  # Higher seed wins
        '0-4': 0, '1-4': 0, '2-4': 0, '3-4': 0   # Lower seed wins
    }
    
    # Get outcome distributions
    higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(ensemble_prob)
    
    # Run simulations - now determining series winner based on probability with home ice advantage
    for _ in range(n_simulations):
        # Determine if higher seed wins the series
        higher_seed_wins_series = np.random.random() < win_prob
        
        if higher_seed_wins_series:
            higher_seed_wins += 1
            # Select a series outcome based on historical distribution
            outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                      p=[higher_seed_outcome_dist['4-0'], 
                                         higher_seed_outcome_dist['4-1'],
                                         higher_seed_outcome_dist['4-2'],
                                         higher_seed_outcome_dist['4-3']])
            win_distribution[outcome] += 1
        else:
            # Select a series outcome for lower seed winning
            outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                      p=[lower_seed_outcome_dist['0-4'], 
                                         lower_seed_outcome_dist['1-4'],
                                         lower_seed_outcome_dist['2-4'],
                                         lower_seed_outcome_dist['3-4']])
            win_distribution[outcome] += 1
    
    # Calculate win percentage
    win_pct = higher_seed_wins / n_simulations
    
    # Calculate confidence interval
    z = 1.96  # 95% confidence
    ci_width = z * np.sqrt((win_pct * (1 - win_pct)) / n_simulations)
    ci_lower = max(0, win_pct - ci_width)
    ci_upper = min(1, win_pct + ci_width)
    
    # Format results
    results = {
        'top_seed': top_seed,
        'bottom_seed': bottom_seed,
        'win_probability': win_pct,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'win_distribution': win_distribution,
        'model_mode': models.get('mode', 'default'),
        'lr_probability': lr_prob,
        'xgb_probability': xgb_prob,
        'combined_base_probability': ensemble_prob,
        'applied_home_ice_boost': home_ice_boost,
        'final_probability_with_boost': win_prob,
        'series_length_breakdown': {
            '4 games': 14.0, 
            '5 games': 24.3, 
            '6 games': 33.6, 
            '7 games': 28.1
        },
        'n_simulations': n_simulations
    }
    
    return results

def verify_matchup_data(matchup_df, models):
    """Verify that matchup data has all required features for prediction
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information and objects
        
    Returns:
        bool: True if data passes verification, False otherwise
    """
    if matchup_df is None or matchup_df.empty:
        print("Error: Empty matchup dataframe")
        return False
    
    # Check for required columns
    required_cols = ['top_seed_abbrev', 'bottom_seed_abbrev']
    missing_cols = [col for col in required_cols if col not in matchup_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return False
    
    # Check for _diff features which are used by the models
    diff_features = [col for col in matchup_df.columns if col.endswith('_diff')]
    if not diff_features:
        print("Error: No differential features available for prediction")
        return False
    
    # Check LR model features if available
    if 'models' in models and 'lr' in models['models']:
        lr_model = models['models']['lr']
        if hasattr(lr_model, 'feature_names_'):
            lr_features = [f for f in lr_model.feature_names_ if f in matchup_df.columns]
            if len(lr_features) < len(lr_model.feature_names_):
                missing_features = [f for f in lr_model.feature_names_ if f not in matchup_df.columns]
                print(f"Warning: Missing {len(missing_features)}/{len(lr_model.feature_names_)} LR model features")
                if len(missing_features) > len(lr_model.feature_names_) // 2:
                    print("Error: More than half of LR features are missing!")
                    return False
    
    # Check XGB model features if available
    if 'models' in models and 'xgb' in models['models']:
        xgb_model = models['models']['xgb']
        if hasattr(xgb_model, 'feature_names_'):
            xgb_features = [f for f in xgb_model.feature_names_ if f in matchup_df.columns]
            if len(xgb_features) < len(xgb_model.feature_names_):
                missing_features = [f for f in xgb_model.feature_names_ if f not in matchup_df.columns]
                print(f"Warning: Missing {len(missing_features)}/{len(xgb_model.feature_names_)} XGB model features")
                if len(missing_features) > len(xgb_model.feature_names_) // 2:
                    print("Error: More than half of XGB features are missing!")
                    return False
    
    # Check for NaN values in features
    nan_cols = [col for col in diff_features if matchup_df[col].isna().any()]
    if nan_cols:
        print(f"Warning: NaN values found in features: {nan_cols}")
    
    # Check for key prediction features
    critical_features = [
        'PP%_rel_diff', 'PK%_rel_diff', 'xGoalsPercentage_diff',
        'possTypeAdjGiveawaysPctg_diff', 'reboundxGoalsPctg_diff', 
        'goalDiff/G_diff', 'playoff_performance_score_diff'
    ]
    
    missing_critical = [f for f in critical_features if f not in matchup_df.columns]
    if missing_critical:
        print(f"Warning: Missing critical features: {missing_critical}")
        if len(missing_critical) > len(critical_features) // 2:
            print("Error: More than half of critical features are missing!")
            return False
    
    return True

import matchup_utils

def simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=1000, detailed_tracking=True):
    """Simulate the entire playoff bracket with comprehensive tracking and results."""
    # Track advancement for each team
    team_advancement = {}
    
    # Track all possible matchups for each round if detailed_tracking enabled
    round_matchups = {
        'round_2': {},  # Division Finals
        'conf_final': {},  # Conference Finals
        'final': {}  # Stanley Cup Final
    }
    
    # Initialize dictionaries for tracking detailed results
    potential_matchups = {}
    bracket_results_count = {}
    
    # Initialize all playoff teams
    for conference, matchups in playoff_matchups.items():
        for series_id, matchup in matchups.items():
            top_seed = matchup['top_seed']['teamAbbrev']
            bottom_seed = matchup['bottom_seed']['teamAbbrev']
            
            if top_seed not in team_advancement:
                team_advancement[top_seed] = {
                    'round_1': 0,
                    'round_2': 0,
                    'conf_final': 0,
                    'final': 0,
                    'champion': 0,
                    'teamName': matchup['top_seed']['teamName'],
                    'total_games': 0  # Track total playoff games played
                }
            
            if bottom_seed not in team_advancement:
                team_advancement[bottom_seed] = {
                    'round_1': 0,
                    'round_2': 0,
                    'conf_final': 0,
                    'final': 0,
                    'champion': 0,
                    'teamName': matchup['bottom_seed']['teamName'],
                    'total_games': 0  # Track total playoff games played
                }

    # Pre-calculate all possible matchups between playoff teams for faster lookups
    print("Pre-calculating all possible playoff matchups...")
    
    # Extract all playoff teams
    playoff_teams = []
    for conference, matchups in playoff_matchups.items():
        for series_id, matchup in matchups.items():
            playoff_teams.append(matchup['top_seed']['teamAbbrev'])
            playoff_teams.append(matchup['bottom_seed']['teamAbbrev'])
    playoff_teams = list(set(playoff_teams))
    
    # Filter team_data to only include playoff teams for faster computation
    playoff_team_data = team_data[team_data['teamAbbrev'].isin(playoff_teams)].copy()
    
    # Generate all matchup combinations
    matchup_dict = matchup_utils.generate_all_matchup_combinations(playoff_team_data)
    
    # Analyze matchup data quality
    print("Analyzing matchup data quality...")
    matchup_analysis = matchup_utils.analyze_matchup_data(matchup_dict)
    print(f"Total matchups calculated: {matchup_analysis['total_matchups']}")
    print("Feature availability:")
    
    critical_features = ['PP%_rel_diff', 'PK%_rel_diff', 'xGoalsPercentage_diff', 
                       'possTypeAdjGiveawaysPctg_diff', 'goalDiff/G_diff']
    
    for feature in critical_features:
        if feature in matchup_analysis['feature_counts']:
            avail_pct = (matchup_analysis['feature_counts'][feature] / matchup_analysis['total_matchups']) * 100
            nan_pct = 0
            if matchup_analysis['feature_counts'][feature] > 0:
                nan_pct = (matchup_analysis['nan_counts'][feature] / matchup_analysis['feature_counts'][feature]) * 100
            print(f"  - {feature}: {avail_pct:.1f}% available, {nan_pct:.1f}% with NaN values")
    
    # Model usage tracking
    model_usage_stats = {
        'lr': 0,
        'xgb': 0,
        'ensemble': 0,
        'points_diff': 0,
        'default': 0
    }

    # Run simulations
    for sim in range(n_simulations):
        if sim % 1000 == 0 and sim > 0:
            print(f"Completed {sim} simulations...")
            
        # Track winners for each round
        round_1_winners = {}
        round_2_winners = {}
        conf_winners = {}
        stanley_cup_winner = None
        
        # Track this simulation's bracket result if detailed tracking is enabled
        current_bracket = [] if detailed_tracking else None
        
        # First round
        for conference, matchups in playoff_matchups.items():
            round_1_winners[conference] = {}
            for series_id, matchup in matchups.items():
                # Get matchup data from pre-calculated dictionary
                top_seed = matchup['top_seed']
                bottom_seed = matchup['bottom_seed']
                matchup_df = matchup_utils.get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data)
                
                # Get prediction for the series
                if matchup_df is not None and not matchup_df.empty:
                    # Get raw win probability from pre-trained model - IMPORTANT: NO HOME ICE BOOST YET
                    ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
                    
                    # Track which model was used
                    if ensemble_prob != 0.5 and lr_prob != 0.5 and xgb_prob != 0.5:
                        model_usage_stats['ensemble'] += 1
                    elif lr_prob != 0.5:
                        model_usage_stats['lr'] += 1
                    elif xgb_prob != 0.5:
                        model_usage_stats['xgb'] += 1
                    elif 'points_diff' in matchup_df.columns:
                        model_usage_stats['points_diff'] += 1
                    else:
                        model_usage_stats['default'] += 1
                    
                    # Apply home ice advantage boost AFTER ensemble calculation
                    home_ice_boost = models.get('home_ice_boost', 0.039)
                    win_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Simulate the series outcome using the boosted probability
                    higher_seed_wins = np.random.random() < win_prob
                    
                    if higher_seed_wins:
                        winner = matchup['top_seed']
                        loser = matchup['bottom_seed']
                    else:
                        winner = matchup['bottom_seed']
                        loser = matchup['top_seed']
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=[0.14, 0.243, 0.336, 0.281])
                    
                    # Track games played for both teams
                    team_advancement[winner['teamAbbrev']]['total_games'] += series_length
                    team_advancement[loser['teamAbbrev']]['total_games'] += series_length
                    
                    # Store winner for next round
                    round_1_winners[conference][series_id] = {
                        'team': winner,
                        'original_matchup': series_id
                    }
                    
                    # Record advancement
                    team_advancement[winner['teamAbbrev']]['round_1'] += 1
                    
                    # Add to current bracket if detailed tracking is enabled
                    if detailed_tracking:
                        current_bracket.append(f"{winner['teamAbbrev']} over {loser['teamAbbrev']}")

        # Process Second round (Division Finals)
        for conference, r1_winners in round_1_winners.items():
            round_2_winners[conference] = {}
            
            # Extract division information from series_ids
            divisions = set()
            for key in r1_winners.keys():
                if len(key) >= 1 and '_' in key:
                    divisions.add(key.split('_')[0])
            
            # Create second round matchups
            for division in divisions:
                # Find matchup winners from this division's first round
                div_winners = [r1_winners[k] for k in r1_winners if k.startswith(division)]
                
                if len(div_winners) >= 2:  # Need at least two teams for a matchup
                    # Identify teams
                    team1 = div_winners[0]['team']
                    team2 = div_winners[1]['team']
                    
                    # Determine seeding (using existing logic)
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
                        
                        # Also add to potential_matchups
                        if matchup_key not in potential_matchups:
                            potential_matchups[matchup_key] = {
                                'top_seed': top_seed['teamAbbrev'],
                                'bottom_seed': bottom_seed['teamAbbrev'],
                                'top_seed_name': top_seed['teamName'],
                                'bottom_seed_name': bottom_seed['teamName'],
                                'wins_top': 0,
                                'wins_bottom': 0,
                                'count': 0,
                                'round': 'second_round',
                                'conference': conference
                            }
                        potential_matchups[matchup_key]['count'] += 1
                    
                    # Use pre-calculated matchup data
                    matchup_df = matchup_utils.get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data)
                    
                    if matchup_df is not None and not matchup_df.empty:
                        # Get win probability 
                        ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
                        
                        # Track which model was used
                        if ensemble_prob != 0.5 and lr_prob != 0.5 and xgb_prob != 0.5:
                            model_usage_stats['ensemble'] += 1
                        elif lr_prob != 0.5:
                            model_usage_stats['lr'] += 1
                        elif xgb_prob != 0.5:
                            model_usage_stats['xgb'] += 1
                        elif 'points_diff' in matchup_df.columns:
                            model_usage_stats['points_diff'] += 1
                        else:
                            model_usage_stats['default'] += 1
                        
                        # Apply home ice advantage boost
                        home_ice_boost = models.get('home_ice_boost', 0.039)
                        win_prob = min(1.0, ensemble_prob + home_ice_boost)
                        
                        # Simulate the outcome
                        higher_seed_wins = np.random.random() < win_prob
                        winner = top_seed if higher_seed_wins else bottom_seed
                        loser = bottom_seed if higher_seed_wins else top_seed
                        
                        # Record the win in the detailed tracking
                        if detailed_tracking:
                            if higher_seed_wins:
                                round_matchups['round_2'][matchup_key]['top_seed_wins'] += 1
                                potential_matchups[matchup_key]['wins_top'] += 1
                            else:
                                potential_matchups[matchup_key]['wins_bottom'] += 1
                        
                        # Simulate series length based on historical distribution
                        series_length = np.random.choice([4, 5, 6, 7], p=[0.14, 0.243, 0.336, 0.281])
                        
                        # Track games played for both teams
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
        
        # Conference Finals - follow the same pattern, using matchup_utils.get_matchup_data
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
                        team1_data_df = team_data[team_data['teamAbbrev'] == team1['teamAbbrev']]
                        team2_data_df = team_data[team_data['teamAbbrev'] == team2['teamAbbrev']]
                        
                        points1 = team1_data_df['points'].iloc[0] if not team1_data_df.empty and 'points' in team1_data_df.columns else 0
                        points2 = team2_data_df['points'].iloc[0] if not team2_data_df.empty and 'points' in team2_data_df.columns else 0
                        
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
                        
                        # Also add to potential_matchups
                        if matchup_key not in potential_matchups:
                            potential_matchups[matchup_key] = {
                                'top_seed': top_seed['teamAbbrev'],
                                'bottom_seed': bottom_seed['teamAbbrev'],
                                'top_seed_name': top_seed['teamName'],
                                'bottom_seed_name': bottom_seed['teamName'],
                                'wins_top': 0,
                                'wins_bottom': 0,
                                'count': 0,
                                'round': 'conf_final',
                                'conference': conference
                            }
                        potential_matchups[matchup_key]['count'] += 1
                    
                    # Use pre-calculated matchup data
                    matchup_df = matchup_utils.get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data)
                    
                    if matchup_df is not None and not matchup_df.empty:
                        # Get win probability
                        ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
                        
                        # Track which model was used
                        if ensemble_prob != 0.5 and lr_prob != 0.5 and xgb_prob != 0.5:
                            model_usage_stats['ensemble'] += 1
                        elif lr_prob != 0.5:
                            model_usage_stats['lr'] += 1
                        elif xgb_prob != 0.5:
                            model_usage_stats['xgb'] += 1
                        elif 'points_diff' in matchup_df.columns:
                            model_usage_stats['points_diff'] += 1
                        else:
                            model_usage_stats['default'] += 1
                        
                        # Apply home ice advantage boost
                        home_ice_boost = models.get('home_ice_boost', 0.039)
                        win_prob = min(1.0, ensemble_prob + home_ice_boost)
                        
                        # Simulate the outcome
                        higher_seed_wins = np.random.random() < win_prob
                        winner = top_seed if higher_seed_wins else bottom_seed
                        loser = bottom_seed if higher_seed_wins else top_seed
                        
                        # Record the win if detailed tracking is enabled
                        if detailed_tracking:
                            if higher_seed_wins:
                                round_matchups['conf_final'][matchup_key]['top_seed_wins'] += 1
                                potential_matchups[matchup_key]['wins_top'] += 1
                            else:
                                potential_matchups[matchup_key]['wins_bottom'] += 1
                        
                        # Simulate series length based on historical distribution
                        series_length = np.random.choice([4, 5, 6, 7], p=[0.14, 0.243, 0.336, 0.281])
                        
                        # Track games played for both teams
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
                team1_data_df = team_data[team_data['teamAbbrev'] == team1['teamAbbrev']]
                team2_data_df = team_data[team_data['teamAbbrev'] == team2['teamAbbrev']]
                
                points1 = team1_data_df['points'].iloc[0] if not team1_data_df.empty and 'points' in team1_data_df.columns else 0
                points2 = team2_data_df['points'].iloc[0] if not team2_data_df.empty and 'points' in team2_data_df.columns else 0
                
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
                    
                    # Also add to potential_matchups
                    if matchup_key not in potential_matchups:
                        potential_matchups[matchup_key] = {
                            'top_seed': top_seed['teamAbbrev'],
                            'bottom_seed': bottom_seed['teamAbbrev'],
                            'top_seed_name': top_seed['teamName'],
                            'bottom_seed_name': bottom_seed['teamName'],
                            'wins_top': 0,
                            'wins_bottom': 0,
                            'count': 0,
                            'round': 'finals',
                            'conference': 'finals'
                        }
                    potential_matchups[matchup_key]['count'] += 1
                
                # Use pre-calculated matchup data
                matchup_df = matchup_utils.get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data)
                
                if matchup_df is not None and not matchup_df.empty:
                    # Get win probability
                    ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
                    
                    # Track which model was used
                    if ensemble_prob != 0.5 and lr_prob != 0.5 and xgb_prob != 0.5:
                        model_usage_stats['ensemble'] += 1
                    elif lr_prob != 0.5:
                        model_usage_stats['lr'] += 1
                    elif xgb_prob != 0.5:
                        model_usage_stats['xgb'] += 1
                    elif 'points_diff' in matchup_df.columns:
                        model_usage_stats['points_diff'] += 1
                    else:
                        model_usage_stats['default'] += 1
                    
                    # Apply home ice advantage boost
                    home_ice_boost = models.get('home_ice_boost', 0.039)
                    win_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Simulate the outcome
                    higher_seed_wins = np.random.random() < win_prob
                    winner = top_seed if higher_seed_wins else bottom_seed
                    loser = bottom_seed if higher_seed_wins else top_seed
                    
                    # Record the win if detailed tracking is enabled
                    if detailed_tracking:
                        if higher_seed_wins:
                            round_matchups['final'][matchup_key]['top_seed_wins'] += 1
                            potential_matchups[matchup_key]['wins_top'] += 1
                        else:
                            potential_matchups[matchup_key]['wins_bottom'] += 1
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=[0.14, 0.243, 0.336, 0.281])
                    
                    # Track games played for both teams
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
    
    # Before sorting, ensure the 'champion' column exists
    if 'champion' not in results_df.columns or results_df.empty:
        # Initialize with zeros if missing
        results_df['champion'] = 0.0
    
    # Now sort safely
    results_df = results_df.sort_values('champion', ascending=False)
    
    # Prepare return values based on detailed tracking option
    results = {
        'team_advancement': results_df
    }
    
    # Add detailed tracking results if enabled
    if detailed_tracking:
        # Find the most common bracket result
        most_common_bracket = max(bracket_results_count.items(), key=lambda x: x[1]) if bracket_results_count else ("", 0)
        
        # Convert round matchup counts to DataFrames
        round2_df = pd.DataFrame([
            {
                'conference': data['conference'],
                'division': data.get('division', 'N/A'),
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0
            }
            for data in round_matchups['round_2'].values()
        ]) if round_matchups['round_2'] else pd.DataFrame()
        
        conf_final_df = pd.DataFrame([
            {
                'conference': data['conference'],
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0
            }
            for data in round_matchups['conf_final'].values()
        ]) if round_matchups['conf_final'] else pd.DataFrame()
        
        final_df = pd.DataFrame([
            {
                'matchup': f"{data['top_seed_name']} vs {data['bottom_seed_name']}",
                'count': data['count'],
                'probability': data['count']/n_simulations*100,
                'top_seed_win_pct': data['top_seed_wins']/data['count']*100 if data['count'] > 0 else 0
            }
            for data in round_matchups['final'].values()
        ]) if round_matchups['final'] else pd.DataFrame()
        
        # Add win probabilities to potential_matchups
        for key in potential_matchups:
            if potential_matchups[key]['count'] > 0:
                potential_matchups[key]['probability'] = potential_matchups[key]['wins_top'] / potential_matchups[key]['count']
        
        results.update({
            'most_common_bracket': {
                'bracket': most_common_bracket[0].split('|') if most_common_bracket[0] else [],
                'count': most_common_bracket[1],
                'probability': most_common_bracket[1]/n_simulations*100
            },
            'round2_matchups': round2_df.sort_values('probability', ascending=False) if not round2_df.empty else pd.DataFrame(),
            'conf_final_matchups': conf_final_df.sort_values('probability', ascending=False) if not conf_final_df.empty else pd.DataFrame(),
            'final_matchups': final_df.sort_values('probability', ascending=False) if not final_df.empty else pd.DataFrame(),
            'potential_matchups': potential_matchups
        })
    
    # Print model usage statistics
    print("\nModel Usage Statistics:")
    total_predictions = sum(model_usage_stats.values())
    for model_type, count in model_usage_stats.items():
        print(f"  - {model_type}: {count} ({count/total_predictions*100:.1f}%)")
    
    # Add model usage statistics to results 
    results['model_usage'] = {
        'counts': model_usage_stats,
        'percentages': {k: v/total_predictions*100 for k, v in model_usage_stats.items()}
    }

    return results

def simulate_single_bracket(playoff_matchups, team_data, models):
    """
    Simulate a single playoff bracket from start to finish
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team data
        models: Dictionary with model data
        
    Returns:
        dict: Results of the simulation
    """
    if not playoff_matchups:
        st.error("No playoff matchups available to simulate.")
        return None
    
    # Initialize tracking dictionaries
    team_results = {}
    bracket_progression = {
        'First Round': [],
        'Second Round': [],
        'Conference Finals': [],
        'Stanley Cup Final': []
    }
    
    # Initialize all playoff teams
    for conference, matchups in playoff_matchups.items():
        for series_id, matchup in matchups.items():
            top_seed = matchup['top_seed']['teamAbbrev']
            bottom_seed = matchup['bottom_seed']['teamAbbrev']
            
            # Add teams if not already present
            for team, team_data_dict in [
                (top_seed, matchup['top_seed']), 
                (bottom_seed, matchup['bottom_seed'])
            ]:
                if team not in team_results:
                    team_results[team] = {
                        'team_name': team_data_dict['teamName'],
                        'conference': conference,
                        'division': team_data_dict.get('division', 'Unknown'),
                        'round_1': False,
                        'round_2': False,
                        'conf_final': False,
                        'final': False,
                        'champion': False,
                        'series_wins': []
                    }

    # FIRST ROUND
    round_1_winners = {}
    for conference, matchups in playoff_matchups.items():
        round_1_winners[conference] = {}
        
        for series_id, matchup in matchups.items():
            # Create matchup data
            matchup_df = model_utils.create_matchup_data(matchup['top_seed'], matchup['bottom_seed'], team_data)
            
            # Simulate the series
            if not matchup_df.empty:
                # Get raw win probability from pre-trained model
                ensemble_prob, _, _ = model_utils.predict_series_winner(matchup_df, models)
                
                # Apply home ice advantage boost
                home_ice_boost = models.get('home_ice_boost', 0.039)
                win_prob = min(1.0, ensemble_prob + home_ice_boost)
                
                # Determine the winner
                higher_seed_wins = np.random.random() < win_prob
                
                top_seed_abbrev = matchup['top_seed']['teamAbbrev']
                bottom_seed_abbrev = matchup['bottom_seed']['teamAbbrev']
                winner_abbrev = top_seed_abbrev if higher_seed_wins else bottom_seed_abbrev
                loser_abbrev = bottom_seed_abbrev if higher_seed_wins else top_seed_abbrev
                
                # Generate series result (4-0, 4-1, etc.)
                higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(ensemble_prob)
                
                if higher_seed_wins:
                    outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                            p=[higher_seed_outcome_dist['4-0'], 
                                               higher_seed_outcome_dist['4-1'],
                                               higher_seed_outcome_dist['4-2'],
                                               higher_seed_outcome_dist['4-3']])
                else:
                    lower_outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                                p=[lower_seed_outcome_dist['0-4'], 
                                                   lower_seed_outcome_dist['1-4'],
                                                   lower_seed_outcome_dist['2-4'],
                                                   lower_seed_outcome_dist['3-4']])
                    # Convert to winner's perspective
                    games = int(lower_outcome[-1])
                    losses = int(lower_outcome[0])
                    outcome = f"{games}-{losses}"
                
                # Store results
                winner = matchup['top_seed'] if higher_seed_wins else matchup['bottom_seed']
                loser = matchup['bottom_seed'] if higher_seed_wins else matchup['top_seed']
                
                # Update team advancement record
                team_results[winner_abbrev]['round_1'] = True
                team_results[winner_abbrev]['series_wins'].append({
                    'round': 'First Round',
                    'opponent': team_results[loser_abbrev]['team_name'],
                    'result': outcome
                })
                
                # Store winner for next round
                round_1_winners[conference][series_id] = {
                    'team': winner,
                    'original_matchup': series_id
                }
                
                # Track in bracket progression
                bracket_progression['First Round'].append({
                    'winner': team_results[winner_abbrev]['team_name'],
                    'loser': team_results[loser_abbrev]['team_name'],
                    'result': outcome,
                    'conference': conference,
                    'series_id': series_id
                })

    # SECOND ROUND Processing
    # Similar logic to the full playoff bracket simulation
    round_2_winners = {}
    for conference, r1_winners in round_1_winners.items():
        round_2_winners[conference] = {}
        
        # Extract division information
        divisions = set()
        for key in r1_winners.keys():
            if '_' in key:
                divisions.add(key.split('_')[0])
        
        # Create second round matchups by division
        for division in divisions:
            # Find winners from this division
            div_winners = [r1_winners[k] for k in r1_winners if k.startswith(division)]
            
            if len(div_winners) >= 2:
                team1 = div_winners[0]['team']
                team2 = div_winners[1]['team']
                
                # Create matchup data 
                matchup_df = model_utils.create_matchup_data(team1, team2, team_data)
                
                # Get prediction and simulate the outcome
                ensemble_prob, _, _ = model_utils.predict_series_winner(matchup_df, models)
                
                # Apply home ice advantage boost
                home_ice_boost = models.get('home_ice_boost', 0.039)
                win_prob = min(1.0, ensemble_prob + home_ice_boost)
                
                higher_seed_wins = np.random.random() < win_prob
                
                # Determine winner and loser
                winner = team1 if higher_seed_wins else team2
                loser = team2 if higher_seed_wins else team1
                
                # Generate series result
                higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(ensemble_prob)
                
                if higher_seed_wins:
                    outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                            p=[higher_seed_outcome_dist['4-0'], 
                                               higher_seed_outcome_dist['4-1'],
                                               higher_seed_outcome_dist['4-2'],
                                               higher_seed_outcome_dist['4-3']])
                else:
                    lower_outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                                    p=[lower_seed_outcome_dist['0-4'], 
                                                       lower_seed_outcome_dist['1-4'],
                                                       lower_seed_outcome_dist['2-4'],
                                                       lower_seed_outcome_dist['3-4']])
                    # Convert to winner's perspective
                    games = int(lower_outcome[-1])
                    losses = int(lower_outcome[0])
                    outcome = f"{games}-{losses}"
                
                # Update team advancement record
                team_results[winner['teamAbbrev']]['round_2'] = True
                team_results[winner['teamAbbrev']]['series_wins'].append({
                    'round': 'Second Round',
                    'opponent': team_results[loser['teamAbbrev']]['team_name'],
                    'result': outcome
                })
                
                # Store winner for next round
                round_2_winners[conference][f"{division}_final"] = {
                    'team': winner,
                    'division': division
                }
                
                # Track in bracket progression
                bracket_progression['Second Round'].append({
                    'winner': team_results[winner['teamAbbrev']]['team_name'],
                    'loser': team_results[loser['teamAbbrev']]['team_name'],
                    'result': outcome,
                    'conference': conference
                })

    # CONFERENCE FINALS Processing
    conf_winners = {}
    for conference, r2_winners in round_2_winners.items():
        if len(r2_winners) >= 2:
            # Get the two division winners
            div_winners = list(r2_winners.values())
            
            if len(div_winners) == 2:
                team1 = div_winners[0]['team']
                team2 = div_winners[1]['team']
                
                # Create matchup data
                matchup_df = model_utils.create_matchup_data(team1, team2, team_data)
                
                # Get prediction and simulate the outcome
                ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
                
                # Apply home ice advantage boost
                home_ice_boost = models.get('home_ice_boost', 0.039)
                win_prob = min(1.0, ensemble_prob + home_ice_boost)
                
                higher_seed_wins = np.random.random() < win_prob
                
                # Determine winner and loser
                winner = team1 if higher_seed_wins else team2
                loser = team2 if higher_seed_wins else team1
                
                # Generate series result
                higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(ensemble_prob)
                
                if higher_seed_wins:
                    outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                            p=[higher_seed_outcome_dist['4-0'], 
                                               higher_seed_outcome_dist['4-1'],
                                               higher_seed_outcome_dist['4-2'],
                                               higher_seed_outcome_dist['4-3']])
                else:
                    lower_outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                                    p=[lower_seed_outcome_dist['0-4'], 
                                                       lower_seed_outcome_dist['1-4'],
                                                       lower_seed_outcome_dist['2-4'],
                                                       lower_seed_outcome_dist['3-4']])
                    # Convert to winner's perspective
                    games = int(lower_outcome[-1])
                    losses = int(lower_outcome[0])
                    outcome = f"{games}-{losses}"
                
                # Update team advancement record
                team_results[winner['teamAbbrev']]['conf_final'] = True
                team_results[winner['teamAbbrev']]['series_wins'].append({
                    'round': 'Conference Finals',
                    'opponent': team_results[loser['teamAbbrev']]['team_name'],
                    'result': outcome
                })
                
                # Store winner for final
                conf_winners[conference] = {
                    'team': winner,
                    'conference': conference
                }
                
                # Track in bracket progression
                bracket_progression['Conference Finals'].append({
                    'winner': team_results[winner['teamAbbrev']]['team_name'],
                    'loser': team_results[loser['teamAbbrev']]['team_name'],
                    'result': outcome,
                    'conference': conference
                })

    # STANLEY CUP FINAL Processing
    champion_abbrev = None
    runner_up_abbrev = None
    if len(conf_winners) == 2:
        # Get the two conference champions
        conf_champs = list(conf_winners.values())
        
        if len(conf_champs) == 2:
            team1 = conf_champs[0]['team']
            team2 = conf_champs[1]['team']
            
            # Create matchup data
            matchup_df = model_utils.create_matchup_data(team1, team2, team_data)
            
            # Get prediction and simulate the outcome  
            ensemble_prob, lr_prob, xgb_prob = model_utils.predict_series_winner(matchup_df, models)
            
            # Apply home ice advantage boost
            home_ice_boost = models.get('home_ice_boost', 0.039)
            win_prob = min(1.0, ensemble_prob + home_ice_boost)
            
            higher_seed_wins = np.random.random() < win_prob
            
            # Determine winner and loser
            winner = team1 if higher_seed_wins else team2
            loser = team2 if higher_seed_wins else team1
            
            # Generate series result
            higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(ensemble_prob)
            
            if higher_seed_wins:
                outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                        p=[higher_seed_outcome_dist['4-0'], 
                                           higher_seed_outcome_dist['4-1'],
                                           higher_seed_outcome_dist['4-2'],
                                           higher_seed_outcome_dist['4-3']])
            else:
                lower_outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                               p=[lower_seed_outcome_dist['0-4'], 
                                                  lower_seed_outcome_dist['1-4'],
                                                  lower_seed_outcome_dist['2-4'],
                                                  lower_seed_outcome_dist['3-4']])
                # Convert to winner's perspective
                games = int(lower_outcome[-1])
                losses = int(lower_outcome[0])
                outcome = f"{games}-{losses}"
            
            # Update team advancement record
            team_results[winner['teamAbbrev']]['final'] = True
            team_results[winner['teamAbbrev']]['champion'] = True
            team_results[loser['teamAbbrev']]['final'] = True
            
            team_results[winner['teamAbbrev']]['series_wins'].append({
                'round': 'Stanley Cup Final',
                'opponent': team_results[loser['teamAbbrev']]['team_name'],
                'result': outcome
            })
            
            # Track champion
            champion_abbrev = winner['teamAbbrev']
            runner_up_abbrev = loser['teamAbbrev']
            
            # Track in bracket progression
            bracket_progression['Stanley Cup Final'].append({
                'winner': team_results[winner['teamAbbrev']]['team_name'], 
                'loser': team_results[loser['teamAbbrev']]['team_name'],
                'result': outcome
            })
    
    # Prepare the final results
    champion_name = team_results[champion_abbrev]['team_name'] if champion_abbrev else "No Champion"
    champion_path = []
    
    # Create champion's path through the playoffs
    if champion_abbrev:
        champion_path = team_results[champion_abbrev]['series_wins']
    
    return {
        'team_results': team_results,
        'bracket_progression': bracket_progression,
        'champion': {
            'abbrev': champion_abbrev,
            'name': champion_name,
            'path': champion_path
        },
        'runner_up': runner_up_abbrev
    }

def get_simulation_results():
    """Get current simulation results or run new simulations if needed"""
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    season_str = f"{current_season}{current_season+1}"
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # If we already have daily simulations in session state, return those
    if 'daily_simulations' in st.session_state and st.session_state.daily_simulations is not None:
        return st.session_state.daily_simulations
    
    # Otherwise, try to load from file
    try:
        results_df = data_handlers.load_data(f"playoff_sim_results_{season_str}.csv", data_folder)
        if results_df is not None and not results_df.empty:
            # We have results but only partial data, so create a basic results object
            results = {'team_advancement': results_df}
            
            # Try to load additional data if available
            try:
                round2_df = data_handlers.load_data(f"playoff_round2_matchups_{season_str}.csv", data_folder)
                conf_final_df = data_handlers.load_data(f"playoff_conf_finals_matchups_{season_str}.csv", data_folder)
                final_df = data_handlers.load_data(f"playoff_stanley_cup_matchups_{season_str}.csv", data_folder)
                
                if round2_df is not None and conf_final_df is not None and final_df is not None:
                    results.update({
                        'round2_matchups': round2_df,
                        'conf_final_matchups': conf_final_df,
                        'final_matchups': final_df
                    })
            except Exception as e:
                # If we can't load the additional data, just continue with the basic results
                st.warning(f"Could not load detailed simulation results: {str(e)}")
            
            # Store in session state for future use
            st.session_state.daily_simulations = results
            
            return results
        else:
            st.warning("No saved simulation results found. Running new simulations...")
    except Exception as e:
        st.warning(f"Error loading simulation results: {str(e)}")
        
    # If we reach here, we need to run new simulations
    if update_daily_simulations(force=True):
        return st.session_state.daily_simulations
    else:
        st.error("Failed to run simulations.")
        return None

def format_simulation_results(sim_results, top_n=10):
    """Format simulation results for display in Streamlit app"""
    if not sim_results or 'team_advancement' not in sim_results:
        return None
    
    results_df = sim_results['team_advancement']
    
    # Get top teams by championship probability
    top_teams = results_df.sort_values('champion', ascending=False).head(top_n)
    
    # Format percentages for better readability
    results_display = top_teams.copy()
    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
        results_display[col] = (results_display[col] * 100).round(1)
    results_display['avg_games_played'] = results_display['avg_games_played'].round(1)
    
    return results_display
