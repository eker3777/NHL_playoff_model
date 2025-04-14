import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

from nhlpy.nhl_client import NHLClient
from nhlpy.api.query.filters.franchise import FranchiseQuery
from nhlpy.api.query.filters.shoot_catch import ShootCatchesQuery
from nhlpy.api.query.filters.draft import DraftQuery
from nhlpy.api.query.filters.season import SeasonQuery
from nhlpy.api.query.filters.game_type import GameTypeQuery
from nhlpy.api.query.filters.position import PositionQuery, PositionTypes
from nhlpy.api.query.filters.status import StatusQuery
from nhlpy.api.query.filters.opponent import OpponentQuery
from nhlpy.api.query.filters.home_road import HomeRoadQuery
from nhlpy.api.query.filters.experience import ExperienceQuery
from nhlpy.api.query.filters.decision import DecisionQuery
from nhlpy.api.query.builder import QueryBuilder, QueryContext

# Set page configuration
st.set_page_config(
    page_title="NHL Playoff Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Folder paths
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Create folders if they don't exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Initialize NHL client
client = NHLClient()

# Constants
current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
season_str = f"{current_season}{current_season+1}"

# Cache data to avoid excessive API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_standings_data():
    """Get current NHL standings data"""
    try:
        standings_data = client.standings.get_standings()
        return standings_data
    except Exception as e:
        st.error(f"Error fetching standings data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_team_stats_data(start_season=current_season, end_season=current_season):
    """Get team summary stats data"""
    try:
        stats_data = client.stats.team_summary(
            start_season, 
            end_season, 
        )
        return stats_data
    except Exception as e:
        st.error(f"Error fetching team stats data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_standings_data(standings_data):
    """Process NHL standings data into a DataFrame"""
    all_standings = []
    
    if isinstance(standings_data, dict):
        # Handle format with 'records' key
        if "records" in standings_data:
            # Loop through each division record
            for division_record in standings_data["records"]:
                division_name = division_record.get("division", {}).get("name", "Unknown")
                conference_name = division_record.get("conference", {}).get("name", "Unknown") if "conference" in division_record else "Unknown"
                
                for team_record in division_record["teamRecords"]:
                    # Start with season and division/conference info
                    team_data = {
                        "season": season_str,
                        "division": division_name,
                        "conference": conference_name
                    }
                    # Process team info
                    team_data["teamId"] = team_record["team"]["id"]
                    team_data["teamName"] = team_record["team"]["name"]
                    team_data["teamAbbrev"] = team_record["team"].get("abbreviation", "")
                    team_data["teamLogo"] = team_record["team"].get("logo", "")
                    
                    # Process all other fields
                    for key, value in team_record.items():
                        if key == "team":  # Already handled above
                            continue
                        
                        if isinstance(value, dict):
                            # Handle nested dictionaries by flattening them
                            for sub_key, sub_value in value.items():
                                team_data[f"{key}_{sub_key}"] = sub_value
                        else:
                            # Regular fields
                            team_data[key] = value
                    
                    all_standings.append(team_data)
        else:
            st.warning(f"Unknown standings data format: {list(standings_data.keys())}")
    else:
        st.warning("Non-dictionary standings data response")
    
    # Convert to DataFrame
    if all_standings:
        return pd.DataFrame(all_standings)
    return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_team_stats_data(stats_data):
    """Process team stats data into a DataFrame"""
    all_stats_data = []
    
    if isinstance(stats_data, list):
        for team_stats in stats_data:
            team_data = {}
            for key, value in team_stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        team_data[f"{key}_{sub_key}"] = sub_value
                else:
                    team_data[key] = value
            all_stats_data.append(team_data)
    else:
        st.warning("Unexpected team stats data format")
    
    # Convert to DataFrame
    if all_stats_data:
        stats_df = pd.DataFrame(all_stats_data)
        
        # Select and rename key columns
        if not stats_df.empty:
            # Format percentages
            pct_cols = [col for col in stats_df.columns if 'Pct' in col]
            for col in pct_cols:
                if col in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[col]):
                    stats_df[col] = (stats_df[col] * 100).round(1).astype(str) + '%'
            
            # Rename columns for better readability
            column_rename = {
                'seasonId': 'season',
                'teamFullName': 'team_name',
                'teamAbbrev': 'team',
                'gamesPlayed': 'GP',
                'wins': 'W',
                'losses': 'L',
                'otLosses': 'OTL',
                'points': 'PTS',
                'pointPct': 'PTS%',
                'goalsFor': 'GF',
                'goalsAgainst': 'GA',
                'goalsForPerGame': 'GF/G',
                'goalsAgainstPerGame': 'GA/G',
                'powerPlayPct': 'PP%',
                'penaltyKillPct': 'PK%',
                'faceoffWinPct': 'FO%'
            }
            
            # Apply renaming for columns that exist
            rename_dict = {k: v for k, v in column_rename.items() if k in stats_df.columns}
            stats_df = stats_df.rename(columns=rename_dict)
            
            return stats_df
    return pd.DataFrame()

# Function to get and process advanced stats data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_advanced_stats_data():
    """
    Get advanced stats from MoneyPuck or another source.
    This is a placeholder - you'll need to implement your own data source.
    """
    # Placeholder for MoneyPuck data or other advanced stats sources
    # In a real implementation, you would fetch this data from an API or webpage
    
    # For now, return an empty DataFrame with the expected columns
    columns = [
        'team', 'season', 'situation', 'games_played',
        'xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage',
        'flurryScoreVenueAdjustedxGoalsAgainst', 'goalsAgainst',
        'flurryScoreVenueAdjustedxGoalsFor', 'goalsFor',
        'hitsFor', 'hitsAgainst', 'takeawaysFor', 'takeawaysAgainst',
        'giveawaysFor', 'giveawaysAgainst', 'dZoneGiveawaysFor', 'dZoneGiveawaysAgainst',
        'reboundxGoalsFor', 'reboundxGoalsAgainst', 'iceTime'
    ]
    
    # TODO: Replace this with actual API call to get advanced stats
    return pd.DataFrame(columns=columns)

# Function to engineer features from raw data
def engineer_features(combined_data):
    """Engineer features for the prediction model"""
    # Create a copy of the data to avoid modifying original
    df = combined_data.copy()
    
    # Convert percentage strings to floats where needed
    for col in df.columns:
        if isinstance(df[col].iloc[0], str) and '%' in df[col].iloc[0]:
            df[col] = df[col].str.rstrip('%').astype(float) / 100
    
    # Calculate additional metrics from the data
    if all(col in df.columns for col in ['goalDifferential', 'gamesPlayed']):
        df['goalDiff/G'] = df['goalDifferential'] / df['gamesPlayed']
    
    if all(col in df.columns for col in ['homeRegulationWins', 'gamesPlayed']):
        df['homeRegulationWin%'] = df['homeRegulationWins'] / df['gamesPlayed']
    
    if all(col in df.columns for col in ['roadRegulationWins', 'gamesPlayed']):
        df['roadRegulationWin%'] = df['roadRegulationWins'] / df['gamesPlayed']
    
    # Advanced metrics - add checks to ensure columns exist before calculating
    if all(col in df.columns for col in ['flurryScoreVenueAdjustedxGoalsAgainst', 'goalsAgainst', 'iceTime']):
        df['adjGoalsSavedAboveX/60'] = (df['flurryScoreVenueAdjustedxGoalsAgainst'] - df['goalsAgainst']) / df['iceTime'] * 60
    
    if all(col in df.columns for col in ['goalsFor', 'flurryScoreVenueAdjustedxGoalsFor', 'iceTime']):
        df['adjGoalsScoredAboveX/60'] = (df['goalsFor'] - df['flurryScoreVenueAdjustedxGoalsFor']) / df['iceTime'] * 60
    
    # Calculate possession-normalized metrics if columns exist
    cols_for_hits = ['hitsFor', 'hitsAgainst']
    if all(col in df.columns for col in cols_for_hits):
        df['hitsPctg'] = df['hitsFor'] / (df['hitsAgainst'] + df['hitsFor'])
    
    cols_for_takeaways = ['takeawaysFor', 'takeawaysAgainst']
    if all(col in df.columns for col in cols_for_takeaways):
        df['takeawaysPctg'] = df['takeawaysFor'] / (df['takeawaysAgainst'] + df['takeawaysFor'])
    
    cols_for_giveaways = ['giveawaysFor', 'giveawaysAgainst']
    if all(col in df.columns for col in cols_for_giveaways):
        df['giveawaysPctg'] = df['giveawaysFor'] / (df['giveawaysAgainst'] + df['giveawaysFor'])
    
    cols_for_dzone = ['dZoneGiveawaysFor', 'dZoneGiveawaysAgainst']
    if all(col in df.columns for col in cols_for_dzone):
        df['dZoneGiveawaysPctg'] = df['dZoneGiveawaysFor'] / (df['dZoneGiveawaysAgainst'] + df['dZoneGiveawaysFor'])
    
    # Apply possession adjustment if corsiPercentage exists
    if 'corsiPercentage' in df.columns and 'hitsPctg' in df.columns:
        df['possAdjHitsPctg'] = df['hitsPctg'] * (0.5 / (1 - df['corsiPercentage'].clip(0.01, 0.99)))
    
    if 'corsiPercentage' in df.columns and 'takeawaysPctg' in df.columns:
        df['possAdjTakeawaysPctg'] = df['takeawaysPctg'] * (0.5 / (1 - df['corsiPercentage'].clip(0.01, 0.99)))
    
    if 'corsiPercentage' in df.columns and 'giveawaysPctg' in df.columns:
        df['possAdjGiveawaysPctg'] = df['giveawaysPctg'] * (0.5 / df['corsiPercentage'].clip(0.01, 0.99))
    
    if 'corsiPercentage' in df.columns and 'dZoneGiveawaysPctg' in df.columns:
        df['possAdjdZoneGiveawaysPctg'] = df['dZoneGiveawaysPctg'] * (0.5 / df['corsiPercentage'].clip(0.01, 0.99))
    
    if all(col in df.columns for col in ['possAdjGiveawaysPctg', 'possAdjdZoneGiveawaysPctg']):
        df['possTypeAdjGiveawaysPctg'] = df['possAdjGiveawaysPctg'] * 1/3 + df['possAdjdZoneGiveawaysPctg'] * 2/3
    
    if all(col in df.columns for col in ['reboundxGoalsFor', 'reboundxGoalsAgainst']):
        df['reboundxGoalsPctg'] = df['reboundxGoalsFor'] / (df['reboundxGoalsFor'] + df['reboundxGoalsAgainst'])
    
    # Calculate special teams metrics if they exist
    if all(col in df.columns for col in ['PP%', 'PK%']):
        # First, convert percentage strings to floats if needed
        for col in ['PP%', 'PK%']:
            if isinstance(df[col].iloc[0], str) and '%' in df[col].iloc[0]:
                df[col] = df[col].str.rstrip('%').astype(float) / 100
        
        # Calculate league averages
        league_avg_pp = df['PP%'].mean()
        league_avg_pk = df['PK%'].mean()
        
        # Calculate relative metrics
        df['PP%_rel'] = df['PP%'] - league_avg_pp
        df['PK%_rel'] = df['PK%'] - league_avg_pk
        
        # Create composite special teams metric
        df['special_teams_composite'] = df['PP%_rel'] + df['PK%_rel']
    
    # Return the dataframe with engineered features
    return df

# Function to determine playoff teams and create matchups
def determine_playoff_teams(standings_df):
    """
    Determine which teams make the playoffs based on NHL rules.
    
    Returns:
        dict: Dictionary with playoff matchups
    """
    # Ensure standings DataFrame is sorted properly
    if standings_df.empty:
        st.error("No standings data available")
        return {}
    
    # Make sure we have the required columns
    required_cols = ['conference', 'division', 'teamName', 'teamAbbrev', 'points', 'teamLogo']
    if not all(col in standings_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in standings_df.columns]
        st.error(f"Missing required columns in standings data: {missing}")
        return {}
    
    # Group by conference and division
    conferences = standings_df['conference'].unique()
    playoff_matchups = {}
    
    for conference in conferences:
        conf_teams = standings_df[standings_df['conference'] == conference].copy()
        divisions = conf_teams['division'].unique()
        
        division_qualifiers = []
        wildcard_candidates = []
        
        # Get top 3 from each division
        for division in divisions:
            div_teams = conf_teams[conf_teams['division'] == division].copy()
            div_teams = div_teams.sort_values(by=['points', 'row'], ascending=False)
            
            # Top 3 teams in division qualify automatically
            division_top3 = div_teams.head(3).copy()
            division_top3['seed_type'] = 'division'
            division_top3['division_rank'] = range(1, 4)
            
            division_qualifiers.append(division_top3)
            
            # Remaining teams are wildcard candidates
            wildcard_candidates.append(div_teams.iloc[3:].copy())
        
        # Combine division qualifiers
        division_qualifiers_df = pd.concat(division_qualifiers)
        
        # Combine and rank wildcard candidates
        if wildcard_candidates:
            wildcard_df = pd.concat(wildcard_candidates)
            wildcard_df = wildcard_df.sort_values(by=['points', 'row'], ascending=False)
            
            # Top 2 wildcards make it
            wildcards = wildcard_df.head(2).copy()
            wildcards['seed_type'] = 'wildcard'
            wildcards['wildcard_rank'] = range(1, 3)
            
            # Combine all playoff teams for this conference
            conference_playoff_teams = pd.concat([division_qualifiers_df, wildcards])
            
            # Create matchups based on NHL playoff format
            # Get division winners
            div_winners = {}
            for division in divisions:
                div_winner = division_qualifiers_df[
                    (division_qualifiers_df['division'] == division) & 
                    (division_qualifiers_df['division_rank'] == 1)
                ].iloc[0]
                div_winners[division] = div_winner
            
            # Determine which wildcard plays which division winner
            # Better division winner plays worst wildcard
            div_winner_list = sorted(
                list(div_winners.values()), 
                key=lambda x: (x['points'], x['row']), 
                reverse=True
            )
            
            if len(wildcards) >= 2 and len(div_winner_list) >= 2:
                # Create matchups
                matchups = {}
                
                # A1 vs WC2
                matchups['A1_WC2'] = {
                    'top_seed': div_winner_list[0],
                    'bottom_seed': wildcards[wildcards['wildcard_rank'] == 2].iloc[0]
                }
                
                # B1 vs WC1
                matchups['B1_WC1'] = {
                    'top_seed': div_winner_list[1],
                    'bottom_seed': wildcards[wildcards['wildcard_rank'] == 1].iloc[0]
                }
                
                # Create other divisional matchups (A2 vs A3, B2 vs B3)
                for division in divisions:
                    div_teams = division_qualifiers_df[division_qualifiers_df['division'] == division]
                    
                    if len(div_teams) >= 3:
                        div_2 = div_teams[div_teams['division_rank'] == 2].iloc[0]
                        div_3 = div_teams[div_teams['division_rank'] == 3].iloc[0]
                        
                        matchups[f"{division[0]}2_{division[0]}3"] = {
                            'top_seed': div_2,
                            'bottom_seed': div_3
                        }
                
                playoff_matchups[conference] = matchups
    
    return playoff_matchups

# Function to create team comparison data for matchups
def create_matchup_data(top_seed, bottom_seed, team_data):
    """
    Create matchup data for model input
    
    Args:
        top_seed: Top seeded team info
        bottom_seed: Bottom seeded team info
        team_data: DataFrame with team stats and features
        
    Returns:
        DataFrame with matchup features
    """
    # Create a single row DataFrame for this matchup
    matchup_data = {}
    
    # Base matchup information
    matchup_data['season'] = current_season
    matchup_data['round'] = 1
    matchup_data['round_name'] = 'First Round'
    matchup_data['series_letter'] = 'TBD'
    matchup_data['top_seed_abbrev'] = top_seed['teamAbbrev']
    matchup_data['bottom_seed_abbrev'] = bottom_seed['teamAbbrev']
    matchup_data['top_seed_rank'] = top_seed.get('division_rank', top_seed.get('wildcard_rank', 0))
    matchup_data['bottom_seed_rank'] = bottom_seed.get('division_rank', bottom_seed.get('wildcard_rank', 0))
    matchup_data['top_seed_wins'] = 0
    matchup_data['bottom_seed_wins'] = 0
    
    # Get team data for each team
    top_seed_data = team_data[team_data['team'] == top_seed['teamAbbrev']].iloc[0] if len(team_data[team_data['team'] == top_seed['teamAbbrev']]) > 0 else None
    bottom_seed_data = team_data[team_data['team'] == bottom_seed['teamAbbrev']].iloc[0] if len(team_data[team_data['team'] == bottom_seed['teamAbbrev']]) > 0 else None
    
    # Feature columns to use for prediction
    feature_cols = [
        'PP%_rel', 'PK%_rel', 'FO%', 'special_teams_composite',
        'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
        'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
        'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
        'adjGoalsScoredAboveX/60'
    ]
    
    # Add features for each team if available
    if top_seed_data is not None and bottom_seed_data is not None:
        for col in feature_cols:
            if col in top_seed_data and col in bottom_seed_data:
                matchup_data[f"{col}_top"] = top_seed_data[col]
                matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
    
    return pd.DataFrame([matchup_data])

# Function to simulate a playoff series
def simulate_playoff_series(matchup_data, model, n_simulations=1000):
    """
    Simulate a playoff series using Monte Carlo simulation
    
    Args:
        matchup_data: DataFrame with matchup features
        model: Trained model for predictions
        n_simulations: Number of simulations to run
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Extract team abbreviations
    top_seed = matchup_data['top_seed_abbrev'].iloc[0]
    bottom_seed = matchup_data['bottom_seed_abbrev'].iloc[0]
    
    # If we have a model, use it to predict
    if model is not None:
        try:
            # Get probability of higher seed winning
            higher_seed_prob = model.predict_proba(matchup_data[model['features']])[:, 1][0]
            
            # Track results
            higher_seed_wins = 0
            win_distribution = {
                '4-0': 0, '4-1': 0, '4-2': 0, '4-3': 0,  # Higher seed wins
                '0-4': 0, '1-4': 0, '2-4': 0, '3-4': 0,  # Lower seed wins
            }
            
            # Run simulations
            for _ in range(n_simulations):
                higher_seed_score = 0
                lower_seed_score = 0
                
                # Simulate up to 7 games
                for game in range(1, 8):
                    # Adjust probability based on home ice
                    if game in [1, 2, 5, 7]:  # Higher seed has home ice
                        game_prob = min(higher_seed_prob + 0.05, 1.0)  # Home ice boost
                    else:  # Lower seed has home ice
                        game_prob = max(higher_seed_prob - 0.05, 0.0)  # Away disadvantage
                    
                    # Simulate game
                    if np.random.random() < game_prob:
                        higher_seed_score += 1
                    else:
                        lower_seed_score += 1
                    
                    # Check if series is over
                    if higher_seed_score == 4 or lower_seed_score == 4:
                        break
                
                # Record result
                if higher_seed_score > lower_seed_score:
                    higher_seed_wins += 1
                    win_distribution[f'4-{lower_seed_score}'] += 1
                else:
                    win_distribution[f'{higher_seed_score}-4'] += 1
            
            # Calculate win percentage
            higher_seed_win_pct = higher_seed_wins / n_simulations
            
            # Calculate confidence interval
            z = 1.96  # 95% confidence
            ci_width = z * np.sqrt((higher_seed_win_pct * (1 - higher_seed_win_pct)) / n_simulations)
            ci_lower = max(0, higher_seed_win_pct - ci_width)
            ci_upper = min(1, higher_seed_win_pct + ci_width)
            
            # Format results
            results = {
                'top_seed': top_seed,
                'bottom_seed': bottom_seed,
                'win_probability': higher_seed_win_pct,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'win_distribution': win_distribution
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in simulation: {str(e)}")
            return None
    else:
        # If no model, use simple probability based on points difference
        top_points = matchup_data.get('points_top', 0)
        bottom_points = matchup_data.get('points_bottom', 0)
        points_diff = top_points - bottom_points
        
        # Simple logistic function to convert points difference to probability
        higher_seed_win_pct = 1 / (1 + np.exp(-0.05 * points_diff))
        
        return {
            'top_seed': top_seed,
            'bottom_seed': bottom_seed,
            'win_probability': higher_seed_win_pct,
            'ci_lower': higher_seed_win_pct - 0.1,
            'ci_upper': higher_seed_win_pct + 0.1,
            'win_distribution': {}
        }

# Function to load team logos
@st.cache_data
def load_team_logo(logo_url):
    """Load team logo from URL"""
    try:
        response = requests.get(logo_url)
        return Image.open(BytesIO(response.content))
    except:
        # Return placeholder image if logo can't be loaded
        return None

# Function to simulate an entire playoff bracket
def simulate_bracket(playoff_matchups, team_data, model, n_simulations=100):
    """
    Simulate the entire playoff bracket
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team stats and features
        model: Trained model for predictions
        n_simulations: Number of simulations to run
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Track advancement for each team
    team_advancement = {}
    
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
                    'champion': 0
                }
            
            if bottom_seed not in team_advancement:
                team_advancement[bottom_seed] = {
                    'round_1': 0,
                    'round_2': 0,
                    'conf_final': 0,
                    'final': 0,
                    'champion': 0
                }
    
    # Run simulations
    for sim in range(n_simulations):
        # Track winners for each round
        round_1_winners = {}
        round_2_winners = {}
        conf_winners = {}
        champion = None
        
        # First round
        for conference, matchups in playoff_matchups.items():
            round_1_winners[conference] = {}
            
            for series_id, matchup in matchups.items():
                # Create matchup data
                matchup_df = create_matchup_data(matchup['top_seed'], matchup['bottom_seed'], team_data)
                
                # Simulate series
                sim_result = simulate_playoff_series(matchup_df, model, 1)
                
                if sim_result:
                    # Higher seed wins?
                    higher_seed_wins = np.random.random() < sim_result['win_probability']
                    winner = matchup['top_seed']['teamAbbrev'] if higher_seed_wins else matchup['bottom_seed']['teamAbbrev']
                    loser = matchup['bottom_seed']['teamAbbrev'] if higher_seed_wins else matchup['top_seed']['teamAbbrev']
                    
                    # Record advance
                    round_1_winners[conference][series_id] = winner
                    team_advancement[winner]['round_1'] += 1
        
        # Second round (Division Finals)
        for conference, r1_winners in round_1_winners.items():
            round_2_winners[conference] = {}
            
            # Determine matchups
            div_matchups = {}
            
            # For each conference, we'll have two divisional finals
            # The winners of A1/WC2 will play winner of A2/A3
            # The winners of B1/WC1 will play winner of B2/B3
            
            if len(r1_winners) >= 4:  # Make sure we have all 4 series results
                keys = list(r1_winners.keys())
                
                # Find division identifiers
                divisions = set()
                for key in keys:
                    if key[0].isalpha() and key[1].isdigit():
                        divisions.add(key[0])
                
                # Create second round matchups
                for div in divisions:
                    # Find first matchup winner (div1 vs wildcard)
                    div1_wc = next((k for k in keys if k.startswith(f"{div}1_")), None)
                    # Find second matchup winner (div2 vs div3)
                    div2_3 = next((k for k in keys if k.startswith(f"{div}2_")), None)
                    
                    if div1_wc and div2_3:
                        winner1 = r1_winners.get(div1_wc)
                        winner2 = r1_winners.get(div2_3)
                        
                        if winner1 and winner2:
                            # Get team data
                            team1 = team_data[team_data['team'] == winner1].iloc[0] if len(team_data[team_data['team'] == winner1]) > 0 else None
                            team2 = team_data[team_data['team'] == winner2].iloc[0] if len(team_data[team_data['team'] == winner2]) > 0 else None
                            
                            if team1 is not None and team2 is not None:
                                # Determine higher seed
                                points1 = team1.get('PTS', 0)
                                points2 = team2.get('PTS', 0)
                                
                                if points1 >= points2:
                                    top_seed = {'teamAbbrev': winner1}
                                    bottom_seed = {'teamAbbrev': winner2}
                                else:
                                    top_seed = {'teamAbbrev': winner2}
                                    bottom_seed = {'teamAbbrev': winner1}
                                
                                # Create matchup data
                                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                                
                                # Simulate series
                                sim_result = simulate_playoff_series(matchup_df, model, 1)
                                
                                if sim_result:
                                    # Higher seed wins?
                                    higher_seed_wins = np.random.random() < sim_result['win_probability']
                                    winner = top_seed['teamAbbrev'] if higher_seed_wins else bottom_seed['teamAbbrev']
                                    loser = bottom_seed['teamAbbrev'] if higher_seed_wins else top_seed['teamAbbrev']
                                    
                                    # Record advance
                                    round_2_winners[conference][f"{div}_final"] = winner
                                    team_advancement[winner]['round_2'] += 1
        
        # Conference Finals
        for conference, r2_winners in round_2_winners.items():
            if len(r2_winners) >= 2:  # Make sure we have both divisional final results
                # Get the two winners
                winners = list(r2_winners.values())
                
                if len(winners) == 2:
                    # Get team data
                    team1 = team_data[team_data['team'] == winners[0]].iloc[0] if len(team_data[team_data['team'] == winners[0]]) > 0 else None
                    team2 = team_data[team_data['team'] == winners[1]].iloc[0] if len(team_data[team_data['team'] == winners[1]]) > 0 else None
                    
                    if team1 is not None and team2 is not None:
                        # Determine higher seed
                        points1 = team1.get('PTS', 0)
                        points2 = team2.get('PTS', 0)
                        
                        if points1 >= points2:
                            top_seed = {'teamAbbrev': winners[0]}
                            bottom_seed = {'teamAbbrev': winners[1]}
                        else:
                            top_seed = {'teamAbbrev': winners[1]}
                            bottom_seed = {'teamAbbrev': winners[0]}
                        
                        # Create matchup data
                        matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                        
                        # Simulate series
                        sim_result = simulate_playoff_series(matchup_df, model, 1)
                        
                        if sim_result:
                            # Higher seed wins?
                            higher_seed_wins = np.random.random() < sim_result['win_probability']
                            winner = top_seed['teamAbbrev'] if higher_seed_wins else bottom_seed['teamAbbrev']
                            loser = bottom_seed['teamAbbrev'] if higher_seed_wins else top_seed['teamAbbrev']
                            
                            # Record advance
                            conf_winners[conference] = winner
                            team_advancement[winner]['conf_final'] += 1
        
        # Stanley Cup Final
        if len(conf_winners) == 2:
            # Get the two conference champions
            finalists = list(conf_winners.values())
            
            if len(finalists) == 2:
                # Get team data
                team1 = team_data[team_data['team'] == finalists[0]].iloc[0] if len(team_data[team_data['team'] == finalists[0]]) > 0 else None
                team2 = team_data[team_data['team'] == finalists[1]].iloc[0] if len(team_data[team_data['team'] == finalists[1]]) > 0 else None
                
                if team1 is not None and team2 is not None:
                    # Determine higher seed
                    points1 = team1.get('PTS', 0)
                    points2 = team2.get('PTS', 0)
                    
                    if points1 >= points2:
                        top_seed = {'teamAbbrev': finalists[0]}
                        bottom_seed = {'teamAbbrev': finalists[1]}
                    else:
                        top_seed = {'teamAbbrev': finalists[1]}
                        bottom_seed = {'teamAbbrev': finalists[0]}
                    
                    # Create matchup data
                    matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                    
                    # Simulate series
                    sim_result = simulate_playoff_series(matchup_df, model, 1)
                    
                    if sim_result:
                        # Higher seed wins?
                        higher_seed_wins = np.random.random() < sim_result['win_probability']
                        winner = top_seed['teamAbbrev'] if higher_seed_wins else bottom_seed['teamAbbrev']
                        loser = bottom_seed['teamAbbrev'] if higher_seed_wins else top_seed['teamAbbrev']
                        
                        # Record advance
                        champion = winner
                        team_advancement[winner]['champion'] += 1
                        
                        # Record finalist
                        team_advancement[winner]['final'] += 1
                        team_advancement[loser]['final'] += 1
    
    # Calculate advancement percentages
    for team, rounds in team_advancement.items():
        for round_name in rounds:
            team_advancement[team][round_name] = rounds[round_name] / n_simulations
    
    return team_advancement

# Load trained model
@st.cache_resource
def load_model(model_path=os.path.join(model_folder, 'playoff_model.pkl')):
    """Load the trained model"""
    try:
        model_data = joblib.load(model_path)
        return model_data
    except:
        st.warning(f"Model file not found at {model_path}. Using simplified predictions.")
        return None

# Main app
def main():
    # App title and description
    st.title("NHL Playoff Predictor")
    st.write("Predict playoff outcomes based on team statistics and advanced metrics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Playoff Bracket", "Team Comparison", "Simulation Results", "About"])
    
    # Get data
    with st.spinner("Fetching NHL data..."):
        # Get standings data
        standings_data = get_standings_data()
        standings_df = process_standings_data(standings_data)
        
        # Get team stats data
        stats_data = get_team_stats_data()
        stats_df = process_team_stats_data(stats_data)
        
        # Get advanced stats data
        advanced_stats_df = get_advanced_stats_data()
        
        # Combine data
        if not standings_df.empty and not stats_df.empty:
            # Merge standings and stats data
            team_data = pd.merge(standings_df, stats_df, on=['season', 'team'], how='left')
            
            # Merge with advanced stats if available
            if not advanced_stats_df.empty:
                team_data = pd.merge(team_data, advanced_stats_df, on=['season', 'team'], how='left')
            
            # Engineer features for prediction
            team_data = engineer_features(team_data)
        else:
            team_data = pd.DataFrame()
    
    # Load model
    model = load_model()
    
    # Determine playoff teams and create matchups
    playoff_matchups = determine_playoff_teams(standings_df)
    
    # Playoff Bracket page
    if page == "Playoff Bracket":
        st.header("NHL Playoff Bracket")
        
        # Check if we have matchups
        if playoff_matchups:
            # Button to run simulation
            if st.button("Simulate Playoff Bracket", key="sim_bracket"):
                with st.spinner("Running playoff simulation..."):
                    # Simulate entire bracket
                    bracket_results = simulate_bracket(playoff_matchups, team_data, model)
                    
                    # Display results
                    st.subheader("Playoff Advancement Probabilities")
                    
                    # Convert results to DataFrame for display
                    results_df = pd.DataFrame.from_dict(bracket_results, orient='index')
                    results_df = results_df.sort_values(by='champion', ascending=False)
                    
                    # Format percentages
                    for col in results_df.columns:
                        results_df[col] = (results_df[col] * 100).round(1).astype(str) + '%'
                    
                    # Rename columns for display
                    results_df = results_df.rename(columns={
                        'round_1': 'First Round',
                        'round_2': 'Second Round',
                        'conf_final': 'Conf. Finals',
                        'final': 'Finals',
                        'champion': 'Champion'
                    })
                    
                    # Display as table
                    st.table(results_df)
            
            # Display current playoff matchups
            st.subheader("Current Playoff Matchups")
            
            # Display each conference's matchups
            for conference, matchups in playoff_matchups.items():
                st.write(f"### {conference}ern Conference")
                
                for series_id, matchup in matchups.items():
                    top_seed = matchup['top_seed']
                    bottom_seed = matchup['bottom_seed']
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.write(f"**{top_seed['teamName']}** ({top_seed['teamAbbrev']})")
                        # Display logo if available
                        if 'teamLogo' in top_seed and top_seed['teamLogo']:
                            logo = load_team_logo(top_seed['teamLogo'])
                            if logo:
                                st.image(logo, width=100)
                    
                    with col2:
                        st.write("**vs**")
                    
                    with col3:
                        st.write(f"**{bottom_seed['teamName']}** ({bottom_seed['teamAbbrev']})")
                        # Display logo if available
                        if 'teamLogo' in bottom_seed and bottom_seed['teamLogo']:
                            logo = load_team_logo(bottom_seed['teamLogo'])
                            if logo:
                                st.image(logo, width=100)
                    
                    # Create matchup data for this series
                    matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                    
                    # Simulate this series specifically
                    if not matchup_df.empty and model is not None:
                        sim_result = simulate_playoff_series(matchup_df, model)
                        
                        if sim_result:
                            # Display prediction
                            win_pct = sim_result['win_probability'] * 100
                            
                            st.write(f"**Prediction:** {top_seed['teamAbbrev']} has a **{win_pct:.1f}%** chance of winning the series")
                            
                            # Display win distribution if available
                            if sim_result.get('win_distribution'):
                                dist = sim_result['win_distribution']
                                
                                # Convert to percentages
                                total = sum(dist.values())
                                if total > 0:
                                    for k, v in dist.items():
                                        dist[k] = (v / total) * 100
                                
                                # Create bar chart of win distribution
                                fig, ax = plt.subplots(figsize=(8, 3))
                                
                                # Higher seed outcomes
                                higher_outcomes = ['4-0', '4-1', '4-2', '4-3']
                                higher_values = [dist.get(outcome, 0) for outcome in higher_outcomes]
                                
                                # Lower seed outcomes
                                lower_outcomes = ['0-4', '1-4', '2-4', '3-4']
                                lower_values = [dist.get(outcome, 0) for outcome in lower_outcomes]
                                
                                # Plot
                                x = np.arange(4)
                                width = 0.35
                                
                                ax.bar(x - width/2, higher_values, width, label=top_seed['teamAbbrev'])
                                ax.bar(x + width/2, lower_values, width, label=bottom_seed['teamAbbrev'])
                                
                                ax.set_xticks(x)
                                ax.set_xticklabels(['4-0', '4-1', '4-2', '4-3'])
                                ax.set_ylabel('Probability (%)')
                                ax.set_title('Series Outcome Distribution')
                                ax.legend()
                                
                                st.pyplot(fig)
                    
                    st.write("---")
        else:
            st.warning("Could not determine playoff matchups from current standings data")
    
    # Team Comparison page
    elif page == "Team Comparison":
        st.header("Team Comparison")
        
        # Get list of teams
        if not team_data.empty:
            teams = team_data['teamName'].unique()
            
            # Team selection
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Select first team", teams, key='team1')
            
            with col2:
                team2 = st.selectbox("Select second team", teams, index=1, key='team2')
            
            # Get team data
            if team1 != team2:
                team1_data = team_data[team_data['teamName'] == team1].iloc[0]
                team2_data = team_data[team_data['teamName'] == team2].iloc[0]
                
                # Display team logos if available
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"### {team1}")
                    if 'teamLogo' in team1_data and team1_data['teamLogo']:
                        logo = load_team_logo(team1_data['teamLogo'])
                        if logo:
                            st.image(logo, width=150)
                
                with col2:
                    st.write(f"### {team2}")
                    if 'teamLogo' in team2_data and team2_data['teamLogo']:
                        logo = load_team_logo(team2_data['teamLogo'])
                        if logo:
                            st.image(logo, width=150)
                
                # Select stats to compare
                st.subheader("Select Stats to Compare")
                
                # Group stats into categories
                basic_stats = ['PTS', 'W', 'L', 'OTL', 'GP', 'GF', 'GA', 'GF/G', 'GA/G']
                
                if st.checkbox("Basic Team Stats", value=True):
                    # Create comparison table
                    stats_to_compare = []
                    for stat in basic_stats:
                        if stat in team1_data and stat in team2_data:
                            stats_to_compare.append(stat)
                    
                    if stats_to_compare:
                        compare_df = pd.DataFrame({
                            'Stat': stats_to_compare,
                            team1: [team1_data[stat] for stat in stats_to_compare],
                            team2: [team2_data[stat] for stat in stats_to_compare]
                        })
                        
                        # Determine which team has better stats (higher is better by default)
                        compare_df['Better'] = compare_df.apply(
                            lambda row: team1 if row[team1] > row[team2] else
                                       (team2 if row[team2] > row[team1] else 'Tie'),
                            axis=1
                        )
                        
                        # For certain stats, lower is better
                        lower_is_better = ['GA', 'GA/G', 'L', 'OTL']
                        for stat in lower_is_better:
                            if stat in stats_to_compare:
                                idx = compare_df[compare_df['Stat'] == stat].index
                                compare_df.loc[idx, 'Better'] = team1 if team1_data[stat] < team2_data[stat] else (team2 if team2_data[stat] < team1_data[stat] else 'Tie')
                        
                        # Display the comparison
                        st.subheader("Team Stats Comparison")
                        for _, row in compare_df.iterrows():
                            stat = row['Stat']
                            val1 = row[team1]
                            val2 = row[team2]
                            better = row['Better']
                            
                            col1, col2, col3 = st.columns([2, 1, 2])
                            
                            with col1:
                                if better == team1:
                                    st.markdown(f"<p style='color:green'><b>{val1}</b></p>", unsafe_allow_html=True)
                                else:
                                    st.write(val1)
                            
                            with col2:
                                st.write(f"**{stat}**")
                            
                            with col3:
                                if better == team2:
                                    st.markdown(f"<p style='color:green'><b>{val2}</b></p>", unsafe_allow_html=True)
                                else:
                                    st.write(val2)
                
                # Special teams comparison
                if st.checkbox("Special Teams", value=True):
                    special_teams = ['PP%', 'PK%', 'PP%_rel', 'PK%_rel', 'special_teams_composite', 'FO%']
                    stats_to_compare = []
                    
                    for stat in special_teams:
                        if stat in team1_data and stat in team2_data:
                            stats_to_compare.append(stat)
                    
                    if stats_to_compare:
                        st.subheader("Special Teams Comparison")
                        
                        for stat in stats_to_compare:
                            val1 = team1_data[stat]
                            val2 = team2_data[stat]
                            
                            col1, col2, col3 = st.columns([2, 1, 2])
                            
                            with col1:
                                if val1 > val2:
                                    st.markdown(f"<p style='color:green'><b>{val1}</b></p>", unsafe_allow_html=True)
                                else:
                                    st.write(val1)
                            
                            with col2:
                                st.write(f"**{stat}**")
                            
                            with col3:
                                if val2 > val1:
                                    st.markdown(f"<p style='color:green'><b>{val2}</b></p>", unsafe_allow_html=True)
                                else:
                                    st.write(val2)
                
                # Advanced stats comparison if available
                advanced_stats = [
                    'xGoalsPercentage', 'adjGoalsSavedAboveX/60', 'adjGoalsScoredAboveX/60',
                    'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
                    'reboundxGoalsPctg', 'goalDiff/G'
                ]
                
                available_advanced = [stat for stat in advanced_stats if stat in team1_data and stat in team2_data]
                
                if available_advanced and st.checkbox("Advanced Stats", value=True):
                    st.subheader("Advanced Stats Comparison")
                    
                    for stat in available_advanced:
                        val1 = team1_data[stat]
                        val2 = team2_data[stat]
                        
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            if val1 > val2:
                                st.markdown(f"<p style='color:green'><b>{val1:.3f}</b></p>", unsafe_allow_html=True)
                            else:
                                st.write(f"{val1:.3f}")
                        
                        with col2:
                            st.write(f"**{stat}**")
                        
                        with col3:
                            if val2 > val1:
                                st.markdown(f"<p style='color:green'><b>{val2:.3f}</b></p>", unsafe_allow_html=True)
                            else:
                                st.write(f"{val2:.3f}")
                
                # Simulate head-to-head series
                st.subheader("Head-to-Head Series Prediction")
                
                # Create artificial matchup data
                top_seed = {'teamAbbrev': team1_data['team'], 'division_rank': 1}
                bottom_seed = {'teamAbbrev': team2_data['team'], 'division_rank': 2}
                
                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                
                # Simulate series
                if not matchup_df.empty and model is not None:
                    sim_result = simulate_playoff_series(matchup_df, model)
                    
                    if sim_result:
                        # Display prediction
                        win_pct = sim_result['win_probability'] * 100
                        
                        st.write(f"In a playoff series between these teams, **{team1}** would have a **{win_pct:.1f}%** chance of winning.")
                        st.write(f"**{team2}** would have a **{(100-win_pct):.1f}%** chance of winning.")
                        
                        # Display win distribution if available
                        if sim_result.get('win_distribution'):
                            dist = sim_result['win_distribution']
                            
                            # Convert to percentages
                            total = sum(dist.values())
                            if total > 0:
                                for k, v in dist.items():
                                    dist[k] = (v / total) * 100
                            
                            # Create bar chart of win distribution
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            # Higher seed outcomes
                            higher_outcomes = ['4-0', '4-1', '4-2', '4-3']
                            higher_values = [dist.get(outcome, 0) for outcome in higher_outcomes]
                            
                            # Lower seed outcomes
                            lower_outcomes = ['0-4', '1-4', '2-4', '3-4']
                            lower_values = [dist.get(outcome, 0) for outcome in lower_outcomes]
                            
                            # Plot
                            x = np.arange(4)
                            width = 0.35
                            
                            ax.bar(x - width/2, higher_values, width, label=team1)
                            ax.bar(x + width/2, lower_values, width, label=team2)
                            
                            ax.set_xticks(x)
                            ax.set_xticklabels(['4-0', '4-1', '4-2', '4-3'])
                            ax.set_ylabel('Probability (%)')
                            ax.set_title('Predicted Series Outcome Distribution')
                            ax.legend()
                            
                            st.pyplot(fig)
                    else:
                        st.warning("Could not simulate series outcome")
                else:
                    st.warning("Insufficient data to predict series outcome")
            
            else:
                st.warning("Please select two different teams")
        else:
            st.error("No team data available")
    
    # Simulation Results page
    elif page == "Simulation Results":
        st.header("Playoff Simulation Results")
        
        # Simulation options
        st.subheader("Simulation Settings")
        n_simulations = st.slider("Number of simulations", 100, 10000, 1000, 100)
        
        if st.button("Run Full Playoff Simulation", key="run_sim"):
            if playoff_matchups and not team_data.empty:
                with st.spinner(f"Running {n_simulations} playoff simulations..."):
                    # Simulate entire bracket
                    bracket_results = simulate_bracket(playoff_matchups, team_data, model, n_simulations)
                    
                    # Display results
                    st.subheader("Playoff Advancement Probabilities")
                    
                    # Convert results to DataFrame for display
                    results_df = pd.DataFrame.from_dict(bracket_results, orient='index')
                    
                    # Add team names
                    team_name_map = {}
                    for _, row in team_data.iterrows():
                        team_name_map[row['team']] = row['teamName']
                    
                    results_df['Team'] = results_df.index.map(lambda x: team_name_map.get(x, x))
                    
                    # Sort by championship probability
                    results_df = results_df.sort_values(by='champion', ascending=False)
                    
                    # Format percentages
                    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
                        results_df[col] = (results_df[col] * 100).round(1)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot championship probabilities
                    teams = results_df.index.tolist()
                    championship_probs = results_df['champion'].tolist()
                    
                    # Sort for visualization
                    teams_sorted = [x for _, x in sorted(zip(championship_probs, teams), reverse=True)]
                    probs_sorted = sorted(championship_probs, reverse=True)
                    
                    # Only show top 10 teams for clarity
                    teams_sorted = teams_sorted[:10]
                    probs_sorted = probs_sorted[:10]
                    
                    ax.barh(teams_sorted, probs_sorted)
                    ax.set_xlabel('Championship Probability (%)')
                    ax.set_title('Stanley Cup Championship Odds')
                    
                    # Display the chart
                    st.pyplot(fig)
                    
                    # Rename columns for display
                    results_df = results_df.rename(columns={
                        'round_1': 'First Round %',
                        'round_2': 'Second Round %',
                        'conf_final': 'Conf. Finals %',
                        'final': 'Finals %',
                        'champion': 'Champion %'
                    })
                    
                    # Reorder columns
                    cols = ['Team', 'Champion %', 'Finals %', 'Conf. Finals %', 'Second Round %', 'First Round %']
                    results_df = results_df[cols]
                    
                    # Display as table
                    st.table(results_df)
            else:
                st.error("Playoff matchups or team data not available for simulation")
    
    # About page
    elif page == "About":
        st.header("About the NHL Playoff Predictor")
        st.write("""
        This application predicts NHL playoff outcomes based on team statistics and advanced metrics.
        
        The model uses features such as:
        - Special teams performance
        - Possession metrics
        - Goal differentials
        - Advanced stats like expected goals
        
        Each playoff matchup is simulated multiple times to account for the randomness inherent in playoff hockey.
        
        The predictions are based on historical playoff data and current season team performance.
        """)
        
        st.subheader("Data Sources")
        st.write("""
        - Team statistics: NHL API
        - Advanced metrics: MoneyPuck (when available)
        - Historical playoff data: Used to train the prediction model
        """)
        
        st.subheader("How the Model Works")
        st.write("""
        The model predicts the probability of the higher seed winning a playoff series based on
        the difference in team metrics. For each playoff matchup, the model:
        
        1. Calculates the difference in key metrics between teams
        2. Applies a trained machine learning model to predict win probability
        3. Simulates the series multiple times using Monte Carlo methods
        
        The simulation accounts for home ice advantage and other factors that influence playoff success.
        """)

# Run the app
if __name__ == "__main__":
    main()




