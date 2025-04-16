import subprocess
import sys
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

# Check if the NHL API package is installed, install if missing
try:
    # Try importing the module (package is installed as nhl-api-py, but imported as nhlpy)
    import nhlpy
    from nhlpy.nhl_client import NHLClient
except ImportError:
    st.write("Installing NHL API package...")
    try:
        # Install the package (pip install name is nhl-api-py)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nhl-api-py"])
        # Import after installation (package is imported as nhlpy)
        import nhlpy
        from nhlpy.nhl_client import NHLClient
    except Exception as e:
        st.error(f"Error installing or importing NHL API package: {str(e)}")
        NHLClient = None

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
try:
    client = NHLClient()
except Exception as e:
    st.error(f"Could not initialize NHL client: {str(e)}")
    client = None

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
    """Get team summary stats data directly from NHL API"""
    try:
        # Construct the URL with season parameters
        url = f"https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId>={start_season}{start_season+1} and seasonId<={end_season}{end_season+1}"
        # Make direct API request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        stats_data = response.json().get('data', [])
        return stats_data
    except Exception as e:
        st.error(f"Error fetching team stats data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_standings_data(standings_data):
    """Process NHL standings data into a DataFrame"""
    all_standings = []
    
    if isinstance(standings_data, dict):
        # Handle format with 'records' key (old format)
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
                    
                    # Handle nested structure for teamName and teamAbbrev
                    if "team" in team_record:
                        if isinstance(team_record["team"].get("name"), dict) and "default" in team_record["team"].get("name", {}):
                            team_data["teamName"] = team_record["team"]["name"]["default"]
                        else:
                            team_data["teamName"] = team_record["team"].get("name", "")
                            
                        if isinstance(team_record["team"].get("abbreviation"), dict) and "default" in team_record["team"].get("abbreviation", {}):
                            team_data["teamAbbrev"] = team_record["team"]["abbreviation"]["default"]
                        else:
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
        # Handle new format with 'standings' key
        elif "standings" in standings_data:
            for team_record in standings_data["standings"]:
                # Start with season info
                team_data = {
                    "season": season_str
                }
                
                # Extract division and conference info if available
                division_name = team_record.get("divisionName", "Unknown")
                conference_name = team_record.get("conferenceName", "Unknown")
                team_data["division"] = division_name
                team_data["conference"] = conference_name
                
                # Process team info - extract the 'default' value from nested JSON
                team_data["teamId"] = team_record.get("teamId", 0)
                
                if isinstance(team_record.get("teamName"), dict):
                    team_data["teamName"] = team_record["teamName"].get("default", "Unknown")
                else:
                    team_data["teamName"] = team_record.get("teamName", "Unknown")
                    
                if isinstance(team_record.get("teamAbbrev"), dict):
                    team_data["teamAbbrev"] = team_record["teamAbbrev"].get("default", "")
                else:
                    team_data["teamAbbrev"] = team_record["teamAbbrev", ""]
                    
                team_data["teamLogo"] = team_record.get("teamLogo", "")
                
                # Process all other fields
                for key, value in team_record.items():
                    # Skip already processed keys
                    if key in ["teamId", "teamName", "teamAbbrev", "teamLogo", "divisionName", "conferenceName"]:
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
        
        if not stats_df.empty:
            # Format percentages
            pct_cols = [col for col in stats_df.columns if 'Pct' in col]
            for col in pct_cols:
                if col in stats_df.columns and pd.api.types.is_numeric_dtype(stats_df[col]):
                    stats_df[col] = (stats_df[col] * 100).round(1).astype(str) + '%'
            
            # Basic column renaming - keep teamName as is
            rename_dict = {
                'seasonId': 'season',
                'teamName': 'teamName',  # Keep teamName as is
                'teamFullName': 'teamName',  # Rename teamFullName to teamName
                'gamesPlayed': 'GP',
                'wins': 'W',
                'losses': 'L',
                'otLosses': 'OTL',
                'points': 'PTS',
                'pointPct': 'PTS%',
                'goalsFor': 'GF',
                'goalsAgainst': 'GA',
                'powerPlayPct': 'PP%',
                'penaltyKillPct': 'PK%',
                'faceoffWinPct': 'FO%'
            }
            
            # Apply renaming for columns that exist
            rename_dict = {k: v for k, v in rename_dict.items() if k in stats_df.columns}
            stats_df = stats_df.rename(columns=rename_dict)
            
            # Rename team_name to teamName if it exists (from previous processing)
            if 'team_name' in stats_df.columns:
                stats_df = stats_df.rename(columns={'team_name': 'teamName'})
            
            return stats_df
    return pd.DataFrame()

# Function to get and process advanced stats data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_advanced_stats_data(season=current_season, situation='5on5'):
    """
    Get advanced stats from MoneyPuck or another source.
    
    Args:
        season (int): NHL season starting year (e.g., 2023 for 2023-2024 season)
        situation (str): Game situation to filter (default: '5on5')
        
    Returns:
        DataFrame: Advanced stats data for the specified season and situation
    """
    # Construct the MoneyPuck URL for the current season
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
    csv_path = os.path.join(data_folder, f"moneypuck_regular_{season}.csv")
    
    try:
        # First try to load from local cache if it exists
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    return process_advanced_stats(df, season_str, situation)
            except Exception as cache_error:
                st.warning(f"Error loading from cache: {str(cache_error)}")
        
        # If no cache or cache failed, try downloading with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if successful
        if response.status_code == 200:
            # Save the raw content to a temporary file and read it with pandas
            with open(os.path.join(data_folder, "temp_data.csv"), "wb") as f:
                f.write(response.content)
            
            df = pd.read_csv(os.path.join(data_folder, "temp_data.csv"))
            
            # Save a local copy for future use
            df.to_csv(csv_path, index=False)
            
            return process_advanced_stats(df, season_str, situation)
        else:
            st.warning(f"Request failed with status code: {response.status_code}")
            return create_empty_advanced_stats_df()
                
    except Exception as e:
        st.warning(f"Error in get_advanced_stats_data: {str(e)}")
        return create_empty_advanced_stats_df()

def process_advanced_stats(df, season_str, situation):
    """Process the advanced stats dataframe while preserving all required columns."""
    # Check if we have data
    if df.empty:
        return df
    
    # Filter by situation if available
    if 'situation' in df.columns:
        situation_values = df['situation'].unique()
        matching_situations = [s for s in situation_values if situation.lower() in s.lower()]
        
        if matching_situations:
            situation_filter = matching_situations[0]
            df = df[df['situation'] == situation_filter].copy()
        else:
            # Just use the first situation in this case
            if len(situation_values) > 0:
                df = df[df['situation'] == situation_values[0]].copy()
    
    # Add season information if not present
    if 'season' not in df.columns:
        df['season'] = season_str

    # Clean up unnecessary columns
    columns_to_drop = ['name', 'team.1', 'position']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
        
    # Rename columns to remove spaces and special characters
    df.columns = df.columns.str.replace(' ', '_').str.replace('.', '_').str.replace('-', '_')
    # Remove any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Convert columns to numeric where possible
    for col in df.columns:
        if col not in ['team', 'situation', 'season']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_empty_advanced_stats_df():
    """Create an empty DataFrame with all required advanced stats columns."""
    columns = [
        'team', 'season', 'situation', 'games_played',
        'xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage', 'iceTime',
        'xOnGoalFor', 'xGoalsFor', 'xReboundsFor', 'xFreezeFor', 'xPlayStoppedFor',
        'xPlayContinuedInZoneFor', 'xPlayContinuedOutsideZoneFor', 'flurryAdjustedxGoalsFor',
        'scoreVenueAdjustedxGoalsFor', 'flurryScoreVenueAdjustedxGoalsFor', 'shotsOnGoalFor',
        'missedShotsFor', 'blockedShotAttemptsFor', 'shotAttemptsFor', 'goalsFor',
        'reboundsFor', 'reboundGoalsFor', 'freezeFor', 'playStoppedFor',
        'playContinuedInZoneFor', 'playContinuedOutsideZoneFor', 'savedShotsOnGoalFor',
        'savedUnblockedShotAttemptsFor', 'penaltiesFor', 'penalityMinutesFor',
        'faceOffsWonFor', 'hitsFor', 'takeawaysFor', 'giveawaysFor',
        'lowDangerShotsFor', 'mediumDangerShotsFor', 'highDangerShotsFor',
        'lowDangerxGoalsFor', 'mediumDangerxGoalsFor', 'highDangerxGoalsFor',
        'lowDangerGoalsFor', 'mediumDangerGoalsFor', 'highDangerGoalsFor',
        'scoreAdjustedShotsAttemptsFor', 'unblockedShotAttemptsFor',
        'scoreAdjustedUnblockedShotAttemptsFor', 'dZoneGiveawaysFor',
        'xGoalsFromxReboundsOfShotsFor', 'xGoalsFromActualReboundsOfShotsFor',
        'reboundxGoalsFor', 'totalShotCreditFor', 'scoreAdjustedTotalShotCreditFor',
        'scoreFlurryAdjustedTotalShotCreditFor', 'xOnGoalAgainst', 'xGoalsAgainst',
        'xReboundsAgainst', 'xFreezeAgainst', 'xPlayStoppedAgainst',
        'xPlayContinuedInZoneAgainst', 'xPlayContinuedOutsideZoneAgainst',
        'flurryAdjustedxGoalsAgainst', 'scoreVenueAdjustedxGoalsAgainst',
        'flurryScoreVenueAdjustedxGoalsAgainst', 'shotsOnGoalAgainst',
        'missedShotsAgainst', 'blockedShotAttemptsAgainst', 'shotAttemptsAgainst',
        'goalsAgainst', 'reboundsAgainst', 'reboundGoalsAgainst', 'freezeAgainst',
        'playStoppedAgainst', 'playContinuedInZoneAgainst', 'playContinuedOutsideZoneAgainst',
        'savedShotsOnGoalAgainst', 'savedUnblockedShotAttemptsAgainst',
        'penaltiesAgainst', 'penalityMinutesAgainst', 'faceOffsWonAgainst',
        'hitsAgainst', 'takeawaysAgainst', 'giveawaysAgainst', 'lowDangerShotsAgainst',
        'mediumDangerShotsAgainst', 'highDangerShotsAgainst', 'lowDangerxGoalsAgainst',
        'mediumDangerxGoalsAgainst', 'highDangerxGoalsAgainst', 'lowDangerGoalsAgainst',
        'mediumDangerGoalsAgainst', 'highDangerGoalsAgainst',
        'scoreAdjustedShotsAttemptsAgainst', 'unblockedShotAttemptsAgainst',
        'scoreAdjustedUnblockedShotAttemptsAgainst', 'dZoneGiveawaysAgainst',
        'xGoalsFromxReboundsOfShotsAgainst', 'xGoalsFromActualReboundsOfShotsAgainst',
        'reboundxGoalsAgainst', 'totalShotCreditAgainst',
        'scoreAdjustedTotalShotCreditAgainst', 'scoreFlurryAdjustedTotalShotCreditAgainst'
    ]
    return pd.DataFrame(columns=columns)

# Function to engineer features from raw data
def engineer_features(combined_data):
    """Engineer features for the prediction model"""
    # Create a copy of the data to avoid modifying original
    df = combined_data.copy()
    
    # Convert percentage strings to floats where needed
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # Check if this column contains percentage strings
                if df[col].astype(str).str.contains('%').any():
                    # Convert percentage strings to floats
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
            except Exception as e:
                # Skip columns that can't be processed
                pass
    
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
            if df[col].dtype == object and df[col].str.contains('%').any():
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

# Add function to incorporate playoff history metrics
def add_playoff_history_metrics(team_data):
    """Add playoff history metrics to the combined team data."""
    try:
        # Load the playoff history data with proper path handling
        playoff_history_path = os.path.join(data_folder, "nhl_playoff_wins_2005_present.csv")
        playoff_history = pd.read_csv(playoff_history_path)
        
        # Add rounds_won if it doesn't exist already
        if 'rounds_won' not in playoff_history.columns:
            playoff_history['rounds_won'] = ((playoff_history['wins'] - (playoff_history['wins'] % 4)) / 4).astype(int)
        
        # Calculate playoff history metrics
        seasons = team_data['season'].unique()
        history_data = calculate_playoff_history(playoff_history, seasons, team_data)
        
        # Merge playoff history metrics with team data
        team_data_with_history = pd.merge(
            team_data,
            history_data,
            on=['teamAbbrev', 'season'],
            how='left'
        )
        
        # Fill NaN values for teams with no playoff history
        for col in ['weighted_playoff_wins', 'weighted_playoff_rounds', 'playoff_performance_score']:
            if col in team_data_with_history.columns:
                team_data_with_history[col] = team_data_with_history[col].fillna(0)
        
        return team_data_with_history
        
    except FileNotFoundError:
        st.info(f"Playoff history data file not found - continuing without playoff history metrics")
        return team_data
    except Exception as e:
        st.warning(f"Error adding playoff history: {str(e)}")
        return team_data

def calculate_playoff_history(playoff_df, seasons, team_data, num_seasons=2):
    """
    Calculate playoff history metrics for all teams across multiple seasons
    
    Parameters:
    -----------
    playoff_df: DataFrame with playoff history data
    seasons: List of unique seasons to process
    team_data: DataFrame with team data including teamAbbrev and season
    num_seasons: Number of prior seasons to consider
    
    Returns:
    --------
    DataFrame with weighted playoff metrics for each team-season
    """
    weights = [0.6, 0.4]  # Weights for previous seasons
    history_data = []
    
    # Create a mapping between team and teamAbbrev from team_data
    team_abbrev_map = {}
    if 'teamAbbrev' in team_data.columns:
        # Get unique team-abbrev pairs
        team_abbrev_pairs = team_data[['teamName', 'teamAbbrev']].drop_duplicates()
        team_abbrev_map = dict(zip(team_abbrev_pairs['teamName'], team_abbrev_pairs['teamAbbrev']))
    
    for season in seasons:
        # Get current year from season
        current_year = int(str(season)[:4]) if len(str(season)) >= 4 else int(season)
        
        # Get all teams for this season from the team_data
        current_teams_df = team_data[team_data['season'] == season]
        
        if 'teamAbbrev' not in current_teams_df.columns:
            continue
            
        # Process each team
        for _, team_row in current_teams_df.iterrows():
            team_abbrev = team_row['teamAbbrev']
            team_name = team_row['teamName']
            
            weighted_wins = 0
            weighted_rounds = 0
            
            # Look at prior seasons
            for i in range(1, num_seasons + 1):
                if i <= len(weights):  # Make sure we have a weight for this season
                    prev_year = current_year - i
                    prev_season = int(f"{prev_year}{prev_year+1}")
                    
                    # Try to find a match by team name first
                    prev_record = playoff_df[(playoff_df['team'] == team_name) & 
                                           (playoff_df['season'] == prev_season)]
                    
                    # If no match, try abbreviation
                    if prev_record.empty and team_abbrev:
                        prev_record = playoff_df[(playoff_df['team'] == team_abbrev) & 
                                               (playoff_df['season'] == prev_season)]
                    
                    if not prev_record.empty:
                        # Get playoff wins and rounds won
                        wins = prev_record['wins'].values[0]
                        rounds = prev_record['rounds_won'].values[0]
                        
                        # Apply weights
                        weighted_wins += wins * weights[i-1]
                        weighted_rounds += rounds * weights[i-1]
            
            # Store results
            history_data.append({
                'teamAbbrev': team_abbrev,
                'season': season,
                'weighted_playoff_wins': weighted_wins,
                'weighted_playoff_rounds': weighted_rounds,
                'playoff_performance_score': weighted_wins/4*.75 + weighted_rounds*1.25
            })
    
    return pd.DataFrame(history_data)

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
            
            # Determine available tiebreaker columns
            sort_columns = ['points']
            tiebreakers = ['regulationWins', 'row', 'wins', 'goalDifferential']
            
            for tiebreaker in tiebreakers:
                if tiebreaker in div_teams.columns:
                    sort_columns.append(tiebreaker)
                    
            # Sort using available columns
            div_teams = div_teams.sort_values(by=sort_columns, ascending=False)
            
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
            
            # Use the same tiebreaker logic for wildcards
            sort_columns = ['points']
            tiebreakers = ['regulationWins', 'row', 'wins', 'goalDifferential']
            
            for tiebreaker in tiebreakers:
                if tiebreaker in wildcard_df.columns:
                    sort_columns.append(tiebreaker)
                    
            wildcard_df = wildcard_df.sort_values(by=sort_columns, ascending=False)
            
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
            # Use the same tiebreaker logic for division winners
            if 'points' in div_winners[list(div_winners.keys())[0]]:
                sort_key = lambda x: tuple(x[col] if col in x else 0 for col in sort_columns)
                div_winner_list = sorted(
                    list(div_winners.values()), 
                    key=sort_key, 
                    reverse=True
                )
            else:
                # Fallback to simpler sorting if points not available
                div_winner_list = list(div_winners.values())
            
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
    
    # Get team data for each team - use teamAbbrev to filter
    top_team_data = team_data[team_data['teamAbbrev'] == top_seed['teamAbbrev']]
    bottom_team_data = team_data[team_data['teamAbbrev'] == bottom_seed['teamAbbrev']]
    
    # Check if team data exists
    if not top_team_data.empty and not bottom_team_data.empty:
        top_seed_data = top_team_data.iloc[0]
        bottom_seed_data = bottom_team_data.iloc[0]
        
        # Add points for seeding comparison
        if 'points' in top_seed_data and 'points' in bottom_seed_data:
            matchup_data['points_top'] = top_seed_data['points']
            matchup_data['points_bottom'] = bottom_seed_data['points']
            matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
        
        # Feature columns to use for prediction - now include playoff history if available
        feature_cols = [
            'PP%_rel', 'PK%_rel', 'FO%', 'special_teams_composite',
            'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
            'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
            'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
            'adjGoalsScoredAboveX/60', 'playoff_performance_score'
        ]
        
        # Add all other potentially useful features
        additional_features = [
            'corsiPercentage', 'fenwickPercentage', 'weighted_playoff_wins',
            'weighted_playoff_rounds', 'PP%', 'PK%', 'home_wins', 'road_wins'
        ]
        
        # Combine all feature lists
        all_features = feature_cols + additional_features
        
        # Feature engineering: calculate missing features if possible
        if 'PP%' in top_seed_data and 'PP%' not in feature_cols:
            # Convert percentage strings to floats if needed
            for team_data_item in [top_seed_data, bottom_seed_data]:
                for col in ['PP%', 'PK%']:
                    if team_data[col].dtype == object and team_data[col].str.contains('%').any():
                        team_data[col] = team_data[col].str.rstrip('%').astype(float) / 100
        
            # Calculate league averages
            if 'PP%' in team_data.columns and 'PK%' in team_data.columns:
                league_avg_pp = team_data['PP%'].mean()
                league_avg_pk = team_data['PK%'].mean()
                
                # Calculate relative metrics
                top_seed_data['PP%_rel'] = top_seed_data['PP%'] - league_avg_pp
                bottom_seed_data['PP%_rel'] = bottom_seed_data['PP%'] - league_avg_pp
                top_seed_data['PK%_rel'] = top_seed_data['PK%'] - league_avg_pk
                bottom_seed_data['PK%_rel'] = bottom_seed_data['PK%'] - league_avg_pk
                
                # Create composite special teams metric
                top_seed_data['special_teams_composite'] = top_seed_data['PP%_rel'] + top_seed_data['PK%_rel']
                bottom_seed_data['special_teams_composite'] = bottom_seed_data['PP%_rel'] + bottom_seed_data['PK%_rel']
        
        # Calculate home/road win percentages if missing
        if 'homeRegulationWin%' not in top_seed_data and 'homeRegulationWins' in top_seed_data and 'gamesPlayed' in top_seed_data:
            top_seed_data['homeRegulationWin%'] = top_seed_data['homeRegulationWins'] / top_seed_data['gamesPlayed']
            bottom_seed_data['homeRegulationWin%'] = bottom_seed_data['homeRegulationWins'] / bottom_seed_data['gamesPlayed']
        
        if 'roadRegulationWin%' not in top_seed_data and 'roadRegulationWins' in top_seed_data and 'gamesPlayed' in top_seed_data:
            top_seed_data['roadRegulationWin%'] = top_seed_data['roadRegulationWins'] / top_seed_data['gamesPlayed']
            bottom_seed_data['roadRegulationWin%'] = bottom_seed_data['roadRegulationWins'] / bottom_seed_data['gamesPlayed']
        
        # Add features for each team if available
        for col in all_features:
            if col in top_seed_data and col in bottom_seed_data:
                matchup_data[f"{col}_top"] = top_seed_data[col]
                matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
    else:
        st.warning(f"Team data not found for {top_seed['teamAbbrev']} or {bottom_seed['teamAbbrev']}")
    
    return pd.DataFrame([matchup_data])

# Function to load trained models
@st.cache_resource
def load_models():
    """Load the trained machine learning models"""
    model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    # Ensure model directory exists
    os.makedirs(model_folder, exist_ok=True)
    
    models = {}
    
    # Define model paths
    lr_path = os.path.join(model_folder, 'logistic_regression_model_final.pkl')
    xgb_path = os.path.join(model_folder, 'xgboost_playoff_model_final.pkl')
    default_path = os.path.join(model_folder, 'playoff_model.pkl')
    
    # List available files for debugging
    available_files = os.listdir(model_folder) if os.path.exists(model_folder) else []
    st.sidebar.info(f"Available model files: {', '.join(available_files) if available_files else 'None'}")
    
    # Check if xgboost is installed, install if missing
    try:
        import xgboost
    except ImportError:
        st.sidebar.warning("Installing XGBoost package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
            import xgboost
            st.sidebar.success("✅ XGBoost installed successfully")
        except Exception as e:
            st.sidebar.error(f"Error installing XGBoost: {str(e)}")
            # We'll continue without XGBoost models
    
    # Try to load the models
    try:
        # Check if LR model exists and load it
        if os.path.exists(lr_path):
            try:
                models['lr'] = joblib.load(lr_path)
                st.sidebar.success("✓ Loaded Logistic Regression model")
            except Exception as lr_error:
                st.sidebar.warning(f"Failed to load LR model: {str(lr_error)}")
        
        # Check if XGB model exists and load it
        if os.path.exists(xgb_path):
            try:
                models['xgb'] = joblib.load(xgb_path)
                st.sidebar.success("✓ Loaded XGBoost model")
            except Exception as xgb_error:
                st.sidebar.warning(f"Failed to load XGB model: {str(xgb_error)}")
            
        # Check if default model exists and load it
        if os.path.exists(default_path):
            try:
                models['default'] = joblib.load(default_path)
                st.sidebar.success("✓ Loaded default model")
            except Exception as default_error:
                st.sidebar.warning(f"Failed to load default model: {str(default_error)}")
            
        if not models:
            st.sidebar.warning("No models found - using basic predictions")
            models['default'] = {
                'model': None,
                'features': [],
                'home_ice_boost': 0.039
            }
            models['mode'] = 'default'
        else:
            # Determine which mode to use based on available models
            if 'lr' in models and 'xgb' in models:
                models['mode'] = 'ensemble'
            elif 'default' in models:
                models['mode'] = 'default'
            elif len(models) > 0:
                models['mode'] = 'single'
            else:
                models['mode'] = 'default'
            
            # Ensure home_ice_boost is set
            if 'home_ice_boost' not in models:
                models['home_ice_boost'] = 0.039
        
        return models
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        # Return a default placeholder model
        return {
            'default': {
                'model': None,
                'features': [],
                'home_ice_boost': 0.039
            },
            'mode': 'default',
            'home_ice_boost': 0.039
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def simulate_playoff_series(matchup_data, _model_data, n_simulations=1000):
    """
    Simulate a playoff series using Monte Carlo simulation
    
    Args:
        matchup_data: DataFrame with matchup features
        _model_data: Dictionary with model information and features (not hashed by Streamlit)
        n_simulations: Number of simulations to run
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Extract team abbreviations
    top_seed = matchup_data['top_seed_abbrev'].iloc[0]
    bottom_seed = matchup_data['bottom_seed_abbrev'].iloc[0]
    
    # If we have a model, use it for predictions
    if _model_data is not None:
        try:
            # Check which model mode we're using
            model_mode = _model_data.get('mode', 'default')
            
            # Get base probability for higher seed winning THE SERIES (not individual games)
            base_prob = 0.5  # Default if no models available
            lr_prob = 0.5
            xgb_prob = 0.5
            
            if model_mode == 'ensemble' and 'lr' in _model_data and 'xgb' in _model_data:
                # Use ensemble of logistic regression and XGBoost models
                lr_model = _model_data['lr'].get('model')
                lr_features = [feat for feat in _model_data['lr'].get('features', []) if feat in matchup_data.columns]
                
                if lr_model is not None and len(lr_features) == len(_model_data['lr'].get('features', [])):
                    lr_prob = lr_model.predict_proba(matchup_data[lr_features])[:, 1][0]
                else:
                    lr_prob = 0.5  # Default value
                
                xgb_model = _model_data['xgb'].get('model')
                xgb_features = [feat for feat in _model_data['xgb'].get('features', []) if feat in matchup_data.columns]
                
                if xgb_model is not None and len(xgb_features) == len(_model_data['xgb'].get('features', [])):
                    xgb_prob = xgb_model.predict_proba(matchup_data[xgb_features])[:, 1][0]
                else:
                    xgb_prob = 0.5  # Default value
                
                # Average the two models for ensemble prediction
                base_prob = (lr_prob + xgb_prob) / 2
                
            elif model_mode in ['single', 'default']:
                # Use a single model
                model_key = 'default' if model_mode == 'default' or 'default' in _model_data else 'xgb' if 'xgb' in _model_data else 'lr'
                if model_key in _model_data:
                    single_model = _model_data[model_key].get('model')
                    features = [feat for feat in _model_data[model_key].get('features', []) if feat in matchup_data.columns]
                    
                    if single_model is not None and len(features) == len(_model_data[model_key].get('features', [])):
                        base_prob = single_model.predict_proba(matchup_data[features])[:, 1][0]
                        if model_key == 'xgb':
                            xgb_prob = base_prob
                        elif model_key == 'lr':
                            lr_prob = base_prob
            
            # Get home ice boost - already included in the model's series prediction
            series_home_ice_boost = _model_data.get('home_ice_boost', 0.039)
            
            # Track results
            higher_seed_wins = 0
            win_distribution = {
                '4-0': 0, '4-1': 0, '4-2': 0, '4-3': 0,  # Higher seed wins
                '0-4': 0, '1-4': 0, '2-4': 0, '3-4': 0   # Lower seed wins
            }
            
            # Historical NHL playoff series length distribution:
            # 4 games: 14.0%, 5 games: 24.3%, 6 games: 33.6%, 7 games: 28.1%
            
            # Calculate scaled outcome distributions based on win probability
            # This scales the length distribution based on team strength
            # Higher win probability -> more likely to win in fewer games
            def get_outcome_distributions(win_prob):
                # Base distribution by series length
                base_sweep = 0.140     # 4 games: 14.0%
                base_five = 0.243      # 5 games: 24.3%
                base_six = 0.336       # 6 games: 33.6%
                base_seven = 0.281     # 7 games: 28.1%
                
                # Scaling factors - how much to adjust based on win probability
                # The further from 0.5, the more skewed the distribution
                p_factor = (win_prob - 0.5) * 2  # -1 to 1 scale
                
                # For the favorite (win_prob > 0.5):
                # - Increase probability of shorter series
                # - Decrease probability of longer series
                if win_prob >= 0.5:
                    # Scale between 0 and 0.8 based on win probability
                    scaling = p_factor * 0.8
                    
                    # Adjust distribution for favorite (higher seed)
                    higher_sweep = base_sweep * (1 + scaling)
                    higher_five = base_five * (1 + scaling * 0.5)
                    higher_six = base_six * (1 - scaling * 0.3)
                    higher_seven = base_seven * (1 - scaling * 0.5)
                    
                    # Calculate the complementary distribution for the underdog
                    lower_sweep = base_sweep * (1 - scaling * 0.5)
                    lower_five = base_five * (1 - scaling * 0.3)
                    lower_six = base_six * (1 + scaling * 0.4)
                    lower_seven = base_seven * (1 + scaling * 0.6)
                else:
                    # For the underdog (win_prob < 0.5)
                    # Scale between 0 and 0.8 based on inverse of win probability
                    scaling = -p_factor * 0.8
                    
                    # Adjust distribution for underdog (higher seed is underdog)
                    higher_sweep = base_sweep * (1 - scaling * 0.5)
                    higher_five = base_five * (1 - scaling * 0.3)
                    higher_six = base_six * (1 + scaling * 0.4)
                    higher_seven = base_seven * (1 + scaling * 0.6)
                    
                    # Calculate the complementary distribution for the favorite
                    lower_sweep = base_sweep * (1 + scaling)
                    lower_five = base_five * (1 + scaling * 0.5)
                    lower_six = base_six * (1 - scaling * 0.3)
                    lower_seven = base_seven * (1 - scaling * 0.5)
                
                # Normalize to ensure probabilities sum to 1
                higher_total = higher_sweep + higher_five + higher_six + higher_seven
                higher_sweep /= higher_total
                higher_five /= higher_total
                higher_six /= higher_total
                higher_seven /= higher_total
                
                lower_total = lower_sweep + lower_five + lower_six + lower_seven
                lower_sweep /= lower_total
                lower_five /= lower_total
                lower_six /= lower_total
                lower_seven /= lower_total
                
                # Return the distributions
                higher_seed_outcome_dist = {
                    '4-0': higher_sweep, 
                    '4-1': higher_five, 
                    '4-2': higher_six, 
                    '4-3': higher_seven
                }
                
                lower_seed_outcome_dist = {
                    '0-4': lower_sweep, 
                    '1-4': lower_five, 
                    '2-4': lower_six, 
                    '3-4': lower_seven
                }
                
                return higher_seed_outcome_dist, lower_seed_outcome_dist
            
            # Get the outcome distributions based on win probability
            higher_seed_outcome_dist, lower_seed_outcome_dist = get_outcome_distributions(base_prob)
            
            # Run simulations - now determining series winner based on probability
            for _ in range(n_simulations):
                # Determine if higher seed wins the series
                higher_seed_wins_series = np.random.random() < base_prob
                
                if higher_seed_wins_series:
                    higher_seed_wins += 1
                    # Randomly select a series outcome based on scaled historical distribution
                    outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                             p=[higher_seed_outcome_dist['4-0'], 
                                                higher_seed_outcome_dist['4-1'],
                                                higher_seed_outcome_dist['4-2'],
                                                higher_seed_outcome_dist['4-3']])
                    win_distribution[outcome] += 1
                else:
                    # Randomly select a series outcome for lower seed winning
                    outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                             p=[lower_seed_outcome_dist['0-4'], 
                                                lower_seed_outcome_dist['1-4'],
                                                lower_seed_outcome_dist['2-4'],
                                                lower_seed_outcome_dist['3-4']])
                    win_distribution[outcome] += 1
            
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
                'win_distribution': win_distribution,
                'model_mode': _model_data.get('mode', 'default'),
                'lr_probability': lr_prob,
                'xgb_probability': xgb_prob,
                'combined_base_probability': base_prob,
                'home_ice_boost': series_home_ice_boost,
                'series_length_breakdown': {
                    '4 games': 0.140, 
                    '5 games': 0.243, 
                    '6 games': 0.336, 
                    '7 games': 0.281
                },
                'simulation_note': "Series-level simulation using historical NHL playoff series length distribution"
            }
            
            return results
        except Exception as e:
            st.error(f"Error in simulation: {str(e)}")
            # Return None if there was an error
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
            'win_distribution': {},
            'model_mode': 'points-based',
            'points_based': True
        }

# Now let's improve the simulate_bracket function to properly advance teams
@st.cache_data(ttl=3600)  # Cache for 1 hour
def simulate_bracket(playoff_matchups, team_data, _model_data, n_simulations=100):
    """
    Simulate the entire playoff bracket
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team stats and features
        _model_data: Dictionary with model information and features (not hashed by Streamlit)
        n_simulations: Number of simulations to run
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Call the more comprehensive simulation function with detailed_tracking=False to keep results format consistent
    sim_results = simulate_playoff_bracket(playoff_matchups, team_data, _model_data, n_simulations, detailed_tracking=False)
    
    # Extract just the team_advancement results to maintain backward compatibility
    team_advancement = {}
    if 'team_advancement' in sim_results:
        # Convert DataFrame to the expected dictionary format
        for _, row in sim_results['team_advancement'].iterrows():
            team = row['teamAbbrev']
            team_advancement[team] = {
                'round_1': row['round_1'],
                'round_2': row['round_2'], 
                'conf_final': row['conf_final'],
                'final': row['final'],
                'champion': row['champion']
            }
    
    return team_advancement

# Function to create team comparison data for matchups - ensure all features are computed
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
    
    # Get team data for each team - use teamAbbrev to filter
    top_team_data = team_data[team_data['teamAbbrev'] == top_seed['teamAbbrev']]
    bottom_team_data = team_data[team_data['teamAbbrev'] == bottom_seed['teamAbbrev']]
    
    # Check if team data exists
    if not top_team_data.empty and not bottom_team_data.empty:
        top_seed_data = top_team_data.iloc[0]
        bottom_seed_data = bottom_team_data.iloc[0]
        
        # Add points for seeding comparison
        if 'points' in top_seed_data and 'points' in bottom_seed_data:
            matchup_data['points_top'] = top_seed_data['points']
            matchup_data['points_bottom'] = bottom_seed_data['points']
            matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
        
        # Feature columns to use for prediction - now include playoff history if available
        feature_cols = [
            'PP%_rel', 'PK%_rel', 'FO%', 'special_teams_composite',
            'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
            'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
            'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
            'adjGoalsScoredAboveX/60', 'playoff_performance_score'
        ]
        
        # Add all other potentially useful features
        additional_features = [
            'corsiPercentage', 'fenwickPercentage', 'weighted_playoff_wins',
            'weighted_playoff_rounds', 'PP%', 'PK%', 'home_wins', 'road_wins'
        ]
        
        # Combine all feature lists
        all_features = feature_cols + additional_features
        
        # Feature engineering: calculate missing features if possible
        if 'PP%' in top_seed_data and 'PP%' not in feature_cols:
            # Convert percentage strings to floats if needed
            for team_data_item in [top_seed_data, bottom_seed_data]:
                for col in ['PP%', 'PK%']:
                    if team_data[col].dtype == object and team_data[col].str.contains('%').any():
                        team_data[col] = team_data[col].str.rstrip('%').astype(float) / 100
        
            # Calculate league averages
            if 'PP%' in team_data.columns and 'PK%' in team_data.columns:
                league_avg_pp = team_data['PP%'].mean()
                league_avg_pk = team_data['PK%'].mean()
                
                # Calculate relative metrics
                top_seed_data['PP%_rel'] = top_seed_data['PP%'] - league_avg_pp
                bottom_seed_data['PP%_rel'] = bottom_seed_data['PP%'] - league_avg_pp
                top_seed_data['PK%_rel'] = top_seed_data['PK%'] - league_avg_pk
                bottom_seed_data['PK%_rel'] = bottom_seed_data['PK%'] - league_avg_pk
                
                # Create composite special teams metric
                top_seed_data['special_teams_composite'] = top_seed_data['PP%_rel'] + top_seed_data['PK%_rel']
                bottom_seed_data['special_teams_composite'] = bottom_seed_data['PP%_rel'] + bottom_seed_data['PK%_rel']
        
        # Calculate home/road win percentages if missing
        if 'homeRegulationWin%' not in top_seed_data and 'homeRegulationWins' in top_seed_data and 'gamesPlayed' in top_seed_data:
            top_seed_data['homeRegulationWin%'] = top_seed_data['homeRegulationWins'] / top_seed_data['gamesPlayed']
            bottom_seed_data['homeRegulationWin%'] = bottom_seed_data['homeRegulationWins'] / bottom_seed_data['gamesPlayed']
        
        if 'roadRegulationWin%' not in top_seed_data and 'roadRegulationWins' in top_seed_data and 'gamesPlayed' in top_seed_data:
            top_seed_data['roadRegulationWin%'] = top_seed_data['roadRegulationWins'] / top_seed_data['gamesPlayed']
            bottom_seed_data['roadRegulationWin%'] = bottom_seed_data['roadRegulationWins'] / bottom_seed_data['gamesPlayed']
        
        # Add features for each team if available
        for col in all_features:
            if col in top_seed_data and col in bottom_seed_data:
                matchup_data[f"{col}_top"] = top_seed_data[col]
                matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
    else:
        st.warning(f"Team data not found for {top_seed['teamAbbrev']} or {bottom_seed['teamAbbrev']}")
    
    return pd.DataFrame([matchup_data])

# Update the Playoff Bracket page display to show separate model predictions
def display_playoff_bracket(playoff_matchups, team_data, model_data):
    """Display the playoff bracket with detailed model predictions and selectable comparison metrics"""
    # Define list of available comparison metrics
    available_metrics = {
        "Points": "points",
        "Goal Differential": "goalDifferential",
        "Goals For": "GF",
        "Goals Against": "GA",
        "Power Play %": "PP%",
        "Penalty Kill %": "PK%",
        "5v5 xGoals %": "xGoalsPercentage", 
        "5v5 Corsi %": "corsiPercentage",
        "Regulation Wins": "regulationWins",
        "Road Wins": "road_wins",
        "Home Wins": "home_wins",
        "Face-off Win %": "FO%",
        "Save %": "savePct",
        "Playoff Experience": "playoff_performance_score"
    }
    
    # Let user select metrics to compare
    st.subheader("Select Comparison Metrics")
    
    # Create columns for select/deselect all buttons
    col1, col2 = st.columns([1, 10])
    with col1:
        select_all = st.button("Select All")
    with col2:
        deselect_all = st.button("Clear All")
    
    # Default selections
    default_selections = ["Points", "Goal Differential", "Power Play %", "Penalty Kill %"]
    
    # Handle select all / deselect all
    if select_all:
        selected_metrics = list(available_metrics.keys())
    elif deselect_all:
        selected_metrics = []
    else:
        selected_metrics = st.multiselect(
            "Choose metrics to compare:",
            options=list(available_metrics.keys()),
            default=default_selections
        )
    
    # Check if we have matchups
    if playoff_matchups:
        # Display each conference's matchups
        for conference, matchups in playoff_matchups.items():
            st.write(f"### {conference} Conference")
            
            for series_id, matchup in matchups.items():
                top_seed = matchup['top_seed']
                bottom_seed = matchup['bottom_seed']
                
                # Create a container for each matchup
                matchup_container = st.container()
                
                with matchup_container:
                    # Team names and logos
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
                    if not matchup_df.empty and model_data is not None:
                        sim_result = simulate_playoff_series(matchup_df, model_data)
                        
                        if sim_result:
                            # Get team data for comparison
                            top_team_data = team_data[team_data['teamAbbrev'] == top_seed['teamAbbrev']].iloc[0] if sum(team_data['teamAbbrev'] == top_seed['teamAbbrev']) > 0 else None
                            bottom_team_data = team_data[team_data['teamAbbrev'] == bottom_seed['teamAbbrev']].iloc[0] if sum(team_data['teamAbbrev'] == bottom_seed['teamAbbrev']) > 0 else None
                            
                            # Model predictions section
                            st.write("#### Model Predictions")
                            
                            # Create a comparison table for different model predictions
                            model_cols = st.columns(4)
                            
                            # Show LR model prediction (raw)
                            with model_cols[0]:
                                lr_prob = sim_result.get('lr_probability', 0.5)
                                st.metric("LR Model", f"{lr_prob*100:.1f}%")
                                st.caption(f"Logistic Regression (Raw)")
                            
                            # Show XGB model prediction (raw)
                            with model_cols[1]:
                                xgb_prob = sim_result.get('xgb_probability', 0.5)
                                st.metric("XGB Model", f"{xgb_prob*100:.1f}%")
                                st.caption(f"XGBoost (Raw)")
                            
                            # Show ensemble prediction (raw)
                            with model_cols[2]:
                                ensemble_prob = sim_result.get('combined_base_probability', (lr_prob + xgb_prob) / 2)
                                st.metric("Ensemble", f"{ensemble_prob*100:.1f}%")
                                st.caption(f"Ensemble w/o Home Ice")
                            
                            # Show combined prediction with home ice boost
                            with model_cols[3]:
                                win_pct = sim_result['win_probability'] * 100
                                st.metric("Final Prediction", f"{win_pct:.1f}%")
                                st.caption(f"With Home Ice Boost")
                            
                            # Determine predicted winner
                            predicted_winner = top_seed['teamName'] if win_pct >= 50 else bottom_seed['teamName']
                            predicted_winner_pct = win_pct if win_pct >= 50 else 100 - win_pct
                            
                            # Display predicted winner prominently
                            st.info(f"**Predicted Winner: {predicted_winner}** ({predicted_winner_pct:.1f}% chance)")
                            
                            # Display win distribution as a table instead of a chart
                            if sim_result.get('win_distribution'):
                                st.write("#### Series Outcome Distribution")
                                dist = sim_result['win_distribution']
                                
                                # Convert to percentages
                                total = sum(dist.values())
                                
                                # Create a DataFrame for the series outcome distribution
                                outcome_data = []
                                
                                # Higher seed outcomes
                                for outcome in ['4-0', '4-1', '4-2', '4-3']:
                                    pct = (dist.get(outcome, 0) / total) * 100 if total > 0 else 0
                                    outcome_data.append({
                                        'Team': top_seed['teamName'],
                                        'Outcome': outcome,
                                        'Probability': f"{pct:.1f}%"
                                    })
                                
                                # Lower seed outcomes
                                for team1_outcome, team2_outcome in zip(['0-4', '1-4', '2-4', '3-4'], ['4-0', '4-1', '4-2', '4-3']):
                                    pct = (dist.get(team1_outcome, 0) / total) * 100 if total > 0 else 0
                                    outcome_data.append({
                                        'Team': bottom_seed['teamName'],
                                        'Outcome': team2_outcome,
                                        'Probability': f"{pct:.1f}%"
                                    })
                                
                                # Convert to DataFrame and display as table
                                outcome_df = pd.DataFrame(outcome_data)
                                st.table(outcome_df)
                            
                            # Show comparison metrics if both teams have data and metrics are selected
                            if top_team_data is not None and bottom_team_data is not None and selected_metrics:
                                st.write("#### Team Comparison")
                                
                                # Create a DataFrame for the comparison table
                                comparison_data = []
                                
                                for display_name in selected_metrics:
                                    col_name = available_metrics[display_name]
                                    
                                    # Check if the metric exists for both teams
                                    if col_name in top_team_data and col_name in bottom_team_data:
                                        top_val = top_team_data[col_name]
                                        bottom_val = bottom_team_data[col_name]
                                        
                                        # Format values appropriately
                                        if isinstance(top_val, (float, np.float64)):
                                            if col_name in ['PP%', 'PK%', 'FO%', 'savePct'] and not isinstance(top_val, str):
                                                top_val_display = f"{top_val*100:.1f}%" if top_val <= 1 else f"{top_val:.1f}%"
                                                bottom_val_display = f"{bottom_val*100:.1f}%" if bottom_val <= 1 else f"{bottom_val:.1f}%"
                                            else:
                                                top_val_display = f"{top_val:.1f}" if top_val % 1 != 0 else f"{int(top_val)}"
                                                bottom_val_display = f"{bottom_val:.1f}" if bottom_val % 1 != 0 else f"{int(bottom_val)}"
                                        else:
                                            top_val_display = str(top_val)
                                            bottom_val_display = str(bottom_val)
                                        
                                        # Add to comparison data
                                        comparison_data.append({
                                            'Metric': display_name,
                                            top_seed['teamName']: top_val_display,
                                            bottom_seed['teamName']: bottom_val_display
                                        })
                                
                                # Convert to DataFrame and display
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.table(comparison_df)
                                else:
                                    st.warning("No comparison metrics available for both teams")
                    
                    st.write("---")
    else:
        st.warning("Could not determine playoff matchups from current standings data")

# Add this function before the display_playoff_bracket function
def load_team_logo(logo_url):
    """
    Load team logo from URL and return as PIL Image.
    
    Args:
        logo_url: URL of the team logo
        
    Returns:
        Image object or None if loading fails
    """
    try:
        # Validate URL
        if not logo_url or not isinstance(logo_url, str) or not logo_url.startswith('http'):
            return None
            
        # Add user-agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(logo_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            try:
                # Save to BytesIO and try to open
                img_data = BytesIO(response.content)
                # Verify it's a valid image before returning
                img = Image.open(img_data)
                # Quick check to ensure it's a valid image
                img.verify()
                # Reopen the image after verify closes it
                img_data.seek(0)
                return Image.open(img_data)
            except Exception:
                # If the image is invalid, return None silently
                return None
        else:
            # Only show warning for significant errors
            return None
    except Exception:
        # Skip logging the error and just return None
        return None
    
# Now let's update the main function to use our enhanced display function for the Playoff Bracket page
def main():
    # App title and description
    st.title("NHL Playoff Predictor")
    st.write("Predict playoff outcomes based on team statistics and advanced metrics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "First Round Matchups", 
        "Full Simulation Results", 
        "Head-to-Head Comparison", 
        "Sim Your Own Bracket",
        "About"
    ])
    
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
        
        # Combine standings and stats data
        if not standings_df.empty and not stats_df.empty:
            # Make sure season columns are strings
            standings_df['season'] = standings_df['season'].astype(str)
            stats_df['season'] = stats_df['season'].astype(str)
            
            # First merge standings and stats data
            team_data = pd.merge(standings_df, stats_df, 
                                 left_on=['season', 'teamName'], 
                                 right_on=['season', 'teamName'], 
                                 how='left')
            
            # Merge with advanced stats if available
            if not advanced_stats_df.empty:
                # Make sure season columns have the same format
                if 'season' in advanced_stats_df.columns:
                    # Convert season from "2024" to "20242025" format if needed
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str).apply(
                        lambda x: f"{x}{int(x)+1}" if len(x) == 4 and x.isdigit() else x
                    )
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str)
                
                # Perform the merge with the correct columns
                team_data = pd.merge(team_data, advanced_stats_df, 
                                     left_on=['season', 'teamAbbrev'],
                                     right_on=['season', 'team'],
                                     how='left')
            
            # Engineer features for prediction
            team_data = engineer_features(team_data)
            
            # Add playoff history metrics
            team_data = add_playoff_history_metrics(team_data)
        else:
            team_data = pd.DataFrame()
    
    # Load models
    model_data = load_models()
    
    # Determine playoff teams and create matchups
    playoff_matchups = determine_playoff_teams(standings_df)
    
    # Add model info to sidebar
    if model_data:
        st.sidebar.title("Model Information")
        st.sidebar.write(f"Model mode: {model_data.get('mode', 'default')}")
        st.sidebar.write(f"Home ice advantage: {model_data.get('home_ice_boost', 0.039)*100:.1f}%")
    
    # First Round Matchups page - renamed from "Playoff Bracket"
    if page == "First Round Matchups":
        st.header("First Round Playoff Matchups")
        display_playoff_bracket(playoff_matchups, team_data, model_data)
    
    # Team Comparison page
    elif page == "Team Comparison":
        st.header("Team Comparison")
        
        # Team selection
        st.subheader("Select Teams to Compare")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Team 1", sorted(team_data['teamName'].unique()), key="team1")
        
        with col2:
            # Filter to exclude team1
            available_teams = sorted([team for team in team_data['teamName'].unique() if team != team1])
            team2 = st.selectbox("Select Team 2", available_teams, key="team2")
        
        # Display team comparison if two different teams are selected
        if team1 != team2:
            team1_data = team_data[team_data['teamName'] == team1].iloc[0]
            team2_data = team_data[team_data['teamName'] == team2].iloc[0]
            
            # Display team logos if available
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(team1)
                if 'teamLogo' in team1_data and team1_data['teamLogo']:
                    logo = load_team_logo(team1_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            with col2:
                st.subheader(team2)
                if 'teamLogo' in team2_data and team2_data['teamLogo']:
                    logo = load_team_logo(team2_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            # Add the rest of your team comparison logic here
            # ...existing team comparison code...
        else:
            st.warning("Please select two different teams")
    
    # Simulation Results page
    elif page == "Full Simulation Results":
        st.header("Full Playoff Simulation Results")
        
        # Simulation options in sidebar
        st.sidebar.subheader("Simulation Settings")
        n_simulations = st.sidebar.slider("Number of simulations", 1000, 20000, 10000, 1000)
        
        # Run the simulation once per day (cached)
        if playoff_matchups and not team_data.empty:
            # Check if we have cached simulation results
            sim_results_key = f"sim_results_{season_str}"
            if sim_results_key in st.session_state and st.session_state[sim_results_key].get('n_simulations') == n_simulations:
                bracket_results = st.session_state[sim_results_key]['results']
                st.success(f"Using cached simulation results ({n_simulations} simulations)")
            else:
                # Run new simulation
                st.info(f"Running new simulation with {n_simulations} iterations...")
                bracket_results = run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations)
                
                # Cache the results in session state
                st.session_state[sim_results_key] = {
                    'results': bracket_results,
                    'n_simulations': n_simulations,
                    'timestamp': datetime.now()
                }
            
            # Display comprehensive simulation results
            display_simulation_results(bracket_results, n_simulations)
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

# Fix 2: Add the missing simulate_playoff_bracket function
def simulate_playoff_bracket(playoff_matchups, team_data, _model_data, n_simulations=1000, detailed_tracking=True):
    """Simulate the entire playoff bracket with comprehensive tracking and results.
    
    Args:
        playoff_matchups: Dictionary of first round matchups by conference
        team_data: DataFrame with team stats data
        _model_data: Dictionary of trained models for predictions
        n_simulations: Number of simulations to run
        detailed_tracking: Whether to track detailed matchup stats (potential matchups, etc.)
        
    Returns:
        Dictionary containing comprehensive simulation results
    """
    # Track advancement for each team
    team_advancement = {}
    
    # Track all possible matchups for each round if detailed tracking enabled
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
    
    # Updated historical distribution of NHL playoff series outcomes
    # 4 games: 14.0%, 5 games: 24.3%, 6 games: 33.6%, 7 games: 28.1%
    series_length_dist = [0.140, 0.243, 0.336, 0.281]
    
    # Store the most common full bracket result if detailed tracking is enabled
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
        
        # Track this simulation's bracket result if detailed tracking is enabled
        current_bracket = [] if detailed_tracking else None
        
        # First round
        for conference, matchups in playoff_matchups.items():
            round_1_winners[conference] = {}
            for series_id, matchup in matchups.items():
                # Create matchup data
                matchup_df = create_matchup_data(matchup['top_seed'], matchup['bottom_seed'], team_data)
                
                # Get prediction for the series
                if not matchup_df.empty:
                    # Get win probability from pre-trained model
                    win_prob = 0.5  # default
                    
                    # Use the model's prediction if available
                    if 'lr' in _model_data and 'model' in _model_data['lr'] and 'features' in _model_data['lr']:
                        lr_features = [f for f in _model_data['lr']['features'] if f in matchup_df.columns]
                        
                        if len(lr_features) == len(_model_data['lr']['features']):
                            try:
                                lr_prob = _model_data['lr']['model'].predict_proba(matchup_df[lr_features])[:, 1][0]
                                win_prob = lr_prob
                            except Exception:
                                pass
                    
                    # Use XGBoost if available
                    if 'xgb' in _model_data and 'model' in _model_data['xgb'] and 'features' in _model_data['xgb']:
                        xgb_features = [f for f in _model_data['xgb']['features'] if f in matchup_df.columns]
                        
                        if len(xgb_features) == len(_model_data['xgb']['features']):
                            try:
                                xgb_prob = _model_data['xgb']['model'].predict_proba(matchup_df[xgb_features])[:, 1][0]
                                # Use ensemble if both models are available
                                if win_prob != 0.5:
                                    win_prob = (win_prob + xgb_prob) / 2
                                else:
                                    win_prob = xgb_prob
                            except Exception:
                                pass
                    
                    # Fallback to points difference if models failed
                    if win_prob == 0.5 and 'points_diff' in matchup_df.columns:
                        points_diff = matchup_df['points_diff'].iloc[0]
                        win_prob = 1 / (1 + np.exp(-0.05 * points_diff))
                    
                    # Simulate the series outcome
                    higher_seed_wins = np.random.random() < win_prob
                    
                    if higher_seed_wins:
                        winner = matchup['top_seed']
                        loser = matchup['bottom_seed']
                    else:
                        winner = matchup['bottom_seed']
                        loser = matchup['top_seed']
                    
                    # Simulate series length based on historical distribution
                    series_length = np.random.choice([4, 5, 6, 7], p=series_length_dist)
                    
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
                        
        # Process remaining rounds (Second round, Conference Finals, Stanley Cup Final)
        # ...existing playoff simulation code...
    
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
    
    # Prepare return values based on detailed tracking option
    results = {
        'team_advancement': results_df
    }
    
    # Add detailed tracking results if enabled and populated
    if detailed_tracking:
        # Add additional tracking information to results
        # ...existing detailed tracking code...
        pass
    
    return results

# Fix 3: Define display_head_to_head function which is called but missing
def display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data):
    """Display head-to-head prediction for two teams"""
    st.subheader("Head-to-Head Prediction")
    
    # Create a mock playoff matchup for prediction
    mock_top_seed = {
        'teamAbbrev': team1_data['teamAbbrev'],
        'teamName': team1_data['teamName'],
        'division_rank': 1  # Mock value
    }
    
    mock_bottom_seed = {
        'teamAbbrev': team2_data['teamAbbrev'],
        'teamName': team2_data['teamName'],
        'division_rank': 2  # Mock value
    }
    
    # Create matchup data
    matchup_df = create_matchup_data(mock_top_seed, mock_bottom_seed, team_data)
    
    # Simulate a series between these teams
    if not matchup_df.empty:
        sim_result = simulate_playoff_series(matchup_df, model_data)
        
        if sim_result:
            # Get win probability
            win_prob = sim_result['win_probability'] * 100
            predicted_winner = team1 if win_prob >= 50 else team2
            winner_prob = win_prob if win_prob >= 50 else 100 - win_prob
            
            # Display prediction
            st.info(f"**In a playoff series, {predicted_winner} would win {winner_prob:.1f}% of the time**")
            
            # Show model details
            expander = st.expander("See prediction details")
            with expander:
                st.write(f"**Model used:** {sim_result.get('model_mode', 'default')}")
                
                if 'lr_probability' in sim_result:
                    lr_pct = sim_result['lr_probability'] * 100
                    st.write(f"**Logistic Regression:** {lr_pct:.1f}% chance for {team1} to win")
                
                if 'xgb_probability' in sim_result:
                    xgb_pct = sim_result['xgb_probability'] * 100
                    st.write(f"**XGBoost:** {xgb_pct:.1f}% chance for {team1} to win")
                
                if 'win_distribution' in sim_result and sim_result['win_distribution']:
                    st.write("**Series outcome distribution:**")
                    dist = sim_result['win_distribution']
                    total = sum(dist.values())
                    
                    # Display each potential series outcome
                    cols = st.columns(2)
                    with cols[0]:
                        st.write(f"**{team1} wins:**")
                        for outcome in ['4-0', '4-1', '4-2', '4-3']:
                            pct = (dist.get(outcome, 0) / total) * 100 if total > 0 else 0
                            st.write(f"{outcome}: {pct:.1f}%")
                    
                    with cols[1]:
                        st.write(f"**{team2} wins:**")
                        for team1_outcome, team2_outcome in zip(['0-4', '1-4', '2-4', '3-4'], ['4-0', '4-1', '4-2', '4-3']):
                            pct = (dist.get(team1_outcome, 0) / total) * 100 if total > 0 else 0
                            st.write(f"{team2_outcome}: {pct:.1f}%")

# Enhanced function to handle daily simulation caching
@st.cache_data(ttl=86400)  # Cache for 24 hours (daily refresh)
def run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations=10000):
    """
    Run a full playoff bracket simulation with comprehensive results
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team stats and features
        model_data: Dictionary with model information
        n_simulations: Number of simulations to run (default: 10,000)
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Call the comprehensive simulation function
    with st.spinner(f"Running {n_simulations} playoff simulations..."):
        return simulate_playoff_bracket(playoff_matchups, team_data, model_data, n_simulations, detailed_tracking=True)

# Function to display color-coded tables
def display_colored_table(df, color_columns, cmap='Blues', text_color_threshold=0.7):
    """
    Display a DataFrame as a table with cell coloring based on values.
    
    Args:
        df: DataFrame to display
        color_columns: List of column names to apply coloring
        cmap: Matplotlib colormap name (default: 'Blues')
        text_color_threshold: Threshold value for switching to light text (0-1)
    """
    # Create a copy of the DataFrame for styling
    styled_df = df.copy()
    
    # Format percentage columns and apply styling
    for col in color_columns:
        if col in styled_df.columns:
            # Format as percentage strings if not already
            if not styled_df[col].dtype == object:
                styled_df[col] = styled_df[col].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else '')
    
    # Create Pandas Styler object
    styler = styled_df.style
    
    # Create a style function for each column
    def style_column(col):
        def color_scale(val):
            if isinstance(val, str) and '%' in val:
                try:
                    # Extract numeric value from percentage string
                    num_val = float(val.strip('%')) / 100
                except ValueError:
                    return ''
            elif isinstance(val, (int, float)):
                num_val = val / 100 if val > 1 else val  # Normalize if needed
            else:
                return ''
                
            # Get the color from colormap
            cmap_obj = plt.cm.get_cmap(cmap)
            rgba = cmap_obj(num_val)
            
            # Determine text color based on background darkness
            background_lightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = 'white' if background_lightness < text_color_threshold else 'black'
            
            # Convert rgba to css rgb string
            rgb = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{0.7})'
            
            return f'background-color: {rgb}; color: {text_color}'
        
        return styler.applymap(color_scale, subset=[col])
    
    # Apply styling to each column
    for col in color_columns:
        if col in styled_df.columns:
            styler = style_column(col)
            
    # Return the styled DataFrame for display
    return styler

# Function to create and display charts for matchups
def display_matchup_charts(matchup_df, matchup_type, top_n=10):
    """
    Create and display bar charts for potential playoff matchups
    
    Args:
        matchup_df: DataFrame with matchup data
        matchup_type: String describing the matchup type (e.g., "Second Round", "Conference Finals")
        top_n: Number of matchups to display (default: 10)
    """
    if matchup_df.empty:
        st.write(f"No {matchup_type} matchup data available")
        return
        
    # Get the top N matchups
    top_matchups = matchup_df.head(top_n) if len(matchup_df) > top_n else matchup_df
    
    # Create figure with sufficient height based on number of matchups
    fig_height = max(6, len(top_matchups) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Create horizontal bar chart
    bars = ax.barh(top_matchups['matchup'], top_matchups['probability'], color='skyblue')
    
    # Add percentage labels to bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=10)
    
    # Set title and labels
    ax.set_title(f"Top {matchup_type} Matchups")
    ax.set_xlabel('Probability (%)')
    ax.set_xlim(0, max(top_matchups['probability']) * 1.1)  # Add some padding
    
    # Remove unnecessary box lines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

# Function to generate the simulation results page
def display_simulation_results(bracket_results, n_simulations):
    """
    Display comprehensive simulation results with visualizations
    
    Args:
        bracket_results: Dictionary with simulation results
        n_simulations: Number of simulations that were run
    """
    if not bracket_results or 'team_advancement' not in bracket_results:
        st.error("No simulation results available")
        return
    
    # Get team advancement data
    results_df = bracket_results['team_advancement'].copy()
    
    # Convert advancement columns to percentages
    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
        if col in results_df.columns:
            results_df[col] = results_df[col] * 100
    
    # Display Stanley Cup championship odds
    st.subheader("Stanley Cup Championship Odds")
    
    # Create two columns for chart and table
    chart_col, table_col = st.columns([3, 2])
    
    with chart_col:
        # Create a bar chart of Stanley Cup odds
        fig, ax = plt.subplots(figsize=(10, 8))
        top_teams = results_df.sort_values('champion', ascending=False).head(16)  # Show playoff teams
        
        # Create horizontal bar chart with team names
        bars = ax.barh(top_teams['teamName'], top_teams['champion'], color='lightblue')
        
        # Add percentage labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    va='center', fontsize=10)
        
        ax.set_xlabel('Championship Probability (%)')
        ax.set_title(f'Stanley Cup Championship Odds (Based on {n_simulations} Simulations)')
        ax.set_xlim(0, max(top_teams['champion']) * 1.1)  # Add some padding
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
    
    with table_col:
        # Display the table with color coding
        # Rename columns for better display
        display_df = results_df.copy()
        column_mapping = {
            'teamName': 'Team',
            'champion': 'Cup %',
            'final': 'Finals %',
            'conf_final': 'Conf Finals %',
            'round_2': 'Round 2 %',
            'round_1': 'Round 1 %',
            'avg_games_played': 'Avg Games'
        }
        
        # Rename columns
        display_df = display_df.rename(columns=column_mapping)
        
        # Order columns as desired
        display_columns = ['Team', 'Cup %', 'Finals %', 'Conf Finals %', 'Round 2 %', 'Round 1 %', 'Avg Games']
        available_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[available_columns].sort_values('Cup %', ascending=False)
        
        # Format avg_games_played
        if 'Avg Games' in display_df.columns:
            display_df['Avg Games'] = display_df['Avg Games'].round(1)
        
        # Display the color-coded table
        color_columns = [col for col in display_df.columns if '%' in col]
        st.write("### Playoff Advancement Table")
        styled_table = display_colored_table(display_df.head(16), color_columns)
        st.dataframe(styled_table, use_container_width=True)
    
    # Create the round-by-round advancement visualization
    st.subheader("Round-by-Round Advancement")
    
    # Create a stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select playoff teams to plot
    teams_to_plot = results_df.sort_values('champion', ascending=False).iloc[:16]
    rounds_to_plot = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
    round_labels = ['First Round', 'Second Round', 'Conf Finals', 'Stanley Cup Final', 'Champion']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create stacked bars
    bottom = np.zeros(len(teams_to_plot))
    width = 0.6
    
    # Plot each round
    bars = []
    for i, (col, label) in enumerate(zip(rounds_to_plot, round_labels)):
        if col in teams_to_plot.columns:
            bar = ax.barh(teams_to_plot['teamName'], teams_to_plot[col], 
                          left=0, height=width, label=label, color=colors[i])
            bars.append(bar)
    
    ax.set_xlabel('Probability (%)')
    ax.set_title('Round-by-Round Playoff Advancement Probabilities')
    ax.legend(loc='upper right')
    
    st.pyplot(fig)
    
    # Display detailed matchup information if available
    if all(key in bracket_results for key in ['round2_matchups', 'conf_final_matchups', 'final_matchups']):
        # Subheader for potential matchups
        st.subheader("Most Likely Playoff Matchups")
        
        # Create tabs for different rounds
        round_tabs = st.tabs(["Second Round", "Conference Finals", "Stanley Cup Final"])
        
        # Second Round Matchups
        with round_tabs[0]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['round2_matchups'], "Second Round")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['round2_matchups'].empty:
                    display_df = bracket_results['round2_matchups'].copy()
                    display_df = display_df[['matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
        
        # Conference Finals Matchups
        with round_tabs[1]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['conf_final_matchups'], "Conference Finals")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['conf_final_matchups'].empty:
                    display_df = bracket_results['conf_final_matchups'].copy()
                    display_df = display_df[['conference', 'matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Conference', 'Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
        
        # Stanley Cup Final Matchups
        with round_tabs[2]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['final_matchups'], "Stanley Cup Final")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['final_matchups'].empty:
                    display_df = bracket_results['final_matchups'].copy()
                    display_df = display_df[['matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
    
    # Display most common bracket if available
    if 'most_common_bracket' in bracket_results:
        st.subheader("Most Likely Complete Playoff Bracket")
        most_common = bracket_results['most_common_bracket']
        
        # Create an expander to show the details
        with st.expander(f"Most Common Bracket (Occurred in {most_common['probability']:.1f}% of simulations)"):
            # Organize by round
            if most_common['bracket']:
                rounds = {
                    "First Round": most_common['bracket'][:8] if len(most_common['bracket']) > 7 else [],
                    "Second Round": most_common['bracket'][8:12] if len(most_common['bracket']) > 11 else [],
                    "Conference Finals": most_common['bracket'][12:14] if len(most_common['bracket']) > 13 else [],
                    "Stanley Cup Final": most_common['bracket'][14:] if len(most_common['bracket']) > 14 else []
                }
                
                # Display each round
                for round_name, results in rounds.items():
                    if results:
                        st.write(f"**{round_name}:**")
                        for i, result in enumerate(results, 1):
                            st.write(f"{i}. {result}")

# Update the Simulation Results page in the main function
def main():
    # ...existing code...
    
    # Simulation Results page
    elif page == "Full Simulation Results":
        st.header("Full Playoff Simulation Results")
        
        # Simulation options in sidebar
        st.sidebar.subheader("Simulation Settings")
        n_simulations = st.sidebar.slider("Number of simulations", 1000, 20000, 10000, 1000)
        
        # Run the simulation once per day (cached)
        if playoff_matchups and not team_data.empty:
            # Check if we have cached simulation results
            sim_results_key = f"sim_results_{season_str}"
            if sim_results_key in st.session_state and st.session_state[sim_results_key].get('n_simulations') == n_simulations:
                bracket_results = st.session_state[sim_results_key]['results']
                st.success(f"Using cached simulation results ({n_simulations} simulations)")
            else:
                # Run new simulation
                st.info(f"Running new simulation with {n_simulations} iterations...")
                bracket_results = run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations)
                
                # Cache the results in session state
                st.session_state[sim_results_key] = {
                    'results': bracket_results,
                    'n_simulations': n_simulations,
                    'timestamp': datetime.now()
                }
            
            # Display comprehensive simulation results
            display_simulation_results(bracket_results, n_simulations)
        else:
            st.error("Playoff matchups or team data not available for simulation")
    
    # ...existing code...

# ...existing code...

def display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data):
    """Display enhanced head-to-head prediction for two teams"""
    st.subheader("Head-to-Head Prediction")
    
    # Create a mock playoff matchup for prediction
    mock_top_seed = {
        'teamAbbrev': team1_data['teamAbbrev'],
        'teamName': team1_data['teamName'],
        'division_rank': 1  # Mock value
    }
    
    mock_bottom_seed = {
        'teamAbbrev': team2_data['teamAbbrev'],
        'teamName': team2_data['teamName'],
        'division_rank': 2  # Mock value
    }
    
    # Show clear explanation of home ice advantage
    st.info(f"**Note:** {team1} is considered the home team with home ice advantage in this comparison")
    
    # Create matchup data
    matchup_df = create_matchup_data(mock_top_seed, mock_bottom_seed, team_data)
    
    # Check if these teams met in simulations
    simulation_results = None
    if 'recent_simulations' in st.session_state:
        sim_results = st.session_state['recent_simulations']
        # Look for this matchup in potential matchups data
        if 'potential_matchups' in sim_results:
            matchup_key = f"{team1_data['teamAbbrev']}_vs_{team2_data['teamAbbrev']}"
            alt_matchup_key = f"{team2_data['teamAbbrev']}_vs_{team1_data['teamAbbrev']}"
            
            if matchup_key in sim_results['potential_matchups']:
                simulation_results = sim_results['potential_matchups'][matchup_key]
            elif alt_matchup_key in sim_results['potential_matchups']:
                simulation_results = sim_results['potential_matchups'][alt_matchup_key]
                # Flip the results since teams are reversed
                if 'probability' in simulation_results:
                    simulation_results['probability'] = 1 - simulation_results['probability']
    
    # Simulate a series between these teams
    if not matchup_df.empty:
        sim_result = simulate_playoff_series(matchup_df, model_data)
        
        if sim_result:
            # Create columns for model predictions
            model_cols = st.columns(4)
            
            # Show LR model prediction (raw)
            with model_cols[0]:
                lr_prob = sim_result.get('lr_probability', 0.5) * 100
                st.metric("LR Model", f"{lr_prob:.1f}%")
                st.caption(f"Logistic Regression (Raw)")
            
            # Show XGB model prediction (raw)
            with model_cols[1]:
                xgb_prob = sim_result.get('xgb_probability', 0.5) * 100
                st.metric("XGB Model", f"{xgb_prob:.1f}%")
                st.caption(f"XGBoost (Raw)")
            
            # Show ensemble prediction with home ice boost
            with model_cols[2]:
                ensemble_prob = sim_result.get('win_probability', 0.5) * 100
                st.metric("Ensemble", f"{ensemble_prob:.1f}%")
                st.caption(f"With Home Ice")
            
            # Show simulation result if available
            with model_cols[3]:
                if simulation_results and 'probability' in simulation_results:
                    sim_prob = simulation_results['probability'] * 100
                    st.metric("Simulation", f"{sim_prob:.1f}%")
                    st.caption(f"From {st.session_state.get('sim_count', 10000)} sims")
                else:
                    st.metric("Simulation", "N/A")
                    st.caption("Teams did not meet")
            
            # Get win probability
            win_prob = sim_result['win_probability'] * 100
            predicted_winner = team1 if win_prob >= 50 else team2
            winner_prob = win_prob if win_prob >= 50 else 100 - win_prob
            
            # Display prediction summary
            st.success(f"**In a playoff series, {predicted_winner} would win {winner_prob:.1f}% of the time**")
            
            # Show series outcome distribution
            if 'win_distribution' in sim_result and sim_result['win_distribution']:
                st.subheader("Series Outcome Distribution")
                dist = sim_result['win_distribution']
                total = sum(dist.values())
                
                # Create a DataFrame for better visualization
                outcome_data = []
                
                # Team 1 outcomes
                for outcome in ['4-0', '4-1', '4-2', '4-3']:
                    pct = (dist.get(outcome, 0) / total) * 100 if total > 0 else 0
                    outcome_data.append({
                        'Team': team1,
                        'Outcome': outcome,
                        'Games': 4 if outcome == '4-0' else 5 if outcome == '4-1' else 6 if outcome == '4-2' else 7,
                        'Probability': pct
                    })
                
                # Team 2 outcomes
                for team1_outcome, team2_outcome in zip(['0-4', '1-4', '2-4', '3-4'], ['4-0', '4-1', '4-2', '4-3']):
                    pct = (dist.get(team1_outcome, 0) / total) * 100 if total > 0 else 0
                    outcome_data.append({
                        'Team': team2,
                        'Outcome': team2_outcome,
                        'Games': 4 if team2_outcome == '4-0' else 5 if team2_outcome == '4-1' else 6 if team2_outcome == '4-2' else 7,
                        'Probability': pct
                    })
                
                # Convert to DataFrame and display 
                outcome_df = pd.DataFrame(outcome_data)
                
                # Display as a table
                styled_table = display_colored_table(
                    outcome_df[['Team', 'Outcome', 'Probability']].sort_values(['Team', 'Games']),
                    ['Probability'], 
                    cmap='coolwarm'
                )
                st.dataframe(styled_table, use_container_width=True)
                
                # Create a bar chart of outcomes
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Team 1 outcomes in blue
                team1_data = outcome_df[outcome_df['Team'] == team1]
                team1_bars = ax.bar(
                    team1_data['Outcome'], 
                    team1_data['Probability'], 
                    color='royalblue', 
                    alpha=0.7,
                    label=team1
                )
                
                # Team 2 outcomes in red
                team2_data = outcome_df[outcome_df['Team'] == team2]
                team2_bars = ax.bar(
                    [f"{o} ({team2})" for o in team2_data['Outcome']], 
                    team2_data['Probability'], 
                    color='firebrick', 
                    alpha=0.7,
                    label=team2
                )
                
                # Add labels on top of bars
                for bars in [team1_bars, team2_bars]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.5,
                            f'{height:.1f}%',
                            ha='center', 
                            va='bottom',
                            fontsize=9
                        )
                
                ax.set_ylabel('Probability (%)')
                ax.set_title('Series Outcome Probabilities')
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

    # Display team comparison metrics
    display_team_comparison_metrics(team1, team2, team1_data, team2_data)
    
def display_team_comparison_metrics(team1, team2, team1_data, team2_data):
    """Display comprehensive comparison metrics for two teams"""
    st.subheader("Team Comparison Metrics")
    
    # Define categories and metrics to show
    metric_categories = {
        "General": [
            ("Points", "points", None),
            ("Goal Differential", "goalDifferential", None),
            ("GF/G", "goalsFor", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y else None),
            ("GA/G", "goalsAgainst", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y else None)
        ],
        "Special Teams": [
            ("Power Play %", "PP%", None),
            ("Penalty Kill %", "PK%", None),
            ("PP Goals", "powerPlayGoals", None),
            ("SH Goals Against", "shGoalsAgainst", None)
        ],
        "Advanced Stats": [
            ("xGoals %", "xGoalsPercentage", None),
            ("Corsi %", "corsiPercentage", None),
            ("Fenwick %", "fenwickPercentage", None),
            ("High-Danger Chances For", "highDangerShotsFor", None),
            ("High-Danger Chances Against", "highDangerShotsAgainst", None)
        ],
        "Home/Road Performance": [
            ("Home Win %", "homeWin%", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y and 'homeWins' in y else None),
            ("Road Win %", "roadWin%", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y and 'roadWins' in y else None),
            ("Home Regulation Wins", "homeRegulationWins", None),
            ("Road Regulation Wins", "roadRegulationWins", None)
        ],
        "Playoff Experience": [
            ("Playoff Performance", "playoff_performance_score", None),
            ("Recent Playoff Wins", "weighted_playoff_wins", None),
            ("Recent Playoff Rounds", "weighted_playoff_rounds", None)
        ]
    }
    
    # Create tabs for different metric categories
    category_tabs = st.tabs(list(metric_categories.keys()))
    
    # Process each category
    for i, (category, tab) in enumerate(zip(metric_categories.keys(), category_tabs)):
        with tab:
            metrics = metric_categories[category]
            
            # Create a DataFrame for this category's metrics
            comparison_data = []
            
            for display_name, col_name, calculator in metrics:
                # Check if both teams have this metric
                if col_name in team1_data and col_name in team2_data:
                    team1_val = team1_data[col_name]
                    team2_val = team2_data[col_name]
                    
                    # Apply custom calculator if provided
                    if calculator:
                        team1_val = calculator(team1_val, team1_data)
                        team2_val = calculator(team2_val, team2_data)
                    
                    # Handle null values after calculation
                    if pd.isna(team1_val) or pd.isna(team2_val):
                        continue
                    
                    # Format values appropriately
                    if isinstance(team1_val, (float, np.float64)):
                        # Handle percentage values
                        if col_name.endswith('%') or 'Percentage' in col_name:
                            # Convert decimal to percentage if needed
                            if 0 < team1_val < 1:
                                team1_val_display = f"{team1_val*100:.1f}%"
                                team2_val_display = f"{team2_val*100:.1f}%"
                                # Calculate difference as percentage points
                                diff = (team1_val - team2_val) * 100
                            else:
                                team1_val_display = f"{team1_val:.1f}%"
                                team2_val_display = f"{team2_val:.1f}%"
                                diff = team1_val - team2_val
                        else:
                            # Format regular numbers
                            team1_val_display = f"{team1_val:.1f}" if team1_val % 1 != 0 else f"{int(team1_val)}"
                            team2_val_display = f"{team2_val:.1f}" if team2_val % 1 != 0 else f"{int(team2_val)}"
                            diff = team1_val - team2_val
                    else:
                        # Handle string or other types
                        team1_val_display = str(team1_val)
                        team2_val_display = str(team2_val)
                        diff = 0  # No meaningful difference for non-numeric
                    
                    # Add to comparison data
                    comparison_data.append({
                        'Metric': display_name,
                        team1: team1_val_display,
                        team2: team2_val_display,
                        'Difference': diff,
                        'Favors': team1 if diff > 0 else team2 if diff < 0 else "Tie"
                    })
            
            # Display the data if we have any
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                # Sort metrics alphabetically for consistency
                comparison_df = comparison_df.sort_values('Metric')
                
                # Display the basic comparison table
                st.table(comparison_df[['Metric', team1, team2]])
                
                # Create a horizontal bar chart to show the relative strengths
                if len(comparison_data) > 0:
                    # Filter to numeric metrics where we can calculate meaningful differences
                    numeric_metrics = [data for data in comparison_data if isinstance(data['Difference'], (int, float)) and data['Difference'] != 0]
                    
                    if numeric_metrics:
                        # Convert to DataFrame for plotting
                        plot_df = pd.DataFrame(numeric_metrics)
                        
                        # Sort by absolute difference
                        plot_df['AbsDiff'] = plot_df['Difference'].abs()
                        plot_df = plot_df.sort_values('AbsDiff', ascending=True)
                        
                        # Create figure
                        fig_height = max(4, len(plot_df) * 0.4)
                        fig, ax = plt.subplots(figsize=(10, fig_height))
                        
                        # Create colors based on which team is favored
                        colors = ['royalblue' if row['Favors'] == team1 else 'firebrick' for _, row in plot_df.iterrows()]
                        
                        # Create the horizontal bars
                        bars = ax.barh(plot_df['Metric'], plot_df['Difference'], color=colors)
                        
                        # Add a vertical line at 0
                        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
                        
                        # Add labels for the teams
                        ax.text(-max(plot_df['AbsDiff'])/2, -0.6, team2 + " ←", ha='center', va='top', fontsize=10, fontweight='bold')
                        ax.text(max(plot_df['AbsDiff'])/2, -0.6, "→ " + team1, ha='center', va='top', fontsize=10, fontweight='bold')
                        
                        # Customize appearance
                        ax.set_title(f'Relative Team Strengths - {category}')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        st.pyplot(fig)
            else:
                st.write(f"No comparison data available for {category} metrics")

# ...existing code...

def main():
    # ...existing code...
    
    # Head-to-Head Comparison page
    elif page == "Head-to-Head Comparison":
        st.header("Team Head-to-Head Comparison")
        
        st.write("""
        Compare any two teams in a hypothetical playoff matchup. The first team selected is considered 
        the higher seed with home ice advantage.
        """)
        
        # Team selection
        st.subheader("Select Teams to Compare")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Home Team", sorted(team_data['teamName'].unique()), key="team1")
        
        with col2:
            # Filter to exclude team1
            available_teams = sorted([team for team in team_data['teamName'].unique() if team != team1])
            team2 = st.selectbox("Select Road Team", available_teams, key="team2")
        
        # Display team comparison if two different teams are selected
        if team1 != team2:
            team1_data = team_data[team_data['teamName'] == team1].iloc[0]
            team2_data = team_data[team_data['teamName'] == team2].iloc[0]
            
            # Display team logos if available
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(team1)
                if 'teamLogo' in team1_data and team1_data['teamLogo']:
                    logo = load_team_logo(team1_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            with col2:
                st.subheader(team2)
                if 'teamLogo' in team2_data and team2_data['teamLogo']:
                    logo = load_team_logo(team2_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            # Display head-to-head prediction
            display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data)
        else:
            st.warning("Please select two different teams")
    
    # ...existing code...

# ...existing code...

def display_sim_your_own_bracket(playoff_matchups, team_data, model_data):
    """
    Display an interactive bracket simulation where users can run their own bracket simulations
    and track the results across multiple runs.
    
    Args:
        playoff_matchups: Dictionary with first round playoff matchups
        team_data: DataFrame with team data
        model_data: Dictionary with model data
    """
    st.write("""
    Run your own bracket simulations and track the results. Each time you click "Simulate Bracket",
    we'll simulate one complete playoff tournament from the first round through the Stanley Cup Final.
    """)
    
    # Initialize session state variables for tracking simulation history
    if 'user_sim_count' not in st.session_state:
        st.session_state.user_sim_count = 0
    if 'user_sim_results' not in st.session_state:
        st.session_state.user_sim_results = {}
    if 'user_sim_history' not in st.session_state:
        st.session_state.user_sim_history = []
    
    # Create columns for actions
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Button to run a new simulation
        if st.button("🏆 Simulate Bracket", key="run_sim_button"):
            with st.spinner("Simulating playoff bracket..."):
                # Run a single simulation
                sim_result = simulate_single_bracket(playoff_matchups, team_data, model_data)
                
                if sim_result:
                    # Increment counter
                    st.session_state.user_sim_count += 1
                    
                    # Add to simulation history
                    sim_result['sim_num'] = st.session_state.user_sim_count
                    st.session_state.user_sim_history.append(sim_result)
                    
                    # Update team statistics in aggregated results
                    for team, result in sim_result['results'].items():
                        if team not in st.session_state.user_sim_results:
                            st.session_state.user_sim_results[team] = {
                                'team': team,
                                'team_name': result['team_name'],
                                'round_1': 0,
                                'round_2': 0,
                                'conf_final': 0,
                                'final': 0,
                                'champion': 0,
                                'total_sims': 0
                            }
                        
                        # Update counters
                        st.session_state.user_sim_results[team]['total_sims'] += 1
                        for round_key, advanced in result.items():
                            if round_key in ['round_1', 'round_2', 'conf_final', 'final', 'champion'] and advanced:
                                st.session_state.user_sim_results[team][round_key] += 1
    
    with col2:
        # Show simulation count and clear button
        if st.session_state.user_sim_count > 0:
            st.write(f"**{st.session_state.user_sim_count}** bracket simulations run")
            
            # Add clear button inline
            if st.button("Clear Results", key="clear_results_button"):
                st.session_state.user_sim_count = 0
                st.session_state.user_sim_results = {}
                st.session_state.user_sim_history = []
                st.experimental_rerun()
    
    # Display current simulation results if any
    if st.session_state.user_sim_count > 0:
        # Display the latest simulation result
        latest_sim = st.session_state.user_sim_history[-1]
        
        st.subheader(f"Simulation #{latest_sim['sim_num']} Results")
        
        # Create a visual bracket representation
        display_bracket_visual(latest_sim)
        
        # Display aggregated results table
        st.subheader("Aggregated Results")
        
        # Convert results dict to DataFrame for display
        results_df = pd.DataFrame(list(st.session_state.user_sim_results.values()))
        
        # Calculate percentages
        for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
            if col in results_df.columns:
                results_df[f"{col}_pct"] = (results_df[col] / results_df['total_sims'] * 100).round(1)
        
        # Sort by championship percentage
        results_df = results_df.sort_values('champion_pct', ascending=False)
        
        # Reformat for display
        display_df = results_df.copy()
        display_df['Round 1'] = display_df['round_1_pct'].astype(str) + '%'
        display_df['Round 2'] = display_df['round_2_pct'].astype(str) + '%'
        display_df['Conf Final'] = display_df['conf_final_pct'].astype(str) + '%'
        display_df['Finals'] = display_df['final_pct'].astype(str) + '%'
        display_df['Champion'] = display_df['champion_pct'].astype(str) + '%'
        
        # Display table with team results
        st.table(display_df[['team_name', 'Round 1', 'Round 2', 'Conf Final', 'Finals', 'Champion']])
        
        # Show simulation history
        with st.expander("Simulation History"):
            # Create tabs for different simulations
            num_sims_to_show = min(5, len(st.session_state.user_sim_history))
            if num_sims_to_show > 0:
                sim_tabs = st.tabs([f"Sim #{sim['sim_num']}" for sim in st.session_state.user_sim_history[-num_sims_to_show:]])
                
                # Display each simulation
                for i, tab in enumerate(sim_tabs):
                    with tab:
                        sim_idx = len(st.session_state.user_sim_history) - num_sims_to_show + i
                        sim = st.session_state.user_sim_history[sim_idx]
                        st.write(f"**Playoff Path:**")
                        
                        # Show the bracket progression
                        for round_name, matchups in sim['bracket'].items():
                            st.write(f"**{round_name}:**")
                            for matchup in matchups:
                                st.write(f"- {matchup['winner']} defeated {matchup['loser']}")
                        
                        st.write(f"**Champion:** {sim['champion']}")
    else:
        st.info("Click the 'Simulate Bracket' button to run a playoff simulation.")

def simulate_single_bracket(playoff_matchups, team_data, model_data):
    """
    Simulate a single playoff bracket from start to finish
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team data
        model_data: Dictionary with model data
        
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
                        'champion': False
                    }

    # FIRST ROUND
    round_1_winners = {}
    for conference, matchups in playoff_matchups.items():
        round_1_winners[conference] = {}
        
        for series_id, matchup in matchups.items():
            # Create matchup data
            matchup_df = create_matchup_data(matchup['top_seed'], matchup['bottom_seed'], team_data)
            
            # Simulate the series
            if not matchup_df.empty:
                # Get win probability
                win_prob = get_series_win_probability(matchup_df, model_data)
                
                # Determine the winner
                higher_seed_wins = np.random.random() < win_prob
                
                top_seed_abbrev = matchup['top_seed']['teamAbbrev']
                bottom_seed_abbrev = matchup['bottom_seed']['teamAbbrev']
                winner_abbrev = top_seed_abbrev if higher_seed_wins else bottom_seed_abbrev
                loser_abbrev = bottom_seed_abbrev if higher_seed_wins else top_seed_abbrev
                
                # Store results
                winner = matchup['top_seed'] if higher_seed_wins else matchup['bottom_seed']
                loser = matchup['bottom_seed'] if higher_seed_wins else matchup['top_seed']
                
                # Update team advancement record
                team_results[winner_abbrev]['round_1'] = True
                
                # Store winner for next round
                round_1_winners[conference][series_id] = {
                    'team': winner,
                    'original_matchup': series_id
                }
                
                # Track in bracket progression
                bracket_progression['First Round'].append({
                    'winner': team_results[winner_abbrev]['team_name'],
                    'loser': team_results[loser_abbrev]['team_name'],
                    'conference': conference,
                    'series_id': series_id
                })
    
    # SECOND ROUND (Division Finals)
    round_2_winners = {}
    for conference, r1_winners in round_1_winners.items():
        round_2_winners[conference] = {}
        
        # Extract division information from series_ids
        divisions = set()
        for key in r1_winners.keys():
            if len(key) >= 1 and key[0].isalpha():
                divisions.add(key[0])
        
        # Create second round matchups
        for division in divisions:
            # Find matchup winners from this division's first round
            div_winners = [r1_winners[k] for k in r1_winners if k.startswith(division)]
            
            if len(div_winners) >= 2:  # Need at least two teams for a matchup
                # Identify teams
                team1 = div_winners[0]['team']
                team2 = div_winners[1]['team']
                
                # Determine seeding
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
                
                # Create matchup data
                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                
                # Simulate the series
                if not matchup_df.empty:
                    # Get win probability
                    win_prob = get_series_win_probability(matchup_df, model_data)
                    
                    # Determine the winner
                    higher_seed_wins = np.random.random() < win_prob
                    
                    top_seed_abbrev = top_seed['teamAbbrev']
                    bottom_seed_abbrev = bottom_seed['teamAbbrev']
                    winner_abbrev = top_seed_abbrev if higher_seed_wins else bottom_seed_abbrev
                    loser_abbrev = bottom_seed_abbrev if higher_seed_wins else top_seed_abbrev
                    
                    # Store results
                    winner = top_seed if higher_seed_wins else bottom_seed
                    loser = bottom_seed if higher_seed_wins else top_seed
                    
                    # Update team advancement record
                    team_results[winner_abbrev]['round_2'] = True
                    
                    # Store winner for next round
                    round_2_winners[conference][f"{division}_final"] = {
                        'team': winner,
                        'division': division
                    }
                    
                    # Track in bracket progression
                    bracket_progression['Second Round'].append({
                        'winner': team_results[winner_abbrev]['team_name'],
                        'loser': team_results[loser_abbrev]['team_name'],
                        'conference': conference,
                        'division': division
                    })
    
    # CONFERENCE FINALS
    conf_winners = {}
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
                
                # Create matchup data
                matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
                
                # Simulate the series
                if not matchup_df.empty:
                    # Get win probability
                    win_prob = get_series_win_probability(matchup_df, model_data)
                    
                    # Determine the winner
                    higher_seed_wins = np.random.random() < win_prob
                    
                    top_seed_abbrev = top_seed['teamAbbrev']
                    bottom_seed_abbrev = bottom_seed['teamAbbrev']
                    winner_abbrev = top_seed_abbrev if higher_seed_wins else bottom_seed_abbrev
                    loser_abbrev = bottom_seed_abbrev if higher_seed_wins else top_seed_abbrev
                    
                    # Store results
                    winner = top_seed if higher_seed_wins else bottom_seed
                    loser = bottom_seed if higher_seed_wins else top_seed
                    
                    # Update team advancement record
                    team_results[winner_abbrev]['conf_final'] = True
                    
                    # Store winner for final
                    conf_winners[conference] = {
                        'team': winner,
                        'conference': conference
                    }
                    
                    # Track in bracket progression
                    bracket_progression['Conference Finals'].append({
                        'winner': team_results[winner_abbrev]['team_name'],
                        'loser': team_results[loser_abbrev]['team_name'],
                        'conference': conference
                    })
    
    # STANLEY CUP FINAL
    champion_abbrev = None
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
            
            # Create matchup data
            matchup_df = create_matchup_data(top_seed, bottom_seed, team_data)
            
            # Simulate the series
            if not matchup_df.empty:
                # Get win probability
                win_prob = get_series_win_probability(matchup_df, model_data)
                
                # Determine the winner
                higher_seed_wins = np.random.random() < win_prob
                
                top_seed_abbrev = top_seed['teamAbbrev']
                bottom_seed_abbrev = bottom_seed['teamAbbrev']
                winner_abbrev = top_seed_abbrev if higher_seed_wins else bottom_seed_abbrev
                loser_abbrev = bottom_seed_abbrev if higher_seed_wins else top_seed_abbrev
                
                # Store results
                winner = top_seed if higher_seed_wins else bottom_seed
                loser = bottom_seed if higher_seed_wins else top_seed
                
                # Update team advancement record
                team_results[winner_abbrev]['final'] = True
                team_results[winner_abbrev]['champion'] = True
                team_results[loser_abbrev]['final'] = True
                
                # Track in bracket progression and set champion
                bracket_progression['Stanley Cup Final'].append({
                    'winner': team_results[winner_abbrev]['team_name'],
                    'loser': team_results[loser_abbrev]['team_name']
                })
                
                champion_abbrev = winner_abbrev
    
    # Prepare and return the simulation result
    return {
        'results': team_results,
        'bracket': bracket_progression,
        'champion': team_results[champion_abbrev]['team_name'] if champion_abbrev else "No champion determined"
    }

def get_series_win_probability(matchup_df, model_data):
    """Calculate series win probability using available models"""
    win_prob = 0.5  # Default probability
    
    # Try to use models if available
    if model_data and not matchup_df.empty:
        try:
            # Use LR model if available
            if 'lr' in model_data and 'model' in model_data['lr'] and 'features' in model_data['lr']:
                lr_features = [feat for feat in model_data['lr']['features'] if feat in matchup_df.columns]
                
                if len(lr_features) == len(model_data['lr']['features']):
                    lr_prob = model_data['lr']['model'].predict_proba(matchup_df[lr_features])[:, 1][0]
                    win_prob = lr_prob
            
            # Use XGB model if available
            if 'xgb' in model_data and 'model' in model_data['xgb'] and 'features' in model_data['xgb']:
                xgb_features = [feat for feat in model_data['xgb']['features'] if feat in matchup_df.columns]
                
                if len(xgb_features) == len(model_data['xgb']['features']):
                    xgb_prob = model_data['xgb']['model'].predict_proba(matchup_df[xgb_features])[:, 1][0]
                    
                    # Use ensemble if both models available
                    if win_prob != 0.5:
                        win_prob = (win_prob + xgb_prob) / 2  # Ensemble
                    else:
                        win_prob = xgb_prob
            
            # Fallback to points difference
            if win_prob == 0.5 and 'points_diff' in matchup_df.columns:
                points_diff = matchup_df['points_diff'].iloc[0]
                win_prob = 1 / (1 + np.exp(-0.05 * points_diff))
        except Exception as e:
            st.warning(f"Error calculating win probability, using default: {str(e)}")
    
    return win_prob

def display_bracket_visual(sim_result):
    """
    Display a visual representation of a simulated playoff bracket
    
    Args:
        sim_result: Dictionary with simulation results
    """
    # Display the champion prominently
    champion = sim_result['champion']
    st.success(f"🏆 **Stanley Cup Champion: {champion}**")
    
    # Create a visual representation of the playoff bracket
    # Use columns to organize the rounds
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # First Round
    with col1:
        st.markdown("### First Round")
        for matchup in sim_result['bracket']['First Round']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Second Round
    with col2:
        st.markdown("### Second Round")
        for matchup in sim_result['bracket']['Second Round']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Conference Finals
    with col3:
        st.markdown("### Conference Finals")
        for matchup in sim_result['bracket']['Conference Finals']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Stanley Cup Final
    with col4:
        st.markdown("### Stanley Cup Final")
        if sim_result['bracket']['Stanley Cup Final']:
            matchup = sim_result['bracket']['Stanley Cup Final'][0]
            with st.container():
                st.markdown(f"**{matchup['winner']}** 🏆")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")

def display_about_page(model_data):
    """Display enhanced about page with model information and app details"""
    st.subheader("About the NHL Playoff Predictor")
    
    st.write("""
    This application predicts NHL playoff outcomes by leveraging machine learning models trained on historical playoff
    data combined with current season team statistics and advanced analytics metrics.
    
    The prediction system uses an ensemble approach that combines:
    1. A logistic regression model capturing linear relationships between features
    2. An XGBoost model that identifies complex, non-linear patterns in the data
    3. Home ice advantage adjustment based on historical playoff performance
    """)
    
    # Model information section
    st.subheader("Model Information")
    
    # Create columns to display model info side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Logistic Regression Model**")
        if model_data and 'lr' in model_data:
            lr_model = model_data['lr']
            st.write(f"Features: {len(lr_model.get('features', []))} input variables")
            
            top_features = []
            if 'coefficients' in lr_model:
                top_features = [f"{feat}: {coef:.4f}" for feat, coef in lr_model['coefficients'][:5]]
            
            if top_features:
                st.write("Top 5 features by importance:")
                for feat in top_features:
                    st.write(f"- {feat}")
            else:
                st.write("Model feature importance not available")
        else:
            st.write("Logistic Regression model not loaded")
    
    with col2:
        st.write("**XGBoost Model**")
        if model_data and 'xgb' in model_data:
            xgb_model = model_data['xgb']
            st.write(f"Features: {len(xgb_model.get('features', []))} input variables")
            
            top_features = []
            if 'importance' in xgb_model:
                top_features = [f"{feat}: {imp:.4f}" for feat, imp in xgb_model['importance'][:5]]
            
            if top_features:
                st.write("Top 5 features by importance:")
                for feat in top_features:
                    st.write(f"- {feat}")
            else:
                st.write("Model feature importance not available")
        else:
            st.write("XGBoost model not loaded")
    
    # Key features section
    st.subheader("Key Features Used")
    
    feature_categories = {
        "Team Performance": [
            "Goal differential per game",
            "Points percentage",
            "Regulation wins (home and road)",
            "Win streaks and performance trends"
        ],
        "Special Teams": [
            "Power play percentage (relative to league average)",
            "Penalty kill percentage (relative to league average)",
            "Special teams composite score",
            "Short-handed goals"
        ],
        "Advanced Metrics": [
            "Expected goals percentage (5v5)",
            "Corsi and Fenwick percentages",
            "High-danger chances for/against",
            "Possession-adjusted defensive metrics"
        ],
        "Playoff Experience": [
            "Recent playoff performance (weighted)",
            "Previous playoff rounds won",
            "Experience in elimination games"
        ]
    }
    
    # Create tabs for feature categories
    category_tabs = st.tabs(list(feature_categories.keys()))
    
    # Fill each tab with corresponding features
    for tab, category in zip(category_tabs, feature_categories.keys()):
        with tab:
            for feature in feature_categories[category]:
                st.write(f"• {feature}")
    
    # Data sources section
    st.subheader("Data Sources")
    st.write("""
    - Team statistics: NHL API (real-time data)
    - Advanced metrics: MoneyPuck (updated daily)
    - Historical playoff data (2005-present): Used for model training and validation
    """)
    
    # Prediction methodology
    st.subheader("Prediction Methodology")
    st.write("""
    For each matchup, the application:
    1. **Data Collection**: Gathers current team statistics and advanced metrics
    2. **Feature Engineering**: Calculates derived features (e.g., relative metrics, adjusted rates)
    3. **Model Prediction**: Generates independent predictions from both the logistic regression and XGBoost models
    4. **Ensemble Prediction**: Combines model predictions with appropriate weighting
    5. **Home Ice Adjustment**: Applies adjustment based on historical home ice advantage
    6. **Monte Carlo Simulation**: Simulates the series thousands of times to account for randomness
    
    The predicted win probability represents the percentage of simulations in which a team wins the series.
    """)
    
    # Limitations and caveats
    st.expander("Limitations and Caveats").write("""
    - Predictions are based on statistical patterns and cannot account for all factors (e.g., injuries, coaching changes)
    - The model assumes that historical patterns continue to be relevant for current matchups
    - Advanced metrics may have limited predictive power due to the inherent randomness in playoff hockey
    - Home ice advantage is applied as a uniform adjustment based on historical averages
    """)
    
    # App information
    st.subheader("Application Information")
    st.write("""
    - Data refreshed: Once per day (cached for 24 hours)
    - Default simulation count: 10,000 iterations per matchup
    - Model ensemble: Logistic Regression + XGBoost with equal weighting
    - Developed with: Streamlit, pandas, numpy, scikit-learn, XGBoost, matplotlib
    """)
    
    # Version and credits footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
    <p>NHL Playoff Predictor v1.0.0</p>
    <p>© 2023-2024 | All NHL team names, logos and data are property of the NHL and their respective teams</p>
    </div>
    """, unsafe_allow_html=True)

# Fix the main function to properly include all 5 pages
def main():
    # App title and description
    st.title("NHL Playoff Predictor")
    st.write("Predict playoff outcomes based on team statistics and advanced metrics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "First Round Matchups", 
        "Full Simulation Results", 
        "Head-to-Head Comparison", 
        "Sim Your Own Bracket",
        "About"
    ])
    
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
        
        # Combine standings and stats data
        if not standings_df.empty and not stats_df.empty:
            # Make sure season columns are strings
            standings_df['season'] = standings_df['season'].astype(str)
            stats_df['season'] = stats_df['season'].astype(str)
            
            # First merge standings and stats data
            team_data = pd.merge(standings_df, stats_df, 
                                 left_on=['season', 'teamName'], 
                                 right_on=['season', 'teamName'], 
                                 how='left')
            
            # Merge with advanced stats if available
            if not advanced_stats_df.empty:
                # Make sure season columns have the same format
                if 'season' in advanced_stats_df.columns:
                    # Convert season from "2024" to "20242025" format if needed
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str).apply(
                        lambda x: f"{x}{int(x)+1}" if len(x) == 4 and x.isdigit() else x
                    )
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str)
                
                # Perform the merge with the correct columns
                team_data = pd.merge(team_data, advanced_stats_df, 
                                     left_on=['season', 'teamAbbrev'],
                                     right_on=['season', 'team'],
                                     how='left')
            
            # Engineer features for prediction
            team_data = engineer_features(team_data)
            
            # Add playoff history metrics
            team_data = add_playoff_history_metrics(team_data)
        else:
            team_data = pd.DataFrame()
    
    # Load models
    model_data = load_models()
    
    # Determine playoff teams and create matchups
    playoff_matchups = determine_playoff_teams(standings_df)
    
    # Add model info to sidebar
    if model_data:
        st.sidebar.title("Model Information")
        st.sidebar.write(f"Model mode: {model_data.get('mode', 'default')}")
        st.sidebar.write(f"Home ice advantage: {model_data.get('home_ice_boost', 0.039)*100:.1f}%")
    
    # First Round Matchups page
    if page == "First Round Matchups":
        st.header("First Round Playoff Matchups")
        display_playoff_bracket(playoff_matchups, team_data, model_data)
    
    elif page == "Full Simulation Results":
        st.header("Full Playoff Simulation Results")
        
        # Simulation options in sidebar
        st.sidebar.subheader("Simulation Settings")
        n_simulations = st.sidebar.slider("Number of simulations", 1000, 20000, 10000, 1000)
        
        # Run the simulation once per day (cached)
        if playoff_matchups and not team_data.empty:
            # Check if we have cached simulation results
            sim_results_key = f"sim_results_{season_str}"
            if sim_results_key in st.session_state and st.session_state[sim_results_key].get('n_simulations') == n_simulations:
                bracket_results = st.session_state[sim_results_key]['results']
                st.success(f"Using cached simulation results ({n_simulations} simulations)")
            else:
                # Run new simulation
                st.info(f"Running new simulation with {n_simulations} iterations...")
                bracket_results = run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations)
                
                # Cache the results in session state
                st.session_state[sim_results_key] = {
                    'results': bracket_results,
                    'n_simulations': n_simulations,
                    'timestamp': datetime.now()
                }
                
                # Save in another variable for reference in other pages
                st.session_state['recent_simulations'] = bracket_results
                st.session_state['sim_count'] = n_simulations
            
            # Display comprehensive simulation results
            display_simulation_results(bracket_results, n_simulations)
        else:
            st.error("Playoff matchups or team data not available for simulation")
    
    # Head-to-Head Comparison page
    elif page == "Head-to-Head Comparison":
        st.header("Team Head-to-Head Comparison")
        
        st.write("""
        Compare any two teams in a hypothetical playoff matchup. The first team selected is considered 
        the higher seed with home ice advantage.
        """)
        
        # Team selection
        st.subheader("Select Teams to Compare")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Home Team", sorted(team_data['teamName'].unique()), key="team1")
        
        with col2:
            # Filter to exclude team1
            available_teams = sorted([team for team in team_data['teamName'].unique() if team != team1])
            team2 = st.selectbox("Select Road Team", available_teams, key="team2")
        
        # Display team comparison if two different teams are selected
        if team1 != team2:
            team1_data = team_data[team_data['teamName'] == team1].iloc[0]
            team2_data = team_data[team_data['teamName'] == team2].iloc[0]
            
            # Display team logos if available
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(team1)
                if 'teamLogo' in team1_data and team1_data['teamLogo']:
                    logo = load_team_logo(team1_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            with col2:
                st.subheader(team2)
                if 'teamLogo' in team2_data and team2_data['teamLogo']:
                    logo = load_team_logo(team2_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            # Display head-to-head prediction
            display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data)
        else:
            st.warning("Please select two different teams")
            
    elif page == "Sim Your Own Bracket":
        st.header("Simulate Your Own Playoff Bracket")
        display_sim_your_own_bracket(playoff_matchups, team_data, model_data)
        
    elif page == "About":
        st.header("About the NHL Playoff Predictor")
        display_about_page(model_data)

# Add missing function to run full playoff simulation
def run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations=10000):
    """
    Run a full playoff bracket simulation with comprehensive results
    
    Args:
        playoff_matchups: Dictionary with playoff matchups
        team_data: DataFrame with team stats and features
        model_data: Dictionary with model information
        n_simulations: Number of simulations to run (default: 10,000)
        
    Returns:
        dict: Dictionary with simulation results
    """
    # Call the comprehensive simulation function
    with st.spinner(f"Running {n_simulations} playoff simulations..."):
        return simulate_playoff_bracket(playoff_matchups, team_data, model_data, n_simulations, detailed_tracking=True)

# Add the missing display_simulation_results function which is referenced but undefined
def display_simulation_results(bracket_results, n_simulations):
    """
    Display comprehensive simulation results with visualizations
    
    Args:
        bracket_results: Dictionary with simulation results
        n_simulations: Number of simulations that were run
    """
    if not bracket_results or 'team_advancement' not in bracket_results:
        st.error("No simulation results available")
        return
    
    # Get team advancement data
    results_df = bracket_results['team_advancement'].copy()
    
    # Convert advancement columns to percentages
    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
        if col in results_df.columns:
            results_df[col] = results_df[col] * 100
    
    # Display Stanley Cup championship odds
    st.subheader("Stanley Cup Championship Odds")
    
    # Create two columns for chart and table
    chart_col, table_col = st.columns([3, 2])
    
    with chart_col:
        # Create a bar chart of Stanley Cup odds
        fig, ax = plt.subplots(figsize=(10, 8))
        top_teams = results_df.sort_values('champion', ascending=False).head(16)  # Show playoff teams
        
        # Create horizontal bar chart with team names
        bars = ax.barh(top_teams['teamName'], top_teams['champion'], color='lightblue')
        
        # Add percentage labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   va='center', fontsize=10)
        
        ax.set_xlabel('Championship Probability (%)')
        ax.set_title(f'Stanley Cup Championship Odds (Based on {n_simulations} Simulations)')
        ax.set_xlim(0, max(top_teams['champion']) * 1.1)  # Add some padding
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
    
    with table_col:
        # Display the table with color coding
        # Rename columns for better display
        display_df = results_df.copy()
        column_mapping = {
            'teamName': 'Team',
            'champion': 'Cup %',
            'final': 'Finals %',
            'conf_final': 'Conf Finals %',
            'round_2': 'Round 2 %',
            'round_1': 'Round 1 %',
            'avg_games_played': 'Avg Games'
        }
        
        # Rename columns
        display_df = display_df.rename(columns=column_mapping)
        
        # Order columns as desired
        display_columns = ['Team', 'Cup %', 'Finals %', 'Conf Finals %', 'Round 2 %', 'Round 1 %', 'Avg Games']
        available_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[available_columns].sort_values('Cup %', ascending=False)
        
        # Format avg_games_played
        if 'Avg Games' in display_df.columns:
            display_df['Avg Games'] = display_df['Avg Games'].round(1)
        
        # Display the color-coded table
        color_columns = [col for col in display_df.columns if '%' in col]
        st.write("### Playoff Advancement Table")
        styled_table = display_colored_table(display_df.head(16), color_columns)
        st.dataframe(styled_table, use_container_width=True)
    
    # Create the round-by-round advancement visualization
    st.subheader("Round-by-Round Advancement")
    
    # Create a stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select playoff teams to plot
    teams_to_plot = results_df.sort_values('champion', ascending=False).iloc[:16]
    rounds_to_plot = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
    round_labels = ['First Round', 'Second Round', 'Conf Finals', 'Stanley Cup Final', 'Champion']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create stacked bars
    bottom = np.zeros(len(teams_to_plot))
    width = 0.6
    
    # Plot each round
    bars = []
    for i, (col, label) in enumerate(zip(rounds_to_plot, round_labels)):
        if col in teams_to_plot.columns:
            bar = ax.barh(teams_to_plot['teamName'], teams_to_plot[col], 
                          left=0, height=width, label=label, color=colors[i])
            bars.append(bar)
    
    ax.set_xlabel('Probability (%)')
    ax.set_title('Round-by-Round Playoff Advancement Probabilities')
    ax.legend(loc='upper right')
    
    st.pyplot(fig)
    
    # Display detailed matchup information if available
    if all(key in bracket_results for key in ['round2_matchups', 'conf_final_matchups', 'final_matchups']):
        # Subheader for potential matchups
        st.subheader("Most Likely Playoff Matchups")
        
        # Create tabs for different rounds
        round_tabs = st.tabs(["Second Round", "Conference Finals", "Stanley Cup Final"])
        
        # Second Round Matchups
        with round_tabs[0]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['round2_matchups'], "Second Round")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['round2_matchups'].empty:
                    display_df = bracket_results['round2_matchups'].copy()
                    display_df = display_df[['matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
        
        # Conference Finals Matchups
        with round_tabs[1]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['conf_final_matchups'], "Conference Finals")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['conf_final_matchups'].empty:
                    display_df = bracket_results['conf_final_matchups'].copy()
                    display_df = display_df[['conference', 'matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Conference', 'Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
        
        # Stanley Cup Final Matchups
        with round_tabs[2]:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display the chart
                fig = display_matchup_charts(bracket_results['final_matchups'], "Stanley Cup Final")
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Prepare data for display
                if not bracket_results['final_matchups'].empty:
                    display_df = bracket_results['final_matchups'].copy()
                    display_df = display_df[['matchup', 'probability', 'top_seed_win_pct']].sort_values('probability', ascending=False)
                    display_df.columns = ['Matchup', 'Probability', 'Higher Seed Win %']
                    
                    # Display the table
                    styled_table = display_colored_table(display_df, ['Probability', 'Higher Seed Win %'])
                    st.dataframe(styled_table, use_container_width=True)
    
    # Display most common bracket if available
    if 'most_common_bracket' in bracket_results:
        st.subheader("Most Likely Complete Playoff Bracket")
        most_common = bracket_results['most_common_bracket']
        
        # Create an expander to show the details
        with st.expander(f"Most Common Bracket (Occurred in {most_common['probability']:.1f}% of simulations)"):
            # Organize by round
            if most_common['bracket']:
                rounds = {
                    "First Round": most_common['bracket'][:8] if len(most_common['bracket']) > 7 else [],
                    "Second Round": most_common['bracket'][8:12] if len(most_common['bracket']) > 11 else [],
                    "Conference Finals": most_common['bracket'][12:14] if len(most_common['bracket']) > 13 else [],
                    "Stanley Cup Final": most_common['bracket'][14:] if len(most_common['bracket']) > 14 else []
                }
                
                # Display each round
                for round_name, results in rounds.items():
                    if results:
                        st.write(f"**{round_name}:**")
                        for i, result in enumerate(results, 1):
                            st.write(f"{i}. {result}")

def display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data):
    """Display enhanced head-to-head prediction for two teams"""
    st.subheader("Head-to-Head Prediction")
    
    # Create a mock playoff matchup for prediction
    mock_top_seed = {
        'teamAbbrev': team1_data['teamAbbrev'],
        'teamName': team1_data['teamName'],
        'division_rank': 1  # Mock value
    }
    
    mock_bottom_seed = {
        'teamAbbrev': team2_data['teamAbbrev'],
        'teamName': team2_data['teamName'],
        'division_rank': 2  # Mock value
    }
    
    # Show clear explanation of home ice advantage
    st.info(f"**Note:** {team1} is considered the home team with home ice advantage in this comparison")
    
    # Create matchup data
    matchup_df = create_matchup_data(mock_top_seed, mock_bottom_seed, team_data)
    
    # Check if these teams met in simulations
    simulation_results = None
    if 'recent_simulations' in st.session_state:
        sim_results = st.session_state['recent_simulations']
        # Look for this matchup in potential matchups data
        if 'potential_matchups' in sim_results:
            matchup_key = f"{team1_data['teamAbbrev']}_vs_{team2_data['teamAbbrev']}"
            alt_matchup_key = f"{team2_data['teamAbbrev']}_vs_{team1_data['teamAbbrev']}"
            
            if matchup_key in sim_results['potential_matchups']:
                simulation_results = sim_results['potential_matchups'][matchup_key]
            elif alt_matchup_key in sim_results['potential_matchups']:
                simulation_results = sim_results['potential_matchups'][alt_matchup_key]
                # Flip the results since teams are reversed
                if 'probability' in simulation_results:
                    simulation_results['probability'] = 1 - simulation_results['probability']
    
    # Simulate a series between these teams
    if not matchup_df.empty:
        sim_result = simulate_playoff_series(matchup_df, model_data)
        
        if sim_result:
            # Create columns for model predictions
            model_cols = st.columns(4)
            
            # Show LR model prediction (raw)
            with model_cols[0]:
                lr_prob = sim_result.get('lr_probability', 0.5) * 100
                st.metric("LR Model", f"{lr_prob:.1f}%")
                st.caption(f"Logistic Regression (Raw)")
            
            # Show XGB model prediction (raw)
            with model_cols[1]:
                xgb_prob = sim_result.get('xgb_probability', 0.5) * 100
                st.metric("XGB Model", f"{xgb_prob:.1f}%")
                st.caption(f"XGBoost (Raw)")
            
            # Show ensemble prediction with home ice boost
            with model_cols[2]:
                ensemble_prob = sim_result.get('win_probability', 0.5) * 100
                st.metric("Ensemble", f"{ensemble_prob:.1f}%")
                st.caption(f"With Home Ice")
            
            # Show simulation result if available
            with model_cols[3]:
                if simulation_results and 'probability' in simulation_results:
                    sim_prob = simulation_results['probability'] * 100
                    st.metric("Simulation", f"{sim_prob:.1f}%")
                    st.caption(f"From {st.session_state.get('sim_count', 10000)} sims")
                else:
                    st.metric("Simulation", "N/A")
                    st.caption("Teams did not meet")
            
            # Get win probability
            win_prob = sim_result['win_probability'] * 100
            predicted_winner = team1 if win_prob >= 50 else team2
            winner_prob = win_prob if win_prob >= 50 else 100 - win_prob
            
            # Display prediction summary
            st.success(f"**In a playoff series, {predicted_winner} would win {winner_prob:.1f}% of the time**")
            
            # Show series outcome distribution
            if 'win_distribution' in sim_result and sim_result['win_distribution']:
                st.subheader("Series Outcome Distribution")
                dist = sim_result['win_distribution']
                total = sum(dist.values())
                
                # Create a DataFrame for better visualization
                outcome_data = []
                
                # Team 1 outcomes
                for outcome in ['4-0', '4-1', '4-2', '4-3']:
                    pct = (dist.get(outcome, 0) / total) * 100 if total > 0 else 0
                    outcome_data.append({
                        'Team': team1,
                        'Outcome': outcome,
                        'Games': 4 if outcome == '4-0' else 5 if outcome == '4-1' else 6 if outcome == '4-2' else 7,
                        'Probability': pct
                    })
                
                # Team 2 outcomes
                for team1_outcome, team2_outcome in zip(['0-4', '1-4', '2-4', '3-4'], ['4-0', '4-1', '4-2', '4-3']):
                    pct = (dist.get(team1_outcome, 0) / total) * 100 if total > 0 else 0
                    outcome_data.append({
                        'Team': team2,
                        'Outcome': team2_outcome,
                        'Games': 4 if team2_outcome == '4-0' else 5 if team2_outcome == '4-1' else 6 if team2_outcome == '4-2' else 7,
                        'Probability': pct
                    })
                
                # Convert to DataFrame and display 
                outcome_df = pd.DataFrame(outcome_data)
                
                # Display as a table
                styled_table = display_colored_table(
                    outcome_df[['Team', 'Outcome', 'Probability']].sort_values(['Team', 'Games']),
                    ['Probability'], 
                    cmap='coolwarm'
                )
                st.dataframe(styled_table, use_container_width=True)
                
                # Create a bar chart of outcomes
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Team 1 outcomes in blue
                team1_data = outcome_df[outcome_df['Team'] == team1]
                team1_bars = ax.bar(
                    team1_data['Outcome'], 
                    team1_data['Probability'], 
                    color='royalblue', 
                    alpha=0.7,
                    label=team1
                )
                
                # Team 2 outcomes in red
                team2_data = outcome_df[outcome_df['Team'] == team2]
                team2_bars = ax.bar(
                    [f"{o} ({team2})" for o in team2_data['Outcome']], 
                    team2_data['Probability'], 
                    color='firebrick', 
                    alpha=0.7,
                    label=team2
                )
                
                # Add labels on top of bars
                for bars in [team1_bars, team2_bars]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.5,
                            f'{height:.1f}%',
                            ha='center', 
                            va='bottom',
                            fontsize=9
                        )
                
                ax.set_ylabel('Probability (%)')
                ax.set_title('Series Outcome Probabilities')
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

    # Display team comparison metrics
    display_team_comparison_metrics(team1, team2, team1_data, team2_data)

def display_team_comparison_metrics(team1, team2, team1_data, team2_data):
    """Display comprehensive comparison metrics for two teams"""
    st.subheader("Team Comparison Metrics")
    
    # Define categories and metrics to show
    metric_categories = {
        "General": [
            ("Points", "points", None),
            ("Goal Differential", "goalDifferential", None),
            ("GF/G", "goalsFor", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y else None),
            ("GA/G", "goalsAgainst", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y else None)
        ],
        "Special Teams": [
            ("Power Play %", "PP%", None),
            ("Penalty Kill %", "PK%", None),
            ("PP Goals", "powerPlayGoals", None),
            ("SH Goals Against", "shGoalsAgainst", None)
        ],
        "Advanced Stats": [
            ("xGoals %", "xGoalsPercentage", None),
            ("Corsi %", "corsiPercentage", None),
            ("Fenwick %", "fenwickPercentage", None),
            ("High-Danger Chances For", "highDangerShotsFor", None),
            ("High-Danger Chances Against", "highDangerShotsAgainst", None)
        ],
        "Home/Road Performance": [
            ("Home Win %", "homeWin%", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y and 'homeWins' in y else None),
            ("Road Win %", "roadWin%", lambda x, y: x / y['gamesPlayed'] if 'gamesPlayed' in y and 'roadWins' in y else None),
            ("Home Regulation Wins", "homeRegulationWins", None),
            ("Road Regulation Wins", "roadRegulationWins", None)
        ],
        "Playoff Experience": [
            ("Playoff Performance", "playoff_performance_score", None),
            ("Recent Playoff Wins", "weighted_playoff_wins", None),
            ("Recent Playoff Rounds", "weighted_playoff_rounds", None)
        ]
    }
    
    # Create tabs for different metric categories
    category_tabs = st.tabs(list(metric_categories.keys()))
    
    # Process each category
    for i, (category, tab) in enumerate(zip(metric_categories.keys(), category_tabs)):
        with tab:
            metrics = metric_categories[category]
            
            # Create a DataFrame for this category's metrics
            comparison_data = []
            
            for display_name, col_name, calculator in metrics:
                # Check if both teams have this metric
                if col_name in team1_data and col_name in team2_data:
                    team1_val = team1_data[col_name]
                    team2_val = team2_data[col_name]
                    
                    # Apply custom calculator if provided
                    if calculator:
                        team1_val = calculator(team1_val, team1_data)
                        team2_val = calculator(team2_val, team2_data)
                    
                    # Handle null values after calculation
                    if pd.isna(team1_val) or pd.isna(team2_val):
                        continue
                    
                    # Format values appropriately
                    if isinstance(team1_val, (float, np.float64)):
                        # Handle percentage values
                        if col_name.endswith('%') or 'Percentage' in col_name:
                            # Convert decimal to percentage if needed
                            if 0 < team1_val < 1:
                                team1_val_display = f"{team1_val*100:.1f}%"
                                team2_val_display = f"{team2_val*100:.1f}%"
                                # Calculate difference as percentage points
                                diff = (team1_val - team2_val) * 100
                            else:
                                team1_val_display = f"{team1_val:.1f}%"
                                team2_val_display = f"{team2_val:.1f}%"
                                diff = team1_val - team2_val
                        else:
                            # Format regular numbers
                            team1_val_display = f"{team1_val:.1f}" if team1_val % 1 != 0 else f"{int(team1_val)}"
                            team2_val_display = f"{team2_val:.1f}" if team2_val % 1 != 0 else f"{int(team2_val)}"
                            diff = team1_val - team2_val
                    else:
                        # Handle string or other types
                        team1_val_display = str(team1_val)
                        team2_val_display = str(team2_val)
                        diff = 0  # No meaningful difference for non-numeric
                    
                    # Add to comparison data
                    comparison_data.append({
                        'Metric': display_name,
                        team1: team1_val_display,
                        team2: team2_val_display,
                        'Difference': diff,
                        'Favors': team1 if diff > 0 else team2 if diff < 0 else "Tie"
                    })
            
            # Display the data if we have any
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                # Sort metrics alphabetically for consistency
                comparison_df = comparison_df.sort_values('Metric')
                
                # Display the basic comparison table
                st.table(comparison_df[['Metric', team1, team2]])
                
                # Create a horizontal bar chart to show the relative strengths
                if len(comparison_data) > 0:
                    # Filter to numeric metrics where we can calculate meaningful differences
                    numeric_metrics = [data for data in comparison_data if isinstance(data['Difference'], (int, float)) and data['Difference'] != 0]
                    
                    if numeric_metrics:
                        # Convert to DataFrame for plotting
                        plot_df = pd.DataFrame(numeric_metrics)
                        
                        # Sort by absolute difference
                        plot_df['AbsDiff'] = plot_df['Difference'].abs()
                        plot_df = plot_df.sort_values('AbsDiff', ascending=True)
                        
                        # Create figure
                        fig_height = max(4, len(plot_df) * 0.4)
                        fig, ax = plt.subplots(figsize=(10, fig_height))
                        
                        # Create colors based on which team is favored
                        colors = ['royalblue' if row['Favors'] == team1 else 'firebrick' for _, row in plot_df.iterrows()]
                        
                        # Create the horizontal bars
                        bars = ax.barh(plot_df['Metric'], plot_df['Difference'], color=colors)
                        
                        # Add a vertical line at 0
                        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
                        
                        # Add labels for the teams
                        ax.text(-max(plot_df['AbsDiff'])/2, -0.6, team2 + " ←", ha='center', va='top', fontsize=10, fontweight='bold')
                        ax.text(max(plot_df['AbsDiff'])/2, -0.6, "→ " + team1, ha='center', va='top', fontsize=10, fontweight='bold')
                        
                        # Customize appearance
                        ax.set_title(f'Relative Team Strengths - {category}')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        st.pyplot(fig)
            else:
                st.write(f"No comparison data available for {category} metrics")

def display_sim_your_own_bracket(playoff_matchups, team_data, model_data):
    """
    Display an interactive bracket simulation where users can run their own bracket simulations
    and track the results across multiple runs.
    
    Args:
        playoff_matchups: Dictionary with first round playoff matchups
        team_data: DataFrame with team data
        model_data: Dictionary with model data
    """
    st.write("""
    Run your own bracket simulations and track the results. Each time you click "Simulate Bracket",
    we'll simulate one complete playoff tournament from the first round through the Stanley Cup Final.
    """)
    
    # Initialize session state variables for tracking simulation history
    if 'user_sim_count' not in st.session_state:
        st.session_state.user_sim_count = 0
    if 'user_sim_results' not in st.session_state:
        st.session_state.user_sim_results = {}
    if 'user_sim_history' not in st.session_state:
        st.session_state.user_sim_history = []
    
    # Create columns for actions
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Button to run a new simulation
        if st.button("🏆 Simulate Bracket", key="run_sim_button"):
            with st.spinner("Simulating playoff bracket..."):
                # Run a single simulation
                sim_result = simulate_single_bracket(playoff_matchups, team_data, model_data)
                
                if sim_result:
                    # Increment counter
                    st.session_state.user_sim_count += 1
                    
                    # Add to simulation history
                    sim_result['sim_num'] = st.session_state.user_sim_count
                    st.session_state.user_sim_history.append(sim_result)
                    
                    # Update team statistics in aggregated results
                    for team, result in sim_result['results'].items():
                        if team not in st.session_state.user_sim_results:
                            st.session_state.user_sim_results[team] = {
                                'team': team,
                                'team_name': result['team_name'],
                                'round_1': 0,
                                'round_2': 0,
                                'conf_final': 0,
                                'final': 0,
                                'champion': 0,
                                'total_sims': 0
                            }
                        
                        # Update counters
                        st.session_state.user_sim_results[team]['total_sims'] += 1
                        for round_key, advanced in result.items():
                            if round_key in ['round_1', 'round_2', 'conf_final', 'final', 'champion'] and advanced:
                                st.session_state.user_sim_results[team][round_key] += 1
    
    with col2:
        # Show simulation count and clear button
        if st.session_state.user_sim_count > 0:
            st.write(f"**{st.session_state.user_sim_count}** bracket simulations run")
            
            # Add clear button inline
            if st.button("Clear Results", key="clear_results_button"):
                st.session_state.user_sim_count = 0
                st.session_state.user_sim_results = {}
                st.session_state.user_sim_history = []
                st.experimental_rerun()
    
    # Display current simulation results if any
    if st.session_state.user_sim_count > 0:
        # Display the latest simulation result
        latest_sim = st.session_state.user_sim_history[-1]
        
        st.subheader(f"Simulation #{latest_sim['sim_num']} Results")
        
        # Create a visual bracket representation
        display_bracket_visual(latest_sim)
        
        # Display aggregated results table
        st.subheader("Aggregated Results")
        
        # Convert results dict to DataFrame for display
        results_df = pd.DataFrame(list(st.session_state.user_sim_results.values()))
        
        # Calculate percentages
        for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
            if col in results_df.columns:
                results_df[f"{col}_pct"] = (results_df[col] / results_df['total_sims'] * 100).round(1)
        
        # Sort by championship percentage
        results_df = results_df.sort_values('champion_pct', ascending=False)
        
        # Reformat for display
        display_df = results_df.copy()
        display_df['Round 1'] = display_df['round_1_pct'].astype(str) + '%'
        display_df['Round 2'] = display_df['round_2_pct'].astype(str) + '%'
        display_df['Conf Final'] = display_df['conf_final_pct'].astype(str) + '%'
        display_df['Finals'] = display_df['final_pct'].astype(str) + '%'
        display_df['Champion'] = display_df['champion_pct'].astype(str) + '%'
        
        # Display table with team results
        st.table(display_df[['team_name', 'Round 1', 'Round 2', 'Conf Final', 'Finals', 'Champion']])
        
        # Show simulation history
        with st.expander("Simulation History"):
            # Create tabs for different simulations
            num_sims_to_show = min(5, len(st.session_state.user_sim_history))
            if num_sims_to_show > 0:
                sim_tabs = st.tabs([f"Sim #{sim['sim_num']}" for sim in st.session_state.user_sim_history[-num_sims_to_show:]])
                
                # Display each simulation
                for i, tab in enumerate(sim_tabs):
                    with tab:
                        sim_idx = len(st.session_state.user_sim_history) - num_sims_to_show + i
                        sim = st.session_state.user_sim_history[sim_idx]
                        st.write(f"**Playoff Path:**")
                        
                        # Show the bracket progression
                        for round_name, matchups in sim['bracket'].items():
                            st.write(f"**{round_name}:**")
                            for matchup in matchups:
                                st.write(f"- {matchup['winner']} defeated {matchup['loser']}")
                        
                        st.write(f"**Champion:** {sim['champion']}")
    else:
        st.info("Click the 'Simulate Bracket' button to run a playoff simulation.")

def display_bracket_visual(sim_result):
    """
    Display a visual representation of a simulated playoff bracket
    
    Args:
        sim_result: Dictionary with simulation results
    """
    # Display the champion prominently
    champion = sim_result['champion']
    st.success(f"🏆 **Stanley Cup Champion: {champion}**")
    
    # Create a visual representation of the playoff bracket
    # Use columns to organize the rounds
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # First Round
    with col1:
        st.markdown("### First Round")
        for matchup in sim_result['bracket']['First Round']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Second Round
    with col2:
        st.markdown("### Second Round")
        for matchup in sim_result['bracket']['Second Round']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Conference Finals
    with col3:
        st.markdown("### Conference Finals")
        for matchup in sim_result['bracket']['Conference Finals']:
            with st.container():
                st.markdown(f"**{matchup['winner']}** ✓")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")
    
    # Stanley Cup Final
    with col4:
        st.markdown("### Stanley Cup Final")
        if sim_result['bracket']['Stanley Cup Final']:
            matchup = sim_result['bracket']['Stanley Cup Final'][0]
            with st.container():
                st.markdown(f"**{matchup['winner']}** 🏆")
                st.markdown(f"{matchup['loser']}")
                st.markdown("---")

def display_about_page(model_data):
    """Display enhanced about page with model information and app details"""
    st.subheader("About the NHL Playoff Predictor")
    
    st.write("""
    This application predicts NHL playoff outcomes by leveraging machine learning models trained on historical playoff
    data combined with current season team statistics and advanced analytics metrics.
    
    The prediction system uses an ensemble approach that combines:
    1. A logistic regression model capturing linear relationships between features
    2. An XGBoost model that identifies complex, non-linear patterns in the data
    3. Home ice advantage adjustment based on historical playoff performance
    """)
    
    # Model information section
    st.subheader("Model Information")
    
    # Create columns to display model info side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Logistic Regression Model**")
        if model_data and 'lr' in model_data:
            lr_model = model_data['lr']
            st.write(f"Features: {len(lr_model.get('features', []))} input variables")
            
            top_features = []
            if 'coefficients' in lr_model:
                top_features = [f"{feat}: {coef:.4f}" for feat, coef in lr_model['coefficients'][:5]]
            
            if top_features:
                st.write("Top 5 features by importance:")
                for feat in top_features:
                    st.write(f"- {feat}")
            else:
                st.write("Model feature importance not available")
        else:
            st.write("Logistic Regression model not loaded")
    
    with col2:
        st.write("**XGBoost Model**")
        if model_data and 'xgb' in model_data:
            xgb_model = model_data['xgb']
            st.write(f"Features: {len(xgb_model.get('features', []))} input variables")
            
            top_features = []
            if 'importance' in xgb_model:
                top_features = [f"{feat}: {imp:.4f}" for feat, imp in xgb_model['importance'][:5]]
            
            if top_features:
                st.write("Top 5 features by importance:")
                for feat in top_features:
                    st.write(f"- {feat}")
            else:
                st.write("Model feature importance not available")
        else:
            st.write("XGBoost model not loaded")
    
    # Key features section
    st.subheader("Key Features Used")
    
    feature_categories = {
        "Team Performance": [
            "Goal differential per game",
            "Points percentage",
            "Regulation wins (home and road)",
            "Win streaks and performance trends"
        ],
        "Special Teams": [
            "Power play percentage (relative to league average)",
            "Penalty kill percentage (relative to league average)",
            "Special teams composite score",
            "Short-handed goals"
        ],
        "Advanced Metrics": [
            "Expected goals percentage (5v5)",
            "Corsi and Fenwick percentages",
            "High-danger chances for/against",
            "Possession-adjusted defensive metrics"
        ],
        "Playoff Experience": [
            "Recent playoff performance (weighted)",
            "Previous playoff rounds won",
            "Experience in elimination games"
        ]
    }
    
    # Create tabs for feature categories
    category_tabs = st.tabs(list(feature_categories.keys()))
    
    # Fill each tab with corresponding features
    for tab, category in zip(category_tabs, feature_categories.keys()):
        with tab:
            for feature in feature_categories[category]:
                st.write(f"• {feature}")
    
    # Data sources section
    st.subheader("Data Sources")
    st.write("""
    - Team statistics: NHL API (real-time data)
    - Advanced metrics: MoneyPuck (updated daily)
    - Historical playoff data (2005-present): Used for model training and validation
    """)
    
    # Prediction methodology
    st.subheader("Prediction Methodology")
    st.write("""
    For each matchup, the application:
    1. **Data Collection**: Gathers current team statistics and advanced metrics
    2. **Feature Engineering**: Calculates derived features (e.g., relative metrics, adjusted rates)
    3. **Model Prediction**: Generates independent predictions from both the logistic regression and XGBoost models
    4. **Ensemble Prediction**: Combines model predictions with appropriate weighting
    5. **Home Ice Adjustment**: Applies adjustment based on historical home ice advantage
    6. **Monte Carlo Simulation**: Simulates the series thousands of times to account for randomness
    
    The predicted win probability represents the percentage of simulations in which a team wins the series.
    """)
    
    # Limitations and caveats
    st.expander("Limitations and Caveats").write("""
    - Predictions are based on statistical patterns and cannot account for all factors (e.g., injuries, coaching changes)
    - The model assumes that historical patterns continue to be relevant for current matchups
    - Advanced metrics may have limited predictive power due to the inherent randomness in playoff hockey
    - Home ice advantage is applied as a uniform adjustment based on historical averages
    """)
    
    # App information
    st.subheader("Application Information")
    st.write("""
    - Data refreshed: Once per day (cached for 24 hours)
    - Default simulation count: 10,000 iterations per matchup
    - Model ensemble: Logistic Regression + XGBoost with equal weighting
    - Developed with: Streamlit, pandas, numpy, scikit-learn, XGBoost, matplotlib
    """)
    
    # Version and credits footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
    <p>NHL Playoff Predictor v1.0.0</p>
    <p>© 2023-2024 | All NHL team names, logos and data are property of the NHL and their respective teams</p>
    </div>
    """, unsafe_allow_html=True)

# Update the main function to properly include all 5 pages with the specified functionality
def main():
    # App title and description
    st.title("NHL Playoff Predictor")
    st.write("Predict playoff outcomes based on team statistics and advanced metrics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", [
        "First Round Matchups", 
        "Full Simulation Results", 
        "Head-to-Head Comparison", 
        "Sim Your Own Bracket",
        "About"
    ])
    
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
        
        # Combine standings and stats data
        if not standings_df.empty and not stats_df.empty:
            # Make sure season columns are strings
            standings_df['season'] = standings_df['season'].astype(str)
            stats_df['season'] = stats_df['season'].astype(str)
            
            # First merge standings and stats data
            team_data = pd.merge(standings_df, stats_df, 
                                 left_on=['season', 'teamName'], 
                                 right_on=['season', 'teamName'], 
                                 how='left')
            
            # Merge with advanced stats if available
            if not advanced_stats_df.empty:
                # Make sure season columns have the same format
                if 'season' in advanced_stats_df.columns:
                    # Convert season from "2024" to "20242025" format if needed
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str).apply(
                        lambda x: f"{x}{int(x)+1}" if len(x) == 4 and x.isdigit() else x
                    )
                    advanced_stats_df['season'] = advanced_stats_df['season'].astype(str)
                
                # Perform the merge with the correct columns
                team_data = pd.merge(team_data, advanced_stats_df, 
                                     left_on=['season', 'teamAbbrev'],
                                     right_on=['season', 'team'],
                                     how='left')
            
            # Engineer features for prediction
            team_data = engineer_features(team_data)
            
            # Add playoff history metrics
            team_data = add_playoff_history_metrics(team_data)
        else:
            team_data = pd.DataFrame()
    
    # Load models
    model_data = load_models()
    
    # Determine playoff teams and create matchups
    playoff_matchups = determine_playoff_teams(standings_df)
    
    # Add model info to sidebar
    if model_data:
        st.sidebar.title("Model Information")
        st.sidebar.write(f"Model mode: {model_data.get('mode', 'default')}")
        st.sidebar.write(f"Home ice advantage: {model_data.get('home_ice_boost', 0.039)*100:.1f}%")
    
    # First Round Matchups page
    if page == "First Round Matchups":
        st.header("First Round Playoff Matchups")
        display_playoff_bracket(playoff_matchups, team_data, model_data)
    
    # Full Simulation Results page
    elif page == "Full Simulation Results":
        st.header("Full Playoff Simulation Results")
        
        # Simulation options in sidebar
        st.sidebar.subheader("Simulation Settings")
        n_simulations = st.sidebar.slider("Number of simulations", 1000, 20000, 10000, 1000)
        
        # Run the simulation once per day (cached)
        if playoff_matchups and not team_data.empty:
            # Check if we have cached simulation results
            sim_results_key = f"sim_results_{season_str}"
            if sim_results_key in st.session_state and st.session_state[sim_results_key].get('n_simulations') == n_simulations:
                bracket_results = st.session_state[sim_results_key]['results']
                st.success(f"Using cached simulation results ({n_simulations} simulations)")
            else:
                # Run new simulation
                st.info(f"Running new simulation with {n_simulations} iterations...")
                bracket_results = run_full_playoff_simulation(playoff_matchups, team_data, model_data, n_simulations)
                
                # Cache the results in session state
                st.session_state[sim_results_key] = {
                    'results': bracket_results,
                    'n_simulations': n_simulations,
                    'timestamp': datetime.now()
                }
                
                # Save in another variable for reference in other pages
                st.session_state['recent_simulations'] = bracket_results
                st.session_state['sim_count'] = n_simulations
            
            # Display comprehensive simulation results
            display_simulation_results(bracket_results, n_simulations)
        else:
            st.error("Playoff matchups or team data not available for simulation")
    
    # Head-to-Head Comparison page
    elif page == "Head-to-Head Comparison":
        st.header("Team Head-to-Head Comparison")
        
        st.write("""
        Compare any two teams in a hypothetical playoff matchup. The first team selected is considered 
        the higher seed with home ice advantage.
        """)
        
        # Team selection
        st.subheader("Select Teams to Compare")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Home Team", sorted(team_data['teamName'].unique()), key="team1")
        
        with col2:
            # Filter to exclude team1
            available_teams = sorted([team for team in team_data['teamName'].unique() if team != team1])
            team2 = st.selectbox("Select Road Team", available_teams, key="team2")
        
        # Display team comparison if two different teams are selected
        if team1 != team2:
            team1_data = team_data[team_data['teamName'] == team1].iloc[0]
            team2_data = team_data[team_data['teamName'] == team2].iloc[0]
            
            # Display team logos if available
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(team1)
                if 'teamLogo' in team1_data and team1_data['teamLogo']:
                    logo = load_team_logo(team1_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            with col2:
                st.subheader(team2)
                if 'teamLogo' in team2_data and team2_data['teamLogo']:
                    logo = load_team_logo(team2_data['teamLogo'])
                    if logo:
                        st.image(logo, width=150)
            
            # Display head-to-head prediction
            display_head_to_head(team1, team2, team1_data, team2_data, team_data, model_data)
        else:
            st.warning("Please select two different teams")
    
    # Sim Your Own Bracket page        
    elif page == "Sim Your Own Bracket":
        st.header("Simulate Your Own Playoff Bracket")
        display_sim_your_own_bracket(playoff_matchups, team_data, model_data)
    
    # About page        
    elif page == "About":
        st.header("About the NHL Playoff Predictor")
        display_about_page(model_data)

# Run the app
if __name__ == "__main__":
    main()




