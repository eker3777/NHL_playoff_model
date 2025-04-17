import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO
from datetime import datetime, timedelta
import time
import json
try:
    # Try importing the module (package is installed as nhl-api-py, but imported as nhlpy)
    import nhlpy
    from nhlpy.nhl_client import NHLClient
except ImportError:
    st.write("Installing NHL API package...")
    try:
        # Install the package (pip install name is nhl-api-py)
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nhl-api-py"])
        # Import after installation (package is imported as nhlpy)
        import nhlpy
        from nhlpy.nhl_client import NHLClient
    except Exception as e:
        st.error(f"Error installing or importing NHL API package: {str(e)}")
        NHLClient = None

# Initialize NHL client
try:
    client = NHLClient()
except Exception as e:
    st.error(f"Could not initialize NHL client: {str(e)}")
    client = None

# Function to check if data needs to be refreshed (once per day, early morning)
def should_refresh_data():
    if 'last_data_refresh' not in st.session_state or st.session_state.last_data_refresh is None:
        return True
    
    # Get current time in UTC
    now = datetime.utcnow()
    
    # Check if it's been 24 hours since last refresh
    if now - st.session_state.last_data_refresh > timedelta(hours=24):
        return True
    
    # If it's a new day and before 9 AM UTC (early morning), refresh the data
    if now.date() > st.session_state.last_data_refresh.date() and now.hour < 9:
        return True
    
    return False

# Helper function for API requests with retries
def make_request_with_retry(url, headers=None, retries=3, timeout=10):
    """Make HTTP request with retry logic"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                st.warning(f"Request timed out. Retrying ({attempt+1}/{retries})...")
                time.sleep(2)  # Wait before retry
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                st.warning(f"Request failed: {str(e)}. Retrying ({attempt+1}/{retries})...")
                time.sleep(2)  # Wait before retry
            else:
                raise
    
    raise requests.exceptions.RequestException("Max retries exceeded")

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_standings_data():
    """Get current NHL standings data with improved error handling and fallback options"""
    try:
        # Try method 1: Using NHLClient from nhlpy
        if client is not None:
            try:
                standings_data = client.standings.get_standings()
                if standings_data:
                    return standings_data
                
            except Exception as client_error:
                st.warning(f"Primary standings source failed: {str(client_error)}")
                # Continue to alternative methods
        
        # Try method 2: Direct API request to NHL.com
        try:
            url = "https://api-web.nhle.com/v1/standings/now"
            response = make_request_with_retry(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as direct_api_error:
            st.warning(f"Direct API request failed: {str(direct_api_error)}")
        
        # Try method 3: Alternative API endpoint
        try:
            url = "https://api-web.nhle.com/v1/standings/current"
            response = make_request_with_retry(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as alt_api_error:
            st.warning(f"Alternative API request failed: {str(alt_api_error)}")
            
        # Method 4: Last resort - load from cached file
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
        season_str = f"{current_season}{current_season+1}"
        standings_path = os.path.join(data_folder, f"standings_{season_str}.csv")
        
        if os.path.exists(standings_path):
            st.info("Using cached standings data from file")
            standings_df = pd.read_csv(standings_path)
            # Convert DataFrame back to dictionary format for consistency
            return {"standings": standings_df.to_dict(orient='records')}
        
        raise Exception("All methods to fetch standings data failed")
    except Exception as e:
        st.error(f"Error fetching standings data: {str(e)}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_team_stats_data(start_season=None, end_season=None):
    """Get team summary stats data directly from NHL API with improved error handling"""
    try:
        # Use current season if not specified
        if start_season is None:
            current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
            start_season = current_season
            
        if end_season is None:
            end_season = start_season
            
        # Construct the URL with season parameters
        url = f"https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId>={start_season}{start_season+1} and seasonId<={end_season}{end_season+1}"
        
        # Make direct API request with retry logic
        response = make_request_with_retry(url, timeout=15)
        stats_data = response.json().get('data', [])
        
        if not stats_data:
            # Try alternative endpoint if main one returns empty
            alt_url = f"https://api.nhle.com/stats/rest/en/team?isAggregate=true&reportType=basic&isGame=false&reportName=teamsummary&cayenneExp=gameTypeId=2 and seasonId>={start_season}{start_season+1} and seasonId<={end_season}{end_season+1}"
            alt_response = make_request_with_retry(alt_url, timeout=15)
            stats_data = alt_response.json().get('data', [])
        
        # Process the data to standardize season format
        if stats_data and isinstance(stats_data, list):
            for team in stats_data:
                if 'seasonId' in team:
                    # Convert seasonId from number to string if needed
                    if isinstance(team['seasonId'], int):
                        team['seasonId'] = str(team['seasonId'])
        
        return stats_data
    except Exception as e:
        st.error(f"Error fetching team stats data: {str(e)}")
        # Try to load from cache/file as fallback
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
        season_str = f"{current_season}{current_season+1}"
        stats_path = os.path.join(data_folder, f"team_stats_{season_str}.json")
        
        if os.path.exists(stats_path):
            st.info("Using cached team stats data from file")
            try:
                with open(stats_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def process_standings_data(standings_data, season_str):
    """Process NHL standings data into a DataFrame with improved error handling"""
    all_standings = []
    
    if standings_data is None:
        st.error("No standings data provided")
        return pd.DataFrame()
        
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
            # Try to handle any other dictionary format by looking for team data
            potential_team_lists = [v for k, v in standings_data.items() if isinstance(v, list) and len(v) > 0]
            if potential_team_lists:
                # Use the longest list as likely candidates for team records
                teams_list = max(potential_team_lists, key=len)
                
                st.warning(f"Using alternative standings data format: {list(standings_data.keys())}")
                
                for team_record in teams_list:
                    if not isinstance(team_record, dict):
                        continue
                        
                    # Create basic team record
                    team_data = {'season': season_str}
                    
                    # Extract division and conference if available
                    for div_field in ['division', 'divisionName']:
                        if div_field in team_record:
                            team_data['division'] = team_record[div_field]
                            break
                    
                    for conf_field in ['conference', 'conferenceName']:
                        if conf_field in team_record:
                            team_data['conference'] = team_record[conf_field]
                            break
                    
                    # Default values if not found
                    if 'division' not in team_data:
                        team_data['division'] = "Unknown"
                    if 'conference' not in team_data:
                        team_data['conference'] = "Unknown"
                    
                    # Process team basic info
                    for id_field in ['teamId', 'id']:
                        if id_field in team_record:
                            team_data['teamId'] = team_record[id_field]
                            break
                    
                    # Try to extract team name
                    if 'teamName' in team_record:
                        if isinstance(team_record['teamName'], dict):
                            team_data['teamName'] = team_record['teamName'].get('default', '')
                        else:
                            team_data['teamName'] = team_record['teamName']
                    elif 'name' in team_record:
                        team_data['teamName'] = team_record['name']
                    
                    # Try to extract abbreviation
                    if 'teamAbbrev' in team_record:
                        if isinstance(team_record['teamAbbrev'], dict):
                            team_data['teamAbbrev'] = team_record['teamAbbrev'].get('default', '')
                        else:
                            team_data['teamAbbrev'] = team_record['teamAbbrev']
                    elif 'abbrev' in team_record:
                        team_data['teamAbbrev'] = team_record['abbrev']
                    
                    # Add all other fields
                    for key, value in team_record.items():
                        if key not in team_data:
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    team_data[f"{key}_{sub_key}"] = sub_value
                            else:
                                team_data[key] = value
                    
                    all_standings.append(team_data)
            else:
                st.warning(f"Unknown standings data format: {list(standings_data.keys())}")
    elif isinstance(standings_data, list) and len(standings_data) > 0:
        # Try to handle list format directly
        st.warning("Received list format for standings data, attempting to process")
        for team_record in standings_data:
            if not isinstance(team_record, dict):
                continue
                
            # Create basic team record with season
            team_data = {'season': season_str}
            
            # Try to extract key team information
            if 'teamName' in team_record:
                team_data['teamName'] = team_record['teamName']
            elif 'name' in team_record:
                team_data['teamName'] = team_record['name']
                
            # Add all other fields
            for key, value in team_record.items():
                if key not in team_data:
                    team_data[key] = value
            
            all_standings.append(team_data)
    else:
        st.warning("Non-dictionary standings data response")
    
    # Convert to DataFrame
    if all_standings:
        df = pd.DataFrame(all_standings)
        
        # Ensure season is always a string type
        if 'season' in df.columns:
            df['season'] = df['season'].astype(str)
        
        # Add pointPctg if missing
        if 'pointPctg' not in df.columns and 'points' in df.columns and 'gamesPlayed' in df.columns:
            df['pointPctg'] = df['points'] / (df['gamesPlayed'] * 2)
        
        # Save a copy of the raw data for debugging
        try:
            data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(data_folder, exist_ok=True)
            df.to_csv(os.path.join(data_folder, f"standings_raw_{season_str}.csv"), index=False)
        except Exception as save_error:
            st.warning(f"Error saving raw standings data: {str(save_error)}")
            
        return df
    return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
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
            # Ensure season is string type
            if 'seasonId' in stats_df.columns:
                stats_df['seasonId'] = stats_df['seasonId'].astype(str)
            
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
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_advanced_stats_data(data_folder, season, situation='5on5'):
    """Get advanced stats from MoneyPuck"""
    # Construct the MoneyPuck URL for the current season
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
    csv_path = os.path.join(data_folder, f"moneypuck_regular_{season}.csv")
    
    try:
        # First try to load from local cache if it exists
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    return process_advanced_stats(df, f"{season}{season+1}", situation)
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
            
            return process_advanced_stats(df, f"{season}{season+1}", situation)
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
        'team', 'season', 'situation', 'games_played', 'pointPctg',
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
                    print(f"Converting percentage column: {col}")
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
    
    # Log which critical features are available
    critical_features = [
        'PP%_rel', 'PK%_rel', 'FO%','playoff_performance_score',
        'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
        'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
        'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
        'adjGoalsScoredAboveX/60'
    ]
    
    available_features = [f for f in critical_features if f in df.columns]
    print(f"Available critical features: {len(available_features)}/{len(critical_features)}")
    print(f"Missing features: {[f for f in critical_features if f not in available_features]}")
    
    # Calculate additional metrics from the data
    if all(col in df.columns for col in ['goalDifferential', 'gamesPlayed']):
        df['goalDiff/G'] = df['goalDifferential'] / df['gamesPlayed'].clip(lower=1)
    else:
        print("Cannot calculate goalDiff/G - missing required columns")
    
    if all(col in df.columns for col in ['homeRegulationWins', 'gamesPlayed']):
        df['homeRegulationWin%'] = df['homeRegulationWins'] / df['gamesPlayed'].clip(lower=1)
    else:
        print("Cannot calculate homeRegulationWin% - missing required columns")
    
    if all(col in df.columns for col in ['roadRegulationWins', 'gamesPlayed']):
        df['roadRegulationWin%'] = df['roadRegulationWins'] / df['gamesPlayed'].clip(lower=1)
    else:
        print("Cannot calculate roadRegulationWin% - missing required columns")
    
    # Advanced metrics - add checks to ensure columns exist before calculating
    if all(col in df.columns for col in ['flurryScoreVenueAdjustedxGoalsAgainst', 'goalsAgainst', 'iceTime']):
        df['adjGoalsSavedAboveX/60'] = (df['flurryScoreVenueAdjustedxGoalsAgainst'] - df['goalsAgainst']) / df['iceTime'].clip(lower=0.01) * 60
        # Create per-82 column directly 
        df['goalsSavedAboveExpectedPer82'] = df['adjGoalsSavedAboveX/60'] * 82
    else:
        print("Cannot calculate adjGoalsSavedAboveX/60 - missing required columns")
        missing_cols = [col for col in ['flurryScoreVenueAdjustedxGoalsAgainst', 'goalsAgainst', 'iceTime'] if col not in df.columns]
        print(f"Missing columns: {missing_cols}")
    
    if all(col in df.columns for col in ['goalsFor', 'flurryScoreVenueAdjustedxGoalsFor', 'iceTime']):
        df['adjGoalsScoredAboveX/60'] = (df['goalsFor'] - df['flurryScoreVenueAdjustedxGoalsFor']) / df['iceTime'].clip(lower=0.01) * 60
        # Create per-82 column directly
        df['goalsScoredAboveExpectedPer82'] = df['adjGoalsScoredAboveX/60'] * 82
    else:
        print("Cannot calculate adjGoalsScoredAboveX/60 - missing required columns")
        missing_cols = [col for col in ['goalsFor', 'flurryScoreVenueAdjustedxGoalsFor', 'iceTime'] if col not in df.columns]
        print(f"Missing columns: {missing_cols}")
    
    # Calculate possession-normalized metrics if columns exist
    cols_for_hits = ['hitsFor', 'hitsAgainst']
    if all(col in df.columns for col in cols_for_hits):
        df['hitsPctg'] = df['hitsFor'] / (df['hitsAgainst'] + df['hitsFor']).clip(lower=0.01)
    else:
        print("Cannot calculate hitsPctg - missing required columns")
    
    cols_for_takeaways = ['takeawaysFor', 'takeawaysAgainst']
    if all(col in df.columns for col in cols_for_takeaways):
        df['takeawaysPctg'] = df['takeawaysFor'] / (df['takeawaysAgainst'] + df['takeawaysFor']).clip(lower=0.01)
    else:
        print("Cannot calculate takeawaysPctg - missing required columns")
    
    cols_for_giveaways = ['giveawaysFor', 'giveawaysAgainst']
    if all(col in df.columns for col in cols_for_giveaways):
        df['giveawaysPctg'] = df['giveawaysFor'] / (df['giveawaysAgainst'] + df['giveawaysFor']).clip(lower=0.01)
    else:
        print("Cannot calculate giveawaysPctg - missing required columns")
    
    cols_for_dzone = ['dZoneGiveawaysFor', 'dZoneGiveawaysAgainst']
    if all(col in df.columns for col in cols_for_dzone):
        df['dZoneGiveawaysPctg'] = df['dZoneGiveawaysFor'] / (df['dZoneGiveawaysAgainst'] + df['dZoneGiveawaysFor']).clip(lower=0.01)
    else:
        print("Cannot calculate dZoneGiveawaysPctg - missing required columns")
    
    # Apply possession adjustment if corsiPercentage exists
    if 'corsiPercentage' in df.columns and 'hitsPctg' in df.columns:
        # Ensure corsiPercentage is properly clipped to prevent division by zero
        df['possAdjHitsPctg'] = df['hitsPctg'] * (0.5 / (1 - df['corsiPercentage'].clip(0.01, 0.99)))
    else:
        print("Cannot calculate possAdjHitsPctg - missing required columns")
    
    if 'corsiPercentage' in df.columns and 'takeawaysPctg' in df.columns:
        df['possAdjTakeawaysPctg'] = df['takeawaysPctg'] * (0.5 / (1 - df['corsiPercentage'].clip(0.01, 0.99)))
    else:
        print("Cannot calculate possAdjTakeawaysPctg - missing required columns")
    
    if 'corsiPercentage' in df.columns and 'giveawaysPctg' in df.columns:
        df['possAdjGiveawaysPctg'] = df['giveawaysPctg'] * (0.5 / df['corsiPercentage'].clip(0.01, 0.99))
    else:
        print("Cannot calculate possAdjGiveawaysPctg - missing required columns")
    
    if 'corsiPercentage' in df.columns and 'dZoneGiveawaysPctg' in df.columns:
        df['possAdjdZoneGiveawaysPctg'] = df['dZoneGiveawaysPctg'] * (0.5 / df['corsiPercentage'].clip(0.01, 0.99))
    else:
        print("Cannot calculate possAdjdZoneGiveawaysPctg - missing required columns")
    
    if all(col in df.columns for col in ['possAdjGiveawaysPctg', 'possAdjdZoneGiveawaysPctg']):
        df['possTypeAdjGiveawaysPctg'] = df['possAdjGiveawaysPctg'] * 1/3 + df['possAdjdZoneGiveawaysPctg'] * 2/3
    else:
        print("Cannot calculate possTypeAdjGiveawaysPctg - missing required columns")
    
    if all(col in df.columns for col in ['reboundxGoalsFor', 'reboundxGoalsAgainst']):
        df['reboundxGoalsPctg'] = df['reboundxGoalsFor'] / (df['reboundxGoalsFor'] + df['reboundxGoalsAgainst']).clip(lower=0.01)
    else:
        print("Cannot calculate reboundxGoalsPctg - missing required columns")
    
    # Calculate special teams metrics if they exist
    if all(col in df.columns for col in ['PP%', 'PK%']):
        # Calculate league averages
        league_avg_pp = df['PP%'].mean()
        league_avg_pk = df['PK%'].mean()
        
        # Print values for debugging
        print(f"League average PP%: {league_avg_pp:.4f}, PK%: {league_avg_pk:.4f}")
        
        # Check scaling of PP% and PK% - we need to ensure these are in decimal format (0-1)
        if df['PP%'].max() > 1:
            pp_divisor = 100.0
            print(f"PP% is in percentage format (0-100), max value: {df['PP%'].max():.2f}")
        else:
            pp_divisor = 1.0
            print("PP% is already in decimal format (0-1)")
            
        if df['PK%'].max() > 1:
            pk_divisor = 100.0
            print(f"PK% is in percentage format (0-100), max value: {df['PK%'].max():.2f}")
        else:
            pk_divisor = 1.0
            print("PK% is already in decimal format (0-1)")
        
        # Calculate relative metrics - ensuring they are in decimal format
        # The _rel metrics should be on a scale of about ±0.05 (5%)
        df['PP%_rel'] = (df['PP%'] / pp_divisor) - (league_avg_pp / pp_divisor)
        df['PK%_rel'] = (df['PK%'] / pk_divisor) - (league_avg_pk / pk_divisor)
        
        # Log values for verification
        print(f"PP%_rel range: {df['PP%_rel'].min():.4f} to {df['PP%_rel'].max():.4f}")
        print(f"PK%_rel range: {df['PK%_rel'].min():.4f} to {df['PK%_rel'].max():.4f}")
        
        # Create composite special teams metric
        df['special_teams_composite'] = df['PP%_rel'] + df['PK%_rel']
    else:
        print("Cannot calculate special teams metrics - missing required columns")
    
    # Fill NaN values for critical features with reasonable defaults
    for col in critical_features:
        if col in df.columns and df[col].isna().any():
            null_count = df[col].isna().sum()
            print(f"Filling {null_count} NaN values in {col} with column mean")
            df[col] = df[col].fillna(df[col].mean())
    
    # Final verification of critical features
    for col in critical_features:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                print(f"WARNING: {col} still has {null_count} NaN values after processing")
    
    return df


# Add function to incorporate playoff history metrics
def add_playoff_history_metrics(team_data, data_folder):
    """Add playoff history metrics to the combined team data."""
    try:
        # Load the playoff history data with proper path handling
        playoff_history_path = os.path.join(data_folder, "nhl_playoff_wins_2005_present.csv")
        
        # If the file doesn't exist, create a dummy version for testing
        if not os.path.exists(playoff_history_path):
            st.info(f"Playoff history file not found - creating a placeholder file")
            # Create a simple placeholder file
            dummy_data = {
                'season': [f"{datetime.now().year-1}{datetime.now().year}" for _ in range(16)],
                'team': team_data['teamAbbrev'].tolist()[:16] if len(team_data) >= 16 else team_data['teamAbbrev'].tolist() + ['TEAM'+str(i) for i in range(16-len(team_data))],
                'wins': [np.random.randint(0, 16) for _ in range(16)],
                'rounds_won': [np.random.randint(0, 4) for _ in range(16)]
            }
            dummy_df = pd.DataFrame(dummy_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(playoff_history_path), exist_ok=True)
            dummy_df.to_csv(playoff_history_path, index=False)
        
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
    """Calculate playoff history metrics for all teams across multiple seasons"""
    weights = [0.6, 0.4]  # Weights for previous seasons
    history_data = []
    
    # Create a mapping between team and teamAbbrev from team_data
    team_abbrev_map = {}
    if 'teamAbbrev' in team_data.columns and 'teamName' in team_data.columns:
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
            team_name = team_row.get('teamName', '')
            
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
    """Determine which teams make the playoffs based on NHL rules."""
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
        division_qualifiers_df = pd.concat(division_qualifiers) if division_qualifiers else pd.DataFrame()
        
        # Combine and rank wildcard candidates
        wildcard_df = pd.DataFrame()
        if wildcard_candidates:
            wildcard_df = pd.concat(wildcard_candidates)
            
            # Sort by points and tiebreakers
            sort_columns = ['points']
            for tiebreaker in tiebreakers:
                if tiebreaker in wildcard_df.columns:
                    sort_columns.append(tiebreaker)
            
            wildcard_df = wildcard_df.sort_values(by=sort_columns, ascending=False)
            
            # Top 2 wildcards make the playoffs
            wildcards = wildcard_df.head(2).copy()
            wildcards['seed_type'] = 'wildcard'
            wildcards['wildcard_rank'] = range(1, 3)
        else:
            wildcards = pd.DataFrame()
        
        # Combine all playoff teams for this conference
        conference_playoff_teams = pd.concat([division_qualifiers_df, wildcards]) if not wildcards.empty else division_qualifiers_df
        
        if len(conference_playoff_teams) < 8:
            st.warning(f"Not enough teams for playoffs in {conference} conference. Found {len(conference_playoff_teams)} teams.")
            continue
        
        # Create matchups based on NHL playoff format
        matchups = {}
        
        # Group division winners
        div_winners = {}
        for division in divisions:
            div_winner = division_qualifiers_df[
                (division_qualifiers_df['division'] == division) & 
                (division_qualifiers_df['division_rank'] == 1)
            ]
            if not div_winner.empty:
                div_winners[division] = div_winner.iloc[0]
        
        # Sort division winners by points
        if div_winners:
            sorted_div_winners = []
            for div, winner in div_winners.items():
                sorted_div_winners.append((div, winner))
            
            # Sort by points and other criteria
            sorted_div_winners.sort(key=lambda x: (
                x[1].get('points', 0), 
                x[1].get('regulationWins', 0), 
                x[1].get('row', 0), 
                x[1].get('wins', 0)
            ), reverse=True)
            
            sorted_divisions = [div for div, _ in sorted_div_winners]
            
            # Create matchups - division winners vs wildcards
            if len(sorted_divisions) >= 2 and not wildcards.empty:
                if len(wildcards) >= 2:
                    # 1st division plays 2nd wildcard
                    top_div = sorted_divisions[0]
                    matchups[f"{top_div}_WC2"] = {
                        'top_seed': div_winners[top_div].to_dict(),
                        'bottom_seed': wildcards[wildcards['wildcard_rank'] == 2].iloc[0].to_dict()
                    }
                    
                    # 2nd division plays 1st wildcard
                    second_div = sorted_divisions[1]
                    matchups[f"{second_div}_WC1"] = {
                        'top_seed': div_winners[second_div].to_dict(),
                        'bottom_seed': wildcards[wildcards['wildcard_rank'] == 1].iloc[0].to_dict()
                    }
                elif len(wildcards) == 1:
                    # Only one wildcard - match with top division
                    top_div = sorted_divisions[0]
                    matchups[f"{top_div}_WC1"] = {
                        'top_seed': div_winners[top_div].to_dict(),
                        'bottom_seed': wildcards.iloc[0].to_dict()
                    }
        
        # Match 2nd and 3rd place teams in each division
        for division in divisions:
            div_2nd = division_qualifiers_df[
                (division_qualifiers_df['division'] == division) & 
                (division_qualifiers_df['division_rank'] == 2)
            ]
            div_3rd = division_qualifiers_df[
                (division_qualifiers_df['division'] == division) & 
                (division_qualifiers_df['division_rank'] == 3)
            ]
            
            if not div_2nd.empty and not div_3rd.empty:
                matchups[f"{division}_2_3"] = {
                    'top_seed': div_2nd.iloc[0].to_dict(),
                    'bottom_seed': div_3rd.iloc[0].to_dict()
                }
        
        playoff_matchups[conference] = matchups
    
    return playoff_matchups

def create_matchup_data(top_seed, bottom_seed, team_data):
    """Create matchup data for model input"""
    # Create a single row DataFrame for this matchup
    matchup_data = {}
    
    # Base matchup information - set defaults in case keys are missing
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    matchup_data['season'] = current_season
    matchup_data['round'] = 1
    matchup_data['round_name'] = 'First Round'
    matchup_data['series_letter'] = 'TBD'
    
    # Use get() to safely access dictionary keys with defaults
    matchup_data['top_seed_abbrev'] = top_seed.get('teamAbbrev', '')
    matchup_data['bottom_seed_abbrev'] = bottom_seed.get('teamAbbrev', '')
    matchup_data['top_seed_rank'] = top_seed.get('division_rank', top_seed.get('wildcard_rank', 0))
    matchup_data['bottom_seed_rank'] = bottom_seed.get('division_rank', bottom_seed.get('wildcard_rank', 0))
    matchup_data['top_seed_wins'] = 0
    matchup_data['bottom_seed_wins'] = 0
    
    # Get team data for each team - use teamAbbrev to filter
    if 'teamAbbrev' in team_data.columns:
        top_team_filter = team_data['teamAbbrev'] == matchup_data['top_seed_abbrev']
        bottom_team_filter = team_data['teamAbbrev'] == matchup_data['bottom_seed_abbrev']
        
        # Check if team data exists
        if sum(top_team_filter) > 0 and sum(bottom_team_filter) > 0:
            top_seed_data = team_data[top_team_filter].iloc[0]
            bottom_seed_data = team_data[bottom_team_filter].iloc[0]
            
            # Get the points difference for basic prediction
            if 'points' in top_seed_data and 'points' in bottom_seed_data:
                matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
            else:
                print(f"Warning: Missing 'points' data for {matchup_data['top_seed_abbrev']} or {matchup_data['bottom_seed_abbrev']}")
            
            # Feature columns to use for prediction
            feature_cols = [
                'PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score',
                'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
                'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
                'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
                'adjGoalsScoredAboveX/60'
            ]
            
            # Log which features are available
            available_features = [col for col in feature_cols if col in top_seed_data and col in bottom_seed_data]
            print(f"Available features for matchup: {len(available_features)}/{len(feature_cols)}")
            
            # Add features for each team if available
            for col in feature_cols:
                if col in top_seed_data and col in bottom_seed_data:
                    # Only add if both values are not NaN
                    if pd.notna(top_seed_data[col]) and pd.notna(bottom_seed_data[col]):
                        matchup_data[f"{col}_top"] = top_seed_data[col]
                        matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                        matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
                    else:
                        print(f"Missing value for feature {col} in matchup data")
    else:
        print(f"Warning: 'teamAbbrev' column not found in team_data")
    
    matchup_df = pd.DataFrame([matchup_data])
    print(f"Created matchup data with {len([c for c in matchup_df.columns if '_diff' in c])} differential features")
    return matchup_df

def load_data(filename, folder):
    """Load data from a CSV file
    
    Args:
        filename (str): Name of the file to load
        folder (str): Path to the folder containing the file
        
    Returns:
        DataFrame: Loaded data or None if file doesn't exist
    """
    filepath = os.path.join(folder, filename)
    try:
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            return data
        else:
            return None
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}")
        return None

def save_data(data, filename, folder):
    """Save data to a CSV or JSON file
    
    Args:
        data (DataFrame, dict, or list): Data to save
        filename (str): Name of the file to save
        folder (str): Path to the folder to save the file
    """
    filepath = os.path.join(folder, filename)
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            # Save DataFrame with appropriate options
            if filename.endswith('.json'):
                # For JSON, ensure dates and special floats are properly handled
                data.to_json(filepath, orient='records', date_format='iso')
                print(f"Saved DataFrame to JSON: {filepath} ({len(data)} rows)")
            else:
                # For CSV, ensure proper encoding and date formatting
                data.to_csv(filepath, index=False, encoding='utf-8')
                print(f"Saved DataFrame to CSV: {filepath} ({len(data)} rows)")
        elif isinstance(data, dict) or isinstance(data, list):
            # Handle JSON serialization with proper formatting
            if filename.endswith('.json'):
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Handle DataFrames nested inside dictionaries
                    processed_data = {}
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, pd.DataFrame):
                                processed_data[key] = json.loads(value.to_json(orient='records', date_format='iso'))
                            else:
                                processed_data[key] = value
                        json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Saved dictionary/list to JSON: {filepath}")
            else:
                st.warning(f"Cannot save dictionary/list to non-JSON file: {filename}")
        else:
            # For other types, attempt to convert to JSON
            try:
                if filename.endswith('.json'):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, default=lambda o: str(o), indent=2, ensure_ascii=False)
                    print(f"Saved data of type {type(data)} to JSON: {filepath}")
                else:
                    st.warning(f"Unsupported data type for {filename}: {type(data)}")
            except TypeError as te:
                st.error(f"Could not serialize data type {type(data)} to JSON: {str(te)}")
        
        # Verify file was created
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # Size in KB
            print(f"Successfully saved {filepath} ({file_size:.1f} KB)")
        else:
            st.error(f"File {filepath} was not created")
            
    except Exception as e:
        st.error(f"Error saving data to {filepath}: {str(e)}")
        import traceback
        print(f"Save error details: {traceback.format_exc()}")

# Update the daily data in the early morning once per day
def update_daily_data(data_folder, current_season, season_str, force=False):
    """Update NHL data once per day or when forced
    
    Args:
        data_folder (str): Path to the data folder
        current_season (int): Current NHL season starting year (e.g., 2023)
        season_str (str): Season string (e.g., "20232024")
        force (bool): Force update regardless of time since last refresh
    
    Returns:
        bool: True if data was updated, False otherwise
    """
    # Check if data directory exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    
    # Check if we should refresh data
    if should_refresh_data() or force:
        with st.spinner("Updating NHL data..."):
            # Get standings data
            standings_data = get_standings_data()
            standings_df = process_standings_data(standings_data, season_str)
            
            # Get team stats data
            stats_data = get_team_stats_data(current_season)
            stats_df = process_team_stats_data(stats_data)
            
            # Get advanced stats data
            advanced_stats_df = get_advanced_stats_data(data_folder, current_season)
            
            # Combine all data
            if not standings_df.empty and not stats_df.empty:
                try:
                    print("Combining data sources...")
                    # Ensure both dataframes have string season columns
                    if 'season' in standings_df.columns:
                        standings_df['season'] = standings_df['season'].astype(str)
                    if 'season' in stats_df.columns:
                        stats_df['season'] = stats_df['season'].astype(str)
                    
                    # First merge standings and stats
                    merged_df = pd.merge(
                        standings_df,
                        stats_df,
                        on=['season', 'teamName'],
                        how='inner',
                        suffixes=('_x', '')
                    )
                    print(f"Merged standings and stats: {merged_df.shape[0]} teams")
                    
                    # Then add advanced stats if available
                    if not advanced_stats_df.empty:
                        print("Merging advanced stats data...")
                        
                        # Debug team abbreviations
                        print("Team abbreviations in merged data:", merged_df['teamAbbrev'].unique())
                        if 'team' in advanced_stats_df.columns:
                            print("Team abbreviations in advanced stats:", advanced_stats_df['team'].unique())
                        
                        # Ensure proper joining columns and data types
                        if 'teamAbbrev' in merged_df.columns and 'team' in advanced_stats_df.columns:
                            # Make sure team abbreviations match
                            advanced_stats_df['team'] = advanced_stats_df['team'].str.strip()
                            
                            # Convert season format in advanced_stats_df from "2024" to "20242025"
                            if 'season' in advanced_stats_df.columns:
                                # Convert season from "2024" to "20242025" format
                                advanced_stats_df['season'] = advanced_stats_df['season'].astype(str).apply(
                                    lambda x: f"{x}{int(x)+1}" if len(x) == 4 and x.isdigit() else x
                                )
                                print(f"Fixed advanced stats season format. Sample: {advanced_stats_df['season'].iloc[0] if not advanced_stats_df.empty else 'No data'}")
                            
                            # Now convert both to string to ensure consistency
                            if 'season' in merged_df.columns:
                                merged_df['season'] = merged_df['season'].astype(str)
                            if 'season' in advanced_stats_df.columns:
                                advanced_stats_df['season'] = advanced_stats_df['season'].astype(str)
                            
                            # Now merge with consistent season format
                            team_data = pd.merge(
                                merged_df,
                                advanced_stats_df,
                                left_on=['season', 'teamAbbrev'],
                                right_on=['season', 'team'],
                                how='left'
                            )
                            print(f"Final merged data with advanced stats: {team_data.shape[0]} teams")
                            
                            # Check for missing data in key advanced stats columns
                            if 'xGoalsPercentage' in team_data.columns:
                                null_pct = team_data['xGoalsPercentage'].isna().mean() * 100
                                print(f"Missing xGoalsPercentage data: {null_pct:.1f}%")
                            
                            # Engineer features
                            team_data = engineer_features(team_data)
                            
                            # Apply standardized metrics calculation
                            print("Calculating standardized metrics...")
                            team_data = calculate_standard_metrics(team_data)
                            
                            # Debug: Print available metrics for verification
                            print("Checking for per-82 metrics:")
                            per82_metrics = ['goalsSavedAboveExpectedPer82', 'goalsScoredAboveExpectedPer82']
                            for metric in per82_metrics:
                                if metric in team_data.columns:
                                    print(f"- {metric}: {len(team_data[team_data[metric] != 0])} teams have non-zero values")
                                else:
                                    print(f"- {metric}: Not found in data")
                            
                            # Add playoff history metrics
                            print("Adding playoff history metrics...")
                            team_data = add_playoff_history_metrics(team_data, data_folder)
                            
                            # Save the combined data
                            save_data(team_data, f"team_data_{season_str}.csv", data_folder)
                            
                            # Store in session state
                            st.session_state.team_data = team_data
                            
                            # Determine playoff matchups
                            playoff_matchups = determine_playoff_teams(standings_df)
                            if playoff_matchups:
                                save_data(playoff_matchups, "playoff_matchups.json", data_folder)
                                st.session_state.playoff_matchups = playoff_matchups
                            
                            # Update last refresh timestamp
                            st.session_state.last_data_refresh = datetime.utcnow()
                            
                            return True
                        else:
                            print("Missing required columns for merging advanced stats")
                            missing_cols = []
                            if 'teamAbbrev' not in merged_df.columns:
                                missing_cols.append('teamAbbrev in merged_df')
                            if 'team' not in advanced_stats_df.columns:
                                missing_cols.append('team in advanced_stats_df')
                            print(f"Missing columns: {missing_cols}")
                    else:
                        print("No advanced stats available to merge")
                        
                        # Still process the data we have
                        team_data = engineer_features(merged_df)
                        team_data = add_playoff_history_metrics(team_data, data_folder)
                        
                        save_data(team_data, f"team_data_{season_str}.csv", data_folder)
                        st.session_state.team_data = team_data
                        return True
                        
                except Exception as e:
                    print(f"Error combining data: {str(e)}")
                    print(f"Data types - standings_df['season']: {standings_df['season'].dtype}, stats_df['season']: {stats_df['season'].dtype}")
                    print("Attempting fallback method with concat...")
                    try:
                        # Fallback to concat approach if merge fails due to type issues
                        if 'teamName' in standings_df.columns and 'teamName' in stats_df.columns:
                            # Convert season to string type in both dataframes
                            standings_df['season'] = standings_df['season'].astype(str)
                            stats_df['season'] = stats_df['season'].astype(str)
                            
                            # Use outer merge to keep all teams
                            merged_df = pd.merge(
                                standings_df,
                                stats_df,
                                on=['season', 'teamName'],
                                how='outer',
                                suffixes=('_x', '')
                            )
                            
                            # Check for missing teams
                            print(f"Merged data shape: {merged_df.shape}")
                            if len(merged_df) < len(standings_df):
                                print(f"Warning: Lost {len(standings_df) - len(merged_df)} teams in merge")
                                
                            # Process data
                            team_data = engineer_features(merged_df)
                            team_data = add_playoff_history_metrics(team_data, data_folder)
                            
                            save_data(team_data, f"team_data_{season_str}.csv", data_folder)
                            st.session_state.team_data = team_data
                            return True
                    except Exception as fallback_error:
                        print(f"Fallback method failed: {str(fallback_error)}")
    
    # If we didn't refresh, check if we need to load from disk
    if 'team_data' not in st.session_state or st.session_state.team_data is None:
        team_data = load_data(f"team_data_{season_str}.csv", data_folder)
        if team_data is not None:
            st.session_state.team_data = team_data
        
    if 'playoff_matchups' not in st.session_state or st.session_state.playoff_matchups is None:
        try:
            with open(os.path.join(data_folder, "playoff_matchups.json"), 'r') as f:
                playoff_matchups = json.load(f)
                st.session_state.playoff_matchups = playoff_matchups
        except:
            pass
    
    return False

# Additional utility functions from notebook

def load_team_data(data_folder=None):
    """Load team statistics data from file or API."""
    if data_folder is None:
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    team_data_path = os.path.join(data_folder, "team_stats.csv")
    
    # Try loading from file first
    if os.path.exists(team_data_path):
        try:
            teams_df = pd.read_csv(team_data_path)
            # Check if file is not empty and has expected columns
            if not teams_df.empty and 'teamName' in teams_df.columns:
                # Ensure season is a string type
                if 'season' in teams_df.columns:
                    teams_df['season'] = teams_df['season'].astype(str)
                return teams_df
        except Exception as e:
            print(f"Error loading team data from file: {e}")
    
    # If file doesn't exist or is invalid, fetch from API
    try:
        current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
        season_str = f"{current_season}{current_season+1}"
        
        # Try to get data through the update function
        update_daily_data(data_folder, current_season, season_str, force=True)
        
        # Check session state
        if 'team_data' in st.session_state and not st.session_state.team_data.empty:
            team_data = st.session_state.team_data
            # Ensure season is a string type
            if 'season' in team_data.columns:
                team_data['season'] = team_data['season'].astype(str)
            return team_data
            
        # Try loading saved file again
        team_data_path = os.path.join(data_folder, f"team_data_{season_str}.csv")
        if os.path.exists(team_data_path):
            team_data = pd.read_csv(team_data_path)
            # Ensure season is a string type
            if 'season' in team_data.columns:
                team_data['season'] = team_data['season'].astype(str)
            return team_data
            
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching team data from API: {e}")
        # Return empty DataFrame if all methods fail
        return pd.DataFrame()

def load_current_playoff_matchups(data_folder=None):
    """Load current playoff matchups from file or API."""
    if data_folder is None:
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    matchups_path = os.path.join(data_folder, "playoff_matchups.json")
    
    # Try loading from session state first
    if 'playoff_matchups' in st.session_state:
        return st.session_state.playoff_matchups
    
    # Try loading from file next
    if os.path.exists(matchups_path):
        try:
            with open(matchups_path, 'r') as f:
                matchups = json.load(f)
            return matchups
        except Exception as e:
            print(f"Error loading playoff matchups from file: {e}")
    
    # If file doesn't exist or is invalid, fetch from API
    try:
        current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
        season_str = f"{current_season}{current_season+1}"
        
        # Try to get data through the update function
        update_daily_data(data_folder, current_season, season_str, force=True)
        
        # Check if update generated matchups
        if 'playoff_matchups' in st.session_state:
            return st.session_state.playoff_matchups
            
        # Try loading saved file again
        if os.path.exists(matchups_path):
            with open(matchups_path, 'r') as f:
                matchups = json.load(f)
            return matchups
    except Exception as e:
        print(f"Error fetching playoff matchups: {e}")
    
    # Return empty dict if all methods fail
    return {}

def load_simulation_results(data_folder=None):
    """Load most recent simulation results."""
    # Check if we have daily simulations in session state
    if 'daily_simulations' in st.session_state:
        return st.session_state.daily_simulations
    
    if data_folder is None:
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    sim_path = os.path.join(data_folder, "simulation_results.json")
    
    # Try loading from file
    if os.path.exists(sim_path):
        try:
            with open(sim_path, 'r') as f:
                sim_data = json.load(f)
                
            # Convert team results to DataFrame if string
            if 'team_results' in sim_data and isinstance(sim_data['team_results'], str):
                sim_data['team_results'] = pd.DataFrame(json.loads(sim_data['team_results']))
            
            return sim_data
        except Exception as e:
            print(f"Error loading simulation results: {e}")
    
    # Return None if we can't load results
    return None

def calculate_standard_metrics(team_data_df):
    """Calculate and standardize metrics needed across the application
    
    Args:
        team_data_df (DataFrame): Team data
        
    Returns:
        DataFrame: Team data with standardized metrics
    """
    # Make a copy to avoid modifying original
    df = team_data_df.copy()
    
    # Standard metrics mapping - use these as the canonical metric names
    standard_metrics = {
        'pointsPct': ['pointPctg', 'Points%', 'points_percentage', 'pts_pct', 'PTS%'], 
        'goalDiff/G': ['goalDiffPerGame', 'goal_diff_per_game'],
        'PP%': ['powerPlayPct', 'ppPctg'],
        'PK%': ['penaltyKillPct', 'pkPctg'],
        'FO%': ['faceOffWinPct', 'faceoffWinPctg'],
        'xGoalsPercentage': ['expectedGoalsFor'],
        'possAdjHitsPctg': ['hitsPct'],
        'possAdjTakeawaysPctg': ['takeawaysPct'],
        'possTypeAdjGiveawaysPctg': ['giveawaysPct'],
        'reboundxGoalsPctg': ['reboundxGoalsPctg'],
        'goalsSavedAboveExpectedPer82':['goalsSavedAboveExpectedPer82'],
        'goalsScoredAboveExpectedPer82':['goalsScoredAboveExpectedPer82']

    }
    
    # Apply mapping - for each standard metric, try all possible source columns
    for std_metric, alt_metrics in standard_metrics.items():
        # Skip if standard metric already exists with non-null values
        if std_metric in df.columns and not df[std_metric].isna().all():
            continue
            
        # Try all alternative metric names
        for alt_metric in alt_metrics:
            if alt_metric in df.columns and not df[alt_metric].isna().all():
                df[std_metric] = df[alt_metric]
                # Apply percentage scaling if needed (0-1 to 0-100)
                if any(x in std_metric for x in ['Pct', 'Percentage', '%']) and df[std_metric].max() <= 1:
                    df[std_metric] = df[std_metric] * 100
                break
    
    # Make sure all percentage metrics are on 0-100 scale
    pct_cols = [col for col in df.columns if any(x in col for x in ['Pct', 'Percentage', '%'])]
    for col in pct_cols:
        if col in df.columns and df[col].max() <= 1:
            df[col] = df[col] * 100
    
    return df
