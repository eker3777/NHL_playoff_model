"""
Data loading and processing module for NHL playoff predictions.
Extracted from Streamlit app for standalone use.
"""

import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import nhlpy
    from nhlpy.nhl_client import NHLClient
    client = NHLClient()
    logger.info("✓ NHL API client initialized")
except Exception as e:
    logger.warning(f"Could not initialize NHL client: {str(e)}")
    client = None


class NHLDataLoader:
    """NHL data loading and processing class"""
    
    def __init__(self, data_folder: str, current_season: int):
        self.data_folder = data_folder
        self.current_season = current_season
        self.season_str = f"{current_season}{current_season + 1}"
        
        # Ensure data folder exists
        os.makedirs(data_folder, exist_ok=True)
    
    def make_request_with_retry(self, url: str, headers: Optional[Dict] = None, 
                               retries: int = 3, timeout: int = 10) -> requests.Response:
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
                    logger.warning(f"Request timed out. Retrying ({attempt+1}/{retries})...")
                    time.sleep(2)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    logger.warning(f"Request failed: {str(e)}. Retrying ({attempt+1}/{retries})...")
                    time.sleep(2)
                else:
                    raise
        
        raise requests.exceptions.RequestException("Max retries exceeded")
    
    def get_standings_data(self) -> Optional[Dict]:
        """Get current NHL standings data with improved error handling and fallback options"""
        try:
            # Try method 1: Using NHLClient from nhlpy
            if client is not None:
                try:
                    standings_data = client.standings.get_standings()
                    if standings_data:
                        logger.info("✓ Got standings from NHL API client")
                        return standings_data
                        
                except Exception as client_error:
                    logger.warning(f"Primary standings source failed: {str(client_error)}")
            
            # Try method 2: Direct API request to NHL.com
            try:
                url = "https://api-web.nhle.com/v1/standings/now"
                response = self.make_request_with_retry(url, timeout=15)
                if response.status_code == 200:
                    logger.info("✓ Got standings from direct API")
                    return response.json()
            except Exception as direct_api_error:
                logger.warning(f"Direct API request failed: {str(direct_api_error)}")
            
            # Try method 3: Alternative API endpoint
            try:
                url = "https://api-web.nhle.com/v1/standings/current"
                response = self.make_request_with_retry(url, timeout=15)
                if response.status_code == 200:
                    logger.info("✓ Got standings from alternative API")
                    return response.json()
            except Exception as alt_api_error:
                logger.warning(f"Alternative API request failed: {str(alt_api_error)}")
                
            # Method 4: Last resort - load from cached file
            standings_path = os.path.join(self.data_folder, f"standings_{self.season_str}.csv")
            
            if os.path.exists(standings_path):
                logger.info("Using cached standings data from file")
                standings_df = pd.read_csv(standings_path)
                # Convert DataFrame back to dictionary format for consistency
                return {"standings": standings_df.to_dict(orient='records')}
            
            raise Exception("All methods to fetch standings data failed")
        except Exception as e:
            logger.error(f"Error fetching standings data: {str(e)}")
            return None
    
    def get_team_stats_data(self, start_season: Optional[int] = None, 
                           end_season: Optional[int] = None) -> Optional[Dict]:
        """Get team summary stats data directly from NHL API with improved error handling"""
        try:
            # Use current season if not specified
            if start_season is None:
                start_season = self.current_season
                
            if end_season is None:
                end_season = start_season
                
            # Construct the URL with season parameters
            url = f"https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId>={start_season}{start_season+1} and seasonId<={end_season}{end_season+1}"
            
            # Make direct API request with retry logic
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
            }
            
            response = self.make_request_with_retry(url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    logger.info(f"✓ Got stats for {len(data['data'])} teams")
                    return data['data']
                else:
                    logger.warning("No team stats data found in API response")
                    return None
            else:
                logger.error(f"API request failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching team stats data: {str(e)}")
            # Try fallback to cached file
            stats_path = os.path.join(self.data_folder, f"stats_{self.season_str}.csv")
            if os.path.exists(stats_path):
                logger.info("Using cached stats data from file")
                stats_df = pd.read_csv(stats_path)
                return stats_df.to_dict(orient='records')
            return None
    
    def get_advanced_stats_data(self, season: Optional[int] = None, situation: str = '5on5') -> pd.DataFrame:
        """Get advanced stats from MoneyPuck"""
        if season is None:
            season = self.current_season
            
        # Construct the MoneyPuck URL for the current season
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
        csv_path = os.path.join(self.data_folder, f"moneypuck_regular_{season}.csv")
        
        try:
            # First try to load from local cache if it exists
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        logger.info("✓ Loaded advanced stats from cache")
                        return self.process_advanced_stats(df, f"{season}{season+1}", situation)
                except Exception as cache_error:
                    logger.warning(f"Error loading from cache: {str(cache_error)}")
            
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
                temp_path = os.path.join(self.data_folder, "temp_data.csv")
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                df = pd.read_csv(temp_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                if not df.empty:
                    # Cache the successful download
                    df.to_csv(csv_path, index=False)
                    logger.info(f"✓ Downloaded and cached advanced stats for {season}")
                    return self.process_advanced_stats(df, f"{season}{season+1}", situation)
            
            logger.warning(f"Failed to get advanced stats data for {season}")
            return self.create_empty_advanced_stats_df()
            
        except Exception as e:
            logger.error(f"Error getting advanced stats: {str(e)}")
            return self.create_empty_advanced_stats_df()
    
    def process_advanced_stats(self, df: pd.DataFrame, season_str: str, situation: str) -> pd.DataFrame:
        """Process the advanced stats dataframe while preserving all required columns."""
        if df.empty:
            return self.create_empty_advanced_stats_df()
        
        # Filter for the specified situation
        if 'situation' in df.columns:
            df = df[df['situation'] == situation].copy()
        
        # Add season column
        df['season'] = season_str
        
        # Team name mapping for consistency
        team_mapping = {
            'ANA': 'Anaheim Ducks', 'ARI': 'Arizona Coyotes', 'BOS': 'Boston Bruins',
            'BUF': 'Buffalo Sabres', 'CGY': 'Calgary Flames', 'CAR': 'Carolina Hurricanes',
            'CHI': 'Chicago Blackhawks', 'COL': 'Colorado Avalanche', 'CBJ': 'Columbus Blue Jackets',
            'DAL': 'Dallas Stars', 'DET': 'Detroit Red Wings', 'EDM': 'Edmonton Oilers',
            'FLA': 'Florida Panthers', 'LAK': 'Los Angeles Kings', 'MIN': 'Minnesota Wild',
            'MTL': 'Montréal Canadiens', 'NSH': 'Nashville Predators', 'NJD': 'New Jersey Devils',
            'NYI': 'New York Islanders', 'NYR': 'New York Rangers', 'OTT': 'Ottawa Senators',
            'PHI': 'Philadelphia Flyers', 'PIT': 'Pittsburgh Penguins', 'SJS': 'San Jose Sharks',
            'SEA': 'Seattle Kraken', 'STL': 'St. Louis Blues', 'TBL': 'Tampa Bay Lightning',
            'TOR': 'Toronto Maple Leafs', 'UTA': 'Utah Hockey Club', 'VAN': 'Vancouver Canucks',
            'VGK': 'Vegas Golden Knights', 'WSH': 'Washington Capitals', 'WPG': 'Winnipeg Jets'
        }
        
        # Map team abbreviations to full names
        if 'team' in df.columns:
            df['teamName'] = df['team'].map(team_mapping).fillna(df['team'])
        
        return df
    
    def create_empty_advanced_stats_df(self) -> pd.DataFrame:
        """Create an empty DataFrame with all required advanced stats columns."""
        columns = [
            'season', 'teamName', 'xGoalsPercentage', 'flurryScoreVenueAdjustedxGoalsFor',
            'flurryScoreVenueAdjustedxGoalsAgainst', 'possAdjHitsPctg', 'possAdjTakeawaysPctg',
            'possTypeAdjGiveawaysPctg', 'possAdjLowDangerShotAttemptsPctg',
            'possAdjMediumDangerShotAttemptsPctg', 'possAdjHighDangerShotAttemptsPctg',
            'possAdjShotAttemptsPctg', 'possAdjUnblockedShotAttemptsPctg',
            'possAdjShotsPctg', 'possAdjGoalsPctg'
        ]
        return pd.DataFrame(columns=columns)
    
    def process_standings_data(self, standings_data: Dict, season_str: str) -> pd.DataFrame:
        """Process NHL standings data into a DataFrame with improved error handling"""
        if not standings_data:
            logger.error("No standings data provided")
            return pd.DataFrame()
        
        all_standings_data = []
        
        try:
            # Handle different possible structures
            if 'standings' in standings_data:
                standings_list = standings_data['standings']
            elif isinstance(standings_data, list):
                standings_list = standings_data
            else:
                standings_list = [standings_data]
            
            for standing in standings_list:
                if isinstance(standing, dict):
                    # Process the standing data
                    team_data = {'season': season_str}
                    
                    # Extract team information
                    if 'teamAbbrev' in standing:
                        team_data['teamAbbrev'] = standing['teamAbbrev']
                    if 'teamName' in standing:
                        team_data['teamName'] = standing['teamName']
                    elif 'teamCommonName' in standing:
                        team_data['teamName'] = standing['teamCommonName']
                    
                    # Extract standing metrics
                    for key, value in standing.items():
                        if key not in ['teamAbbrev', 'teamName', 'teamCommonName']:
                            team_data[key] = value
                    
                    all_standings_data.append(team_data)
            
            if all_standings_data:
                df = pd.DataFrame(all_standings_data)
                df['season'] = season_str
                logger.info(f"✓ Processed standings for {len(df)} teams")
                return df
            else:
                logger.warning("No valid standings data found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error processing standings data: {str(e)}")
            return pd.DataFrame()
    
    def process_team_stats_data(self, stats_data: list) -> pd.DataFrame:
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
            logger.warning("Unexpected team stats data format")
        
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
                
                # Basic column renaming
                rename_dict = {
                    'seasonId': 'season',
                    'teamName': 'teamName',
                    'teamFullName': 'teamName',
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
                
                # Rename team_name to teamName if it exists
                if 'team_name' in stats_df.columns:
                    stats_df = stats_df.rename(columns={'team_name': 'teamName'})
                
                logger.info(f"✓ Processed stats for {len(stats_df)} teams")
                return stats_df
        
        return pd.DataFrame()