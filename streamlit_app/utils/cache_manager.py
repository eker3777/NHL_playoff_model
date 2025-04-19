"""
Centralized cache management module for NHL Playoff Predictor.
Handles data refreshing, caching of simulation results, and timezone management.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import logging

# Import all constants from config
from streamlit_app.config import (
    REFRESH_HOUR,
    REFRESH_TIMEZONE,
    CACHE_DURATION,
    DATA_DIR,
    CACHE_DIR,
    SIMULATION_RESULTS_DIR,
    DEBUG_MODE
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

def get_current_time():
    """Get current time in the configured timezone
    
    Returns:
        datetime: Current time in configured timezone
    """
    return datetime.now(REFRESH_TIMEZONE)

def should_refresh_data():
    """Check if we should refresh the data based on refresh rules
    
    Returns:
        bool: True if data should be refreshed, False otherwise
    """
    # Check if last_data_refresh exists in session state
    if 'last_data_refresh' not in st.session_state:
        logger.info("No last_data_refresh in session state - refresh needed")
        return True
    
    try:
        # Get current time in configured timezone
        now = get_current_time()
        
        # Get last refresh time from session state
        last_refresh = st.session_state.last_data_refresh
        
        # Handle string timestamps (convert to datetime)
        if isinstance(last_refresh, str):
            try:
                # Try to parse as ISO format
                last_refresh = datetime.fromisoformat(last_refresh)
                # Make it timezone aware if it's not
                if last_refresh.tzinfo is None:
                    last_refresh = REFRESH_TIMEZONE.localize(last_refresh)
            except ValueError:
                logger.error("Invalid last_refresh format")
                return True
        
        # Handle None or invalid types
        if not isinstance(last_refresh, datetime):
            logger.error(f"Invalid last_refresh type: {type(last_refresh)}")
            return True
            
        # Make sure it's timezone aware
        if last_refresh.tzinfo is None:
            last_refresh = REFRESH_TIMEZONE.localize(last_refresh)
        
        # Calculate time since last refresh
        time_since_refresh = now - last_refresh
        
        # Check if it's been more than CACHE_DURATION seconds since last refresh
        if time_since_refresh.total_seconds() > CACHE_DURATION:
            logger.info(f"Cache expired - {time_since_refresh.total_seconds()} seconds since last refresh")
            return True
        
        # Check if it's a new day and after REFRESH_HOUR
        if (now.date() > last_refresh.date() and now.hour >= REFRESH_HOUR):
            logger.info(f"New day after {REFRESH_HOUR}:00 - refresh needed")
            return True
            
        # Check if it's the same day but we haven't refreshed yet after REFRESH_HOUR
        if (now.date() == last_refresh.date() and 
            now.hour >= REFRESH_HOUR and 
            last_refresh.hour < REFRESH_HOUR):
            logger.info(f"Same day after {REFRESH_HOUR}:00 but last refresh was before {REFRESH_HOUR}:00 - refresh needed")
            return True
        
        logger.info("No refresh needed")
        return False
    except Exception as e:
        logger.error(f"Error checking refresh status: {str(e)}", exc_info=True)
        return True  # When in doubt, refresh

def get_next_refresh_time():
    """Calculate the next scheduled refresh time
    
    Returns:
        datetime: Next scheduled refresh time
    """
    now = get_current_time()
    
    # If it's before REFRESH_HOUR today, next refresh is today at REFRESH_HOUR
    if now.hour < REFRESH_HOUR:
        next_refresh = now.replace(hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
    else:
        # Otherwise, next refresh is tomorrow at REFRESH_HOUR
        next_refresh = (now + timedelta(days=1)).replace(hour=REFRESH_HOUR, minute=0, second=0, microsecond=0)
    
    return next_refresh

def cache_simulation_results(results, data_folder=None):
    """Cache simulation results to disk
    
    Args:
        results: Dictionary of simulation results
        data_folder: Path to data folder (uses SIMULATION_RESULTS_DIR from config if None)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if data_folder is None:
        data_folder = SIMULATION_RESULTS_DIR
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    
    try:
        # Generate cache file path
        cache_path = os.path.join(data_folder, "simulation_results.json")
        
        # Process results to make them JSON serializable
        serialized_results = {}
        
        # Handle each key in the results dictionary
        for key, value in results.items():
            # Convert DataFrames to JSON strings
            if isinstance(value, pd.DataFrame):
                serialized_results[key] = value.to_json(orient='records', date_format='iso')
            elif isinstance(value, np.ndarray):
                serialized_results[key] = value.tolist()
            elif isinstance(value, (datetime, pd.Timestamp)):
                serialized_results[key] = value.isoformat()
            else:
                serialized_results[key] = value
        
        # Add timestamp to cache
        serialized_results['cache_timestamp'] = get_current_time().isoformat()
        
        # Write to disk
        with open(cache_path, 'w') as f:
            json.dump(serialized_results, f, indent=2)
        
        logger.info(f"Simulation results cached to {cache_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error caching simulation results: {str(e)}", exc_info=True)
        return False

def load_cached_simulation_results(data_folder=None):
    """Load cached simulation results from disk
    
    Args:
        data_folder: Path to data folder (uses SIMULATION_RESULTS_DIR from config if None)
        
    Returns:
        dict: Simulation results or None if no cache exists
    """
    if data_folder is None:
        data_folder = SIMULATION_RESULTS_DIR
    
    cache_path = os.path.join(data_folder, "simulation_results.json")
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        # Load cached results
        with open(cache_path, 'r') as f:
            serialized_results = json.load(f)
        
        # Process results back to original types
        results = {}
        
        # Handle each key in the serialized results
        for key, value in serialized_results.items():
            # Skip the timestamp
            if key == 'cache_timestamp':
                try:
                    # Convert ISO timestamp string to datetime with timezone
                    timestamp = datetime.fromisoformat(value)
                    # Make timestamp timezone aware if needed
                    if timestamp.tzinfo is None:
                        timestamp = REFRESH_TIMEZONE.localize(timestamp)
                    results[key] = timestamp
                except ValueError:
                    # Fallback if timestamp can't be parsed
                    results[key] = value
                continue
            
            # Try to convert JSON strings back to DataFrames
            if isinstance(value, str) and value.startswith('[{'):
                try:
                    results[key] = pd.read_json(value, orient='records')
                except:
                    results[key] = value
            else:
                results[key] = value
        
        logger.info(f"Loaded cached simulation results from {cache_path}")
        return results
    
    except Exception as e:
        logger.error(f"Error loading cached simulation results: {str(e)}", exc_info=True)
        return None

def is_cache_fresh(cache_timestamp=None):
    """Check if cache is still fresh
    
    Args:
        cache_timestamp: Timestamp of when the cache was created
        
    Returns:
        bool: True if cache is fresh, False otherwise
    """
    if cache_timestamp is None:
        logger.debug("Cache timestamp is None")
        return False
    
    try:
        # Get current time in configured timezone
        now = get_current_time()
        
        # Handle string timestamps (convert to datetime)
        if isinstance(cache_timestamp, str):
            try:
                cache_timestamp = datetime.fromisoformat(cache_timestamp)
                # Make timezone aware if needed
                if cache_timestamp.tzinfo is None:
                    cache_timestamp = REFRESH_TIMEZONE.localize(cache_timestamp)
            except ValueError:
                logger.error("Invalid cache_timestamp format")
                return False
        
        # Make sure timestamp is timezone aware
        if cache_timestamp.tzinfo is None:
            cache_timestamp = REFRESH_TIMEZONE.localize(cache_timestamp)
        
        # Calculate time since cache was created
        time_since_cache = now - cache_timestamp
        
        # Check if it's been more than CACHE_DURATION seconds
        if time_since_cache.total_seconds() > CACHE_DURATION:
            return False
        
        # Check if it's a new day and after REFRESH_HOUR
        if (now.date() > cache_timestamp.date() and now.hour >= REFRESH_HOUR):
            return False
            
        # Check if it's the same day but we're past REFRESH_HOUR and cache is from before
        if (now.date() == cache_timestamp.date() and 
            now.hour >= REFRESH_HOUR and 
            cache_timestamp.hour < REFRESH_HOUR):
            return False
        
        logger.debug("Cache is fresh")
        return True
    except Exception as e:
        logger.error(f"Error checking cache freshness: {str(e)}", exc_info=True)
        return False  # When in doubt, consider cache stale

def get_cache_status(detailed=False):
    """Get status of the cache for UI display
    
    Args:
        detailed: Whether to return detailed status information
        
    Returns:
        dict: Cache status information
    """
    try:
        status = {
            'is_fresh': False,
            'last_refresh': None,
            'next_refresh': get_next_refresh_time(),
            'refresh_needed': True
        }
        
        # Check if we have last_data_refresh in session state
        if 'last_data_refresh' in st.session_state and st.session_state.last_data_refresh is not None:
            last_refresh = st.session_state.last_data_refresh
            status['last_refresh'] = last_refresh
            
            # Only check refresh status if last_refresh is not None
            refresh_needed = should_refresh_data()
            status['is_fresh'] = not refresh_needed
            status['refresh_needed'] = refresh_needed
        
        if detailed:
            # Add detailed information
            status['data_status'] = {}
            
            # Check standings data
            if 'standings_df' in st.session_state:
                status['data_status']['standings'] = {
                    'available': True,
                    'rows': len(st.session_state.standings_df)
                }
            else:
                status['data_status']['standings'] = {
                    'available': False
                }
            
            # Check team data
            if 'team_data' in st.session_state:
                status['data_status']['team_data'] = {
                    'available': True,
                    'rows': len(st.session_state.team_data)
                }
            else:
                status['data_status']['team_data'] = {
                    'available': False
                }
            
            # Check playoff matchups
            if 'playoff_matchups' in st.session_state:
                matchups = st.session_state.playoff_matchups
                conferences = matchups.keys() if isinstance(matchups, dict) else []
                status['data_status']['playoff_matchups'] = {
                    'available': True,
                    'conferences': list(conferences)
                }
            else:
                status['data_status']['playoff_matchups'] = {
                    'available': False
                }
            
            # Check simulation results
            if 'daily_simulations' in st.session_state:
                status['data_status']['simulations'] = {
                    'available': True,
                    'timestamp': st.session_state.get('last_simulation_refresh', None)
                }
            else:
                status['data_status']['simulations'] = {
                    'available': False
                }
        
        logger.debug(f"Cache status: {status}")
        return status
    except Exception as e:
        logger.error(f"Error in get_cache_status(): {str(e)}", exc_info=True)
        return {
            'is_fresh': False,
            'last_refresh': None,
            'next_refresh': None,
            'refresh_needed': True
        }

def display_cache_status():
    """Display cache status in the Streamlit sidebar"""
    status = get_cache_status()
    
    st.sidebar.subheader("Data Status")
    
    if status['is_fresh']:
        st.sidebar.success("Data is up to date")
    else:
        st.sidebar.warning("Data refresh needed")
    
    if status['last_refresh']:
        # Format timestamp for display
        if hasattr(status['last_refresh'], 'strftime'):
            last_refresh_str = status['last_refresh'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            last_refresh_str = str(status['last_refresh'])
        st.sidebar.text(f"Last refresh: {last_refresh_str}")
    
    # Format next refresh time
    if status['next_refresh'] and hasattr(status['next_refresh'], 'strftime'):
        next_refresh_str = status['next_refresh'].strftime('%Y-%m-%d %H:%M:%S')
    else:
        next_refresh_str = str(status['next_refresh'])
    
    st.sidebar.text(f"Next refresh: {next_refresh_str}")
    
    if status['refresh_needed']:
        if st.sidebar.button("Refresh Data Now"):
            st.session_state.force_refresh = True
            st.rerun()

def cache_api_response(endpoint, data, expires_in=CACHE_DURATION):
    """Cache API response to the cache directory
    
    Args:
        endpoint: API endpoint or unique identifier for the data
        data: Data to cache (usually JSON response)
        expires_in: Cache expiration time in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure cache directory exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            
        # Generate a safe filename from the endpoint
        safe_filename = "".join([c if c.isalnum() else "_" for c in endpoint])
        cache_path = os.path.join(CACHE_DIR, f"{safe_filename}.json")
        
        # Prepare data with metadata
        cache_data = {
            "data": data,
            "timestamp": get_current_time().isoformat(),
            "expires_at": (get_current_time() + timedelta(seconds=expires_in)).isoformat()
        }
        
        # Write to disk
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        logger.info(f"API response cached to {cache_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error caching API response: {str(e)}", exc_info=True)
        return False

def load_cached_api_response(endpoint):
    """Load cached API response from the cache directory
    
    Args:
        endpoint: API endpoint or unique identifier for the data
        
    Returns:
        dict: Cached data or None if no cache exists or cache is expired
    """
    try:
        # Generate the same safe filename
        safe_filename = "".join([c if c.isalnum() else "_" for c in endpoint])
        cache_path = os.path.join(CACHE_DIR, f"{safe_filename}.json")
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            logger.debug(f"No cache file found for {endpoint}")
            return None
            
        # Load cache data
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is expired
        if 'expires_at' in cache_data:
            try:
                expires_at = datetime.fromisoformat(cache_data['expires_at'])
                if expires_at.tzinfo is None:
                    expires_at = REFRESH_TIMEZONE.localize(expires_at)
                    
                if get_current_time() > expires_at:
                    logger.debug(f"Cache for {endpoint} is expired")
                    return None
            except (ValueError, TypeError):
                logger.warning(f"Invalid expiration format in cache for {endpoint}")
                return None
                
        logger.info(f"Loaded cached API response for {endpoint}")
        return cache_data.get('data')
        
    except Exception as e:
        logger.error(f"Error loading cached API response: {str(e)}", exc_info=True)
        return None

def clear_all_caches():
    """Clear all cache files from cache and simulation results directories
    
    Returns:
        tuple: (bool, int) - Success status and number of files removed
    """
    try:
        files_removed = 0
        
        # Clear files from cache directory
        if os.path.exists(CACHE_DIR):
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, filename))
                    files_removed += 1
        
        # Clear files from simulation results directory
        if os.path.exists(SIMULATION_RESULTS_DIR):
            for filename in os.listdir(SIMULATION_RESULTS_DIR):
                if filename.endswith('.json'):
                    os.remove(os.path.join(SIMULATION_RESULTS_DIR, filename))
                    files_removed += 1
                    
        logger.info(f"Cleared {files_removed} cache files")
        return True, files_removed
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}", exc_info=True)
        return False, 0
