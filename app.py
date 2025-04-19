"""
NHL Playoff Predictor - Main Application

This is the main entry point for the NHL Playoff Predictor Streamlit application.
It handles page navigation, session state initialization, and app configuration.
"""

import streamlit as st
import os
import pandas as pd
from datetime import datetime
import traceback
import logging

# Import constants from centralized config
from streamlit_app.config import (
    APP_TITLE, 
    APP_VERSION, 
    GITHUB_URL, 
    THEME_COLOR,
    REFRESH_TIMEZONE,
    DATA_DIR,
    MODEL_DIR,
    ENABLE_DEBUG_UI,
    DEBUG_MODE
)

# Import utility modules
from streamlit_app.utils.cache_manager import (
    should_refresh_data, 
    get_current_time,
    get_next_refresh_time,
    get_cache_status
)

from streamlit_app.utils.data_handlers import (
    update_daily_data,
    load_team_data,
    format_percentage_for_display
)
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
def apply_custom_styling():
    """Apply custom styling to the application"""
    custom_css = f"""
        <style>
        .main-header {{
            color: {THEME_COLOR};
        }}
        .stProgress > div > div {{
            background-color: {THEME_COLOR};
        }}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
def initialize_session_state():
    """Initialize session state variables"""
    if 'last_data_refresh' not in st.session_state:
        st.session_state.last_data_refresh = None
        
    if 'last_simulation_refresh' not in st.session_state:
        st.session_state.last_simulation_refresh = None
        
    if 'team_data' not in st.session_state:
        st.session_state.team_data = None
        
    if 'playoff_matchups' not in st.session_state:
        st.session_state.playoff_matchups = None
        
    if 'daily_simulations' not in st.session_state:
        st.session_state.daily_simulations = None
        
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "First Round"

# Display sidebar navigation and information
def display_sidebar():
    """Display sidebar navigation and app information"""
    st.sidebar.title(f"{APP_TITLE} v{APP_VERSION}")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "First Round": "Current Playoff Matchups",
        "Simulation Results": "Playoff Simulation Results",
        "Head to Head": "Team Comparison Tool",
        "Sim Bracket": "Interactive Bracket Simulation",
        "About": "About the App"
    }
    
    # Add debug page if enabled
    if ENABLE_DEBUG_UI:
        pages["Debug"] = "Debug & Diagnostics"
    
    # Page selection
    selected_page = st.sidebar.radio("Navigation", pages.keys(), 
                                    format_func=lambda x: pages[x],
                                    index=list(pages.keys()).index(st.session_state.current_page)
                                    if st.session_state.current_page in pages.keys() else 0)
    
    st.session_state.current_page = selected_page
    
    # Data controls section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Controls")
    
    # Get cache status
    try:
        cache_status = get_cache_status()
        if not isinstance(cache_status, dict):
            raise ValueError("Invalid cache_status format: Expected a dictionary.")
    except Exception as e:
        st.sidebar.error("Error retrieving cache status.")
        logger.error(f"Error in get_cache_status(): {str(e)}", exc_info=True)
        cache_status = {
            'is_fresh': False,
            'last_refresh': None,
            'next_refresh': None,
            'refresh_needed': True
        }
    
    # Show data refresh information
    if cache_status.get('last_refresh') is not None:
        last_refresh = cache_status['last_refresh']
        st.sidebar.info(f"Last data refresh: {last_refresh.strftime('%Y-%m-%d %H:%M')} {REFRESH_TIMEZONE.zone}")
    else:
        st.sidebar.warning("Data not yet loaded")
    
    # Show next scheduled refresh
    next_refresh = cache_status.get('next_refresh')
    if next_refresh:
        st.sidebar.info(f"Next scheduled refresh: {next_refresh.strftime('%Y-%m-%d %H:%M')} {REFRESH_TIMEZONE.zone}")
    
    # Manual refresh button
    if st.sidebar.button("Refresh Data Now"):
        with st.spinner("Refreshing data..."):
            try:
                current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
                season_str = f"{current_season}{current_season+1}"
                
                # Force data refresh
                update_daily_data(DATA_DIR, current_season, season_str, force=True)
                
                if 'last_data_refresh' in st.session_state and st.session_state.last_data_refresh:
                    st.sidebar.success(f"Data refreshed successfully at {st.session_state.last_data_refresh.strftime('%H:%M:%S')}")
                else:
                    st.sidebar.error("Data refresh failed")
            except Exception as e:
                st.sidebar.error(f"Error refreshing data: {str(e)}")
                # Log the error
                with open(os.path.join(DATA_DIR, "error_log.txt"), 'a') as f:
                    f.write(f"{get_current_time().isoformat()}: Data refresh error: {str(e)}\n{traceback.format_exc()}\n")
    
    # App information and links
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Created by: [Your Name](https://github.com/yourusername)  
        [GitHub Repository]({GITHUB_URL})
        """
    )

# Ensure required folders exist
def ensure_folders_exist():
    """Ensure all required data folders exist"""
    folders = [DATA_DIR, MODEL_DIR]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# Load models
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_cached_models():
    """Load machine learning models with caching"""
    try:
        from streamlit_app.utils.model_utils import load_models
        from streamlit_app.config import MODEL_DIR
        
        st.write("Loading NHL playoff prediction models...")
        models = load_models(MODEL_DIR)
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Main application function
def main():
    """Main application entry point"""
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Ensure folders exist
    ensure_folders_exist()
    
    # Ensure models are loaded and stored in session state
    if 'models' not in st.session_state:
        st.session_state.models = load_cached_models()
    
    # Display sidebar
    display_sidebar()
    
    # Refresh data if needed
    try:
        if should_refresh_data():
            # Get current season details
            current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
            season_str = f"{current_season}{current_season+1}"
            
            with st.spinner("Updating NHL data..."):
                update_daily_data(DATA_DIR, current_season, season_str)
        
        # Verify team data is loaded
        if st.session_state.team_data is None:
            with st.spinner("Loading team data..."):
                team_data = load_team_data(DATA_DIR)
                if team_data is not None and not team_data.empty:
                    st.session_state.team_data = team_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Log the error
        with open(os.path.join(DATA_DIR, "error_log.txt"), 'a') as f:
            f.write(f"{get_current_time().isoformat()}: App startup error: {str(e)}\n{traceback.format_exc()}\n")
    
    # Display the selected page
    try:
        if st.session_state.current_page == "First Round":
            # Import inside the function to avoid circular imports
            from streamlit_app.pages.first_round import show_first_round
            show_first_round()
            
        elif st.session_state.current_page == "Simulation Results":
            from streamlit_app.pages.simulation_results import show_simulation_results
            show_simulation_results()
            
        elif st.session_state.current_page == "Head to Head":
            from streamlit_app.pages.head_to_head import show_head_to_head
            show_head_to_head()
            
        elif st.session_state.current_page == "Sim Bracket":
            from streamlit_app.pages.sim_bracket import show_bracket_simulation
            show_bracket_simulation()
            
        elif st.session_state.current_page == "About":
            from streamlit_app.pages.about import show_about
            show_about()
            
        elif st.session_state.current_page == "Debug" and ENABLE_DEBUG_UI:
            from streamlit_app.pages.debug import show_debug
            show_debug()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.error("See details below:")
        st.code(traceback.format_exc())
        
        # Log the error
        with open(os.path.join(DATA_DIR, "error_log.txt"), 'a') as f:
            f.write(f"{get_current_time().isoformat()}: Page error ({st.session_state.current_page}): {str(e)}\n{traceback.format_exc()}\n")

# Run the application
if __name__ == "__main__":
    main()
