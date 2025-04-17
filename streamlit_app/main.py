import streamlit as st
import os
import time
from datetime import datetime, timedelta
from PIL import Image
import pytz  # Add pytz for timezone handling

# Import modules
import data_handlers
import model_utils
import simulation
import visualization
# Change import path to use app_pages instead of pages
from app_pages import first_round, simulation_results, head_to_head, sim_bracket, about

# Set page configuration
st.set_page_config(
    page_title="NHL Playoff Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  # Remove default menu items
)

# Create folders if they don't exist
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Constants
current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
season_str = f"{current_season}{current_season+1}"

# Global state for daily data refresh
if 'last_data_refresh' not in st.session_state:
    st.session_state.last_data_refresh = None

if 'last_simulation_refresh' not in st.session_state:
    st.session_state.last_simulation_refresh = None

if 'daily_simulations' not in st.session_state:
    st.session_state.daily_simulations = None

if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

def add_debug_info(message):
    """Add debugging information to session state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.debug_info.append(f"{timestamp}: {message}")
    if len(st.session_state.debug_info) > 20:  # Keep only the last 20 messages
        st.session_state.debug_info.pop(0)

def convert_to_eastern(dt):
    """Convert a datetime object to Eastern Time"""
    if dt is None:
        return "Never"
    eastern = pytz.timezone('US/Eastern')
    if dt.tzinfo is None:
        # Assume UTC if no timezone is specified
        dt = dt.replace(tzinfo=pytz.UTC)
    eastern_time = dt.astimezone(eastern)
    return eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

def main():
    # App title and description
    st.title("NHL Playoff Predictor")
    st.write("Predict playoff outcomes based on team statistics and advanced metrics")
    
    # Initialize timestamps if they're None
    if st.session_state.last_data_refresh is None:
        st.session_state.last_data_refresh = datetime.now()
        add_debug_info("Initialized data refresh timestamp")
    
    if st.session_state.last_simulation_refresh is None:
        st.session_state.last_simulation_refresh = datetime.now()
        add_debug_info("Initialized simulation refresh timestamp")
    
    # Display last update times instead of buttons for force refresh
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Last data refresh: {convert_to_eastern(st.session_state.last_data_refresh)}")
    with col2:
        st.info(f"Last simulation run: {convert_to_eastern(st.session_state.last_simulation_refresh)}")
    
    # Update daily data in the early morning once per day
    try:
        data_status = data_handlers.update_daily_data(data_folder, current_season, season_str)
        if data_status:
            st.session_state.last_data_refresh = datetime.now()
            add_debug_info("Daily data updated automatically")
    except Exception as e:
        add_debug_info(f"Error in daily data update: {str(e)}")
    
    # Run daily simulations (10,000 simulations once per day)
    try:
        sim_results = simulation.update_daily_simulations(n_simulations=100)
        if sim_results:
            st.session_state.last_simulation_refresh = datetime.now()
        add_debug_info("Daily simulations checked/updated")
    except Exception as e:
        add_debug_info(f"Error in daily simulation update: {str(e)}")
    
    # Load models BEFORE creating the sidebar
    try:
        model_data = model_utils.load_models(model_folder)
        add_debug_info(f"Models loaded: {', '.join(model_data.get('models', {}).keys())}")
    except Exception as e:
        model_data = {}
        add_debug_info(f"Error loading models: {str(e)}")
    
    # Hide default Streamlit elements that might be causing unwanted tabs
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp header {display: none;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Sidebar for navigation - Remove duplicate title/header items
    with st.sidebar:
        # Only use Radio control for navigation, no extra buttons or titles
        st.markdown("## Navigation")
        page = st.radio("", [
            "First Round Matchups", 
            "Full Simulation Results", 
            "Head-to-Head Comparison", 
            "Sim Your Own Bracket",
            "About"
        ])
        
        # Add model info to sidebar
        st.markdown("## Model Information")
        if model_data and 'models' in model_data and model_data['models']:
            st.write(f"Model mode: {model_data.get('mode', 'default')}")
            st.write(f"Home ice advantage: {model_data.get('home_ice_boost', 0.039)*100:.1f}%")
            st.write(f"Models loaded: {', '.join(model_data.get('models', {}).keys())}")
        else:
            st.error("No models loaded")
            add_debug_info("No models available in model_data")
        
        # Debug expander
        with st.expander("Debug Information"):
            if st.session_state.debug_info:
                for msg in st.session_state.debug_info:
                    st.text(msg)
            else:
                st.text("No debug information available")
    
    # Page routing
    if page == "First Round Matchups":
        first_round.display_first_round_matchups(model_data)
    elif page == "Full Simulation Results":
        simulation_results.display_simulation_results(model_data)
    elif page == "Head-to-Head Comparison":
        # Store model_data in session state and call without arguments
        st.session_state.model_data = model_data
        head_to_head.display_head_to_head_comparison()
    elif page == "Sim Your Own Bracket":
        # Store model_data in session state and call without arguments
        st.session_state.model_data = model_data
        sim_bracket.display_sim_your_bracket()
    elif page == "About":
        about.display_about_page(model_data)

if __name__ == "__main__":
    main()