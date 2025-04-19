import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time  # Add missing time import

# Import simulation module
import streamlit_app.models.simulation as simulation
import streamlit_app.utils.data_handlers as data_handlers
import streamlit_app.utils.visualization as visualization  # Add visualization module
import streamlit_app.utils.visualization_utils as visualization_utils  # Add visualization_utils module

# Import HOME_ICE_ADVANTAGE and SERIES_LENGTH_DISTRIBUTION from config
from streamlit_app.config import (
    HOME_ICE_ADVANTAGE,
    SERIES_LENGTH_DISTRIBUTION,
    # ...other imports you already have...
)

def display_sim_your_bracket():
    st.title("Simulate Your Own Bracket")
    st.write("Select matchups and see how your bracket would perform")
    
    # Get model_data from session state
    model_data = st.session_state.model_data
    
    # Extract needed data for simulation
    models = model_data.get('models', {})
    
    # Load team data using data_handlers
    # Use the correct function load_team_data() without arguments
    team_data = data_handlers.load_team_data()
    
    # Get current playoff matchups using the correct function load_current_playoff_matchups()
    # NOT get_playoff_matchups which doesn't exist
    playoff_matchups = data_handlers.load_current_playoff_matchups()
    
    # Check if data was loaded successfully
    if team_data.empty:
        st.error("Could not load team data. Please check the data files.")
        return
        
    if not playoff_matchups:
        st.error("Could not load playoff matchups. Please check the data files.")
        return
    
    # Add button to run the simulation
    if st.button("Run Playoff Bracket Simulation"):
        # Show spinner during simulation
        with st.spinner("Simulating playoff bracket..."):
            # Simulate a single bracket
            bracket_result = simulation.simulate_single_bracket(playoff_matchups, team_data, models)
            
            # Store in session state
            st.session_state.bracket_result = bracket_result
            
            # Add timestamp
            st.session_state.bracket_time = time.time()
    
    # Display bracket result if available
    if 'bracket_result' in st.session_state:
        # Show when it was generated
        if 'bracket_time' in st.session_state:
            time_diff = time.time() - st.session_state.bracket_time
            if time_diff < 60:
                st.success(f"Bracket generated {int(time_diff)} seconds ago")
            else:
                st.success(f"Bracket generated {int(time_diff/60)} minutes ago")
        
        # Display the detailed bracket results directly without requiring checkbox
        st.subheader("Playoff Bracket Results")
        visualization.create_simple_bracket_visual(st.session_state.bracket_result)
        
        # Show champion path if available - keep only this simple bullet point version
        if 'champion' in st.session_state.bracket_result and st.session_state.bracket_result['champion']:
            champion = st.session_state.bracket_result['champion']
            st.subheader(f"🏆 Champion: {champion.get('name', '')}")
            
            # Show champion's path through the playoffs
            if 'path' in champion and champion['path']:
                st.write("Path to the Stanley Cup:")
                for series in champion['path']:
                    st.write(f"• {series['round']}: Defeated {series['opponent']} {series['result']}")

def run_interactive_simulation(matchups, team_data, models):
    # ...existing code...
    
    # If there's any hardcoded home ice advantage value, replace it
    # For example: home_ice = 0.039
    home_ice = HOME_ICE_ADVANTAGE
    
    # ...existing code...

def simulate_series(home_team, away_team, team_data, models):
    # ...existing code...
    
    # Replace any hardcoded series length distribution
    # series_dist = [0.14, 0.24, 0.33, 0.29]  # example
    series_dist = SERIES_LENGTH_DISTRIBUTION
    
    # ...existing code...

def app():
    """Main entry point for the bracket simulator page"""
    # Get models from session state
    models = None
    if 'models' in st.session_state:
        models = st.session_state.models
    
    # Get team data from session state
    team_data = None
    if 'team_data' in st.session_state:
        team_data = st.session_state.team_data
    
    # Get playoff matchups from session state
    playoff_matchups = None
    if 'playoff_matchups' in st.session_state:
        playoff_matchups = st.session_state.playoff_matchups
    
    # Call the display function
    display_bracket_simulator(team_data=team_data, model_data=models, playoff_matchups=playoff_matchups)

def show_bracket_simulation():
    """Entry point for bracket simulation page called from app.py"""
    app()

if __name__ == "__main__":
    # This allows the page to be run directly for testing
    display_bracket_simulator()
