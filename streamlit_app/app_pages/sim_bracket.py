import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time  # Add missing time import

# Import simulation module
import simulation
import data_handlers
import visualization  # Add visualization module
import visualization_utils  # Add visualization_utils module

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
