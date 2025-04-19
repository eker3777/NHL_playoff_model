import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit_app.utils.data_handlers as data_handlers
import streamlit_app.utils.model_utils as model_utils
import os
import streamlit_app.utils.visualization as visualization
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import HOME_ICE_ADVANTAGE and SERIES_LENGTH_DISTRIBUTION from config
from streamlit_app.config import (
    HOME_ICE_ADVANTAGE,
    SERIES_LENGTH_DISTRIBUTION,
    # ...other imports you already have...
)

# Add a caching decorator for expensive operations
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_matchup_predictions(matchup_df, model_data):
    """Cache matchup predictions to avoid recalculation on every page load"""
    # Get model predictions
    ensemble_prob_raw, raw_lr_prob, raw_xgb_prob = model_utils.predict_series_winner(matchup_df, model_data)
    
    # Replace hardcoded value with imported constant
    # Replace: home_ice_factor = 0.038
    home_ice_factor = HOME_ICE_ADVANTAGE
    
    # Get final prediction with home ice boost applied
    matchup_prediction = model_utils.predict_matchup(matchup_df, model_data)
    ensemble_prob = matchup_prediction['home_win_prob']
    
    # Get simulation results
    series_results = model_utils.predict_series(matchup_df, model_data, n_simulations=1000)
    sim_prob = series_results['win_probability']
    
    return {
        'raw_lr_prob': raw_lr_prob,
        'raw_xgb_prob': raw_xgb_prob,
        'ensemble_prob_raw': ensemble_prob_raw,
        'ensemble_prob': ensemble_prob,
        'sim_prob': sim_prob,
        'series_results': series_results
    }

# Cache the playoff matchups determination
@st.cache_data(ttl=3600, show_spinner=False)
def get_playoff_matchups(standings_df):
    """Cache the playoff matchups to avoid recalculation"""
    return data_handlers.determine_playoff_teams(standings_df)

# Cache the matchup data creation
@st.cache_data(ttl=3600, show_spinner=False)
def get_matchup_data(top_seed, bottom_seed, team_data):
    """Cache the matchup data to avoid recalculation"""
    return data_handlers.create_matchup_data(top_seed, bottom_seed, team_data)

def display_first_round_matchups(model_data=None):
    """Display the first round matchups page"""
    st.title("NHL Playoff First Round Matchups")
    
    # Add debug mode toggle for developers - just add this at the top
    with st.sidebar.expander("Developer Options", expanded=False):
        debug_mode = st.checkbox("Debug Mode", value=False)
        st.session_state.debug_mode = debug_mode
    
    # Get current season
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    season_str = f"{current_season}{current_season+1}"
    
    # Get data folder path
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Check if data is already in session state before loading from disk
    if 'standings_df' in st.session_state and 'team_data' in st.session_state:
        standings_df = st.session_state.standings_df
        team_data = st.session_state.team_data
        if st.session_state.debug_mode:
            st.write("Using data from session state")
    else:
        # Load data
        try:
            standings_df = data_handlers.load_data(f"standings_{season_str}.csv", data_folder)
            team_data = data_handlers.load_data(f"team_data_{season_str}.csv", data_folder)
            
            # Store in session state for future use
            if standings_df is not None and not standings_df.empty:
                st.session_state.standings_df = standings_df
            if team_data is not None and not team_data.empty:
                st.session_state.team_data = team_data
            
            if standings_df is None or standings_df.empty or team_data is None or team_data.empty:
                st.error("Required data not available. Please make sure data has been fetched.")
                
                if st.button("Fetch Data Now"):
                    with st.spinner("Fetching NHL data..."):
                        success = data_handlers.update_daily_data(data_folder, current_season, season_str, force=True)
                        if success:
                            st.success("Data fetched successfully! Reloading...")
                            # Load the newly fetched data
                            standings_df = data_handlers.load_data(f"standings_{season_str}.csv", data_folder)
                            team_data = data_handlers.load_data(f"team_data_{season_str}.csv", data_folder)
                            # Update session state
                            if standings_df is not None and not standings_df.empty:
                                st.session_state.standings_df = standings_df
                            if team_data is not None and not team_data.empty:
                                st.session_state.team_data = team_data
                            st.rerun()  # Changed from st.experimental_rerun()
                        else:
                            st.error("Failed to fetch data. Please check logs for details.")
                return
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)
            return
    
    # Use cached function to determine playoff matchups
    try:
        playoff_matchups = get_playoff_matchups(standings_df)
        
        if not playoff_matchups:
            st.warning("Could not determine playoff matchups with the current standings.")
            return
        
        # Display the matchups by conference
        for conference, matchups in playoff_matchups.items():
            st.header(f"{conference} Conference Matchups")
            
            # Loop through matchups one at a time (one matchup per row for more space)
            for series_id, matchup in matchups.items():
                top_seed = matchup['top_seed']
                bottom_seed = matchup['bottom_seed']
                
                # Use cached function to create matchup data
                matchup_df = get_matchup_data(top_seed, bottom_seed, team_data)
                
                st.subheader(f"{top_seed['teamName']} vs {bottom_seed['teamName']}")
                
                # Create two columns: team info, probabilities chart
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show team seeds
                    top_seed_label = f"Division #{top_seed.get('division_rank', '')}" if 'division_rank' in top_seed else f"Wildcard #{top_seed.get('wildcard_rank', '')}"
                    bottom_seed_label = f"Division #{bottom_seed.get('division_rank', '')}" if 'division_rank' in bottom_seed else f"Wildcard #{bottom_seed.get('wildcard_rank', '')}"
                    
                    st.write(f"**{top_seed['teamAbbrev']}** ({top_seed_label}) vs **{bottom_seed['teamAbbrev']}** ({bottom_seed_label})")
                
                    # Show team logos if available
                    if 'teamLogo' in top_seed and 'teamLogo' in bottom_seed:
                        logo_cols = st.columns(2)
                        with logo_cols[0]:
                            st.image(top_seed['teamLogo'], width=100)
                        with logo_cols[1]:
                            st.image(bottom_seed['teamLogo'], width=100)

                with col2:
                    try:
                        # Use cached function to get model predictions
                        predictions = get_matchup_predictions(matchup_df, model_data)
                        
                        raw_lr_prob = predictions['raw_lr_prob']
                        raw_xgb_prob = predictions['raw_xgb_prob']
                        ensemble_prob_raw = predictions['ensemble_prob_raw']
                        ensemble_prob = predictions['ensemble_prob']
                        sim_prob = predictions['sim_prob']
                        series_results = predictions['series_results']
                        
                        # Log probabilities for debugging
                        if st.session_state.get('debug_mode', False):
                            st.write(f"Raw LR: {raw_lr_prob:.4f}, Raw XGB: {raw_xgb_prob:.4f}")
                            st.write(f"Raw Ensemble: {ensemble_prob_raw:.4f}, With HI: {ensemble_prob:.4f}")
                            st.write(f"Simulation: {sim_prob:.4f}")
                    except Exception as e:
                        st.warning(f"Error getting model predictions: {str(e)}")
                        # Use fallback values
                        raw_lr_prob, raw_xgb_prob = 0.5, 0.5
                        ensemble_prob = 0.5
                        sim_prob = 0.5
                        series_results = {"win_distribution": {}}
                    
                    # Use our improved visualization function with team abbrevs for team colors
                    visualization.plot_head_to_head_probabilities(
                        team1=top_seed["teamName"],
                        team2=bottom_seed["teamName"],
                        lr_prob=raw_lr_prob,
                        xgb_prob=raw_xgb_prob,
                        ensemble_prob=ensemble_prob,
                        sim_prob=sim_prob,
                        team1_abbrev=top_seed["teamAbbrev"],  # Pass team abbreviation for proper colors
                        team2_abbrev=bottom_seed["teamAbbrev"]  # Pass team abbreviation for proper colors
                    )

                # Show the series length probabilities in a table
                st.write("### Series Length Probabilities")
                
                if 'win_distribution' in series_results:
                    # Use the visualization function for series length table
                    visualization.plot_series_length_table(
                        win_distribution=series_results['win_distribution'],
                        top_team=top_seed["teamName"],
                        bottom_team=bottom_seed["teamName"]
                    )
                else:
                    st.write("Simulation data not available")
                
                # Display team metrics comparison (shown by default)
                st.write("### Team Stats Comparison")
                
                # Get team data for both teams
                team1_data = team_data[team_data['teamName'] == top_seed['teamName']].iloc[0] if any(team_data['teamName'] == top_seed['teamName']) else None
                team2_data = team_data[team_data['teamName'] == bottom_seed['teamName']].iloc[0] if any(team_data['teamName'] == bottom_seed['teamName']) else None
                
                if team1_data is not None and team2_data is not None:
                    # Define fixed metrics to display in table format with clear, descriptive names
                    metrics = [
                        ("Points Percentage", "pointPctg", True), 
                        ("Goal Differential per Game", "goalDifferential/gamesPlayed", True), 
                        ("5v5 Expected Goals %", "xGoalsPercentage", True),
                        ("Power Play %", "PP%", True),
                        ("Penalty Kill %", "PK%", True),
                        ("Face Off %", "FO%", True),
                        ("Goals Saved Above Expected per 82", "goalsSavedAboveExpectedPer82", True),
                        ("Goals Scored Above Expected per 82", "goalsScoredAboveExpectedPer82", False),
                        ("5v5 Hits %", "possAdjHitsPctg", True),
                        ("5v5 Takeaways %", "possAdjTakeawaysPctg", True),
                        ("5v5 Giveaways %", "possTypeAdjGiveawaysPctg", False),
                        ("5v5 Rebound xGoals %", "reboundxGoalsPctg", True)
                    ]
                    
                    # Display metrics table
                    display_metrics_table(team1_data, team2_data, metrics, top_seed['teamName'], bottom_seed['teamName'])
                else:
                    st.error("Team data not available for comparison")
                
                st.markdown("---")  # Add a separator between matchups
    
    except Exception as e:
        st.error(f"Error displaying matchups: {str(e)}")
        st.exception(e)
        
        if st.button("Refresh Data"):
            with st.spinner("Refreshing NHL data..."):
                success = data_handlers.update_daily_data(data_folder, current_season, season_str, force=True)
                if success:
                    # Clear session state to force reload
                    if 'standings_df' in st.session_state:
                        del st.session_state.standings_df
                    if 'team_data' in st.session_state:
                        del st.session_state.team_data
                    # Also clear the cache to force recalculation
                    st.cache_data.clear()
                    st.success("Data refreshed successfully!")
                    st.rerun()  # Changed from st.experimental_rerun()
                else:
                    st.error("Failed to refresh data.")

def display_metrics_table(team1_data, team2_data, metrics, team1_name, team2_name):
    """Display a table of metrics with the better value highlighted."""
    # Create data for the table
    table_data = []
    
    for display_name, field_name, higher_is_better in metrics:
        # Get values based on the field name
        t1_value = 0
        t2_value = 0
        
        # Special handling for calculated fields
        if '/' in field_name:
            # Example: "goalDifferential/gamesPlayed" - split and calculate
            numerator, denominator = field_name.split('/')
            
            if numerator in team1_data and denominator in team1_data:
                try:
                    t1_value = float(team1_data[numerator]) / max(1, float(team1_data[denominator]))
                except (ValueError, TypeError, ZeroDivisionError):
                    t1_value = 0
                    
            if numerator in team2_data and denominator in team2_data:
                try:
                    t2_value = float(team2_data[numerator]) / max(1, float(team2_data[denominator]))
                except (ValueError, TypeError, ZeroDivisionError):
                    t2_value = 0
        else:
            # Normal fields
            if field_name in team1_data:
                try:
                    t1_value = float(team1_data[field_name])
                except (ValueError, TypeError):
                    t1_value = 0
            
            if field_name in team2_data:
                try:
                    t2_value = float(team2_data[field_name])
                except (ValueError, TypeError):
                    t2_value = 0
        
        # Convert percentages to 0-100 scale if needed
        if "Percentage" in display_name or "%" in display_name:
            if 0 < t1_value <= 1:
                t1_value *= 100
            if 0 < t2_value <= 1:
                t2_value *= 100
        
        # Format the values for display
        if "Percentage" in display_name or "%" in display_name:
            t1_display = f"{t1_value:.1f}%"
            t2_display = f"{t2_value:.1f}%"
        elif "per Game" in display_name:
            t1_display = f"{t1_value:.2f}"
            t2_display = f"{t2_value:.2f}"
        elif "per 82" in display_name:
            t1_display = f"{t1_value:.1f}"
            t2_display = f"{t2_value:.1f}"
        else:
            t1_display = f"{t1_value:.2f}"
            t2_display = f"{t2_value:.2f}"
        
        # Determine which value is better and add highlighting
        if higher_is_better:
            t1_better = t1_value > t2_value
        else:
            t1_better = t1_value < t2_value
        
        if t1_better and abs(t1_value - t2_value) > 0.01:
            t1_display = f"**{t1_display}**"
        elif not t1_better and abs(t2_value - t1_value) > 0.01:
            t2_display = f"**{t2_display}**"
        
        # Add row to table
        table_data.append([display_name, t1_display, t2_display])
    
    # Create DataFrame and display as table
    df = pd.DataFrame(table_data, columns=["Metric", team1_name, team2_name])
    st.table(df)

def app():
    """Main entry point for the first round page"""
    # Get models from session state
    models = None
    if 'models' in st.session_state:
        models = st.session_state.models
    
    # Call the display function
    display_first_round_matchups(model_data=models)

def show_first_round():
    """Entry point for first round page called from app.py"""
    app()

if __name__ == "__main__":
    # This allows the page to be run directly for testing
    display_first_round_matchups()
