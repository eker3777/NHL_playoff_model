import streamlit as st
import pandas as pd
import os
import model_utils
import data_handlers
import visualization
import numpy as np
from datetime import datetime  # Add missing import

def display_head_to_head_comparison():
    """Display head-to-head comparison between two selected teams."""
    st.header("Head-to-Head Playoff Comparison")
    
    # Get the model data from session state
    if 'model_data' not in st.session_state or st.session_state.model_data is None:
        st.error("No model data available. Please load the models first from the main page.")
        return
    
    model_data = st.session_state.model_data
    
    # Extract team data from model_data
    # First try to get from model_data directly
    team_data = model_data.get('team_data')
    
    # If not found or empty, try to load it from the data folder
    if team_data is None or (isinstance(team_data, pd.DataFrame) and team_data.empty):
        try:
            # Get current season
            current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
            season_str = f"{current_season}{current_season+1}"
            
            # Get data folder path
            data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            
            # Try to load team data from file
            team_data = data_handlers.load_data(f"team_data_{season_str}.csv", data_folder)
            
            if team_data is None or team_data.empty:
                st.error("Could not load team data from file. Please ensure data has been fetched.")
                
                # Offer a button to fetch data
                if st.button("Fetch Data Now"):
                    with st.spinner("Fetching NHL data..."):
                        success = data_handlers.update_daily_data(data_folder, current_season, season_str, force=True)
                        if success:
                            st.success("Data fetched successfully! Reloading...")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to fetch data. Please check logs for details.")
                return
        except Exception as e:
            st.error(f"Error loading team data: {str(e)}")
            return
    
    if team_data is None or isinstance(team_data, pd.DataFrame) and team_data.empty:
        st.error("No team data available for comparison. Please ensure data is loaded.")
        return
    
    # Explanatory note about home/away selection
    st.info("**Note:** The first team selected is considered the **Home Team** and the second is the **Away Team**. " 
            "This can affect home-ice advantage calculations.")
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Home Team")
        team1_name = st.selectbox("Select Home Team", 
                                 sorted(team_data['teamName'].unique()),
                                 key="team1_selector")
        team1_data = team_data[team_data['teamName'] == team1_name].iloc[0]
        team1_abbrev = team1_data['teamAbbrev']
        
        # Show team logo if available
        if 'teamLogo' in team1_data and team1_data['teamLogo']:
            logo = visualization.load_team_logo(team1_data['teamLogo'])
            if logo:
                st.image(logo, width=100)
    
    with col2:
        st.markdown("### Away Team")
        # Filter out already selected team
        available_teams = sorted([t for t in team_data['teamName'].unique() if t != team1_name])
        
        team2_name = st.selectbox("Select Away Team",
                                 available_teams,
                                 key="team2_selector")
        team2_data = team_data[team_data['teamName'] == team2_name].iloc[0]
        team2_abbrev = team2_data['teamAbbrev']
        
        # Show team logo if available
        if 'teamLogo' in team2_data and team2_data['teamLogo']:
            logo = visualization.load_team_logo(team2_data['teamLogo'])
            if logo:
                st.image(logo, width=100)
    
    # Create a separator
    st.write("---")
    
    # Create matchup data
    try:
        # Set up home and away teams
        home_team = {
            'teamName': team1_name,
            'teamAbbrev': team1_abbrev
        }
        
        away_team = {
            'teamName': team2_name,
            'teamAbbrev': team2_abbrev
        }
        
        # Add additional data from team_data if available
        for key in team1_data.keys():
            if key not in home_team and not pd.isna(team1_data[key]):
                home_team[key] = team1_data[key]
        
        for key in team2_data.keys():
            if key not in away_team and not pd.isna(team2_data[key]):
                away_team[key] = team2_data[key]
        
        # Ensure division_rank is present for matchup data creation
        if 'division_rank' not in home_team:
            home_team['division_rank'] = 1
        if 'division_rank' not in away_team:
            away_team['division_rank'] = 2
            
        # Create matchup dataframe
        matchup_df = model_utils.create_matchup_data(home_team, away_team, team_data)
        
        if not matchup_df.empty:
            # Check if teams appeared in playoff simulations
            simulation_results = None
            if 'simulation_results' in model_data:
                simulation_results = model_data['simulation_results']
                
            # Show simulation matchup status if available
            show_simulation_status(team1_name, team2_name, simulation_results)
            
            # Make predictions
            try:
                # Pass correct model data structure to prediction function
                prediction_results = model_utils.predict_series_winner(matchup_df, model_data)
                
                # Handle different return formats (tuple or dict)
                if isinstance(prediction_results, tuple) and len(prediction_results) == 3:
                    ensemble_prob, lr_prob, xgb_prob = prediction_results
                elif isinstance(prediction_results, dict):
                    ensemble_prob = prediction_results.get('ensemble', 0.5)
                    lr_prob = prediction_results.get('logistic', 0.5)
                    xgb_prob = prediction_results.get('xgboost', 0.5)
                else:
                    st.error("Unexpected prediction result format")
                    return
                
                # Simulate series
                win_distribution = simulate_series(ensemble_prob)
                
                # Calculate simulated probability based on win distribution
                home_wins = sum(win_distribution[k] for k in ['4-0', '4-1', '4-2', '4-3'])
                away_wins = sum(win_distribution[k] for k in ['0-4', '1-4', '2-4', '3-4'])
                sim_prob = home_wins / (home_wins + away_wins) if (home_wins + away_wins) > 0 else 0.5
                
                # Get simulation matchup probability if available
                simulation_prob = get_simulation_matchup_probability(team1_name, team2_name, simulation_results)
                
                # Set up tabs for different comparison views
                tabs = st.tabs([
                    "Series Prediction", 
                    "Team Stats Comparison", 
                    "Series Details"
                ])
                
                # Series Prediction Tab
                with tabs[0]:
                    st.subheader("Series Prediction")
                    
                    # Display the head to head probabilities chart with team colors
                    visualization.plot_head_to_head_probabilities(
                        team1=team1_name, 
                        team2=team2_name, 
                        lr_prob=lr_prob, 
                        xgb_prob=xgb_prob, 
                        ensemble_prob=ensemble_prob, 
                        sim_prob=sim_prob,
                        team1_abbrev=team1_abbrev,
                        team2_abbrev=team2_abbrev
                    )
                    
                    # Show key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        winner = team1_name if ensemble_prob > 0.5 else team2_name
                        win_pct = max(ensemble_prob, 1-ensemble_prob) * 100
                        st.metric("Predicted Winner", winner, f"{win_pct:.1f}% chance")
                    
                    with col2:
                        # Calculate expected games based on win distribution
                        total_games = 0
                        total_series = 0
                        for outcome, count in win_distribution.items():
                            games = int(outcome[0]) + int(outcome[2])
                            total_games += games * count
                            total_series += count
                        
                        avg_games = total_games / total_series if total_series > 0 else 0
                        st.metric("Expected Series Length", f"{avg_games:.2f} games")
                    
                    with col3:
                        # Most likely outcome
                        likely_outcome = max(win_distribution.items(), key=lambda x: x[1])
                        
                        if likely_outcome[0] in ['4-0', '4-1', '4-2', '4-3']:
                            # Convert the outcome to the actual series length
                            games_played = int(likely_outcome[0][0]) + int(likely_outcome[0][2])
                            outcome_text = f"{team1_name} in {games_played}"
                        else:
                            # For opponent wins, convert to proper series length
                            games_played = int(likely_outcome[0][0]) + int(likely_outcome[0][2])
                            outcome_text = f"{team2_name} in {games_played}"
                            
                        outcome_pct = likely_outcome[1] / sum(win_distribution.values()) * 100
                        st.metric("Most Likely Outcome", outcome_text, f"{outcome_pct:.1f}%")
                
                # Team Stats Comparison Tab
                with tabs[1]:
                    st.subheader("Team Stats Comparison")
                    
                    # Define fixed metrics to display in table format with clear, descriptive names
                    metrics = [
                        ("Points Percentage", "pointPctg", True),  # Changed from pointPctg to pointsPct
                        ("Goal Differential per Game", "goalDifferential/gamesPlayed", True),  # Changed from goalDifferential/gamesPlayed to goalDiff/G
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
                    
                    # Debug metrics availability if in debug mode
                    if st.session_state.get('debug_mode', False):
                        st.write("Available metrics in team1_data:")
                        gsae_metrics = [col for col in team1_data.keys() if 'goal' in str(col).lower() and 'save' in str(col).lower()]
                        gsoe_metrics = [col for col in team1_data.keys() if 'goal' in str(col).lower() and 'score' in str(col).lower()]
                        st.write(f"GSAE metrics: {gsae_metrics}")
                        st.write(f"GSOE metrics: {gsoe_metrics}")
                    
                    # Display metrics table
                    display_metrics_table(team1_data, team2_data, metrics, team1_name, team2_name)
                
                # Series Details Tab
                with tabs[2]:
                    st.subheader("Series Length Probabilities")
                    
                    # Show the series length probabilities
                    visualization.plot_series_length_table(win_distribution, team1_name, team2_name)
                    
                    # Add some commentary on key scenarios
                    st.write("### Key Scenarios")
                    
                    # Sweep probability
                    sweep_prob = (win_distribution['4-0'] + win_distribution['0-4']) / sum(win_distribution.values()) * 100
                    st.write(f"- **Sweep probability:** {sweep_prob:.1f}%")
                    
                    # 7-game series probability
                    game7_prob = (win_distribution['4-3'] + win_distribution['3-4']) / sum(win_distribution.values()) * 100
                    st.write(f"- **7-game series probability:** {game7_prob:.1f}%")
                    
                    # Home team wins in 7 probability
                    home_g7_prob = win_distribution['4-3'] / sum(win_distribution.values()) * 100
                    st.write(f"- **{team1_name} wins in 7 probability:** {home_g7_prob:.1f}%")
                    
                    # Away team wins in 7 probability
                    away_g7_prob = win_distribution['3-4'] / sum(win_distribution.values()) * 100
                    st.write(f"- **{team2_name} wins in 7 probability:** {away_g7_prob:.1f}%")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
        else:
            st.error("Unable to create matchup data for these teams")
    except Exception as e:
        st.error(f"Error in comparison: {str(e)}")
        st.info("Try selecting different teams or check if team data is complete")

def show_simulation_status(team1_name, team2_name, simulation_results):
    """Display whether the teams met in simulations and related information."""
    if simulation_results is None:
        st.warning("No simulation data available. Showing model predictions only.")
        return
    
    # Check if both teams made playoffs in simulations
    team1_in_playoffs = False
    team2_in_playoffs = False
    matchup_occurred = False
    
    # Check if teams made playoffs
    if 'playoff_teams' in simulation_results:
        playoff_teams = simulation_results['playoff_teams']
        team1_in_playoffs = any(team1_name in teams for teams in playoff_teams)
        team2_in_playoffs = any(team2_name in teams for teams in playoff_teams)
    
    # Check if teams faced each other
    if 'matchups' in simulation_results:
        matchups = simulation_results['matchups']
        matchup_occurred = any(
            (team1_name in matchup and team2_name in matchup) 
            for matchup in matchups
        )
    
    # Display status messages
    if not team1_in_playoffs and not team2_in_playoffs:
        st.warning(f"Neither {team1_name} nor {team2_name} made the playoffs in the simulations.")
    elif not team1_in_playoffs:
        st.warning(f"{team1_name} did not make the playoffs in the simulations.")
    elif not team2_in_playoffs:
        st.warning(f"{team2_name} did not make the playoffs in the simulations.")
    elif matchup_occurred:
        st.success(f"{team1_name} and {team2_name} faced each other in the playoff simulations!")
    else:
        st.info(f"Both teams made the playoffs in simulations, but they did not face each other.")

def get_simulation_matchup_probability(team1_name, team2_name, simulation_results):
    """Get the probability of team1 winning against team2 in simulations."""
    if simulation_results is None or 'matchup_results' not in simulation_results:
        return None
    
    matchup_results = simulation_results['matchup_results']
    
    # Look for the exact matchup
    for matchup, results in matchup_results.items():
        teams = matchup.split(' vs ')
        if len(teams) != 2:
            continue
            
        if (teams[0] == team1_name and teams[1] == team2_name):
            # Direct matchup found
            return results.get('win_percentage', None)
        elif (teams[1] == team1_name and teams[0] == team2_name):
            # Reverse matchup found, need to invert probability
            win_pct = results.get('win_percentage', None)
            return 1 - win_pct if win_pct is not None else None
    
    return None

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

def simulate_series(win_prob, n_simulations=10000):
    """Simulate a playoff series with the given win probability."""
    # Initialize counters
    win_distribution = {
        '4-0': 0, '4-1': 0, '4-2': 0, '4-3': 0,  # Higher seed wins
        '0-4': 0, '1-4': 0, '2-4': 0, '3-4': 0   # Lower seed wins
    }
    
    # Updated historical distribution of NHL playoff series outcomes
    # 4 games: 14.0%, 5 games: 24.3%, 6 games: 33.6%, 7 games: 28.1%
    total_percent = 14.0 + 24.3 + 33.6 + 28.1
    
    # Distribution for when higher seed wins
    higher_seed_outcome_dist = {
        '4-0': 14.0/total_percent, 
        '4-1': 24.3/total_percent, 
        '4-2': 33.6/total_percent, 
        '4-3': 28.1/total_percent
    }
    
    # Same distribution for when lower seed wins
    lower_seed_outcome_dist = {
        '0-4': 14.0/total_percent, 
        '1-4': 24.3/total_percent, 
        '2-4': 33.6/total_percent, 
        '3-4': 28.1/total_percent
    }
    
    # Ensure win_prob is a valid probability
    win_prob = max(0.0, min(1.0, win_prob))
    
    # Run simulations
    for _ in range(n_simulations):
        # Determine if higher seed wins the series
        higher_seed_wins_series = np.random.random() < win_prob
        
        if higher_seed_wins_series:
            # Select a series outcome based on historical distribution
            outcome = np.random.choice(
                ['4-0', '4-1', '4-2', '4-3'], 
                p=[
                    higher_seed_outcome_dist['4-0'], 
                    higher_seed_outcome_dist['4-1'],
                    higher_seed_outcome_dist['4-2'],
                    higher_seed_outcome_dist['4-3']
                ]
            )
            win_distribution[outcome] += 1
        else:
            # Select a series outcome for lower seed winning
            outcome = np.random.choice(
                ['0-4', '1-4', '2-4', '3-4'], 
                p=[
                    lower_seed_outcome_dist['0-4'], 
                    lower_seed_outcome_dist['1-4'],
                    lower_seed_outcome_dist['2-4'],
                    lower_seed_outcome_dist['3-4']
                ]
            )
            win_distribution[outcome] += 1
    
    return win_distribution
