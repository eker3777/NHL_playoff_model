"""
Analysis utilities for NHL playoff simulation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def create_advancement_table(simulation_results):
    """Create a DataFrame showing team advancement probabilities.
    
    Args:
        simulation_results: Results from simulate_playoff_bracket
        
    Returns:
        DataFrame: Team advancement probabilities
    """
    # Extract team advancement data
    team_adv = simulation_results["team_advancement"]
    
    # Create records for DataFrame
    records = []
    for team, data in team_adv.items():
        record = {
            "teamAbbrev": team,
            "teamName": data["teamName"],
            "Round 1": 100.0,  # All playoff teams are in round 1
            "Round 2": data["advancement"]["2"] * 100,
            "Conf Finals": data["advancement"]["3"] * 100,
            "Cup Final": data["advancement"]["4"] * 100,
            "Champion": data["advancement"]["champion"] * 100,
            "Avg Games": data["avg_games"]
        }
        records.append(record)
    
    # Create DataFrame and sort by championship probability
    df = pd.DataFrame(records)
    return df.sort_values("Champion", ascending=False)

def plot_championship_odds(simulation_results, n_teams=10):
    """Create a horizontal bar chart of championship probabilities.
    
    Args:
        simulation_results: Results from simulate_playoff_bracket
        n_teams: Number of teams to show (default: 10)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create advancement table
    adv_table = create_advancement_table(simulation_results)
    
    # Select top teams
    top_teams = adv_table.head(n_teams)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bars
    bars = ax.barh(top_teams["teamName"], top_teams["Champion"])
    
    # Add labels and formatting
    ax.set_xlabel("Championship Probability (%)")
    ax.set_title("Stanley Cup Championship Odds")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x = width + 1
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", va="center")
    
    # Add decoration
    plt.tight_layout()
    
    return fig

def plot_advancement_probabilities(simulation_results, n_teams=16):
    """Create a stacked bar chart showing round-by-round advancement.
    
    Args:
        simulation_results: Results from simulate_playoff_bracket
        n_teams: Number of teams to show (default: 16)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create advancement table
    adv_table = create_advancement_table(simulation_results)
    
    # Select teams to show
    teams = adv_table.head(n_teams)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare data for stacked bar chart
    rounds = ["Round 2", "Conf Finals", "Cup Final", "Champion"]
    bottom = np.zeros(len(teams))
    
    # Colors for different rounds
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Create stacked bars
    for i, (round_name, color) in enumerate(zip(rounds, colors)):
        ax.barh(teams["teamName"], teams[round_name], left=bottom, 
                height=0.7, label=round_name, color=color)
        bottom += teams[round_name]
    
    # Add formatting
    ax.set_xlabel("Probability (%)")
    ax.set_title("Round-by-Round Playoff Advancement Probabilities")
    ax.legend(loc="upper right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    
    # Sort teams by championship probability
    ax.invert_yaxis()  # Put highest probability at the top
    
    plt.tight_layout()
    
    return fig

def display_matchup_probabilities(simulation_results, round_num):
    """Create a DataFrame of potential matchups for a given round.
    
    Args:
        simulation_results: Results from simulate_playoff_bracket
        round_num: Round number (2, 3, or 4)
        
    Returns:
        DataFrame: Matchup probabilities
    """
    # Extract matchup data
    round_key = str(round_num)
    matchups = simulation_results["matchup_probabilities"].get(round_key, {})
    
    # Create records for DataFrame
    records = []
    for matchup_key, data in matchups.items():
        record = {
            "Team 1": data["team1_name"],
            "Team 2": data["team2_name"],
            "Probability": data["probability"] * 100,
            "Team 1 Win %": data["team1_win_pct"] * 100,
            "Avg Games": data["avg_games"]
        }
        records.append(record)
    
    # Create DataFrame and sort by probability
    df = pd.DataFrame(records)
    if not df.empty:
        return df.sort_values("Probability", ascending=False)
    return df

def analyze_simulation_results(simulation_results):
    """Create a comprehensive analysis of simulation results for Streamlit.
    
    Args:
        simulation_results: Results from simulate_playoff_bracket
    """
    st.subheader("Playoff Advancement Probabilities")
    
    # Display advancement table
    adv_table = create_advancement_table(simulation_results)
    st.dataframe(adv_table.style.format({
        "Round 2": "{:.1f}%",
        "Conf Finals": "{:.1f}%",
        "Cup Final": "{:.1f}%",
        "Champion": "{:.1f}%",
        "Avg Games": "{:.1f}"
    }))
    
    # Plot championship odds
    st.subheader("Stanley Cup Championship Odds")
    fig1 = plot_championship_odds(simulation_results)
    st.pyplot(fig1)
    
    # Plot advancement probabilities
    st.subheader("Round-by-Round Advancement")
    fig2 = plot_advancement_probabilities(simulation_results)
    st.pyplot(fig2)
    
    # Display potential matchups
    st.subheader("Potential Second Round Matchups")
    r2_matchups = display_matchup_probabilities(simulation_results, 2)
    st.dataframe(r2_matchups.style.format({
        "Probability": "{:.1f}%",
        "Team 1 Win %": "{:.1f}%",
        "Avg Games": "{:.1f}"
    }))
    
    st.subheader("Potential Conference Finals Matchups")
    r3_matchups = display_matchup_probabilities(simulation_results, 3)
    st.dataframe(r3_matchups.style.format({
        "Probability": "{:.1f}%",
        "Team 1 Win %": "{:.1f}%",
        "Avg Games": "{:.1f}"
    }))
    
    st.subheader("Potential Stanley Cup Final Matchups")
    r4_matchups = display_matchup_probabilities(simulation_results, 4)
    st.dataframe(r4_matchups.style.format({
        "Probability": "{:.1f}%",
        "Team 1 Win %": "{:.1f}%",
        "Avg Games": "{:.1f}"
    }))
    
    # Show simulation metadata
    st.subheader("Simulation Information")
    st.write(f"Number of simulations: {simulation_results['n_simulations']:,}")
    st.write(f"Simulation timestamp: {simulation_results['timestamp']}")
