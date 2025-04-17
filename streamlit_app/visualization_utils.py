import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Any

def create_nhl_cmap():
    """Create an NHL-themed colormap (blue to white to red)."""
    # NHL colors (blue/white/red theme)
    colors = [(0.09, 0.27, 0.6),  # Dark Blue
             (1, 1, 1),          # White
             (0.76, 0.09, 0.18)] # Dark Red
    
    return LinearSegmentedColormap.from_list('nhl_cmap', colors, N=100)

def create_comparison_metrics() -> Dict[str, List[Tuple[str, str]]]:
    """Create dictionary of metric sets for team comparisons.
    
    Returns:
        Dict with predefined metric sets
    """
    # Define standard comparison metrics
    metrics = {
        'basic': [
            ('Points', 'points'),
            ('Goal Differential', 'goalDifferential'),
            ('Goals For', 'goalsFor'),
            ('Goals Against', 'goalsAgainst')
        ],
        'special_teams': [
            ('Power Play %', 'PP%'),
            ('Penalty Kill %', 'PK%'),
            ('Power Play Goals', 'powerPlayGoals'),
            ('Shorthanded Goals', 'shortHandedGoals')
        ],
        'advanced': [
            ('Corsi %', 'corsiPercentage'),
            ('Fenwick %', 'fenwickPercentage'),
            ('Expected Goals %', 'xGoalsPercentage'),
            ('Goal Diff/Game', 'goalDiff/G'),
            ('Adj Goals Saved Above X/60', 'adjGoalsSavedAboveX/60')
        ],
        'playoff_history': [
            ('Playoff Performance Score', 'playoff_performance_score'),
            ('Weighted Playoff Wins', 'weighted_playoff_wins'),
            ('Weighted Playoff Rounds', 'weighted_playoff_rounds')
        ]
    }
    
    return metrics

def get_valid_metrics(team_data: pd.DataFrame, metric_set: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter metrics to only include those available in the dataset.
    
    Args:
        team_data (DataFrame): Team stats DataFrame
        metric_set (list): List of (display_name, column_name) tuples to filter
        
    Returns:
        List of valid metrics
    """
    if team_data is None or team_data.empty:
        return []
        
    columns = team_data.columns
    return [(name, col) for name, col in metric_set if col in columns]

def format_percentage(value: float) -> str:
    """Format a decimal value as a percentage string.
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted percentage
    """
    return f"{value * 100:.1f}%"

def create_scatter_comparison(team_data: pd.DataFrame, x_metric: str, y_metric: str, 
                             x_label: str = None, y_label: str = None, 
                             highlight_teams: List[str] = None):
    """Create a scatter plot comparing two metrics across all teams.
    
    Args:
        team_data (DataFrame): Team data
        x_metric (str): Column name for x-axis
        y_metric (str): Column name for y-axis
        x_label (str): Label for x-axis (defaults to x_metric)
        y_label (str): Label for y-axis (defaults to y_metric)
        highlight_teams (list): List of teams to highlight
    """
    if x_metric not in team_data.columns or y_metric not in team_data.columns:
        st.warning(f"Metrics {x_metric} and/or {y_metric} not found in team data")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all teams
    ax.scatter(team_data[x_metric], team_data[y_metric], 
              color='gray', alpha=0.5, s=40)
    
    # Add team labels
    for i, row in team_data.iterrows():
        team_abbr = row['teamAbbrev']
        x_val = row[x_metric] 
        y_val = row[y_metric]
        
        # Highlight specified teams
        if highlight_teams and team_abbr in highlight_teams:
            ax.scatter(x_val, y_val, color='red', s=100, zorder=10)
            ax.text(x_val, y_val, team_abbr, fontsize=12, 
                   ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(x_val, y_val, team_abbr, fontsize=8, 
                   ha='center', va='bottom', alpha=0.7)
    
    # Set labels
    ax.set_xlabel(x_label or x_metric)
    ax.set_ylabel(y_label or y_metric)
    
    # Add title
    ax.set_title(f"{x_label or x_metric} vs {y_label or y_metric}")
    
    # Add reference lines at median values
    ax.axhline(y=team_data[y_metric].median(), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=team_data[x_metric].median(), color='gray', linestyle='--', alpha=0.3)
    
    return fig

def plot_wins_by_round(results_df: pd.DataFrame, team_abbrev: str):
    """Plot stacked bar chart showing team's progression through rounds.
    
    Args:
        results_df (DataFrame): Simulation results DataFrame
        team_abbrev (str): Team abbreviation to plot
    """
    # Filter for the specified team
    team_data = results_df[results_df['teamAbbrev'] == team_abbrev]
    
    if team_data.empty:
        st.warning(f"No data found for team {team_abbrev}")
        return
        
    team_name = team_data['teamName'].iloc[0]
    
    # Extract round probabilities
    rounds = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
    round_names = ['First Round', 'Second Round', 'Conf Finals', 'Finals', 'Champion']
    probs = [team_data[round_col].iloc[0] * 100 for round_col in rounds]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(round_names, probs)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Customize plot
    ax.set_ylim(0, max(probs) * 1.2)  # Add 20% headroom
    ax.set_title(f"{team_name} Playoff Advancement Probabilities")
    ax.set_ylabel("Probability (%)")
    
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any, Union

# Set visual styles for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# -----------------------------------------------------------------------
# Data Processing Utilities
# -----------------------------------------------------------------------

def calculate_win_percentages(data: pd.DataFrame, 
                             team_col: str = 'teamAbbrev', 
                             win_col: str = 'wins', 
                             games_col: str = 'gamesPlayed') -> pd.DataFrame:
    """
    Calculate win percentages for teams based on wins and games played.
    
    Args:
        data: DataFrame with team data
        team_col: Name of the team identifier column
        win_col: Name of the wins column
        games_col: Name of the games played column
        
    Returns:
        DataFrame with additional win percentage column
    """
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # Add win percentage column
    if win_col in result.columns and games_col in result.columns:
        result['win_pct'] = (result[win_col] / result[games_col]).round(3)
    
    return result

def normalize_data(data: pd.DataFrame, 
                 columns: List[str],
                 method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize selected columns in a DataFrame.
    
    Args:
        data: DataFrame with data to normalize
        columns: List of column names to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        DataFrame with normalized columns
    """
    # Create a copy to avoid modifying the original
    result = data.copy()
    
    # Apply normalization
    for col in columns:
        if col in result.columns:
            if method == 'minmax':
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:
                    result[f'{col}_norm'] = (result[col] - min_val) / (max_val - min_val)
                else:
                    result[f'{col}_norm'] = 0.5  # Default if all values are the same
            
            elif method == 'zscore':
                mean_val = result[col].mean()
                std_val = result[col].std()
                if std_val > 0:
                    result[f'{col}_norm'] = (result[col] - mean_val) / std_val
                else:
                    result[f'{col}_norm'] = 0  # Default if standard deviation is zero
    
    return result

def calculate_matchup_stats(series_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics for matchup results.
    
    Args:
        series_results: Dictionary containing series simulation results
        
    Returns:
        Dictionary with calculated statistics
    """
    stats = {}
    
    # Check if we have the necessary data
    if 'win_distribution' in series_results:
        dist = series_results['win_distribution']
        total = sum(dist.values())
        
        if total > 0:
            # Calculate probability of each outcome
            outcome_probs = {k: v/total for k, v in dist.items()}
            
            # Calculate average series length
            avg_length = sum([int(k[0]) + int(k[2]) * v for k, v in dist.items() if '-' in k]) / total
            
            # Calculate sweep probability (4-0 or 0-4)
            sweep_prob = (dist.get('4-0', 0) + dist.get('0-4', 0)) / total
            
            # Calculate game 7 probability (4-3 or 3-4)
            game7_prob = (dist.get('4-3', 0) + dist.get('3-4', 0)) / total
            
            # Store calculated stats
            stats['outcome_probabilities'] = outcome_probs
            stats['avg_series_length'] = avg_length
            stats['sweep_probability'] = sweep_prob
            stats['game7_probability'] = game7_prob
    
    return stats

# -----------------------------------------------------------------------
# Chart Generation Utilities
# -----------------------------------------------------------------------

def create_polar_comparison_chart(team1_data: Dict[str, float], 
                                team2_data: Dict[str, float],
                                team1_name: str,
                                team2_name: str,
                                team1_color: str = '#1f77b4',
                                team2_color: str = '#ff7f0e') -> go.Figure:
    """
    Create a polar (radar) chart comparing two teams.
    
    Args:
        team1_data: Dictionary of metrics for team 1
        team2_data: Dictionary of metrics for team 2
        team1_name: Name of team 1
        team2_name: Name of team 2
        team1_color: Color for team 1
        team2_color: Color for team 2
        
    Returns:
        Plotly figure object
    """
    # Ensure both teams have the same metrics
    common_metrics = sorted(set(team1_data.keys()).intersection(set(team2_data.keys())))
    
    if not common_metrics:
        # Return empty figure if no common metrics
        return go.Figure().update_layout(title="No common metrics found for comparison")
    
    # Prepare data for radar chart
    fig = go.Figure()
    
    # Add trace for team 1
    fig.add_trace(go.Scatterpolar(
        r=[team1_data[m] for m in common_metrics],
        theta=common_metrics,
        fill='toself',
        name=team1_name,
        line_color=team1_color,
        fillcolor=f'rgba{tuple(int(team1_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'
    ))
    
    # Add trace for team 2
    fig.add_trace(go.Scatterpolar(
        r=[team2_data[m] for m in common_metrics],
        theta=common_metrics,
        fill='toself',
        name=team2_name,
        line_color=team2_color,
        fillcolor=f'rgba{tuple(int(team2_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Assume normalized values
            )
        ),
        showlegend=True,
        title=f"{team1_name} vs {team2_name} Comparison"
    )
    
    return fig

def create_win_probability_timeline(win_probs: List[float],
                                   team1_name: str,
                                   team2_name: str,
                                   team1_color: str = '#1f77b4',
                                   team2_color: str = '#ff7f0e') -> go.Figure:
    """
    Create a timeline chart showing win probability changes.
    
    Args:
        win_probs: List of win probabilities for team 1
        team1_name: Name of team 1
        team2_name: Name of team 2
        team1_color: Color for team 1
        team2_color: Color for team 2
        
    Returns:
        Plotly figure object
    """
    # Create timeline data
    time_points = list(range(len(win_probs)))
    team1_probs = win_probs
    team2_probs = [1 - p for p in win_probs]
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for team 1
    fig.add_trace(go.Scatter(
        x=time_points,
        y=team1_probs,
        mode='lines+markers',
        name=team1_name,
        line=dict(color=team1_color, width=3),
        marker=dict(size=8)
    ))
    
    # Add trace for team 2
    fig.add_trace(go.Scatter(
        x=time_points,
        y=team2_probs,
        mode='lines+markers',
        name=team2_name,
        line=dict(color=team2_color, width=3),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title="Win Probability Timeline",
        xaxis_title="Time",
        yaxis_title="Win Probability",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified"
    )
    
    # Add a horizontal line at 0.5
    fig.add_shape(
        type="line",
        x0=min(time_points),
        y0=0.5,
        x1=max(time_points),
        y1=0.5,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    return fig

def create_series_outcome_chart(win_distribution: Dict[str, int],
                               team1_name: str,
                               team2_name: str,
                               team1_color: str = '#1f77b4',
                               team2_color: str = '#ff7f0e') -> go.Figure:
    """
    Create a chart showing the probability of each series outcome.
    
    Args:
        win_distribution: Dictionary with series outcome counts
        team1_name: Name of team 1 (higher seed)
        team2_name: Name of team 2 (lower seed)
        team1_color: Color for team 1
        team2_color: Color for team 2
        
    Returns:
        Plotly figure object
    """
    # Calculate total simulations
    total = sum(win_distribution.values())
    if total == 0:
        return go.Figure().update_layout(title="No simulation data available")
    
    # Prepare data for plotting
    outcomes = []
    probabilities = []
    colors = []
    hover_texts = []
    
    # Process team 1 (higher seed) outcomes
    for outcome in ['4-0', '4-1', '4-2', '4-3']:
        games = int(outcome[0]) + int(outcome[2])
        probability = win_distribution.get(outcome, 0) / total * 100
        
        outcomes.append(f"{team1_name} {outcome}")
        probabilities.append(probability)
        colors.append(team1_color)
        hover_texts.append(f"{team1_name} wins in {games} games: {probability:.1f}%")
    
    # Process team 2 (lower seed) outcomes
    for outcome in ['0-4', '1-4', '2-4', '3-4']:
        games = int(outcome[0]) + int(outcome[2])
        reverse_outcome = f"{outcome[2]}-{outcome[0]}"
        probability = win_distribution.get(outcome, 0) / total * 100
        
        outcomes.append(f"{team2_name} {reverse_outcome}")
        probabilities.append(probability)
        colors.append(team2_color)
        hover_texts.append(f"{team2_name} wins in {games} games: {probability:.1f}%")
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=outcomes,
        y=probabilities,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probabilities],
        textposition='auto',
        hovertext=hover_texts
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Series Outcome Probabilities: {team1_name} vs {team2_name}",
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, max(probabilities) * 1.1]),
        showlegend=False
    )
    
    return fig

def create_bracket_visual(bracket_data: Dict[str, Any]) -> go.Figure:
    """
    Create a visual representation of a playoff bracket.
    
    Args:
        bracket_data: Dictionary containing bracket progression data
        
    Returns:
        Plotly figure object
    """
    # Check if we have the necessary data
    if not bracket_data or 'rounds' not in bracket_data:
        return go.Figure().update_layout(title="No bracket data available")
    
    # Prepare the figure
    fig = go.Figure()
    
    # Define positions and spacing
    rounds = bracket_data['rounds']
    num_rounds = len(rounds)
    round_width = 1 / (num_rounds + 1)
    
    # Process each round
    for round_idx, round_data in enumerate(rounds):
        round_name = round_data.get('name', f"Round {round_idx+1}")
        matchups = round_data.get('matchups', [])
        
        # X position for this round
        x_pos = (round_idx + 1) * round_width
        
        # Process matchups in this round
        for match_idx, matchup in enumerate(matchups):
            winner = matchup.get('winner', '')
            loser = matchup.get('loser', '')
            score = matchup.get('score', '')
            
            # Calculate y position based on number of matchups
            y_spacing = 1 / (len(matchups) + 1)
            y_pos = (match_idx + 1) * y_spacing
            
            # Add winner text
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=f"<b>{winner}</b> {score}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Add loser text below
            fig.add_annotation(
                x=x_pos,
                y=y_pos-0.03,
                text=f"defeated {loser}",
                showarrow=False,
                font=dict(size=10, color='gray')
            )
            
            # Draw connecting lines for next round if not the final round
            if round_idx < num_rounds - 1:
                # Find the winner's next matchup in the next round
                next_round = rounds[round_idx + 1]
                for next_match in next_round.get('matchups', []):
                    if winner in [next_match.get('winner', ''), next_match.get('loser', '')]:
                        # Calculate next position
                        next_match_idx = next_round['matchups'].index(next_match)
                        next_y_spacing = 1 / (len(next_round['matchups']) + 1)
                        next_y_pos = (next_match_idx + 1) * next_y_spacing
                        
                        # Draw connecting line
                        fig.add_shape(
                            type="line",
                            x0=x_pos + 0.02,
                            y0=y_pos,
                            x1=(round_idx + 2) * round_width - 0.02,
                            y1=next_y_pos,
                            line=dict(color="gray", width=1)
                        )
    
    # Add round labels at the top
    for round_idx, round_data in enumerate(rounds):
        round_name = round_data.get('name', f"Round {round_idx+1}")
        x_pos = (round_idx + 1) * round_width
        
        fig.add_annotation(
            x=x_pos,
            y=0.98,
            text=f"<b>{round_name}</b>",
            showarrow=False,
            font=dict(size=14)
        )
    
    # Add trophy emoji for champion
    if num_rounds > 0 and rounds[-1].get('matchups'):
        final_matchup = rounds[-1]['matchups'][0]
        champion = final_matchup.get('winner', '')
        
        fig.add_annotation(
            x=(num_rounds) * round_width + 0.1,
            y=0.5,
            text="🏆",
            showarrow=False,
            font=dict(size=30)
        )
    
    # Set layout
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=600,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.05, 1.05]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[-0.05, 1.05]
        )
    )
    
    return fig

# -----------------------------------------------------------------------
# Advanced Visualization Utilities
# -----------------------------------------------------------------------

def create_team_performance_dashboard(team_data: pd.DataFrame,
                                     team_abbrev: str,
                                     season: str) -> Dict[str, go.Figure]:
    """
    Create a dashboard of visualizations for team performance.
    
    Args:
        team_data: DataFrame with team statistics
        team_abbrev: Team abbreviation to focus on
        season: Season identifier
        
    Returns:
        Dictionary of named Plotly figures
    """
    # Filter to the selected team
    team_info = team_data[team_data['teamAbbrev'] == team_abbrev].iloc[0] if not team_data[team_data['teamAbbrev'] == team_abbrev].empty else None
    
    if team_info is None:
        return {}
    
    # Dictionary to store figures
    figures = {}
    
    # 1. Create win-loss pie chart
    win_loss_fig = go.Figure(go.Pie(
        labels=['Wins', 'Losses', 'OT Losses'],
        values=[team_info.get('wins', 0), team_info.get('losses', 0), team_info.get('otLosses', 0)],
        hole=0.4,
        marker=dict(colors=['#2ca02c', '#d62728', '#ff7f0e'])
    ))
    win_loss_fig.update_layout(
        title=f"{team_info.get('teamName', team_abbrev)} Record ({season})"
    )
    figures['win_loss'] = win_loss_fig
    
    # 2. Create goals for/against bar chart
    goals_fig = go.Figure()
    goals_fig.add_trace(go.Bar(
        x=['Goals For', 'Goals Against'],
        y=[team_info.get('goalFor', 0), team_info.get('goalAgainst', 0)],
        marker_color=['#2ca02c', '#d62728']
    ))
    goals_fig.update_layout(
        title=f"Goals For vs Against ({season})"
    )
    figures['goals'] = goals_fig
    
    # 3. Create home/away win percentage comparison
    home_away_fig = go.Figure()
    home_away_fig.add_trace(go.Bar(
        x=['Home', 'Away'],
        y=[
            team_info.get('homeWins', 0) / team_info.get('homeGamesPlayed', 1) * 100,
            team_info.get('roadWins', 0) / team_info.get('roadGamesPlayed', 1) * 100
        ],
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    home_away_fig.update_layout(
        title="Win Percentage: Home vs Away",
        yaxis=dict(title="Win %", range=[0, 100])
    )
    figures['home_away'] = home_away_fig
    
    # 4. Create special teams chart if available
    if 'PP%' in team_info and 'PK%' in team_info:
        pp_val = float(team_info['PP%'].strip('%')) if isinstance(team_info['PP%'], str) else team_info['PP%'] * 100
        pk_val = float(team_info['PK%'].strip('%')) if isinstance(team_info['PK%'], str) else team_info['PK%'] * 100
        
        special_teams_fig = go.Figure()
        special_teams_fig.add_trace(go.Bar(
            x=['Power Play', 'Penalty Kill'],
            y=[pp_val, pk_val],
            marker_color=['#1f77b4', '#ff7f0e']
        ))
        special_teams_fig.update_layout(
            title="Special Teams Performance",
            yaxis=dict(title="%", range=[0, 100])
        )
        figures['special_teams'] = special_teams_fig
    
    return figures

def create_advancement_probability_timeline(sim_results_over_time: List[pd.DataFrame], 
                                           selected_teams: List[str] = None) -> go.Figure:
    """
    Create a timeline chart showing how advancement probabilities changed over time.
    
    Args:
        sim_results_over_time: List of DataFrames with simulation results at different times
        selected_teams: List of team abbreviations to include (limits to top 8 if None)
        
    Returns:
        Plotly figure object
    """
    # Check if we have data
    if not sim_results_over_time or len(sim_results_over_time) < 2:
        return go.Figure().update_layout(title="Insufficient timeline data")
    
    # Determine which teams to include
    if selected_teams is None:
        # Use the top teams from the most recent simulation
        latest_results = sim_results_over_time[-1]
        selected_teams = latest_results.sort_values('champion', ascending=False).head(8)['teamAbbrev'].tolist()
    
    # Create figure
    fig = go.Figure()
    
    # Create time points
    time_points = list(range(len(sim_results_over_time)))
    
    # Track championship probabilities over time for each team
    for team in selected_teams:
        team_probs = []
        team_name = ""
        
        for result_df in sim_results_over_time:
            team_data = result_df[result_df['teamAbbrev'] == team]
            if not team_data.empty:
                team_probs.append(team_data.iloc[0].get('champion', 0) * 100)
                if not team_name and 'teamName' in team_data.columns:
                    team_name = team_data.iloc[0]['teamName']
            else:
                team_probs.append(0)
        
        # Use team name if available, otherwise abbreviation
        display_name = team_name if team_name else team
        
        # Add trace for this team
        fig.add_trace(go.Scatter(
            x=time_points,
            y=team_probs,
            mode='lines+markers',
            name=display_name,
            hovertemplate='%{y:.1f}%'
        ))
    
    # Update layout
    fig.update_layout(
        title="Championship Probability Timeline",
        xaxis_title="Time",
        yaxis_title="Championship Probability (%)",
        hovermode="x unified"
    )
    
    return fig

def create_series_simulation_display(series_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process series simulation results into display components.
    
    Args:
        series_results: Dictionary containing series simulation results
        
    Returns:
        Dictionary with processed display components
    """
    display_components = {}
    
    # Check if we have the necessary data
    if not series_results:
        return display_components
    
    # Extract key information
    top_seed = series_results.get('top_seed', '')
    bottom_seed = series_results.get('bottom_seed', '')
    win_probability = series_results.get('win_probability', 0.5)
    win_distribution = series_results.get('win_distribution', {})
    
    # 1. Calculate summary statistics
    stats = calculate_matchup_stats(series_results)
    display_components['stats'] = stats
    
    # 2. Create win probability gauge
    from visualization import plot_win_probability_gauge
    gauge_fig = plot_win_probability_gauge(
        win_probability, 
        top_seed, 
        bottom_seed
    )
    display_components['gauge'] = gauge_fig
    
    # 3. Create series outcome chart
    outcome_fig = create_series_outcome_chart(
        win_distribution,
        top_seed,
        bottom_seed
    )
    display_components['outcome_chart'] = outcome_fig
    
    # 4. Format text summaries
    ci_lower = series_results.get('ci_lower', 0) * 100
    ci_upper = series_results.get('ci_upper', 0) * 100
    
    summary_text = f"{top_seed} has a {win_probability*100:.1f}% chance to win the series "
    summary_text += f"(95% CI: {ci_lower:.1f}%-{ci_upper:.1f}%)."
    
    if stats:
        summary_text += f"\nAverage series length: {stats.get('avg_series_length', 0):.1f} games."
        summary_text += f"\nProbability of a sweep: {stats.get('sweep_probability', 0)*100:.1f}%."
        summary_text += f"\nProbability of a Game 7: {stats.get('game7_probability', 0)*100:.1f}%."
    
    display_components['summary_text'] = summary_text
    
    return display_components

# -----------------------------------------------------------------------
# Streamlit-specific visualization helpers
# -----------------------------------------------------------------------

def display_team_header(team_info: Dict[str, Any], logo_column=None):
    """
    Display a team header with logo and key information.
    
    Args:
        team_info: Dictionary with team information
        logo_column: Optional Streamlit column for logo placement
    """
    from visualization import load_team_logo
    
    # Get team information
    team_name = team_info.get('teamName', 'Unknown Team')
    team_record = f"{team_info.get('wins', 0)}-{team_info.get('losses', 0)}-{team_info.get('otLosses', 0)}"
    team_points = team_info.get('points', 0)
    team_logo_url = team_info.get('teamLogo', '')
    
    # Display logo if column provided
    if logo_column is not None:
        with logo_column:
            logo_img = load_team_logo(team_logo_url)
            if logo_img:
                st.image(logo_img, width=100)
            else:
                # Display team abbreviation as fallback
                st.subheader(team_info.get('teamAbbrev', ''))
    
    # Display team information
    st.markdown(f"### {team_name}")
    st.write(f"**Record:** {team_record} ({team_points} points)")
    
    # Add additional team stats if available
    if 'goalDiff/G' in team_info:
        goal_diff = team_info['goalDiff/G']
        color = 'green' if goal_diff > 0 else 'red'
        st.write(f"**Goal Differential/Game:** <span style='color:{color}'>{goal_diff:+.2f}</span>", unsafe_allow_html=True)
    
    if 'PP%' in team_info and 'PK%' in team_info:
        pp_val = team_info['PP%']
        pk_val = team_info['PK%']
        st.write(f"**Special Teams:** PP: {pp_val}, PK: {pk_val}")

def display_simulation_summary(sim_results: Dict[str, Any], detailed: bool = False):
    """
    Display a summary of simulation results.
    
    Args:
        sim_results: Dictionary with simulation results
        detailed: Whether to show detailed information
    """
    # Check if we have simulation results
    if not sim_results or 'team_advancement' not in sim_results:
        st.warning("No simulation results available.")
        return
    
    # Get team advancement data
    team_advancement = sim_results['team_advancement']
    
    # Display overall summary
    st.subheader("Playoff Simulation Summary")
    
    # Format the advancement data for display
    display_df = team_advancement.copy()
    
    # Convert probabilities to percentages
    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).map('{:.1f}%'.format)
    
    # Keep relevant columns
    display_cols = ['teamName', 'round_1', 'round_2', 'conf_final', 'final', 'champion']
    display_cols = [col for col in display_cols if col in display_df.columns]
    
    # Show top teams
    st.dataframe(display_df[display_cols].head(10), use_container_width=True)
    
    # Display most common bracket if available
    if detailed and 'most_common_bracket' in sim_results:
        from visualization import plot_bracket_simulation_summary
        
        bracket_data = sim_results['most_common_bracket']
        n_sims = sim_results.get('n_simulations', 10000)
        
        plot_bracket_simulation_summary(bracket_data, n_sims)
    
    # Display potential matchups if available and detailed view requested
    if detailed:
        if 'round2_matchups' in sim_results and not sim_results['round2_matchups'].empty:
            st.subheader("Potential Second Round Matchups")
            st.dataframe(sim_results['round2_matchups'].head(5), use_container_width=True)
        
        if 'conf_final_matchups' in sim_results and not sim_results['conf_final_matchups'].empty:
            st.subheader("Potential Conference Finals")
            st.dataframe(sim_results['conf_final_matchups'].head(5), use_container_width=True)
        
        if 'final_matchups' in sim_results and not sim_results['final_matchups'].empty:
            st.subheader("Potential Stanley Cup Finals")
            st.dataframe(sim_results['final_matchups'].head(5), use_container_width=True)

def interactive_matchup_selector(team_data: pd.DataFrame) -> Tuple[str, str]:
    """
    Create an interactive selector for team matchups.
    
    Args:
        team_data: DataFrame with team information
        
    Returns:
        Tuple of selected team abbreviations
    """
    if team_data.empty:
        st.error("No team data available")
        return None, None
    
    # Ensure we have the needed columns
    required_columns = ['teamName', 'teamAbbrev', 'conference']
    if not all(col in team_data.columns for col in required_columns):
        st.error("Team data missing required columns")
        return None, None
    
    # Sort teams by name
    sorted_teams = team_data.sort_values('teamName')
    team_options = sorted_teams['teamName'].tolist()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team 1")
        selected_team1 = st.selectbox(
            "Select first team",
            team_options,
            index=0,
            key="team1_select"
        )
    
    with col2:
        st.subheader("Team 2")
        # Filter to exclude first team and ensure different conferences if possible
        team1_data = team_data[team_data['teamName'] == selected_team1].iloc[0]
        team1_conf = team1_data['conference']
        
        # Prefer teams from other conference
        other_conf_teams = team_data[team_data['conference'] != team1_conf]['teamName'].tolist()
        same_conf_teams = team_data[team_data['conference'] == team1_conf]['teamName'].tolist()
        same_conf_teams.remove(selected_team1)  # Remove the first selected team
        
        # Organize options with other conference teams first
        team2_options = other_conf_teams + same_conf_teams
        
        selected_team2 = st.selectbox(
            "Select second team",
            team2_options,
            index=0,
            key="team2_select"
        )
    
    # Get team abbreviations
    team1_abbrev = team_data[team_data['teamName'] == selected_team1]['teamAbbrev'].iloc[0]
    team2_abbrev = team_data[team_data['teamName'] == selected_team2]['teamAbbrev'].iloc[0]
    
    return team1_abbrev, team2_abbrev
