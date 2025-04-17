import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any

# Set visual styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# -------------------------------------------------------------------
# Team logo and image utilities
# -------------------------------------------------------------------

def load_team_logo(logo_url):
    """Load team logo from URL and return as PIL Image.
    
    Args:
        logo_url (str): URL to the team logo
        
    Returns:
        PIL.Image or None: The loaded image or None if failed
    """
    try:
        # Validate URL
        if not logo_url or not isinstance(logo_url, str) or not logo_url.startswith('http'):
            return None
        
        # Add user-agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(logo_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            try:
                img_data = BytesIO(response.content)
                img = Image.open(img_data)
                img.verify()  # Verify it's a valid image
                img_data.seek(0)
                return Image.open(img_data)
            except Exception:
                return None
        return None
    except Exception:
        return None

def get_team_color(team_abbrev):
    """Get team primary color based on team abbreviation.
    
    Args:
        team_abbrev (str): Team abbreviation (e.g., 'TOR', 'NYR', etc.)
        
    Returns:
        str: Hex color code for the team
    """
    # Dictionary of team colors (primary colors for each NHL team)
    team_colors = {
        'ANA': '#F47A38',  # Anaheim Ducks
        'ARI': '#8C2633',  # Arizona Coyotes
        'BOS': '#FFB81C',  # Boston Bruins
        'BUF': '#002654',  # Buffalo Sabres
        'CGY': '#C8102E',  # Calgary Flames
        'CAR': '#CC0000',  # Carolina Hurricanes
        'CHI': '#CF0A2C',  # Chicago Blackhawks
        'COL': '#6F263D',  # Colorado Avalanche
        'CBJ': '#002654',  # Columbus Blue Jackets
        'DAL': '#006847',  # Dallas Stars
        'DET': '#CE1126',  # Detroit Red Wings
        'EDM': '#FF4C00',  # Edmonton Oilers
        'FLA': '#C8102E',  # Florida Panthers
        'LAK': '#111111',  # Los Angeles Kings
        'MIN': '#154734',  # Minnesota Wild
        'MTL': '#AF1E2D',  # Montreal Canadiens
        'NSH': '#FFB81C',  # Nashville Predators
        'NJD': '#CE1126',  # New Jersey Devils
        'NYI': '#00539B',  # New York Islanders
        'NYR': '#0038A8',  # New York Rangers
        'OTT': '#C52032',  # Ottawa Senators
        'PHI': '#F74902',  # Philadelphia Flyers
        'PIT': '#FCB514',  # Pittsburgh Penguins
        'SEA': '#99D9D9',  # Seattle Kraken
        'SJS': '#006D75',  # San Jose Sharks
        'STL': '#002F87',  # St. Louis Blues
        'TBL': '#002868',  # Tampa Bay Lightning
        'TOR': '#00205B',  # Toronto Maple Leafs
        'UTA': '#041E42',  # Utah Hockey Club (formerly Arizona)
        'VAN': '#00205B',  # Vancouver Canucks
        'VGK': '#B4975A',  # Vegas Golden Knights
        'WSH': '#041E42',  # Washington Capitals
        'WPG': '#041E42'   # Winnipeg Jets
    }
    
    # Return color if found, or default to NHL blue
    return team_colors.get(team_abbrev.upper(), '#004C97')

# -------------------------------------------------------------------
# First Round Matchup Visualization Functions
# -------------------------------------------------------------------

def plot_series_probabilities(top_seed, bottom_seed, lr_prob, xgb_prob, ensemble_prob, sim_prob):
    """Plot bar chart comparing different model probabilities for a playoff series.
    
    Args:
        top_seed (dict): Top seeded team info
        bottom_seed (dict): Bottom seeded team info
        lr_prob (float): Logistic regression probability
        xgb_prob (float): XGBoost model probability
        ensemble_prob (float): Ensemble model probability (with home ice boost)
        sim_prob (float): Simulation result probability
    """
    # Prepare data for the plot
    models = ['Logistic Regression', 'XGBoost', 'Ensemble+HI', 'Simulation']
    probs = [lr_prob*100, xgb_prob*100, ensemble_prob*100, sim_prob*100]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate bars with different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(models, probs, color=colors)
    
    # Add labels and title
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'Series Win Probability: {top_seed["teamName"]} vs {bottom_seed["teamName"]}')
    ax.set_ylim(0, 100)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')
    
    # Add a horizontal line at 50%
    ax.axhline(y=50, linestyle='--', color='gray', alpha=0.7)
    
    # Show the plot
    st.pyplot(fig)

def plot_series_length_table(win_distribution, top_team, bottom_team):
    """Create a styled table showing probability of each series length outcome.
    
    Args:
        win_distribution (dict): Dictionary with series outcome counts
        top_team (str): Name of the top seeded team
        bottom_team (str): Name of the bottom seeded team
    """
    # Calculate total number of simulations
    total = sum(win_distribution.values())
    
    # Create series data
    series_data = {
        'Length': ['4 games', '5 games', '6 games', '7 games'],
        f'{top_team} Wins': [
            f"{win_distribution['4-0']/total*100:.1f}%", 
            f"{win_distribution['4-1']/total*100:.1f}%", 
            f"{win_distribution['4-2']/total*100:.1f}%", 
            f"{win_distribution['4-3']/total*100:.1f}%"
        ],
        f'{bottom_team} Wins': [
            f"{win_distribution['0-4']/total*100:.1f}%", 
            f"{win_distribution['1-4']/total*100:.1f}%", 
            f"{win_distribution['2-4']/total*100:.1f}%", 
            f"{win_distribution['3-4']/total*100:.1f}%"
        ]
    }
    
    # Create DataFrame
    series_df = pd.DataFrame(series_data)
    
    # Display the table
    st.table(series_df)

def plot_team_comparison(top_team_data, bottom_team_data, metrics):
    """Create a comparison chart for selected team metrics.
    
    Args:
        top_team_data (Series): Data for the top seeded team
        bottom_team_data (Series): Data for the bottom seeded team
        metrics (list): List of metrics to compare
    """
    # Check if we have valid data
    if top_team_data is None or bottom_team_data is None:
        st.warning("Team data not available for comparison")
        return
    
    # Prepare data for the chart
    comparison_data = []
    for metric_name, column_name in metrics:
        if column_name in top_team_data and column_name in bottom_team_data:
            top_val = top_team_data[column_name]
            bottom_val = bottom_team_data[column_name]
            
            # Convert percentage strings to floats if needed
            if isinstance(top_val, str) and '%' in top_val:
                top_val = float(top_val.strip('%'))
            if isinstance(bottom_val, str) and '%' in bottom_val:
                bottom_val = float(bottom_val.strip('%'))
            
            # Try to convert to float for plotting
            try:
                top_val = float(top_val)
                bottom_val = float(bottom_val)
                
                comparison_data.append({
                    'Metric': metric_name,
                    top_team_data['teamName']: top_val,
                    bottom_team_data['teamName']: bottom_val
                })
            except (ValueError, TypeError):
                pass
    
    if not comparison_data:
        st.warning("No valid metrics available for comparison")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame(comparison_data)
    
    # Reshape for Altair
    df_melted = pd.melt(
        df, 
        id_vars=['Metric'], 
        value_vars=[top_team_data['teamName'], bottom_team_data['teamName']]
    )
    
    # Create Altair chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('value:Q', title='Value'),
        y=alt.Y('Metric:N'),
        color=alt.Color('variable:N', title='Team', 
                       scale=alt.Scale(domain=[top_team_data['teamName'], bottom_team_data['teamName']],
                                      range=['#1f77b4', '#ff7f0e'])),
        tooltip=['variable:N', 'value:Q']
    ).properties(
        title=f"Team Comparison: {top_team_data['teamName']} vs {bottom_team_data['teamName']}",
        width=600,
        height=30 * len(metrics)
    ).configure_axis(
        labelAngle=0
    )
    
    st.altair_chart(chart, use_container_width=True)

def plot_win_probability_gauge(win_probability, team1_name, team2_name, team1_abbrev=None, team2_abbrev=None):
    """Create a gauge chart showing the win probability for a playoff series.
    
    Args:
        win_probability (float): Probability of team1 winning (0.0-1.0)
        team1_name (str): Name of team 1
        team2_name (str): Name of team 2
        team1_abbrev (str): Team 1 abbreviation for color (optional)
        team2_abbrev (str): Team 2 abbreviation for color (optional)
    """
    # Get team colors or use defaults
    team1_color = get_team_color(team1_abbrev) if team1_abbrev else '#1f77b4'
    team2_color = get_team_color(team2_abbrev) if team2_abbrev else '#ff7f0e'
    
    # Create gauge chart using Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = win_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Series Win Probability<br><span style='font-size:0.8em'>{team1_name} vs {team2_name}</span>"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': team1_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': team2_color},
                {'range': [50, 100], 'color': team1_color}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=[
            dict(
                x=0.1,
                y=0.1,
                text=f"{team2_name}",
                showarrow=False
            ),
            dict(
                x=0.9,
                y=0.1,
                text=f"{team1_name}",
                showarrow=False
            )
        ]
    )
    
    return fig

def plot_team_stats_comparison(team1_data, team2_data, selected_stats, team1_abbrev=None, team2_abbrev=None):
    """Create a table comparing selected team statistics.
    
    Args:
        team1_data (Series/dict): Statistics for team 1
        team2_data (Series/dict): Statistics for team 2
        selected_stats (list): List of (display_name, column_name) tuples for stats to show
        team1_abbrev (str): Team 1 abbreviation for color (optional)
        team2_abbrev (str): Team 2 abbreviation for color (optional)
    """
    # Prepare data for the table
    comparison_data = []
    
    for display_name, column_name in selected_stats:
        if column_name in team1_data and column_name in team2_data:
            try:
                # Extract values
                val1 = team1_data[column_name]
                val2 = team2_data[column_name]
                
                # Convert percentage strings to floats if needed
                if isinstance(val1, str) and '%' in val1:
                    val1 = float(val1.strip('%'))
                if isinstance(val2, str) and '%' in val2:
                    val2 = float(val2.strip('%'))
                
                # Format for display
                if "%" in column_name:
                    val1_fmt = f"{val1:.2f}%"
                    val2_fmt = f"{val2:.2f}%"
                else:
                    val1_fmt = f"{val1:.3f}"
                    val2_fmt = f"{val2:.3f}"
                
                # Add to comparison data - removed advantage column
                comparison_data.append({
                    'Metric': display_name,
                    f'{team1_data.get("teamName", "Team 1")}': val1_fmt,
                    f'{team2_data.get("teamName", "Team 2")}': val2_fmt
                })
            except (ValueError, TypeError):
                # Skip metrics that can't be converted to numbers
                pass
    
    # Create DataFrame for display
    comparison_df = pd.DataFrame(comparison_data)
    
    # Check if we have data
    if comparison_df.empty:
        return None
    
    # Display the table
    st.table(comparison_df)
    
    return None  # No figure to return since we're displaying a table

# -------------------------------------------------------------------
# Simulation Results Visualization Functions
# -------------------------------------------------------------------

def plot_championship_odds(results_df, top_n=10):
    """Plot bar chart showing championship odds for the top teams.
    
    Args:
        results_df (DataFrame): DataFrame with simulation results
        top_n (int): Number of teams to include in the chart
    """
    # Get top teams by championship odds
    top_teams = results_df.sort_values('champion', ascending=False).head(top_n)
    
    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars with team colors (could be customized further)
    bars = ax.barh(top_teams['teamName'], top_teams['champion'] * 100)
    
    # Add team name and percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        team_name = top_teams.iloc[i]['teamName']
        percentage = top_teams.iloc[i]['champion'] * 100
        
        # Add percentage label to end of bar
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%', va='center')
    
    # Set labels and title  
    ax.set_xlabel('Championship Probability (%)')
    ax.set_title('Stanley Cup Championship Odds')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit to give room for labels
    ax.set_xlim(0, top_teams['champion'].max() * 100 * 1.2)
    
    # Show the plot
    st.pyplot(fig)

def plot_advancement_heatmap(results_df, top_n=16):
    """Create heatmap showing advancement probabilities for each round.
    
    Args:
        results_df (DataFrame): DataFrame with simulation results
        top_n (int): Number of teams to include in the chart
    """
    # Select top teams and relevant columns
    top_teams = results_df.sort_values('champion', ascending=False).head(top_n)
    rounds = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
    round_labels = ['First Round', 'Second Round', 'Conf Finals', 'Finals', 'Champion']
    
    # Create a pivot table for the heatmap
    heatmap_data = top_teams[['teamName'] + rounds].set_index('teamName')
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))
    
    # Generate heatmap
    ax = sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='Blues',
                     linewidths=.5, cbar_kws={'label': 'Probability (%)'})
    
    # Set labels
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Playoff Advancement Probabilities by Round')
    
    # Rename column labels
    ax.set_xticklabels(round_labels)
    
    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')
    
    # Show plot
    st.pyplot(plt.gcf())

def plot_colored_advancement_table(results_df, top_n=16):
    """Display a colored table showing playoff advancement probabilities.
    
    Args:
        results_df (DataFrame): DataFrame with simulation results
        top_n (int): Number of teams to include in the table
    """
    # Select top teams by championship odds
    top_teams = results_df.sort_values('champion', ascending=False).head(top_n)
    
    # Select columns to display
    display_cols = ['teamName', 'round_1', 'round_2', 'conf_final', 'final', 'champion', 'avg_games_played']
    column_headers = ['Team', 'Round 1', 'Round 2', 'Conf Finals', 'Finals', 'Champion', 'Avg Games']
    
    # Create a copy for display with formatted percentages
    display_df = top_teams[display_cols].copy()
    
    # Format percentage columns
    for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
        display_df[col] = (display_df[col] * 100).round(1)
    
    # Format average games column
    display_df['avg_games_played'] = display_df['avg_games_played'].round(1)
    
    # Rename columns for display
    display_df.columns = column_headers
    
    # Function to apply conditional styling
    def style_df(val, max_value, col_name):
        # Only style percentage columns
        if col_name not in ['Team', 'Avg Games']:
            # Calculate color intensity - darker for higher values
            normalized_val = val / max_value if max_value > 0 else 0
            color = f'background-color: rgba(66, 133, 244, {normalized_val * 0.9})'
            
            # Use darker text for very light backgrounds
            text_color = 'white' if normalized_val > 0.5 else 'black'
            return f'{color}; color: {text_color}'
        return ''
    
    # Calculate max values for each column for normalization
    max_values = {
        'Round 1': display_df['Round 1'].max(),
        'Round 2': display_df['Round 2'].max(),
        'Conf Finals': display_df['Conf Finals'].max(),
        'Finals': display_df['Finals'].max(),
        'Champion': display_df['Champion'].max()
    }
    
    # Create styler object
    styled_df = display_df.style.format({
        'Round 1': '{:.1f}%',
        'Round 2': '{:.1f}%',
        'Conf Finals': '{:.1f}%', 
        'Finals': '{:.1f}%',
        'Champion': '{:.1f}%'
    })
    
    # Apply conditional styling
    for col in max_values.keys():
        styled_df = styled_df.applymap(
            lambda x, col=col, max_val=max_values[col]: style_df(x, max_val, col),
            subset=[col]
        )
    
    # Display the table
    st.dataframe(styled_df, use_container_width=True)

def plot_potential_matchups(matchups_df, title, top_n=10):
    """Create a horizontal bar chart showing the most likely potential matchups.
    
    Args:
        matchups_df (DataFrame): DataFrame with potential matchups info
        title (str): Chart title
        top_n (int): Number of matchups to display
    """
    if matchups_df.empty:
        st.info(f"No {title.lower()} matchup data available")
        return
    
    # Take top N matchups
    top_matchups = matchups_df.sort_values('probability', ascending=False).head(top_n)
    
    # Create Plotly horizontal bar chart
    fig = px.bar(
        top_matchups,
        y='matchup',
        x='probability',
        orientation='h',
        labels={'probability': 'Probability (%)', 'matchup': 'Matchup'},
        title=title,
        text='probability'
    )
    
    # Update layout for better appearance
    fig.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        marker_color='rgb(55, 83, 169)'
    )
    fig.update_layout(
        xaxis_title='Probability (%)',
        yaxis_title='',
        xaxis=dict(range=[0, max(100, top_matchups['probability'].max() * 1.1)]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

def display_matchup_probabilities_table(matchups_df, title, top_n=10):
    """Display a table of potential matchups with their probabilities.
    
    Args:
        matchups_df (DataFrame): DataFrame with potential matchups info
        title (str): Table title
        top_n (int): Number of matchups to display
    """
    if matchups_df.empty:
        return
    
    # Take top N matchups
    top_matchups = matchups_df.sort_values('probability', ascending=False).head(top_n)
    
    # Format table for display
    display_df = top_matchups.copy()
    display_df['probability'] = display_df['probability'].map('{:.1f}%'.format)
    
    if 'top_seed_win_pct' in display_df.columns:
        display_df['top_seed_win_pct'] = display_df['top_seed_win_pct'].map('{:.1f}%'.format)
        display_df = display_df.rename(columns={
            'matchup': 'Matchup', 
            'probability': 'Probability',
            'top_seed_win_pct': 'Top Seed Win %'
        })
        cols_to_show = ['Matchup', 'Probability', 'Top Seed Win %']
    else:
        display_df = display_df.rename(columns={
            'matchup': 'Matchup', 
            'probability': 'Probability'
        })
        cols_to_show = ['Matchup', 'Probability']
    
    # Ensure we only show columns that exist
    cols_to_show = [col for col in cols_to_show if col in display_df.columns]
    
    # Display table with title
    st.subheader(title)
    st.table(display_df[cols_to_show])

# -------------------------------------------------------------------
# Head-to-Head Comparison Visualization Functions
# -------------------------------------------------------------------

def _color_distance(color1, color2):
    """Calculate visual distance between two colors.
    
    Args:
        color1 (str): First color in hex format (e.g., '#FF0000')
        color2 (str): Second color in hex format
        
    Returns:
        float: Visual distance between colors (higher values mean more distinct colors)
    """
    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        # Remove the # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert to RGB 
    try:
        # Handle both hex and rgb() format
        if color1.startswith('rgb'):
            # Extract RGB values from rgb(R, G, B) format
            c1 = tuple(int(x) for x in color1.strip('rgb()').split(','))
        else:
            c1 = hex_to_rgb(color1)
            
        if color2.startswith('rgb'):
            c2 = tuple(int(x) for x in color2.strip('rgb()').split(','))
        else:
            c2 = hex_to_rgb(color2)
            
        # Calculate Euclidean distance in RGB space
        # This is a simple approach - more sophisticated color distance metrics exist
        return sum((a-b)**2 for a, b in zip(c1, c2))**0.5
    except (ValueError, TypeError):
        # Return a small value if there's an error converting colors
        return 0

def plot_head_to_head_probabilities(team1, team2, lr_prob, xgb_prob, ensemble_prob, sim_prob=None, team1_abbrev=None, team2_abbrev=None):
    """Create visual comparison of model predictions for head-to-head matchup.
    
    Args:
        team1 (str): Name of team 1 (home team)
        team2 (str): Name of team 2 (away team)
        lr_prob (float): Logistic regression probability (RAW, no home ice)
        xgb_prob (float): XGBoost model probability (RAW, no home ice)
        ensemble_prob (float): Ensemble model probability (WITH home ice boost)
        sim_prob (float): Probability from simulation (optional)
        team1_abbrev (str): Team 1 abbreviation for color (optional)
        team2_abbrev (str): Team 2 abbreviation for color (optional)
    """
    # Prepare data for the plot
    models = ['Logistic Regression', 'XGBoost', 'Ensemble+HI']
    probs = [lr_prob*100, xgb_prob*100, ensemble_prob*100]
    
    # Add simulation probability if provided
    if sim_prob is not None:
        models.append('Simulation')
        probs.append(sim_prob*100)
    
    # Get team colors if abbreviations are provided
    team1_color = get_team_color(team1_abbrev) if team1_abbrev else 'rgb(55, 83, 169)'
    team2_color = get_team_color(team2_abbrev) if team2_abbrev else 'rgb(219, 64, 82)'
    
    # Check if team colors are too similar
    DEFAULT_GRAY = '#777777'  # Default gray for team 2 if colors are too similar
    COLOR_SIMILARITY_THRESHOLD = 90  # Threshold for considering colors too similar
    
    # Calculate color difference
    color_diff = _color_distance(team1_color, team2_color)
    
    # If colors are too similar, use default gray for team 2
    if color_diff < COLOR_SIMILARITY_THRESHOLD:
        team2_color = DEFAULT_GRAY
    
    # Create Plotly bar chart
    fig = go.Figure()
    
    # Team 1 probabilities
    fig.add_trace(go.Bar(
        x=models,
        y=probs,
        name=team1,
        marker_color=team1_color,
        hoverinfo="x+y+name",
        hovertemplate=f'{team1}: %{{y:.1f}}%<extra></extra>'
    ))
    
    # Team 2 probabilities
    fig.add_trace(go.Bar(
        x=models,
        y=[100-p for p in probs],
        name=team2,
        marker_color=team2_color,
        hoverinfo="x+y+name",
        hovertemplate=f'{team2}: %{{y:.1f}}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Series Win Probability: {team1} vs {team2}',
            font=dict(color='black', size=18),  # Black title text
            xanchor='center',
            x=0.5,
            yanchor='top',
            y=0.95
        ),
        xaxis=dict(
            title=dict(text='Model', font=dict(color='black', size=14)),  # Black axis title
            tickfont=dict(color='black'),  # Black tick labels
            gridcolor='lightgray'  # Light gray grid lines
        ),
        yaxis=dict(
            title=dict(text='Probability (%)', font=dict(color='black', size=14)),  # Black axis title
            tickfont=dict(color='black'),  # Black tick labels
            gridcolor='lightgray',  # Light gray grid lines
            range=[0, 110]  # Keep existing range
        ),
        barmode='stack',
        height=500,
        # Improved legend styling with no border
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='white',  # Solid white background
            bordercolor='white',  # Remove border by matching background
            font=dict(
                size=14,
                color='black'  # Black text for contrast
            )
        ),
        # Set white background for entire plot
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50),
    )
    
    # Draw a horizontal line at 50%
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=50,
        x1=len(models)-0.5,
        y1=50,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    # Add annotations for team percentages
    for i, p in enumerate(probs):
        # Add annotation for team1
        if p >= 5:
            fig.add_annotation(
                x=models[i],
                y=p/2,
                text=f"{p:.1f}%",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=team1_color,
                borderpad=3,
                opacity=0.9
            )
        
        # Add annotation for team2
        if (100-p) >= 5:
            fig.add_annotation(
                x=models[i],
                y=p + (100-p)/2,
                text=f"{(100-p):.1f}%",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=team2_color,
                borderpad=3,
                opacity=0.9
            )
        
        # For very small values, add labels outside
        if p < 5:
            fig.add_annotation(
                x=models[i],
                y=5,
                text=f"{p:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=team1_color,
                font=dict(color="black", size=10),
                bgcolor="white",
                bordercolor=team1_color,
                borderpad=2
            )
        
        if (100-p) < 5:
            fig.add_annotation(
                x=models[i],
                y=95,
                text=f"{(100-p):.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=team2_color,
                font=dict(color="black", size=10),
                bgcolor="white",
                bordercolor=team2_color,
                borderpad=2
            )
    
    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

def plot_head_to_head_metrics(team1_data, team2_data, metrics):
    """Create a radar chart comparing team metrics.
    
    Args:
        team1_data (Series): Data for team 1
        team2_data (Series): Data for team 2
        metrics (list): List of (display_name, column_name) tuples for metrics to compare
    """
    # Prepare data for radar chart
    categories = [m[0] for m in metrics]
    team1_values = []
    team2_values = []
    
    # Extract normalized values for each metric
    for _, col in metrics:
        if col in team1_data and col in team2_data:
            val1 = team1_data[col]
            val2 = team2_data[col]
            
            # Convert to float if string percentage
            if isinstance(val1, str) and '%' in val1:
                val1 = float(val1.strip('%'))
            if isinstance(val2, str) and '%' in val2:
                val2 = float(val2.strip('%'))
            
            # Get max value for normalization
            try:
                val1 = float(val1)
                val2 = float(val2)
                max_val = max(abs(val1), abs(val2))
                
                # Normalize to 0-1 range
                norm_val1 = 0.5 if max_val == 0 else (val1 / max_val + 1) / 2
                norm_val2 = 0.5 if max_val == 0 else (val2 / max_val + 1) / 2
                
                team1_values.append(norm_val1)
                team2_values.append(norm_val2)
            except (ValueError, TypeError):
                # Skip metrics that can't be converted to numbers
                team1_values.append(0.5)
                team2_values.append(0.5)
        else:
            team1_values.append(0.5)
            team2_values.append(0.5)
    
    # Create radar chart using Plotly
    fig = go.Figure()
    
    # Add first team trace
    fig.add_trace(go.Scatterpolar(
        r=team1_values,
        theta=categories,
        fill='toself',
        name=team1_data['teamName'],
        line=dict(color='rgb(55, 83, 169)'),
        fillcolor='rgba(55, 83, 169, 0.3)'
    ))
    
    # Add second team trace
    fig.add_trace(go.Scatterpolar(
        r=team2_values,
        theta=categories,
        fill='toself',
        name=team2_data['teamName'],
        line=dict(color='rgb(219, 64, 82)'),
        fillcolor='rgba(219, 64, 82, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Team Comparison: {team1_data['teamName']} vs {team2_data['teamName']}",
        showlegend=True
    )
    
    # Show chart
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Playoff Bracket Visualization Functions
# -------------------------------------------------------------------

def create_bracket_visual(playoff_results):
    """Create a visual representation of a playoff bracket.
    
    Args:
        playoff_results (dict): Results from simulate_single_bracket function
    """
    if not playoff_results:
        st.error("No playoff results to display")
        return
    
    # Extract data from results
    bracket_progression = playoff_results['bracket_progression']
    champion = playoff_results.get('champion', {})
    team_results = playoff_results.get('team_results', {})
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Remove axis elements
    ax.axis('off')
    
    # Define positions for each round
    round_positions = {
        'First Round': 5,
        'Second Round': 30,
        'Conference Finals': 55,
        'Stanley Cup Final': 80
    }
    
    # Track team vertical positions
    team_positions = {}
    
    # Draw bracket lines and add team names
    y_pos = 95  # Start from top
    spacing = 90 / (len(bracket_progression['First Round']) + 1)
    
    # First round
    for i, matchup in enumerate(bracket_progression['First Round']):
        winner = matchup['winner']
        loser = matchup['loser']
        result = matchup.get('result', '')
        
        # Position the teams
        y_pos = 95 - (i+1) * spacing
        team_positions[winner] = y_pos
        
        # Draw team names and result
        ax.text(round_positions['First Round'], y_pos, f"{winner} ({result})", 
                fontsize=10, color='black', ha='left', fontweight='bold')
        ax.text(round_positions['First Round'], y_pos-3, f"defeated {loser}", 
                fontsize=8, color='gray', ha='left')
    
    # Second round
    for i, matchup in enumerate(bracket_progression['Second Round']):
        winner = matchup['winner']
        loser = matchup['loser']
        result = matchup.get('result', '')
        
        # Find position based on previous round
        y_pos = team_positions.get(winner, 50)  # Default to middle if not found
        team_positions[winner] = y_pos
        
        # Draw team names and result
        ax.text(round_positions['Second Round'], y_pos, f"{winner} ({result})", 
                fontsize=10, color='black', ha='left', fontweight='bold')
        ax.text(round_positions['Second Round'], y_pos-3, f"defeated {loser}", 
                fontsize=8, color='gray', ha='left')
    
    # Conference Finals
    for i, matchup in enumerate(bracket_progression['Conference Finals']):
        winner = matchup['winner']
        loser = matchup['loser']
        result = matchup.get('result', '')
        conference = matchup.get('conference', '')
        
        # Find position based on previous round
        y_pos = team_positions.get(winner, 50)  # Default to middle if not found
        team_positions[winner] = y_pos
        
        # Draw team names and result
        ax.text(round_positions['Conference Finals'], y_pos, 
                f"{winner} ({result})", 
                fontsize=10, color='black', ha='left', fontweight='bold')
        ax.text(round_positions['Conference Finals'], y_pos-3, 
                f"defeated {loser} - {conference} Champion", 
                fontsize=8, color='gray', ha='left')
    
    # Stanley Cup Final
    if bracket_progression['Stanley Cup Final']:
        matchup = bracket_progression['Stanley Cup Final'][0]
        winner = matchup['winner']
        loser = matchup['loser']
        result = matchup.get('result', '')
        
        # Find position based on previous round
        y_pos = team_positions.get(winner, 50)  # Default to middle if not found
        
        # Draw team names and result
        ax.text(round_positions['Stanley Cup Final'], y_pos, 
                f"{winner} ({result})", 
                fontsize=12, color='darkgreen', ha='left', fontweight='bold')
        ax.text(round_positions['Stanley Cup Final'], y_pos-3, 
                f"defeated {loser} - STANLEY CUP CHAMPION", 
                fontsize=10, color='darkgreen', ha='left', fontweight='bold')
        
        # Add a trophy icon or symbol
        ax.text(round_positions['Stanley Cup Final']+15, y_pos-1, '🏆', 
                fontsize=20, ha='center', va='center')
    
    # Draw round labels at the top
    for round_name, x_pos in round_positions.items():
        ax.text(x_pos, 98, round_name, fontsize=12, fontweight='bold', ha='left')
    
    # Show the bracket
    st.pyplot(fig)

def create_simple_bracket_visual(playoff_results):
    """Create a simpler text-based representation of a playoff bracket.
    
    Args:
        playoff_results (dict): Results from simulate_single_bracket function
    """
    if not playoff_results:
        st.error("No playoff results to display")
        return
    
    # Extract data
    bracket_progression = playoff_results['bracket_progression']
    champion = playoff_results.get('champion', {})
    
    # Display round by round results
    for round_name, matchups in bracket_progression.items():
        st.subheader(round_name)
        
        # Create columns for matchups
        cols = st.columns(min(len(matchups), 4))
        
        for i, matchup in enumerate(matchups):
            col_idx = i % len(cols)
            with cols[col_idx]:
                winner = matchup['winner']
                loser = matchup['loser']
                result = matchup.get('result', '')
                
                # Highlight the winner
                st.markdown(f"**{winner}** ({result})")
                st.write(f"defeated {loser}")
                
                # Add conference info for conference finals
                if round_name == 'Conference Finals' and 'conference' in matchup:
                    st.write(f"*{matchup['conference']} Champion*")
                
                # Add special styling for the champion
                if round_name == 'Stanley Cup Final':
                    st.markdown("### 🏆 STANLEY CUP CHAMPION 🏆")
    
    # Display champion's path
    if champion and 'path' in champion and champion['path']:
        st.subheader(f"Champion's Path: {champion['name']}")
        
        # Create a table showing the champion's journey
        path_data = []
        for series in champion['path']:
            path_data.append({
                'Round': series['round'],
                'Opponent': series['opponent'],
                'Result': series['result']
            })
        
        path_df = pd.DataFrame(path_data)
        st.table(path_df)

# -------------------------------------------------------------------
# Utility visualization functions
# -------------------------------------------------------------------

def display_colored_table(df, color_columns, cmap='Blues', text_color_threshold=0.7):
    """Display a DataFrame with background colors based on values.
    
    Args:
        df (DataFrame): DataFrame to display
        color_columns (list): Columns to apply color formatting
        cmap (str): Matplotlib colormap name
        text_color_threshold (float): Threshold for switching to white text
    """
    # Create a copy for display
    styled_df = df.copy()
    
    # Define a function to color the cells
    def color_cells(val, min_val, max_val, cmap_name):
        # Normalize the value
        norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
        
        # Get color from colormap
        cmap = plt.cm.get_cmap(cmap_name)
        color = cmap(norm_val)
        
        # Convert to hex for styling
        hex_color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        # Choose text color based on background darkness
        text_color = 'white' if norm_val > text_color_threshold else 'black'
        
        return f'background-color: {hex_color}; color: {text_color}'
    
    # Apply styling to specified columns
    for col in color_columns:
        if col in styled_df.columns:
            min_val = styled_df[col].min()
            max_val = styled_df[col].max()
            
            styled_df = styled_df.style.applymap(
                lambda x, col=col, min_val=min_val, max_val=max_val: 
                color_cells(x, min_val, max_val, cmap), 
                subset=[col]
            )
    
    return styled_df

def plot_top_teams_chart(teams_to_plot, metric='champion', top_n=10, title=None):
    """Create a horizontal bar chart for the top teams based on the selected metric.
    
    Args:
        teams_to_plot (DataFrame): DataFrame with team data
        metric (str): Column name for the metric to plot
        top_n (int): Number of teams to include
        title (str): Chart title (optional)
    """
    # Get top teams
    if metric not in teams_to_plot.columns:
        st.error(f"Metric '{metric}' not found in data")
        return
    
    # Select top teams
    sorted_teams = teams_to_plot.sort_values(metric, ascending=False).head(top_n)
    
    # Format values as percentages if they're probabilities
    y_values = sorted_teams['teamName']
    x_values = sorted_teams[metric]
    if x_values.max() <= 1.0:
        x_values = x_values * 100
        x_label = 'Probability (%)'
    else:
        x_label = metric.replace('_', ' ').title()
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.barh(y_values, x_values)
    
    # Add value labels
    for i, v in enumerate(x_values):
        plt.text(v + 0.5, i, f'{v:.1f}', va='center')
    
    # Set labels and title
    plt.xlabel(x_label)
    if title:
        plt.title(title)
    else:
        plt.title(f'Top {top_n} Teams by {metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    return fig

def create_round_advancement_chart(teams_to_plot):
    """Create a chart showing advancement through playoff rounds.
    
    Args:
        teams_to_plot (DataFrame): DataFrame with team advancement data
    """
    # Check if we have the necessary columns
    required_cols = ['teamName', 'round_1', 'round_2', 'conf_final', 'final', 'champion']
    for col in required_cols:
        if col not in teams_to_plot.columns:
            st.error(f"Required column '{col}' not found in data")
            return
    
    # Select top teams by championship odds
    top_teams = teams_to_plot.sort_values('champion', ascending=False)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    
    # Define the rounds and their labels
    rounds = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
    round_labels = ['First Round', 'Second Round', 'Conf Finals', 'Stanley Cup Final', 'Champion']
    
    # Set position and width
    y_pos = np.arange(len(top_teams))
    width = 0.6
    
    # Plot each round
    bottom = np.zeros(len(top_teams))
    for i, (col, label) in enumerate(zip(rounds, round_labels)):
        plt.barh(y_pos, top_teams[col]*100, left=bottom, height=width, 
                label=label, alpha=0.8)
        bottom += top_teams[col]*100
    
    # Customize the plot
    plt.yticks(y_pos, top_teams['teamName'])
    plt.xlabel('Probability (%)')
    plt.title('Playoff Round Advancement Probabilities')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    return fig

def plot_championship_odds_plotly(results_df, top_n=10):
    """Create an interactive bar chart showing championship odds using Plotly.
    
    Args:
        results_df (DataFrame): DataFrame with simulation results
        top_n (int): Number of teams to include in the chart
    """
    # Get top teams by championship odds
    top_teams = results_df.sort_values('champion', ascending=False).head(top_n)
    
    # Create bar chart with team colors
    fig = go.Figure()
    
    # Add bars with team colors
    for i, row in top_teams.iterrows():
        team_abbrev = row.get('teamAbbrev', '')
        team_color = get_team_color(team_abbrev)
        
        fig.add_trace(go.Bar(
            x=[row['champion'] * 100],
            y=[row['teamName']],
            orientation='h',
            marker_color=team_color,
            text=f"{row['champion']*100:.1f}%",
            textposition='outside',
            name=row['teamName']
        ))
    
    # Update layout
    fig.update_layout(
        title='Stanley Cup Championship Odds',
        xaxis_title='Championship Probability (%)',
        yaxis_title='',
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, max(top_teams['champion'])*100*1.1]),
    )
    
    return fig

def plot_bracket_simulation_summary(most_common_bracket, total_sims):
    """Display the most common bracket outcome from simulations.
    
    Args:
        most_common_bracket (dict): Dictionary containing the most common bracket info
        total_sims (int): Total number of simulations run
    """
    # Check if we have bracket data
    if not most_common_bracket or 'bracket' not in most_common_bracket:
        st.warning("No bracket data available")
        return
    
    # Display summary info
    st.subheader("Most Likely Playoff Outcome")
    
    # Calculate probability
    count = most_common_bracket.get('count', 0)
    probability = most_common_bracket.get('probability', 0)
    
    # Show info
    st.write(f"Occurred in {count:,} of {total_sims:,} simulations ({probability:.2f}%)")
    
    # Bracket results
    bracket_results = most_common_bracket.get('bracket', [])
    
    # Create a visual representation
    if bracket_results:
        # Determine number of rounds
        num_rounds = 4
        results_per_round = [8, 4, 2, 1]  # Expected results per round
        
        # Create columns for each round
        cols = st.columns(num_rounds)
        
        # Initialize the result index
        result_idx = 0
        
        # Display results by round
        for round_idx, results_count in enumerate(results_per_round):
            round_name = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Final"][round_idx]
            
            with cols[round_idx]:
                st.markdown(f"**{round_name}**")
                
                # Display the results for this round
                for i in range(results_count):
                    if result_idx < len(bracket_results):
                        result = bracket_results[result_idx]
                        parts = result.split(" over ")
                        
                        if len(parts) == 2:
                            winner, loser = parts
                            st.markdown(f"**{winner}** over {loser}")
                        else:
                            st.text(result)
                        
                        result_idx += 1

# -------------------------------------------------------------------
# Utility visualization functions
# -------------------------------------------------------------------

def create_small_multiple_charts(dataframes, plot_func, n_cols=2, **kwargs):
    """Create a grid of small multiple charts using the same plot function.
    
    Args:
        dataframes (list): List of dataframes to plot
        plot_func (callable): Function that takes a dataframe and returns a figure
        n_cols (int): Number of columns in the grid
        **kwargs: Additional keyword arguments for plot_func
    """
    # Calculate number of rows needed
    n_charts = len(dataframes)
    n_rows = (n_charts + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a grid of columns
    for i in range(0, n_charts, n_cols):
        # Create row of columns
        cols = st.columns(min(n_cols, n_charts - i))
        
        # Fill each column with a chart
        for j in range(min(n_cols, n_charts - i)):
            with cols[j]:
                fig = plot_func(dataframes[i + j], **kwargs)
                st.pyplot(fig)