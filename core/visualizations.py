"""
Visualization module for NHL playoff predictions.
Extracted from Streamlit app for standalone use with matplotlib/plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Set visual styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("colorblind")


class NHLVisualizer:
    """NHL data visualization class"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Team colors dictionary
        self.team_colors = {
            'ANA': '#F47A38', 'ARI': '#8C2633', 'BOS': '#FFB81C', 'BUF': '#002654',
            'CGY': '#C8102E', 'CAR': '#CC0000', 'CHI': '#CF0A2C', 'COL': '#6F263D',
            'CBJ': '#002654', 'DAL': '#006847', 'DET': '#CE1126', 'EDM': '#FF4C00',
            'FLA': '#C8102E', 'LAK': '#111111', 'MIN': '#154734', 'MTL': '#AF1E2D',
            'NSH': '#FFB81C', 'NJD': '#CE1126', 'NYI': '#00539B', 'NYR': '#0038A8',
            'OTT': '#C52032', 'PHI': '#F74902', 'PIT': '#FCB514', 'SEA': '#99D9D9',
            'SJS': '#006D75', 'STL': '#002F87', 'TBL': '#002868', 'TOR': '#00205B',
            'UTA': '#041E42', 'VAN': '#00205B', 'VGK': '#B4975A', 'WSH': '#041E42',
            'WPG': '#041E42'
        }
    
    def get_team_color(self, team_abbrev: str) -> str:
        """Get team primary color based on team abbreviation."""
        return self.team_colors.get(team_abbrev, '#1f77b4')  # Default blue
    
    def create_standings_chart(self, standings_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create standings visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Eastern Conference
        east_teams = standings_df[standings_df.get('conferenceAbbrev', '') == 'E'].head(16)
        if not east_teams.empty:
            colors = [self.get_team_color(abbrev) for abbrev in east_teams.get('teamAbbrev', [])]
            bars1 = ax1.barh(range(len(east_teams)), east_teams.get('points', []), color=colors)
            ax1.set_yticks(range(len(east_teams)))
            ax1.set_yticklabels(east_teams.get('teamAbbrev', []))
            ax1.set_xlabel('Points')
            ax1.set_title('Eastern Conference Standings')
            ax1.invert_yaxis()
            
            # Add point values on bars
            for i, (bar, points) in enumerate(zip(bars1, east_teams.get('points', []))):
                ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        str(int(points)), va='center')
        
        # Western Conference
        west_teams = standings_df[standings_df.get('conferenceAbbrev', '') == 'W'].head(16)
        if not west_teams.empty:
            colors = [self.get_team_color(abbrev) for abbrev in west_teams.get('teamAbbrev', [])]
            bars2 = ax2.barh(range(len(west_teams)), west_teams.get('points', []), color=colors)
            ax2.set_yticks(range(len(west_teams)))
            ax2.set_yticklabels(west_teams.get('teamAbbrev', []))
            ax2.set_xlabel('Points')
            ax2.set_title('Western Conference Standings')
            ax2.invert_yaxis()
            
            # Add point values on bars
            for i, (bar, points) in enumerate(zip(bars2, west_teams.get('points', []))):
                ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        str(int(points)), va='center')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'standings_chart.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved standings chart: {save_path}")
        return save_path
    
    def create_team_comparison_chart(self, team_data: pd.DataFrame, team1: str, team2: str,
                                   metrics: List[Tuple[str, str]], save_path: Optional[str] = None) -> str:
        """Create team comparison radar/bar chart"""
        
        # Filter data for the two teams
        team1_data = team_data[team_data['teamAbbrev'] == team1]
        team2_data = team_data[team_data['teamAbbrev'] == team2]
        
        if team1_data.empty or team2_data.empty:
            logger.warning(f"No data found for teams {team1} or {team2}")
            return ""
        
        team1_row = team1_data.iloc[0]
        team2_row = team2_data.iloc[0]
        
        # Extract metric values
        metric_names = []
        team1_values = []
        team2_values = []
        
        for display_name, column_name in metrics:
            if column_name in team1_row and column_name in team2_row:
                metric_names.append(display_name)
                team1_values.append(float(team1_row[column_name]) if pd.notna(team1_row[column_name]) else 0)
                team2_values.append(float(team2_row[column_name]) if pd.notna(team2_row[column_name]) else 0)
        
        if not metric_names:
            logger.warning(f"No valid metrics found for comparison")
            return ""
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, team1_values, width, label=team1, 
                      color=self.get_team_color(team1), alpha=0.8)
        bars2 = ax.bar(x + width/2, team2_values, width, label=team2, 
                      color=self.get_team_color(team2), alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title(f'{team1} vs {team2} - Statistical Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{team1}_vs_{team2}_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved team comparison chart: {save_path}")
        return save_path
    
    def create_playoff_bracket_chart(self, playoff_matchups: List[Dict], save_path: Optional[str] = None) -> str:
        """Create playoff bracket visualization"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define bracket structure
        rounds = ['First Round', 'Second Round', 'Conference Finals', 'Stanley Cup Final']
        round_x_positions = [0.1, 0.35, 0.6, 0.85]
        
        # Draw bracket structure
        for round_idx, round_name in enumerate(rounds):
            x_pos = round_x_positions[round_idx]
            
            # Round label
            ax.text(x_pos, 0.95, round_name, fontsize=14, fontweight='bold', 
                   ha='center', transform=ax.transAxes)
            
            # Draw matchups for this round
            matchups_in_round = [m for m in playoff_matchups if m.get('round_name') == round_name]
            
            if matchups_in_round:
                num_matchups = len(matchups_in_round)
                y_positions = np.linspace(0.1, 0.8, num_matchups)
                
                for i, matchup in enumerate(matchups_in_round):
                    y_pos = y_positions[i]
                    
                    # Team 1
                    team1 = matchup.get('top_seed_abbrev', 'TBD')
                    team1_prob = matchup.get('prediction_prob', 0.5)
                    
                    # Team 2  
                    team2 = matchup.get('bottom_seed_abbrev', 'TBD')
                    team2_prob = 1 - team1_prob
                    
                    # Draw matchup box
                    box_height = 0.08
                    box_width = 0.2
                    
                    # Team 1 box
                    rect1 = mpatches.Rectangle((x_pos - box_width/2, y_pos), 
                                             box_width, box_height/2,
                                             facecolor=self.get_team_color(team1), 
                                             alpha=0.7, edgecolor='black')
                    ax.add_patch(rect1)
                    
                    # Team 2 box
                    rect2 = mpatches.Rectangle((x_pos - box_width/2, y_pos - box_height/2), 
                                             box_width, box_height/2,
                                             facecolor=self.get_team_color(team2), 
                                             alpha=0.7, edgecolor='black')
                    ax.add_patch(rect2)
                    
                    # Add team labels and probabilities
                    ax.text(x_pos, y_pos + box_height/4, f"{team1}\n{team1_prob:.1%}", 
                           ha='center', va='center', fontweight='bold', fontsize=10)
                    ax.text(x_pos, y_pos - box_height/4, f"{team2}\n{team2_prob:.1%}", 
                           ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('NHL Playoff Bracket Predictions', fontsize=18, fontweight='bold', pad=20)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'playoff_bracket.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved playoff bracket chart: {save_path}")
        return save_path
    
    def create_prediction_confidence_chart(self, predictions_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create prediction confidence visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Confidence distribution
        confidences = predictions_df.get('confidence', [])
        if len(confidences) > 0:
            ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Prediction Confidence')
            ax1.set_ylabel('Number of Matchups')
            ax1.set_title('Distribution of Prediction Confidence')
            ax1.grid(True, alpha=0.3)
        
        # Model agreement
        if 'lr_prob' in predictions_df.columns and 'xgb_prob' in predictions_df.columns:
            lr_probs = predictions_df['lr_prob']
            xgb_probs = predictions_df['xgb_prob']
            
            ax2.scatter(lr_probs, xgb_probs, alpha=0.7, s=100)
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')
            ax2.set_xlabel('Logistic Regression Probability')
            ax2.set_ylabel('XGBoost Probability')
            ax2.set_title('Model Agreement Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_confidence.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved prediction confidence chart: {save_path}")
        return save_path
    
    def create_feature_importance_chart(self, feature_importance_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create feature importance visualization"""
        if feature_importance_df.empty:
            logger.warning("No feature importance data available")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(feature_importance_df)), 
                      feature_importance_df['importance'], 
                      color='steelblue', alpha=0.7)
        
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels(feature_importance_df['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Model Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved feature importance chart: {save_path}")
        return save_path
    
    def create_season_performance_chart(self, team_data: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create season performance overview chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Points vs Goal Differential
        if 'points' in team_data.columns and 'goalDifferential' in team_data.columns:
            colors = [self.get_team_color(abbrev) for abbrev in team_data.get('teamAbbrev', [])]
            ax1.scatter(team_data['goalDifferential'], team_data['points'], 
                       c=colors, s=100, alpha=0.7)
            ax1.set_xlabel('Goal Differential')
            ax1.set_ylabel('Points')
            ax1.set_title('Points vs Goal Differential')
            ax1.grid(True, alpha=0.3)
            
            # Add team labels for top teams
            top_teams = team_data.nlargest(5, 'points')
            for _, team in top_teams.iterrows():
                ax1.annotate(team.get('teamAbbrev', ''), 
                           (team.get('goalDifferential', 0), team.get('points', 0)),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Power Play vs Penalty Kill
        if 'PP%' in team_data.columns and 'PK%' in team_data.columns:
            # Convert percentage strings to floats if needed
            pp_pct = team_data['PP%']
            pk_pct = team_data['PK%']
            
            # Handle string percentages
            if pd.api.types.is_object_dtype(pp_pct):
                pp_pct = pd.to_numeric(pp_pct.str.rstrip('%'), errors='coerce') / 100
            if pd.api.types.is_object_dtype(pk_pct):
                pk_pct = pd.to_numeric(pk_pct.str.rstrip('%'), errors='coerce') / 100
            
            colors = [self.get_team_color(abbrev) for abbrev in team_data.get('teamAbbrev', [])]
            ax2.scatter(pp_pct, pk_pct, c=colors, s=100, alpha=0.7)
            ax2.set_xlabel('Power Play %')
            ax2.set_ylabel('Penalty Kill %')
            ax2.set_title('Special Teams Performance')
            ax2.grid(True, alpha=0.3)
        
        # Goals For vs Goals Against
        if 'goalsFor' in team_data.columns and 'goalsAgainst' in team_data.columns:
            colors = [self.get_team_color(abbrev) for abbrev in team_data.get('teamAbbrev', [])]
            ax3.scatter(team_data['goalsFor'], team_data['goalsAgainst'], 
                       c=colors, s=100, alpha=0.7)
            ax3.set_xlabel('Goals For')
            ax3.set_ylabel('Goals Against')
            ax3.set_title('Offensive vs Defensive Performance')
            ax3.grid(True, alpha=0.3)
        
        # Wins Distribution
        if 'wins' in team_data.columns:
            ax4.hist(team_data['wins'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            ax4.set_xlabel('Wins')
            ax4.set_ylabel('Number of Teams')
            ax4.set_title('Distribution of Wins')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'season_performance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved season performance chart: {save_path}")
        return save_path