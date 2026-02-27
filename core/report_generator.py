"""
Report generation module for NHL playoff predictions.
Creates comprehensive markdown reports with analysis and visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NHLReportGenerator:
    """NHL playoff prediction report generator"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.season_str = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    season_str: str,
                                    team_data: pd.DataFrame,
                                    standings_data: pd.DataFrame,
                                    predictions: List[Dict],
                                    chart_paths: Dict[str, str],
                                    model_info: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report
        
        Args:
            season_str: Season string (e.g., '20242025')
            team_data: DataFrame with team statistics
            standings_data: DataFrame with standings
            predictions: List of prediction dictionaries
            chart_paths: Dictionary mapping chart names to file paths
            model_info: Model information and metadata
            
        Returns:
            str: Path to generated markdown report
        """
        self.season_str = season_str
        
        # Create report content
        report_content = self._generate_header()
        report_content += self._generate_executive_summary(predictions, model_info)
        report_content += self._generate_season_overview(team_data, standings_data)
        report_content += self._generate_predictions_section(predictions)
        report_content += self._generate_model_analysis(model_info)
        report_content += self._generate_visualizations_section(chart_paths)
        report_content += self._generate_methodology_section()
        report_content += self._generate_appendix(team_data)
        
        # Save report
        report_path = os.path.join(self.output_dir, f"NHL_Playoff_Report_{season_str}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✓ Generated comprehensive report: {report_path}")
        return report_path
    
    def _generate_header(self) -> str:
        """Generate report header"""
        return f"""# NHL Playoff Prediction Report
## Season {self.season_str}

**Generated:** {self.timestamp}
**Analysis Type:** Machine Learning Playoff Predictions
**Data Sources:** NHL API, MoneyPuck Advanced Statistics

---

"""
    
    def _generate_executive_summary(self, predictions: List[Dict], model_info: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        content = """## Executive Summary

This report provides comprehensive NHL playoff predictions using advanced machine learning models and statistical analysis. Our ensemble approach combines multiple algorithms to predict playoff matchup outcomes with high accuracy.

### Key Findings

"""
        
        # Calculate summary statistics
        if predictions:
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in predictions])
            high_confidence_predictions = len([p for p in predictions if p.get('confidence', 0.5) > 0.7])
            
            content += f"- **Total Predictions Generated:** {len(predictions)}\n"
            content += f"- **Average Prediction Confidence:** {avg_confidence:.1%}\n"
            content += f"- **High Confidence Predictions:** {high_confidence_predictions}\n"
        
        # Model information
        models_used = list(model_info.get('models', {}).keys())
        if models_used:
            content += f"- **Models Utilized:** {', '.join(models_used).upper()}\n"
            content += f"- **Prediction Mode:** {model_info.get('mode', 'Default').title()}\n"
        
        content += "\n"
        return content
    
    def _generate_season_overview(self, team_data: pd.DataFrame, standings_data: pd.DataFrame) -> str:
        """Generate season overview section"""
        content = """## Season Overview

### Current Standings Summary

"""
        
        if not standings_data.empty:
            # Eastern Conference top teams
            east_teams = standings_data[standings_data.get('conferenceAbbrev', '') == 'E'].head(8)
            if not east_teams.empty:
                content += "#### Eastern Conference Playoff Picture\n\n"
                content += "| Rank | Team | Points | Wins | Losses | OT/SO |\n"
                content += "|------|------|--------|------|--------|-------|\n"
                
                for i, (_, team) in enumerate(east_teams.iterrows(), 1):
                    content += f"| {i} | {team.get('teamAbbrev', 'TBD')} | {team.get('points', 0)} | {team.get('wins', 0)} | {team.get('losses', 0)} | {team.get('otLosses', 0)} |\n"
                
                content += "\n"
            
            # Western Conference top teams
            west_teams = standings_data[standings_data.get('conferenceAbbrev', '') == 'W'].head(8)
            if not west_teams.empty:
                content += "#### Western Conference Playoff Picture\n\n"
                content += "| Rank | Team | Points | Wins | Losses | OT/SO |\n"
                content += "|------|------|--------|------|--------|-------|\n"
                
                for i, (_, team) in enumerate(west_teams.iterrows(), 1):
                    content += f"| {i} | {team.get('teamAbbrev', 'TBD')} | {team.get('points', 0)} | {team.get('wins', 0)} | {team.get('losses', 0)} | {team.get('otLosses', 0)} |\n"
                
                content += "\n"
        
        # Season statistics
        if not team_data.empty:
            content += "### Season Statistics Overview\n\n"
            
            # Top performers
            if 'points' in team_data.columns:
                top_team = team_data.loc[team_data['points'].idxmax()]
                content += f"- **League Leader (Points):** {top_team.get('teamAbbrev', 'TBD')} ({top_team.get('points', 0)} points)\n"
            
            if 'goalsFor' in team_data.columns:
                top_offense = team_data.loc[team_data['goalsFor'].idxmax()]
                content += f"- **Top Offense:** {top_offense.get('teamAbbrev', 'TBD')} ({top_offense.get('goalsFor', 0)} goals for)\n"
            
            if 'goalsAgainst' in team_data.columns:
                best_defense = team_data.loc[team_data['goalsAgainst'].idxmin()]
                content += f"- **Best Defense:** {best_defense.get('teamAbbrev', 'TBD')} ({best_defense.get('goalsAgainst', 0)} goals against)\n"
            
            content += "\n"
        
        return content
    
    def _generate_predictions_section(self, predictions: List[Dict]) -> str:
        """Generate predictions section"""
        content = """## Playoff Predictions

### First Round Matchup Predictions

"""
        
        if predictions:
            content += "| Matchup | Higher Seed Win Probability | Confidence | Model Consensus |\n"
            content += "|---------|----------------------------|------------|------------------|\n"
            
            for pred in predictions:
                top_seed = pred.get('top_seed_abbrev', 'TBD')
                bottom_seed = pred.get('bottom_seed_abbrev', 'TBD')
                probability = pred.get('ensemble_prob', 0.5)
                confidence = pred.get('confidence', 0.5)
                
                # Model consensus
                lr_prob = pred.get('lr_prob', 0.5)
                xgb_prob = pred.get('xgb_prob', 0.5)
                consensus = "High" if abs(lr_prob - xgb_prob) < 0.1 else "Medium" if abs(lr_prob - xgb_prob) < 0.2 else "Low"
                
                content += f"| {top_seed} vs {bottom_seed} | {probability:.1%} | {confidence:.1%} | {consensus} |\n"
            
            content += "\n"
            
            # Upset alerts
            content += "### Upset Alerts\n\n"
            upset_candidates = [p for p in predictions if p.get('ensemble_prob', 0.5) < 0.6]
            
            if upset_candidates:
                content += "Teams with upset potential (lower seed win probability > 40%):\n\n"
                for pred in upset_candidates:
                    top_seed = pred.get('top_seed_abbrev', 'TBD')
                    bottom_seed = pred.get('bottom_seed_abbrev', 'TBD')
                    upset_prob = 1 - pred.get('ensemble_prob', 0.5)
                    content += f"- **{bottom_seed}** has a {upset_prob:.1%} chance against **{top_seed}**\n"
            else:
                content += "No significant upset potential detected in current matchups.\n"
            
            content += "\n"
        
        return content
    
    def _generate_model_analysis(self, model_info: Dict[str, Any]) -> str:
        """Generate model analysis section"""
        content = """## Model Analysis

### Model Performance Overview

"""
        
        models = model_info.get('models', {})
        mode = model_info.get('mode', 'default')
        
        content += f"**Prediction Mode:** {mode.title()}\n\n"
        
        if models:
            content += "#### Available Models\n\n"
            for model_name, model_data in models.items():
                if isinstance(model_data, dict):
                    features = model_data.get('features', [])
                    content += f"- **{model_name.upper()}:** {len(features)} features\n"
                else:
                    content += f"- **{model_name.upper()}:** Loaded successfully\n"
        
        content += f"\n**Home Ice Advantage:** +{model_info.get('home_ice_boost', 0.039):.1%}\n\n"
        
        # Feature importance (if available)
        content += "### Key Predictive Features\n\n"
        content += "Our models utilize advanced hockey analytics including:\n\n"
        content += "- **Expected Goals Percentage (xG%):** Advanced metric measuring shot quality\n"
        content += "- **Special Teams Performance:** Power play and penalty kill effectiveness\n"
        content += "- **Possession Metrics:** Corsi, Fenwick, and shot attempt percentages\n"
        content += "- **Playoff History:** Historical playoff performance weighting\n"
        content += "- **Regulation Win Percentage:** Home and road regulation win rates\n"
        content += "- **Advanced Defensive Metrics:** Goals saved above expected\n\n"
        
        return content
    
    def _generate_visualizations_section(self, chart_paths: Dict[str, str]) -> str:
        """Generate visualizations section"""
        content = """## Data Visualizations

The following charts provide visual analysis of the current season and predictions:

"""
        
        # Add each chart if it exists
        if 'standings' in chart_paths and chart_paths['standings']:
            content += "### Current Standings\n\n"
            content += f"![Standings Chart]({os.path.basename(chart_paths['standings'])})\n\n"
            content += "*Current NHL standings by conference showing point totals and playoff positioning.*\n\n"
        
        if 'season_performance' in chart_paths and chart_paths['season_performance']:
            content += "### Season Performance Analysis\n\n"
            content += f"![Season Performance]({os.path.basename(chart_paths['season_performance'])})\n\n"
            content += "*Comprehensive analysis of team performance across multiple statistical categories.*\n\n"
        
        if 'playoff_bracket' in chart_paths and chart_paths['playoff_bracket']:
            content += "### Predicted Playoff Bracket\n\n"
            content += f"![Playoff Bracket]({os.path.basename(chart_paths['playoff_bracket'])})\n\n"
            content += "*Predicted playoff bracket with win probabilities for each matchup.*\n\n"
        
        if 'prediction_confidence' in chart_paths and chart_paths['prediction_confidence']:
            content += "### Prediction Confidence Analysis\n\n"
            content += f"![Prediction Confidence]({os.path.basename(chart_paths['prediction_confidence'])})\n\n"
            content += "*Analysis of prediction confidence levels and model agreement.*\n\n"
        
        if 'feature_importance' in chart_paths and chart_paths['feature_importance']:
            content += "### Model Feature Importance\n\n"
            content += f"![Feature Importance]({os.path.basename(chart_paths['feature_importance'])})\n\n"
            content += "*Relative importance of different statistical features in the prediction models.*\n\n"
        
        return content
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section"""
        return """## Methodology

### Data Sources

1. **NHL Official API:** Team standings, basic statistics, and game results
2. **MoneyPuck:** Advanced analytics including expected goals, Corsi, and possession metrics
3. **Historical Playoff Data:** Multi-year playoff performance weighting

### Model Approach

Our ensemble approach combines multiple machine learning algorithms:

1. **Logistic Regression:** Baseline linear model with feature engineering
2. **XGBoost:** Gradient boosting model capturing non-linear relationships
3. **Ensemble Weighting:** Combines predictions with optimized weights (40% LR, 60% XGB)

### Feature Engineering

Key engineered features include:

- **Differential Metrics:** Head-to-head statistical comparisons
- **Relative Percentages:** Performance relative to league average
- **Playoff Performance Score:** Weighted historical playoff success
- **Advanced Possession Metrics:** Venue-adjusted advanced statistics

### Validation

Models are validated using:

- **Historical Cross-Validation:** Testing on previous seasons
- **Feature Importance Analysis:** Identifying key predictive variables
- **Confidence Scoring:** Measuring prediction reliability

---

"""
    
    def _generate_appendix(self, team_data: pd.DataFrame) -> str:
        """Generate appendix with detailed statistics"""
        content = """## Appendix

### Complete Team Statistics

"""
        
        if not team_data.empty:
            # Select key columns for the appendix table
            key_columns = ['teamAbbrev', 'points', 'wins', 'losses', 'otLosses', 
                          'goalsFor', 'goalsAgainst', 'PP%', 'PK%']
            
            available_columns = [col for col in key_columns if col in team_data.columns]
            
            if available_columns:
                appendix_df = team_data[available_columns].copy()
                
                # Sort by points
                if 'points' in appendix_df.columns:
                    appendix_df = appendix_df.sort_values('points', ascending=False)
                
                # Convert to markdown table
                content += appendix_df.to_markdown(index=False)
                content += "\n\n"
        
        content += """### Glossary

- **GP:** Games Played
- **W:** Wins
- **L:** Losses  
- **OTL:** Overtime/Shootout Losses
- **PTS:** Points (2 for win, 1 for OT/SO loss)
- **GF:** Goals For
- **GA:** Goals Against
- **PP%:** Power Play Percentage
- **PK%:** Penalty Kill Percentage
- **xG%:** Expected Goals Percentage
- **CF%:** Corsi For Percentage
- **FF%:** Fenwick For Percentage

---

*This report was generated using the NHL Playoff Prediction System - an advanced machine learning platform for hockey analytics.*

**Contact:** For questions about this analysis or the underlying methodology, please refer to the project documentation.

**Disclaimer:** These predictions are for analytical purposes only and should not be used for gambling or wagering activities.
"""
        
        return content