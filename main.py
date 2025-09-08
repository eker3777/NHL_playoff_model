"""
Main pipeline script for NHL Playoff Prediction System.

This script orchestrates the complete data loading, modeling, and reporting pipeline
for NHL playoff predictions, generating a comprehensive markdown report with visualizations.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.data_loader import NHLDataLoader
from core.model_predictor import NHLModelPredictor
from core.visualizations import NHLVisualizer
from core.report_generator import NHLReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nhl_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(season: int = None) -> Dict[str, Any]:
    """Load all required NHL data"""
    logger.info("=== Starting Data Loading Phase ===")
    
    if season is None:
        season = config.current_season
    
    season_str = f"{season}{season + 1}"
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    
    # Initialize data loader
    data_loader = NHLDataLoader(data_folder, season)
    
    # Load standings data
    logger.info("Loading standings data...")
    standings_raw = data_loader.get_standings_data()
    standings_df = data_loader.process_standings_data(standings_raw, season_str) if standings_raw else None
    
    # Load team stats data
    logger.info("Loading team statistics...")
    stats_raw = data_loader.get_team_stats_data(season)
    stats_df = data_loader.process_team_stats_data(stats_raw) if stats_raw else None
    
    # Load advanced stats data
    logger.info("Loading advanced statistics...")
    advanced_df = data_loader.get_advanced_stats_data(season)
    
    # Combine all data
    logger.info("Combining data sources...")
    team_data = combine_team_data(standings_df, stats_df, advanced_df, season_str)
    
    # Save processed data
    if team_data is not None and not team_data.empty:
        data_path = os.path.join(data_folder, f"team_data_{season_str}.csv")
        team_data.to_csv(data_path, index=False)
        logger.info(f"✓ Saved combined team data: {data_path}")
    
    return {
        'team_data': team_data,
        'standings_data': standings_df,
        'season_str': season_str,
        'season': season
    }


def combine_team_data(standings_df, stats_df, advanced_df, season_str):
    """Combine all team data sources"""
    import pandas as pd
    
    if standings_df is None or standings_df.empty:
        logger.warning("No standings data available")
        return pd.DataFrame()
    
    team_data = standings_df.copy()
    
    # Merge with stats data
    if stats_df is not None and not stats_df.empty:
        logger.info("Merging team statistics...")
        # Ensure both have season columns
        if 'season' in stats_df.columns:
            stats_df['season'] = stats_df['season'].astype(str)
        team_data['season'] = team_data['season'].astype(str)
        
        # Merge on teamName and season
        team_data = pd.merge(
            team_data, stats_df,
            on=['season', 'teamName'],
            how='left',
            suffixes=('', '_stats')
        )
        logger.info(f"✓ Merged with stats data: {len(team_data)} teams")
    
    # Merge with advanced stats
    if advanced_df is not None and not advanced_df.empty:
        logger.info("Merging advanced statistics...")
        advanced_df['season'] = advanced_df['season'].astype(str)
        
        # Merge on teamName and season
        team_data = pd.merge(
            team_data, advanced_df,
            on=['season', 'teamName'],
            how='left',
            suffixes=('', '_advanced')
        )
        logger.info(f"✓ Merged with advanced stats: {len(team_data)} teams")
    
    # Engineer features
    team_data = engineer_features(team_data)
    
    return team_data


def engineer_features(df):
    """Engineer features for the prediction model"""
    import pandas as pd
    import numpy as np
    
    logger.info("Engineering features...")
    
    if df.empty:
        return df
    
    # Convert percentage strings to floats where needed
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # Check if this column contains percentage strings
                if df[col].astype(str).str.contains('%').any():
                    logger.debug(f"Converting percentage column: {col}")
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
            except Exception as e:
                logger.debug(f"Error converting column {col}: {str(e)}")
    
    # Calculate goal differential per game
    if all(col in df.columns for col in ['goalsFor', 'goalsAgainst', 'gamesPlayed']):
        df['goalDiff/G'] = (df['goalsFor'] - df['goalsAgainst']) / df['gamesPlayed'].clip(lower=1)
    
    # Calculate home/road regulation win percentages
    if all(col in df.columns for col in ['homeRegulationWins', 'gamesPlayed']):
        df['homeRegulationWin%'] = df['homeRegulationWins'] / df['gamesPlayed'].clip(lower=1)
    
    if all(col in df.columns for col in ['roadRegulationWins', 'gamesPlayed']):
        df['roadRegulationWin%'] = df['roadRegulationWins'] / df['gamesPlayed'].clip(lower=1)
    
    # Calculate relative percentages for special teams
    if 'PP%' in df.columns:
        league_avg_pp = df['PP%'].mean()
        df['PP%_rel'] = df['PP%'] - league_avg_pp
    
    if 'PK%' in df.columns:
        league_avg_pk = df['PK%'].mean()
        df['PK%_rel'] = df['PK%'] - league_avg_pk
    
    # Add playoff history metrics (simplified)
    df['playoff_performance_score'] = 0.5  # Default neutral score
    
    logger.info(f"✓ Feature engineering completed: {len(df.columns)} total features")
    return df


def generate_predictions(team_data, models_folder):
    """Generate playoff predictions"""
    logger.info("=== Starting Prediction Phase ===")
    
    # Initialize model predictor
    model_predictor = NHLModelPredictor(models_folder)
    
    # Generate sample playoff matchups (simplified for demo)
    predictions = generate_sample_matchups(team_data, model_predictor)
    
    logger.info(f"✓ Generated {len(predictions)} predictions")
    return predictions, model_predictor.models


def generate_sample_matchups(team_data, model_predictor):
    """Generate sample playoff matchups and predictions"""
    import pandas as pd
    
    if team_data.empty:
        logger.warning("No team data available for predictions")
        return []
    
    predictions = []
    
    # Sort teams by points to simulate playoff seeding
    if 'points' in team_data.columns:
        sorted_teams = team_data.sort_values('points', ascending=False)
        
        # Create sample first round matchups
        matchups = [
            (0, 7), (1, 6), (2, 5), (3, 4),  # Eastern Conference
            (8, 15), (9, 14), (10, 13), (11, 12)  # Western Conference
        ]
        
        for i, (top_idx, bottom_idx) in enumerate(matchups):
            if top_idx < len(sorted_teams) and bottom_idx < len(sorted_teams):
                top_seed = sorted_teams.iloc[top_idx]
                bottom_seed = sorted_teams.iloc[bottom_idx]
                
                # Create matchup data
                matchup_df = model_predictor.create_matchup_data(
                    team_data, 
                    {'teamAbbrev': top_seed.get('teamAbbrev', f'Team{top_idx+1}')},
                    {'teamAbbrev': bottom_seed.get('teamAbbrev', f'Team{bottom_idx+1}')}
                )
                
                # Generate prediction
                ensemble_prob, lr_prob, xgb_prob = model_predictor.predict_series_winner(matchup_df)
                
                # Calculate confidence (how far from 50/50)
                confidence = abs(ensemble_prob - 0.5) * 2
                
                prediction = {
                    'top_seed_abbrev': top_seed.get('teamAbbrev', f'Team{top_idx+1}'),
                    'bottom_seed_abbrev': bottom_seed.get('teamAbbrev', f'Team{bottom_idx+1}'),
                    'ensemble_prob': ensemble_prob,
                    'lr_prob': lr_prob,
                    'xgb_prob': xgb_prob,
                    'confidence': confidence,
                    'round_name': 'First Round'
                }
                
                predictions.append(prediction)
                
                logger.debug(f"Prediction: {prediction['top_seed_abbrev']} vs {prediction['bottom_seed_abbrev']} - {ensemble_prob:.1%}")
    
    return predictions


def generate_visualizations(team_data, standings_data, predictions, reports_dir):
    """Generate all visualizations"""
    logger.info("=== Starting Visualization Phase ===")
    
    visualizer = NHLVisualizer(reports_dir)
    chart_paths = {}
    
    # Generate standings chart
    if standings_data is not None and not standings_data.empty:
        chart_paths['standings'] = visualizer.create_standings_chart(standings_data)
    
    # Generate season performance chart
    if team_data is not None and not team_data.empty:
        chart_paths['season_performance'] = visualizer.create_season_performance_chart(team_data)
    
    # Generate playoff bracket chart
    if predictions:
        chart_paths['playoff_bracket'] = visualizer.create_playoff_bracket_chart(predictions)
    
    # Generate prediction confidence chart
    if predictions:
        import pandas as pd
        predictions_df = pd.DataFrame(predictions)
        chart_paths['prediction_confidence'] = visualizer.create_prediction_confidence_chart(predictions_df)
    
    # Generate feature importance chart (if available)
    # This would require actual feature importance data from models
    # chart_paths['feature_importance'] = visualizer.create_feature_importance_chart(feature_df)
    
    logger.info(f"✓ Generated {len(chart_paths)} visualizations")
    return chart_paths


def generate_report(data_dict, predictions, model_info, chart_paths, reports_dir):
    """Generate final markdown report"""
    logger.info("=== Starting Report Generation Phase ===")
    
    report_generator = NHLReportGenerator(reports_dir)
    
    report_path = report_generator.generate_comprehensive_report(
        season_str=data_dict['season_str'],
        team_data=data_dict['team_data'],
        standings_data=data_dict['standings_data'],
        predictions=predictions,
        chart_paths=chart_paths,
        model_info=model_info
    )
    
    logger.info(f"✓ Generated comprehensive report: {report_path}")
    return report_path


def main(season: int = None):
    """Main pipeline execution"""
    start_time = datetime.now()
    logger.info(f"=== NHL Playoff Prediction Pipeline Started ===")
    logger.info(f"Timestamp: {start_time}")
    
    try:
        # Set up directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(base_dir, "data")
        models_folder = os.path.join(base_dir, "models")
        reports_dir = os.path.join(base_dir, "reports")
        
        # Create directories
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(models_folder, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Phase 1: Load Data
        data_dict = load_data(season)
        
        # Phase 2: Generate Predictions  
        predictions, model_info = generate_predictions(data_dict['team_data'], models_folder)
        
        # Phase 3: Generate Visualizations
        chart_paths = generate_visualizations(
            data_dict['team_data'], 
            data_dict['standings_data'], 
            predictions, 
            reports_dir
        )
        
        # Phase 4: Generate Report
        report_path = generate_report(data_dict, predictions, model_info, chart_paths, reports_dir)
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=== Pipeline Completed Successfully ===")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Final report: {report_path}")
        logger.info(f"Charts generated: {len(chart_paths)}")
        logger.info(f"Predictions made: {len(predictions)}")
        
        return {
            'success': True,
            'report_path': report_path,
            'chart_paths': chart_paths,
            'predictions': predictions,
            'execution_time': duration
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Allow season override from command line
    season = None
    if len(sys.argv) > 1:
        try:
            season = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid season provided. Using current season.")
    
    result = main(season)
    
    if result['success']:
        print(f"\n✓ Pipeline completed successfully!")
        print(f"Report: {result['report_path']}")
        print(f"Execution time: {result['execution_time']}")
    else:
        print(f"\n✗ Pipeline failed: {result['error']}")
        sys.exit(1)