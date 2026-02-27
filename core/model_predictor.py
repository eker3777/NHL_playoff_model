"""
Model loading and prediction module for NHL playoff predictions.
Extracted from Streamlit app for standalone use.
"""

import os
import joblib
import numpy as np
import pandas as pd
import sys
import subprocess
import json
import warnings
from typing import Dict, Any, Tuple, Optional
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class NHLModelPredictor:
    """NHL model loading and prediction class"""
    
    def __init__(self, model_folder: str):
        self.model_folder = model_folder
        self.models = {'models': {}}
        
        # Ensure model directory exists
        os.makedirs(model_folder, exist_ok=True)
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self) -> Dict[str, Any]:
        """Load the trained machine learning models for playoff predictions.
        
        Returns:
            dict: Dictionary containing model information and objects
        """
        
        # Define model paths
        lr_path = os.path.join(self.model_folder, 'logistic_regression_model_final.pkl')
        xgb_path = os.path.join(self.model_folder, 'xgboost_playoff_model_final.pkl')
        default_path = os.path.join(self.model_folder, 'playoff_model.pkl')
        
        # List available files for debugging
        available_files = os.listdir(self.model_folder) if os.path.exists(self.model_folder) else []
        logger.info(f"Available model files: {', '.join(available_files) if available_files else 'None'}")
        
        # Check if xgboost is installed, install if missing
        try:
            import xgboost
        except ImportError:
            logger.info("Installing XGBoost package...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
                import xgboost
                logger.info("XGBoost installed successfully")
            except Exception as e:
                logger.error(f"Error installing XGBoost: {str(e)}")
        
        # Try to load the models
        try:
            # Check if LR model exists and load it
            if os.path.exists(lr_path):
                try:
                    lr_model_package = joblib.load(lr_path)
                    
                    # Check if we have a dictionary with model inside or direct model
                    if isinstance(lr_model_package, dict) and 'model' in lr_model_package:
                        # New format: dictionary with model and features
                        self.models['models']['lr'] = lr_model_package
                    else:
                        # Old format: direct model object
                        # Wrap it in our expected format
                        self.models['models']['lr'] = {
                            'model': lr_model_package,
                            'features': [col for col in lr_model_package.feature_names_in_] if hasattr(lr_model_package, 'feature_names_in_') else []
                        }
                        
                    # Verify the model has predict_proba method
                    model_obj = self.models['models']['lr']['model']
                    if not hasattr(model_obj, 'predict_proba'):
                        logger.warning("LR model doesn't have predict_proba method!")
                    else:
                        logger.info("✓ Loaded Logistic Regression model with predict_proba")
                        
                except Exception as lr_error:
                    logger.error(f"Failed to load LR model: {str(lr_error)}")
            
            # Check if XGB model exists and load it
            if os.path.exists(xgb_path):
                try:
                    xgb_model_package = joblib.load(xgb_path)
                    
                    # Check if we have a dictionary with model inside or direct model
                    if isinstance(xgb_model_package, dict) and 'model' in xgb_model_package:
                        # New format: dictionary with model and features
                        self.models['models']['xgb'] = xgb_model_package
                    else:
                        # Old format: direct model object
                        # Wrap it in our expected format
                        self.models['models']['xgb'] = {
                            'model': xgb_model_package,
                            'features': [col for col in xgb_model_package.feature_names_in_] if hasattr(xgb_model_package, 'feature_names_in_') else []
                        }
                        
                    # Verify the model has predict_proba method
                    model_obj = self.models['models']['xgb']['model']
                    if not hasattr(model_obj, 'predict_proba'):
                        logger.warning("XGB model doesn't have predict_proba method!")
                    else:
                        logger.info("✓ Loaded XGBoost model with predict_proba")
                        
                except Exception as xgb_error:
                    logger.error(f"Failed to load XGB model: {str(xgb_error)}")
                
            # Check if default model exists and load it
            if os.path.exists(default_path):
                try:
                    default_model = joblib.load(default_path)
                    # If combined is a dict with models inside, merge with our models
                    if isinstance(default_model, dict) and 'models' in default_model:
                        for model_key, model_value in default_model['models'].items():
                            self.models['models'][model_key] = model_value
                        # Copy other keys
                        for key, value in default_model.items():
                            if key != 'models':
                                self.models[key] = value
                        logger.info("✓ Loaded combined model package")
                    else:
                        # Otherwise just add as 'combined'
                        self.models['models']['combined'] = default_model
                        logger.info("✓ Loaded default model")
                except Exception as default_error:
                    logger.error(f"Failed to load default model: {str(default_error)}")
                
            # Set model mode and parameters
            if not self.models['models']:
                logger.warning("No models found - using basic predictions")
                self.models['mode'] = 'default'
                self.models['home_ice_boost'] = 0.039
            else:
                # Determine which mode to use based on available models
                if 'lr' in self.models['models'] and 'xgb' in self.models['models']:
                    self.models['mode'] = 'ensemble'
                    logger.info("Using ensemble mode (LR + XGB)")
                elif 'lr' in self.models['models']:
                    self.models['mode'] = 'lr'
                    logger.info("Using Logistic Regression model only")
                elif 'xgb' in self.models['models']:
                    self.models['mode'] = 'xgb'
                    logger.info("Using XGBoost model only")
                else:
                    self.models['mode'] = 'default'
                    logger.info("Using default model mode")
                
                # Ensure home_ice_boost is set
                if 'home_ice_boost' not in self.models:
                    self.models['home_ice_boost'] = 0.039
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models = {
                'models': {},
                'mode': 'default',
                'home_ice_boost': 0.039
            }
            return self.models
    
    def create_matchup_data(self, team_data: pd.DataFrame, top_seed: Any, bottom_seed: Any) -> pd.DataFrame:
        """Create matchup data for prediction with feature engineering
        
        Args:
            team_data: DataFrame with team statistics
            top_seed: Higher seeded team information (dict or string)
            bottom_seed: Lower seeded team information (dict or string)
            
        Returns:
            DataFrame: Matchup data with engineered features
        """
        # Initialize matchup data dictionary
        matchup_data = {}
        matchup_data['round_name'] = 'First Round'
        matchup_data['series_letter'] = 'TBD'
        
        # Ensure top_seed and bottom_seed are dicts with required keys
        top_abbrev = top_seed.get('teamAbbrev', '') if isinstance(top_seed, dict) else top_seed
        bottom_abbrev = bottom_seed.get('teamAbbrev', '') if isinstance(bottom_seed, dict) else bottom_seed
        
        # If teamAbbrev key not found, treat the entire input as the abbreviation
        if top_abbrev == '' and isinstance(top_seed, str):
            top_abbrev = top_seed
        if bottom_abbrev == '' and isinstance(bottom_seed, str):
            bottom_abbrev = bottom_seed
            
        matchup_data['top_seed_abbrev'] = top_abbrev
        matchup_data['bottom_seed_abbrev'] = bottom_abbrev
        matchup_data['top_seed_rank'] = top_seed.get('division_rank', top_seed.get('wildcard_rank', 0)) if isinstance(top_seed, dict) else 0
        matchup_data['bottom_seed_rank'] = bottom_seed.get('division_rank', bottom_seed.get('wildcard_rank', 0)) if isinstance(bottom_seed, dict) else 0
        matchup_data['top_seed_wins'] = 0
        matchup_data['bottom_seed_wins'] = 0
        
        logger.debug(f"Creating matchup data for {top_abbrev} vs {bottom_abbrev}")
        
        # Get team data for each team - use teamAbbrev to filter
        if 'teamAbbrev' in team_data.columns:
            top_team_filter = team_data['teamAbbrev'] == top_abbrev
            bottom_team_filter = team_data['teamAbbrev'] == bottom_abbrev
            
            # Check if team data exists
            if sum(top_team_filter) > 0 and sum(bottom_team_filter) > 0:
                top_seed_data = team_data[top_team_filter].iloc[0]
                bottom_seed_data = team_data[bottom_team_filter].iloc[0]
                
                # Points difference (useful as a fallback)
                if 'points' in top_seed_data and 'points' in bottom_seed_data:
                    matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
                
                # Feature columns to use for prediction
                feature_cols = [
                    'PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score',
                    'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%',
                    'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg',
                    'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60',
                    'adjGoalsScoredAboveX/60'
                ]
                
                # Add features for each team if available
                features_added = 0
                for col in feature_cols:
                    if col in top_seed_data and col in bottom_seed_data:
                        # Only add if both values are not NaN
                        if pd.notna(top_seed_data[col]) and pd.notna(bottom_seed_data[col]):
                            matchup_data[f"{col}_top"] = top_seed_data[col]
                            matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                            matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
                            features_added += 1
                        else:
                            logger.debug(f"Skipping feature {col} - missing values")
                
                # Add basic features that are always available
                basic_features = ['points', 'wins', 'losses', 'goalsFor', 'goalsAgainst']
                for col in basic_features:
                    if col in top_seed_data and col in bottom_seed_data:
                        if pd.notna(top_seed_data[col]) and pd.notna(bottom_seed_data[col]):
                            matchup_data[f"{col}_top"] = top_seed_data[col]
                            matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                            matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
                            features_added += 1
                
                logger.debug(f"Added {features_added} features for matchup {top_abbrev} vs {bottom_abbrev}")
                    
                # Add fallback features if necessary
                if features_added == 0:
                    logger.warning(f"No model features available for {top_abbrev} vs {bottom_abbrev}")
                    logger.debug(f"Using points difference as fallback: {matchup_data.get('points_diff', 'N/A')}")
            else:
                logger.warning(f"Team data not found for {top_abbrev} or {bottom_abbrev}")
        else:
            logger.warning("'teamAbbrev' column not found in team_data")
        
        matchup_df = pd.DataFrame([matchup_data])
        
        # Check and report on the quality of the matchup data
        diff_features = [col for col in matchup_df.columns if col.endswith('_diff')]
        logger.debug(f"Created matchup data with {len(diff_features)} differential features")
        
        return matchup_df
    
    def predict_series_winner(self, matchup_df: pd.DataFrame) -> Tuple[float, float, float]:
        """Predict the winner of a playoff series using available models
        
        Args:
            matchup_df: DataFrame with matchup features
            
        Returns:
            tuple: (ensemble_prob, lr_prob, xgb_prob) - Raw probabilities of top seed winning (no home ice boost)
        """
        # Default probabilities
        lr_prob = 0.5
        xgb_prob = 0.5
        ensemble_prob = 0.5
        
        # Check for empty dataframe
        if matchup_df is None or matchup_df.empty:
            logger.warning("Empty matchup dataframe provided")
            return ensemble_prob, lr_prob, xgb_prob
        
        # Get raw model predictions
        lr_prob, xgb_prob = self.get_raw_model_predictions(matchup_df)
        
        # Calculate ensemble prediction
        if lr_prob != 0.5 and xgb_prob != 0.5:
            # Weight the predictions (can be adjusted)
            ensemble_prob = 0.4 * lr_prob + 0.6 * xgb_prob
        elif lr_prob != 0.5:
            ensemble_prob = lr_prob
        elif xgb_prob != 0.5:
            ensemble_prob = xgb_prob
        else:
            # Fallback: Use points difference if available
            if 'points_diff' in matchup_df.columns and not pd.isna(matchup_df['points_diff'].iloc[0]):
                points_diff = matchup_df['points_diff'].iloc[0]
                # Convert points difference to probability (roughly)
                ensemble_prob = 0.5 + (points_diff / 200)  # Adjust scaling as needed
                ensemble_prob = max(0.1, min(0.9, ensemble_prob))  # Clamp between 0.1 and 0.9
            
        logger.debug(f"Predictions - LR: {lr_prob:.3f}, XGB: {xgb_prob:.3f}, Ensemble: {ensemble_prob:.3f}")
        
        return ensemble_prob, lr_prob, xgb_prob
    
    def get_raw_model_predictions(self, matchup_df: pd.DataFrame) -> Tuple[float, float]:
        """Get raw model predictions without home ice advantage
        
        Args:
            matchup_df: DataFrame with matchup features
            
        Returns:
            tuple: (lr_prob, xgb_prob) - Raw probabilities from each model
        """
        # Default probabilities
        lr_prob = 0.5
        xgb_prob = 0.5
        
        # Check for empty dataframe
        if matchup_df is None or matchup_df.empty:
            return lr_prob, xgb_prob
        
        # Check which models we have available and make predictions
        if 'models' in self.models:
            # Try Logistic Regression model
            if 'lr' in self.models['models']:
                try:
                    # Extract model and features if available
                    lr_model = self.models['models']['lr']['model']
                    
                    # Get features if available in model
                    if 'features' in self.models['models']['lr']:
                        lr_features = [f for f in self.models['models']['lr']['features'] if f in matchup_df.columns]
                    elif hasattr(lr_model, 'feature_names_in_'):
                        lr_features = [f for f in lr_model.feature_names_in_ if f in matchup_df.columns]
                    else:
                        # Default features - columns ending with '_diff'
                        lr_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                    
                    # Make prediction if we have features
                    if lr_features and len(lr_features) > 0 and hasattr(lr_model, 'predict_proba'):
                        try:
                            prediction_data = matchup_df[lr_features].fillna(0)  # Fill missing with 0
                            lr_prob = lr_model.predict_proba(prediction_data)[:, 1][0]
                        except Exception as pred_error:
                            logger.warning(f"LR prediction failed with available features: {pred_error}")
                            # Use default features if available
                            basic_features = [col for col in matchup_df.columns if col.endswith('_diff') and 'points' in col or 'goals' in col]
                            if basic_features:
                                lr_prob = lr_model.predict_proba(matchup_df[basic_features[:1]].fillna(0))[:, 1][0]
                except Exception as e:
                    logger.error(f"Error in LR prediction: {str(e)}")
            
            # Try XGBoost model
            if 'xgb' in self.models['models']:
                try:
                    # Extract model and features
                    xgb_model = self.models['models']['xgb']['model']
                    
                    # Get features if available in model
                    if 'features' in self.models['models']['xgb']:
                        xgb_features = [f for f in self.models['models']['xgb']['features'] if f in matchup_df.columns]
                    elif hasattr(xgb_model, 'feature_names_in_'):
                        xgb_features = [f for f in xgb_model.feature_names_in_ if f in matchup_df.columns]
                    else:
                        # Default features - columns ending with '_diff'
                        xgb_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                    
                    # Make prediction if we have features
                    if xgb_features and len(xgb_features) > 0 and hasattr(xgb_model, 'predict_proba'):
                        try:
                            prediction_data = matchup_df[xgb_features].fillna(0)  # Fill missing with 0
                            xgb_prob = xgb_model.predict_proba(prediction_data)[:, 1][0]
                        except Exception as pred_error:
                            logger.warning(f"XGB prediction failed with available features: {pred_error}")
                            # Use default features if available
                            basic_features = [col for col in matchup_df.columns if col.endswith('_diff') and ('points' in col or 'goals' in col)]
                            if basic_features:
                                xgb_prob = xgb_model.predict_proba(matchup_df[basic_features[:1]].fillna(0))[:, 1][0]
                except Exception as e:
                    logger.error(f"Error in XGB prediction: {str(e)}")
        
        return lr_prob, xgb_prob
    
    def predict_with_home_ice(self, ensemble_prob: float) -> float:
        """Apply home ice advantage to prediction
        
        Args:
            ensemble_prob: Raw ensemble probability
            
        Returns:
            float: Probability with home ice advantage applied
        """
        home_ice_boost = self.models.get('home_ice_boost', 0.039)
        return min(0.95, ensemble_prob + home_ice_boost)