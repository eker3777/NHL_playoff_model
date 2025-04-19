import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys
import subprocess
import json
import warnings
warnings.filterwarnings('ignore')

# Import constants from central config
from streamlit_app.config import (
    HOME_ICE_ADVANTAGE,
    SERIES_LENGTH_DISTRIBUTION,
    CRITICAL_FEATURES,
    MODEL_DIR,
    DATA_DIR,
    DEFAULT_MODEL_MODE
)

# Import cache utilities
from streamlit_app.utils.cache_manager import get_current_time

# Function to load trained models
@st.cache_resource
def load_models(model_folder=None):
    """Load the trained machine learning models for playoff predictions.
    
    Args:
        model_folder (str): Path to the directory containing model files
        
    Returns:
        dict: Dictionary containing model information and objects
    """
    if model_folder is None:
        model_folder = MODEL_DIR
        
    # Ensure model directory exists
    os.makedirs(model_folder, exist_ok=True)
    
    # Dictionary to return with models
    models = {'models': {}}
    
    # Define model paths
    lr_path = os.path.join(model_folder, 'logistic_regression_model_final.pkl')
    xgb_path = os.path.join(model_folder, 'xgboost_playoff_model_final.pkl')
    ensemble_path = os.path.join(model_folder, 'ensemble_model.pkl')
    
    # Print actual paths for debugging
    print(f"Looking for models at:")
    print(f"LR model path: {lr_path}")
    print(f"XGB model path: {xgb_path}")
    print(f"Ensemble model path: {ensemble_path}")
    
    # List available files for debugging
    available_files = os.listdir(model_folder) if os.path.exists(model_folder) else []
    print(f"Available model files: {', '.join(available_files) if available_files else 'None'}")

    # Check if expected files exist
    if not os.path.exists(lr_path):
        print(f"WARNING: LR model file not found at {lr_path}")
    if not os.path.exists(xgb_path):
        print(f"WARNING: XGB model file not found at {xgb_path}")
    if not os.path.exists(ensemble_path):
        print(f"WARNING: Ensemble model file not found at {ensemble_path}")
        
    # Check if xgboost is installed, install if missing
    try:
        import xgboost
    except ImportError:
        print("Installing XGBoost package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
            import xgboost
            print("XGBoost installed successfully")
        except Exception as e:
            print(f"Error installing XGBoost: {str(e)}")
    
    # Try to load the models
    try:
        # Check if LR model exists and load it
        if os.path.exists(lr_path):
            try:
                lr_model_package = joblib.load(lr_path)
                
                # Check if we have a dictionary with model inside or direct model
                if isinstance(lr_model_package, dict) and 'model' in lr_model_package:
                    # New format: dictionary with model and features
                    models['models']['lr'] = lr_model_package
                else:
                    # Old format: direct model object
                    # Wrap it in our expected format
                    models['models']['lr'] = {
                        'model': lr_model_package,
                        'features': [col for col in lr_model_package.feature_names_in_] if hasattr(lr_model_package, 'feature_names_in_') else []
                    }
                    
                # Verify the model has predict_proba method
                model_obj = models['models']['lr']['model']
                if not hasattr(model_obj, 'predict_proba'):
                    print("WARNING: LR model doesn't have predict_proba method!")
                else:
                    print("✓ Loaded Logistic Regression model with predict_proba")
                    
            except Exception as lr_error:
                print(f"Failed to load LR model: {str(lr_error)}")
        
        # Check if XGB model exists and load it
        if os.path.exists(xgb_path):
            try:
                xgb_model_package = joblib.load(xgb_path)
                
                # Check if we have a dictionary with model inside or direct model
                if isinstance(xgb_model_package, dict) and 'model' in xgb_model_package:
                    # New format: dictionary with model and features
                    models['models']['xgb'] = xgb_model_package
                else:
                    # Old format: direct model object
                    # Wrap it in our expected format
                    models['models']['xgb'] = {
                        'model': xgb_model_package,
                        'features': [col for col in xgb_model_package.feature_names_in_] if hasattr(xgb_model_package, 'feature_names_in_') else []
                    }
                    
                # Verify the model has predict_proba method
                model_obj = models['models']['xgb']['model']
                if not hasattr(model_obj, 'predict_proba'):
                    print("WARNING: XGB model doesn't have predict_proba method!")
                else:
                    print("✓ Loaded XGBoost model with predict_proba")
                    
            except Exception as xgb_error:
                print(f"Failed to load XGB model: {str(xgb_error)}")
        
        # Check if ensemble model exists and load it
        if os.path.exists(ensemble_path):
            try:
                ensemble_model = joblib.load(ensemble_path)
                models['models']['ensemble'] = ensemble_model
                print("✓ Loaded ensemble model")
            except Exception as ensemble_error:
                print(f"Failed to load ensemble model: {str(ensemble_error)}")
        else:
            # Create ensemble model if we have both LR and XGB models
            if 'lr' in models['models'] and 'xgb' in models['models']:
                print("Creating ensemble model from LR and XGB models")
                try:
                    # Create a simple ensemble model that averages the predictions
                    ensemble_model = {
                        'lr_model': models['models']['lr'],
                        'xgb_model': models['models']['xgb'],
                        'features': list(set(
                            models['models']['lr'].get('features', []) + 
                            models['models']['xgb'].get('features', [])
                        )),
                        'creation_time': get_current_time().isoformat()
                    }
                    models['models']['ensemble'] = ensemble_model
                    
                    # Save the ensemble model for future use
                    try:
                        joblib.dump(ensemble_model, ensemble_path)
                        print(f"✓ Saved ensemble model to {ensemble_path}")
                    except Exception as save_error:
                        print(f"Error saving ensemble model: {str(save_error)}")
                        
                except Exception as create_error:
                    print(f"Error creating ensemble model: {str(create_error)}")
            
        # Set model mode and parameters
        if not models['models']:
            print("No models found - using basic predictions")
            models['mode'] = 'default'
            models['home_ice_boost'] = HOME_ICE_ADVANTAGE
        else:
            # Determine which mode to use based on available models and config
            if 'ensemble' in models['models']:
                models['mode'] = 'ensemble'
                print("Using ensemble model")
            elif 'lr' in models['models'] and 'xgb' in models['models']:
                models['mode'] = 'ensemble'
                print("Using ensemble mode (LR + XGB)")
            elif 'lr' in models['models']:
                models['mode'] = 'lr'
                print("Using Logistic Regression model only")
            elif 'xgb' in models['models']:
                models['mode'] = 'xgb'
                print("Using XGBoost model only")
            else:
                models['mode'] = 'default'
                print("Using default model mode")
            
            # Ensure home_ice_boost is set
            if 'home_ice_boost' not in models:
                models['home_ice_boost'] = HOME_ICE_ADVANTAGE
                
        # If we got this far without valid models, create a fallback model
        if not models['models'] or all(not hasattr(m.get('model', {}), 'predict_proba') 
                                      for m in models['models'].values() if isinstance(m, dict)):
            print("No valid models found - creating a basic logistic regression fallback model")
            try:
                from sklearn.linear_model import LogisticRegression
                # Create a simple model that predicts based on points_diff
                fallback_model = LogisticRegression()
                X = np.array([-20, -10, -5, 0, 5, 10, 20]).reshape(-1, 1)  # Points difference
                y = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])         # Win probability
                fallback_model.fit(X, y > 0.5)  # Train on binary outcomes
                # Store the fallback model
                models['models']['fallback'] = {
                    'model': fallback_model,
                    'features': ['points_diff']
                }
                models['mode'] = 'fallback'
                print("Created fallback logistic regression model")
            except Exception as e:
                print(f"Failed to create fallback model: {str(e)}")
        
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        # Return a default placeholder model
        return {
            'mode': 'default',
            'home_ice_boost': HOME_ICE_ADVANTAGE,
            'models': {}
        }

def create_matchup_data(top_seed, bottom_seed, team_data):
    """Create matchup data for model input
    
    Args:
        top_seed (dict): Higher seed team information
        bottom_seed (dict): Lower seed team information
        team_data (DataFrame): Team statistics data
        
    Returns:
        DataFrame: Single row with matchup data for prediction
    """
    # Create a single row DataFrame for this matchup
    matchup_data = {}
    
    # Base matchup information
    from datetime import datetime
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    matchup_data['season'] = current_season
    matchup_data['round'] = 1
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
    
    # Print debug information
    debug_enabled = False
    if debug_enabled:
        print(f"Creating matchup data for {top_abbrev} vs {bottom_abbrev}")
    
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
            
            # Feature columns to use for prediction - use the list from config
            feature_cols = CRITICAL_FEATURES
            
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
                    elif debug_enabled:
                        print(f"Skipping feature {col} - missing values")
            
            if debug_enabled:
                print(f"Added {features_added} features for matchup {top_abbrev} vs {bottom_abbrev}")
                
            # Add fallback features if necessary
            if features_added == 0:
                if debug_enabled:
                    print(f"WARNING: No model features available for {top_abbrev} vs {bottom_abbrev}")
                    print(f"Using points difference as fallback: {matchup_data.get('points_diff', 'N/A')}")
        else:
            if debug_enabled:
                print(f"Warning: Team data not found for {top_abbrev} or {bottom_abbrev}")
    else:
        if debug_enabled:
            print(f"Warning: 'teamAbbrev' column not found in team_data")
    
    matchup_df = pd.DataFrame([matchup_data])
    
    # Check and report on the quality of the matchup data
    if debug_enabled:
        diff_features = [col for col in matchup_df.columns if col.endswith('_diff')]
        print(f"Created matchup data with {len(diff_features)} differential features")
    
    return matchup_df

def predict_series_winner(matchup_df, models):
    """Predict the winner of a playoff series using available models
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information and objects
        
    Returns:
        tuple: (ensemble_prob, lr_prob, xgb_prob) - Raw probabilities of top seed winning (no home ice boost)
    
    Note: This function DOES NOT apply home ice advantage - this should be applied separately
          by the calling function. See simulate_playoff_bracket() for correct usage.
    """
    # Default probabilities
    lr_prob = 0.5
    xgb_prob = 0.5
    ensemble_prob = 0.5
    
    # Check for empty dataframe
    if matchup_df is None or matchup_df.empty:
        print("Warning: Empty matchup dataframe provided to predict_series_winner")
        return ensemble_prob, lr_prob, xgb_prob
    
    # Debug info - what teams are we predicting?
    # Set to True to enable more verbose debugging
    debug_enabled = False
    if debug_enabled and 'top_seed_abbrev' in matchup_df.columns and 'bottom_seed_abbrev' in matchup_df.columns:
        top = matchup_df['top_seed_abbrev'].iloc[0]
        bottom = matchup_df['bottom_seed_abbrev'].iloc[0]
        print(f"Predicting matchup: {top} vs {bottom}")
    
    # List available differential features for debugging
    diff_features = [col for col in matchup_df.columns if col.endswith('_diff')]
    if debug_enabled:
        print(f"Available differential features for prediction: {len(diff_features)}")
        if len(diff_features) > 0:
            print(f"Features: {diff_features}")
        else:
            print("WARNING: No differential features available for prediction!")
    
    # Ensure we have the points_diff feature for fallback
    has_points_diff = 'points_diff' in matchup_df.columns
    
    # Check which models we have available and make predictions
    if 'models' in models:
        # First check if we have an ensemble model
        if 'ensemble' in models['models']:
            try:
                ensemble_model = models['models']['ensemble']
                # If we have a custom ensemble predict method, use it
                if hasattr(ensemble_model, 'predict'):
                    features = ensemble_model.get('features', diff_features)
                    available_features = [f for f in features if f in matchup_df.columns]
                    if available_features:
                        prediction_data = matchup_df[available_features].fillna(0)
                        ensemble_prob = ensemble_model.predict(prediction_data)[0]
                        if debug_enabled:
                            print(f"Ensemble model prediction: {ensemble_prob:.4f}")
                        return ensemble_prob, lr_prob, xgb_prob
                # Otherwise use the component models
                elif 'lr_model' in ensemble_model and 'xgb_model' in ensemble_model:
                    # Get individual predictions from component models
                    lr_model = ensemble_model['lr_model'].get('model')
                    xgb_model = ensemble_model['xgb_model'].get('model')
                    
                    if lr_model and hasattr(lr_model, 'predict_proba'):
                        lr_features = ensemble_model['lr_model'].get('features', [])
                        available_lr_features = [f for f in lr_features if f in matchup_df.columns]
                        if available_lr_features:
                            prediction_data = matchup_df[available_lr_features].fillna(0)
                            lr_prob = lr_model.predict_proba(prediction_data)[:, 1][0]
                            if debug_enabled:
                                print(f"LR component prediction: {lr_prob:.4f}")
                    
                    if xgb_model and hasattr(xgb_model, 'predict_proba'):
                        xgb_features = ensemble_model['xgb_model'].get('features', [])
                        available_xgb_features = [f for f in xgb_features if f in matchup_df.columns]
                        if available_xgb_features:
                            prediction_data = matchup_df[available_xgb_features].fillna(0)
                            xgb_prob = xgb_model.predict_proba(prediction_data)[:, 1][0]
                            if debug_enabled:
                                print(f"XGB component prediction: {xgb_prob:.4f}")
                    
                    # Average the predictions for ensemble result
                    if lr_prob != 0.5 and xgb_prob != 0.5:
                        ensemble_prob = (lr_prob + xgb_prob) / 2
                        if debug_enabled:
                            print(f"Ensemble average: {ensemble_prob:.4f}")
                        return ensemble_prob, lr_prob, xgb_prob
            except Exception as ensemble_error:
                print(f"Error using ensemble model: {str(ensemble_error)}")
        
        # Try using individual models if ensemble failed or doesn't exist
        # Try Logistic Regression model
        if 'lr' in models['models'] and isinstance(models['models']['lr'], dict) and 'model' in models['models']['lr']:
            try:
                # Extract model and features
                lr_model = models['models']['lr']['model']
                
                # Get features if available in model definition
                if 'features' in models['models']['lr']:
                    expected_lr_features = models['models']['lr']['features']
                    lr_features = [f for f in expected_lr_features if f in matchup_df.columns]
                    if debug_enabled:
                        print(f"LR model expects {len(expected_lr_features)} features, found {len(lr_features)}")
                        missing_features = [f for f in expected_lr_features if f not in matchup_df.columns]
                        if missing_features:
                            print(f"Missing features for LR model: {missing_features}")
                elif hasattr(lr_model, 'feature_names_in_'):
                    expected_lr_features = lr_model.feature_names_in_
                    lr_features = [f for f in expected_lr_features if f in matchup_df.columns]
                    if debug_enabled:
                        print(f"LR model expects {len(expected_lr_features)} features, found {len(lr_features)}")
                        missing_features = [f for f in expected_lr_features if f not in matchup_df.columns]
                        if missing_features:
                            print(f"Missing features for LR model: {missing_features}")
                else:
                    # Default features - columns ending with '_diff'
                    lr_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                    if debug_enabled:
                        print(f"Using {len(lr_features)} differential features for LR model")
                
                # Make prediction if we have features and predict_proba method
                if lr_features and len(lr_features) > 0 and hasattr(lr_model, 'predict_proba'):
                    # Fill NaN values with 0
                    prediction_data = matchup_df[lr_features].fillna(0)
                    
                    # Make prediction
                    lr_prob = lr_model.predict_proba(prediction_data)[:, 1][0]
                    if debug_enabled:
                        print(f"LR model prediction: {lr_prob:.4f}")
                else:
                    if debug_enabled:
                        print("LR model does not have predict_proba method or no features available")
            except Exception as e:
                if debug_enabled:
                    print(f"Error in LR prediction: {str(e)}")
        
        # Try XGBoost model
        if 'xgb' in models['models'] and isinstance(models['models']['xgb'], dict) and 'model' in models['models']['xgb']:
            try:
                # Extract model and features
                xgb_model = models['models']['xgb']['model']
                
                # Get features if available in model definition
                if 'features' in models['models']['xgb']:
                    expected_xgb_features = models['models']['xgb']['features']
                    xgb_features = [f for f in expected_xgb_features if f in matchup_df.columns]
                    if debug_enabled:
                        print(f"XGB model expects {len(expected_xgb_features)} features, found {len(xgb_features)}")
                        missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                        if missing_features:
                            print(f"Missing features for XGB model: {missing_features}")
                elif hasattr(xgb_model, 'feature_names_in_'):
                    expected_xgb_features = xgb_model.feature_names_in_
                    xgb_features = [f for f in expected_xgb_features if f in matchup_df.columns]
                    if debug_enabled:
                        print(f"XGB model expects {len(expected_xgb_features)} features, found {len(xgb_features)}")
                        missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                        if missing_features:
                            print(f"Missing features for XGB model: {missing_features}")
                else:
                    # Default features - columns ending with '_diff'
                    xgb_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                    if debug_enabled:
                        print(f"Using {len(xgb_features)} differential features for XGB model")
                
                # Make prediction if we have features and predict_proba method
                if xgb_features and len(xgb_features) > 0 and hasattr(xgb_model, 'predict_proba'):
                    # Fill NaN values with 0 for prediction
                    prediction_data = matchup_df[xgb_features].fillna(0)
                    
                    # Make prediction
                    xgb_prob = xgb_model.predict_proba(prediction_data)[:, 1][0]
                    if debug_enabled:
                        print(f"XGB model prediction: {xgb_prob:.4f}")
                else:
                    if debug_enabled:
                        print("XGB model does not have predict_proba method or no features available")
            except Exception as e:
                if debug_enabled:
                    print(f"Error in XGB prediction: {str(e)}")
                
        # Try fallback model if available
        if 'fallback' in models['models'] and lr_prob == 0.5 and xgb_prob == 0.5:
            try:
                fallback_model = models['models']['fallback']['model']
                if has_points_diff and hasattr(fallback_model, 'predict_proba'):
                    points_diff = matchup_df['points_diff'].iloc[0]
                    fallback_prob = fallback_model.predict_proba(np.array([[points_diff]]))[:, 1][0]
                    if debug_enabled:
                        print(f"Fallback model prediction: {fallback_prob:.4f} (based on points_diff: {points_diff})")
                    ensemble_prob = fallback_prob
                    return ensemble_prob, lr_prob, xgb_prob
            except Exception as e:
                if debug_enabled:
                    print(f"Error in fallback model prediction: {str(e)}")
    
    # Calculate ensemble probability based on the available models and mode
    if models.get('mode') == 'ensemble' and lr_prob != 0.5 and xgb_prob != 0.5:
        ensemble_prob = (lr_prob + xgb_prob) / 2
        if debug_enabled:
            print(f"Using ensemble mode: {ensemble_prob:.4f} = ({lr_prob:.4f} + {xgb_prob:.4f}) / 2")
    elif models.get('mode') == 'lr' and lr_prob != 0.5:
        ensemble_prob = lr_prob
        if debug_enabled:
            print(f"Using LR mode: {ensemble_prob:.4f}")
    elif models.get('mode') == 'xgb' and xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        if debug_enabled:
            print(f"Using XGB mode: {ensemble_prob:.4f}")
    elif lr_prob != 0.5:
        ensemble_prob = lr_prob
        if debug_enabled:
            print(f"Falling back to LR: {ensemble_prob:.4f}")
    elif xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        if debug_enabled:
            print(f"Falling back to XGB: {ensemble_prob:.4f}")
    else:
        # Use points difference as fallback
        if 'points_diff' in matchup_df.columns:
            points_diff = matchup_df['points_diff'].iloc[0]
            ensemble_prob = 1 / (1 + np.exp(-0.05 * points_diff))
            if debug_enabled:
                print(f"Using points difference fallback: {ensemble_prob:.4f} (diff: {points_diff})")
        else:
            if debug_enabled:
                print("Using default 0.5 probability (no model predictions or points difference)")
    
    # IMPORTANT: The home ice boost is NOT applied here
    # This function returns the raw probabilities
    # The boost must be applied separately in the calling function
    
    return ensemble_prob, lr_prob, xgb_prob

# New helper function to check feature availability and values
def check_matchup_features(matchup_df, models):
    """Check the availability and values of features needed for model prediction
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information and objects
        
    Returns:
        dict: Statistics on feature availability and values
    """
    if matchup_df is None or matchup_df.empty:
        return {"error": "Empty matchup dataframe"}
    
    result = {
        "teams": f"{matchup_df['top_seed_abbrev'].iloc[0]} vs {matchup_df['bottom_seed_abbrev'].iloc[0]}",
        "total_columns": len(matchup_df.columns),
        "diff_features": [col for col in matchup_df.columns if col.endswith('_diff')],
        "feature_checks": {}
    }
    
    # Check for critical features
    critical_features = [
        'goalDiff/G_diff', 'adjGoalsScoredAboveX/60_diff', 'PK%_rel_diff', 
        'xGoalsPercentage_diff', 'possAdjHitsPctg_diff', 'reboundxGoalsPctg_diff'
    ]
    
    # Get expected features from models
    if 'models' in models:
        if 'lr' in models['models'] and 'features' in models['models']['lr']:
            result["expected_lr_features"] = models['models']['lr']['features']
        
        if 'xgb' in models['models'] and 'features' in models['models']['xgb']:
            result["expected_xgb_features"] = models['models']['xgb']['features']
    
    # Check each critical feature
    for feature in critical_features:
        if feature in matchup_df.columns:
            value = matchup_df[feature].iloc[0]
            result["feature_checks"][feature] = {
                "present": True, 
                "value": value,
                "is_null": pd.isna(value)
            }
        else:
            result["feature_checks"][feature] = {
                "present": False, 
                "value": None,
                "is_null": True
            }
    
    # Calculate feature availability stats
    present_features = [f for f, data in result["feature_checks"].items() if data["present"]]
    null_features = [f for f, data in result["feature_checks"].items() if data["present"] and data["is_null"]]
    
    result["feature_stats"] = {
        "total_critical": len(critical_features),
        "present_critical": len(present_features),
        "null_critical": len(null_features),
        "missing_critical": len(critical_features) - len(present_features),
        "usable_critical": len(present_features) - len(null_features)
    }
    
    return result

def predict_matchup(matchup_df, models):
    """Predict matchup outcome using loaded models
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information
        
    Returns:
        dict: Prediction results and metadata
    """
    if matchup_df is None or matchup_df.empty:
        return {'home_win_prob': 0.5, 'model_used': 'No prediction available',
                'away_win_prob': 0.5, 'raw_win_prob': 0.5}
        
    # Get team names
    home_team = matchup_df['top_seed_abbrev'].iloc[0]
    away_team = matchup_df['bottom_seed_abbrev'].iloc[0]
    
    # Get raw prediction (no home ice advantage) using silent version
    ensemble_prob, lr_prob, xgb_prob, debug_info = silent_predict_series_winner(matchup_df, models)
    
    # Apply home ice advantage here - AFTER ensemble calculation
    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
    win_prob = min(1.0, max(0.0, ensemble_prob + home_ice_boost))
    
    # Determine which model was used for the prediction
    if models.get('mode') == 'ensemble' and lr_prob != 0.5 and xgb_prob != 0.5:
        model_used = 'Ensemble (LR + XGB)'
    elif models.get('mode') == 'lr' and lr_prob != 0.5:
        model_used = 'Logistic Regression'
    elif models.get('mode') == 'xgb' and xgb_prob != 0.5:
        model_used = 'XGBoost'
    elif 'points_diff' in matchup_df.columns:
        model_used = 'Points-based'
    else:
        model_used = 'No prediction available'
        # If we didn't get a valid probability from any model, use 50%
        if ensemble_prob == 0.5 and 'points_diff' not in matchup_df.columns:
            win_prob = 0.5
    
    # Return prediction and metadata
    return {
        'home_win_prob': win_prob,        # Final probability WITH home ice boost
        'away_win_prob': 1 - win_prob,    # Away team probability
        'home_team': home_team,
        'away_team': away_team,
        'model_used': model_used,
        'lr_prob': lr_prob,               # Raw LR model probability
        'xgb_prob': xgb_prob,             # Raw XGB model probability
        'ensemble_prob': ensemble_prob,    # Raw ensemble probability (before boost)
        'raw_win_prob': ensemble_prob,     # Same as ensemble_prob for clarity
        'home_ice_boost': home_ice_boost,  # The boost value that was applied
        'debug_info': debug_info           # Include debug info for reference
    }

def predict_series(matchup_df, models, n_simulations=1000):
    """Predict series outcome using the pre-trained models with updated historical series length distribution
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information
        n_simulations: Number of series simulations to run
        
    Returns:
        dict: Results including win probability and series outcome distribution
    """
    # Extract team abbreviations for display
    top_seed = matchup_df['top_seed_abbrev'].iloc[0]
    bottom_seed = matchup_df['bottom_seed_abbrev'].iloc[0]
    
    # Get basic prediction for the matchup
    base_prediction = predict_matchup(matchup_df, models)
    base_prob = base_prediction['home_win_prob']
    model_used = base_prediction['model_used']
    
    # Get higher seed and lower seed outcome distributions
    higher_seed_outcome_dist, lower_seed_outcome_dist = get_series_outcome_distributions()
    
    # Initialize counters
    higher_seed_wins = 0
    win_distribution = {
        '4-0': 0, '4-1': 0, '4-2': 0, '4-3': 0,  # Higher seed wins
        '0-4': 0, '1-4': 0, '2-4': 0, '3-4': 0   # Lower seed wins
    }
    
    # Run simulations - determine series winner based on probability
    for _ in range(n_simulations):
        # Determine if higher seed wins the series
        higher_seed_wins_series = np.random.random() < base_prob
        
        if higher_seed_wins_series:
            higher_seed_wins += 1
            # Select a series outcome based on historical distribution
            outcome = np.random.choice(['4-0', '4-1', '4-2', '4-3'], 
                                      p=[higher_seed_outcome_dist['4-0'], 
                                         higher_seed_outcome_dist['4-1'],
                                         higher_seed_outcome_dist['4-2'],
                                         higher_seed_outcome_dist['4-3']])
            win_distribution[outcome] += 1
        else:
            # Select a series outcome for lower seed winning
            outcome = np.random.choice(['0-4', '1-4', '2-4', '3-4'], 
                                      p=[lower_seed_outcome_dist['0-4'], 
                                         lower_seed_outcome_dist['1-4'],
                                         lower_seed_outcome_dist['2-4'],
                                         lower_seed_outcome_dist['3-4']])
            win_distribution[outcome] += 1
    
    # Calculate win percentage and confidence interval
    win_pct = higher_seed_wins / n_simulations
    z = 1.96  # 95% confidence interval
    ci_width = z * np.sqrt((win_pct * (1 - win_pct)) / n_simulations)
    ci_lower = max(0, win_pct - ci_width)
    ci_upper = min(1, win_pct + ci_width)
    
    # Format results
    results = {
        'top_seed': top_seed,
        'bottom_seed': bottom_seed,
        'win_probability': win_pct,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'win_distribution': win_distribution,
        'model_used': model_used,
        'lr_probability': base_prediction['lr_prob'],
        'xgb_probability': base_prediction['xgb_prob'],
        'ensemble_probability': base_prediction['ensemble_prob'],
        'n_simulations': n_simulations
    }
    
    return results

def get_series_outcome_distributions():
    """Get the series outcome distributions for playoffs based on historical data.
    
    Returns:
        tuple: (higher_seed_dist, lower_seed_dist) - Distributions for series outcomes
    """
    # Use normalized SERIES_LENGTH_DISTRIBUTION
    # 4 games: 14.0%, 5 games: 24.3%, 6 games: 33.6%, 7 games: 28.1%
    total_percent = sum(SERIES_LENGTH_DISTRIBUTION)
    
    # Distribution for when higher seed wins (maintain relative proportions)
    higher_seed_outcome_dist = {
        '4-0': SERIES_LENGTH_DISTRIBUTION[0]/total_percent, 
        '4-1': SERIES_LENGTH_DISTRIBUTION[1]/total_percent, 
        '4-2': SERIES_LENGTH_DISTRIBUTION[2]/total_percent, 
        '4-3': SERIES_LENGTH_DISTRIBUTION[3]/total_percent
    }
    
    # Same distribution for when lower seed wins
    lower_seed_outcome_dist = {
        '0-4': SERIES_LENGTH_DISTRIBUTION[0]/total_percent, 
        '1-4': SERIES_LENGTH_DISTRIBUTION[1]/total_percent, 
        '2-4': SERIES_LENGTH_DISTRIBUTION[2]/total_percent, 
        '3-4': SERIES_LENGTH_DISTRIBUTION[3]/total_percent
    }
    
    return higher_seed_outcome_dist, lower_seed_outcome_dist

def get_raw_model_predictions(matchup_df, models):
    """Get raw model predictions without home ice advantage
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information and objects
        
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
    if 'models' in models:
        # Try Logistic Regression model
        if 'lr' in models['models']:
            try:
                # Extract model and features if available
                lr_model = models['models']['lr']['model']
                
                # Get features if available in model
                if 'features' in models['models']['lr']:
                    lr_features = [f for f in models['models']['lr']['features'] if f in matchup_df.columns]
                elif hasattr(lr_model, 'feature_names_in_'):
                    lr_features = [f for f in lr_model.feature_names_in_ if f in matchup_df.columns]
                else:
                    # Default features - columns ending with '_diff'
                    lr_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                
                # Make prediction if we have features
                if lr_features and len(lr_features) > 0 and hasattr(lr_model, 'predict_proba'):
                    lr_prob = lr_model.predict_proba(matchup_df[lr_features])[:, 1][0]
            except Exception as e:
                print(f"Error in LR prediction: {str(e)}")
        
        # Try XGBoost model
        if 'xgb' in models['models']:
            try:
                # Extract model and features
                xgb_model = models['models']['xgb']['model']
                
                # Get features if available in model
                if 'features' in models['models']['xgb']:
                    xgb_features = [f for f in models['models']['xgb']['features'] if f in matchup_df.columns]
                elif hasattr(xgb_model, 'feature_names_in_'):
                    xgb_features = [f for f in xgb_model.feature_names_in_ if f in matchup_df.columns]
                else:
                    # Default features - columns ending with '_diff'
                    xgb_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                
                # Make prediction if we have features
                if xgb_features and len(xgb_features) > 0 and hasattr(xgb_model, 'predict_proba'):
                    xgb_prob = xgb_model.predict_proba(matchup_df[xgb_features])[:, 1][0]
            except Exception as e:
                print(f"Error in XGB prediction: {str(e)}")
    
    return lr_prob, xgb_prob

def predict_lr(model, features):
    """Make a prediction using the Linear Regression model."""
    # Check for PK%_rel_diff scaling issue (assuming it's the second feature)
    if len(features) >= 2 and abs(features[1]) > 0.5:  # If PK%_rel_diff is too large
        # Apply scaling to bring it to the expected range
        features[1] = features[1] * 0.01
        
    # Make prediction
    try:
        return model.predict([features])[0]
    except Exception as e:
        import streamlit as st
        if 'st' in globals():
            st.error(f"LR prediction error: {str(e)}")
        return 0.5  # Default to 50% if prediction fails

def predict_xgb(model, features):
    """Make a prediction using the XGBoost model."""
    import numpy as np
    # Create a proper numpy array for XGBoost
    feature_array = np.array([features])
    
    # Make prediction
    try:
        return model.predict(feature_array)[0]
    except Exception as e:
        import streamlit as st
        if 'st' in globals():
            st.error(f"XGB prediction error: {str(e)}")
        return 0.5  # Default to 50% if prediction fails

def get_prediction_features(home_team_features, away_team_features, model_type):
    """Extract the correct features for prediction based on model type."""
    feature_diff = {}
    for key in home_team_features:
        if key in away_team_features:
            feature_diff[f"{key}_diff"] = home_team_features[key] - away_team_features[key]
    
    # Get feature lists based on model type
    if model_type == 'LR':
        # Example LR features - replace with your actual feature list
        lr_features = ['goalDiff/G_diff', 'PK%_rel_diff', 'adjGoalsScoredAboveX/60_diff']
        return [feature_diff.get(feat, 0) for feat in lr_features]
    
    elif model_type == 'XGB':
        # Example XGB features - replace with your actual feature list
        xgb_features = ['points_diff', 'PP%_rel_diff', 'PK%_rel_diff', 
                         'goalDiff/G_diff', 'adjGoalsScoredAboveX/60_diff', 'adjGoalsSavedAboveX/60_diff']
        return [feature_diff.get(feat, 0) for feat in xgb_features]
    
    return []

# Add these functions to support the debug page

def predict_lr(model, features):
    """Make a prediction using the Linear Regression model."""
    # Check for PK%_rel_diff scaling issue (assuming it's in features)
    for i, feat_val in enumerate(features):
        # If any value is suspiciously large (PK%_rel_diff typically)
        if abs(feat_val) > 0.5 and abs(feat_val) < 100:
            # Apply scaling to bring it to the expected range
            features[i] = feat_val * 0.01
    
    # Make prediction
    try:
        return model.predict([features])[0]
    except Exception as e:
        import streamlit as st
        st.error(f"LR prediction error: {str(e)}")
        return 0.5  # Default to 50% if prediction fails

def predict_xgb(model, features):
    """Make a prediction using the XGBoost model."""
    import numpy as np
    
    # Create a proper numpy array for XGBoost
    feature_array = np.array([features])
    
    # Make prediction
    try:
        return model.predict(feature_array)[0]
    except Exception as e:
        import streamlit as st
        st.error(f"XGB prediction error: {str(e)}")
        return 0.5  # Default to 50% if prediction fails

def get_model_features(model_data, model_type):
    """Get the feature list for a specific model type."""
    if 'feature_sets' in model_data and model_type in model_data['feature_sets']:
        return model_data['feature_sets'][model_type]
    
    # Default feature sets if not found in model_data
    if model_type == 'LR':
        return ['goalDiff/G_diff', 'PK%_rel_diff', 'adjGoalsScoredAboveX/60_diff']
    elif model_type == 'XGB':
        return ['points_diff', 'PP%_rel_diff', 'PK%_rel_diff', 
                'goalDiff/G_diff', 'adjGoalsScoredAboveX/60_diff', 'adjGoalsSavedAboveX/60_diff']
    
    return []

def check_pk_scaling_issues(feature_dict):
    """Check for PK%_rel scaling issues in the data."""
    results = {}
    for team, features in feature_dict.items():
        if 'PK%_rel' in features and abs(features['PK%_rel']) > 0.5:
            results[team] = {
                'original': features['PK%_rel'],
                'should_be': features['PK%_rel'] * 0.01
            }
    return results

# Add this function to the end of model_utils.py
def silent_predict_series_winner(matchup_df, models):
    """Silent version of predict_series_winner that doesn't print debug info.
    
    Args:
        matchup_df: DataFrame with matchup features
        models: Dictionary with model information and objects
        
    Returns:
        tuple: (ensemble_prob, lr_prob, xgb_prob, debug_info) - 
               Raw probabilities and debug information dictionary
    """
    # Capture debug info in a dictionary instead of printing to terminal
    debug_info = {
        "matchup": "",
        "diff_features": [],
        "diff_feature_count": 0,
        "lr": {"features": [], "prediction": 0.5, "data": {}, "scaling_applied": False},
        "xgb": {"features": [], "prediction": 0.5, "data": {}},
        "ensemble": {"method": "", "prediction": 0.5}
    }
    
    # Default probabilities
    lr_prob = 0.5
    xgb_prob = 0.5
    ensemble_prob = 0.5
    
    # Check for empty dataframe
    if matchup_df is None or matchup_df.empty:
        debug_info["error"] = "Empty matchup dataframe"
        return ensemble_prob, lr_prob, xgb_prob, debug_info
    
    # Extract matchup info
    if 'top_seed_abbrev' in matchup_df.columns and 'bottom_seed_abbrev' in matchup_df.columns:
        top = matchup_df['top_seed_abbrev'].iloc[0]
        bottom = matchup_df['bottom_seed_abbrev'].iloc[0]
        debug_info["matchup"] = f"{top} vs {bottom}"
    
    # Get differential features
    diff_features = [col for col in matchup_df.columns if col.endswith('_diff')]
    debug_info["diff_features"] = diff_features
    debug_info["diff_feature_count"] = len(diff_features)
    
    # Ensure we have the points_diff feature for fallback
    has_points_diff = 'points_diff' in matchup_df.columns
    
    # Check which models we have available and make predictions
    if 'models' in models:
        # First check if we have an ensemble model
        if 'ensemble' in models['models']:
            try:
                ensemble_model = models['models']['ensemble']
                # If we have a custom ensemble predict method, use it
                if hasattr(ensemble_model, 'predict'):
                    features = ensemble_model.get('features', diff_features)
                    available_features = [f for f in features if f in matchup_df.columns]
                    if available_features:
                        prediction_data = matchup_df[available_features].fillna(0)
                        ensemble_prob = ensemble_model.predict(prediction_data)[0]
                        debug_info["ensemble"]["method"] = "ensemble_model"
                        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
                        return ensemble_prob, lr_prob, xgb_prob, debug_info
                # Otherwise use the component models
                elif 'lr_model' in ensemble_model and 'xgb_model' in ensemble_model:
                    # Get individual predictions from component models
                    lr_model = ensemble_model['lr_model'].get('model')
                    xgb_model = ensemble_model['xgb_model'].get('model')
                    
                    # Get LR prediction
                    if lr_model and hasattr(lr_model, 'predict_proba'):
                        lr_features = ensemble_model['lr_model'].get('features', [])
                        available_lr_features = [f for f in lr_features if f in matchup_df.columns]
                        if available_lr_features:
                            prediction_data = matchup_df[available_lr_features].fillna(0)
                            lr_prob = lr_model.predict_proba(prediction_data)[:, 1][0]
                            debug_info["lr"]["prediction"] = float(lr_prob)
                    
                    # Get XGB prediction
                    if xgb_model and hasattr(xgb_model, 'predict_proba'):
                        xgb_features = ensemble_model['xgb_model'].get('features', [])
                        available_xgb_features = [f for f in xgb_features if f in matchup_df.columns]
                        if available_xgb_features:
                            prediction_data = matchup_df[available_xgb_features].fillna(0)
                            xgb_prob = xgb_model.predict_proba(prediction_data)[:, 1][0]
                            debug_info["xgb"]["prediction"] = float(xgb_prob)
                    
                    # Average the predictions for ensemble result
                    if lr_prob != 0.5 and xgb_prob != 0.5:
                        ensemble_prob = (lr_prob + xgb_prob) / 2
                        debug_info["ensemble"]["method"] = "ensemble_average"
                        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
                        debug_info["ensemble"]["components"] = {"lr": float(lr_prob), "xgb": float(xgb_prob)}
                        return ensemble_prob, lr_prob, xgb_prob, debug_info
            except Exception as ensemble_error:
                debug_info["ensemble"]["error"] = str(ensemble_error)
        
        # Try using individual models if ensemble failed or doesn't exist
        # Try Logistic Regression model
        if 'LR' in models['models'] and 'model' in models['models']['LR']:
            try:
                # Extract model and features
                lr_model = models['models']['LR']['model']
                
                # Get features if available in model definition
                if 'features' in models['models']['LR']:
                    expected_lr_features = models['models']['LR']['features']
                    lr_features = [f for f in expected_lr_features if f in matchup_df.columns]
                    debug_info["lr"]["expected_features"] = expected_lr_features
                    debug_info["lr"]["available_features"] = lr_features
                    
                    missing_features = [f for f in expected_lr_features if f not in matchup_df.columns]
                    if missing_features:
                        debug_info["lr"]["missing_features"] = missing_features
                elif hasattr(lr_model, 'feature_names_in_'):
                    expected_lr_features = lr_model.feature_names_in_
                    lr_features = [f for f in expected_lr_features if f in matchup_df.columns]
                    debug_info["lr"]["expected_features"] = list(expected_lr_features)
                    debug_info["lr"]["available_features"] = lr_features
                    
                    missing_features = [f for f in expected_lr_features if f not in matchup_df.columns]
                    if missing_features:
                        debug_info["lr"]["missing_features"] = missing_features
                else:
                    # Default features - columns ending with '_diff'
                    lr_features = get_model_features(models, 'LR')
                    lr_features = [f for f in lr_features if f in matchup_df.columns]
                    debug_info["lr"]["available_features"] = lr_features
                    debug_info["lr"]["note"] = "Using default LR features"
                
                # Make prediction if we have features and predict_proba method
                if lr_features and len(lr_features) > 0:
                    # Check for NaN values in features
                    nan_features = [f for f in lr_features if matchup_df[f].isna().any()]
                    if nan_features:
                        debug_info["lr"]["nan_features"] = nan_features
                    
                    # Apply emergency scaling fix for known problematic features
                    prediction_data = matchup_df[lr_features].copy()
                    
                    # Check and fix PK%_rel_diff if it seems to be on the wrong scale
                    if 'PK%_rel_diff' in prediction_data.columns:
                        pk_value = prediction_data['PK%_rel_diff'].iloc[0]
                        if abs(pk_value) > 0.2:  # If value is too large (e.g., -4.9 instead of -0.049)
                            debug_info["lr"]["scaling_applied"] = True
                            debug_info["lr"]["original_pk"] = pk_value
                            debug_info["lr"]["scaled_pk"] = pk_value/100.0
                            prediction_data['PK%_rel_diff'] = prediction_data['PK%_rel_diff'] / 100.0
                    
                    # Fill remaining NaN values with 0
                    prediction_data = prediction_data.fillna(0)
                    
                    # Record prediction data
                    for feature in lr_features:
                        debug_info["lr"]["data"][feature] = float(prediction_data[feature].iloc[0])
                    
                    # Make prediction
                    try:
                        # Convert to list for prediction
                        feature_values = [prediction_data[feat].iloc[0] for feat in lr_features]
                        lr_prob = predict_lr(lr_model, feature_values)
                        debug_info["lr"]["prediction"] = float(lr_prob)
                    except Exception as e:
                        debug_info["lr"]["error"] = str(e)
            except Exception as e:
                debug_info["lr"]["error"] = str(e)
        
        # Try XGBoost model
        if 'XGB' in models['models'] and 'model' in models['models']['XGB']:
            try:
                # Extract model and features
                xgb_model = models['models']['XGB']['model']
                
                # Get features if available in model definition
                if 'features' in models['models']['XGB']:
                    expected_xgb_features = models['models']['XGB']['features']
                    xgb_features = [f for f in expected_xgb_features if f in matchup_df.columns]
                    debug_info["xgb"]["expected_features"] = expected_xgb_features
                    debug_info["xgb"]["available_features"] = xgb_features
                    
                    missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                    if missing_features:
                        debug_info["xgb"]["missing_features"] = missing_features
                elif hasattr(xgb_model, 'feature_names_in_'):
                    expected_xgb_features = xgb_model.feature_names_in_
                    xgb_features = [f for f in expected_xgb_features if f in matchup_df.columns]
                    debug_info["xgb"]["expected_features"] = list(expected_xgb_features)
                    debug_info["xgb"]["available_features"] = xgb_features
                    
                    missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                    if missing_features:
                        debug_info["xgb"]["missing_features"] = missing_features
                else:
                    # Default features
                    xgb_features = get_model_features(models, 'XGB')
                    xgb_features = [f for f in xgb_features if f in matchup_df.columns]
                    debug_info["xgb"]["available_features"] = xgb_features
                    debug_info["xgb"]["note"] = "Using default XGB features"
                
                # Make prediction if we have features
                if xgb_features and len(xgb_features) > 0:
                    # Check for NaN values in features
                    nan_features = [f for f in xgb_features if matchup_df[f].isna().any()]
                    if nan_features:
                        debug_info["xgb"]["nan_features"] = nan_features
                    
                    # Fill NaN values with 0 for prediction
                    prediction_data = matchup_df[xgb_features].fillna(0)
                    
                    # Record prediction data
                    for feature in xgb_features:
                        debug_info["xgb"]["data"][feature] = float(prediction_data[feature].iloc[0])
                    
                    # Make prediction
                    try:
                        # Convert to list for prediction
                        feature_values = [prediction_data[feat].iloc[0] for feat in xgb_features]
                        xgb_prob = predict_xgb(xgb_model, feature_values)
                        debug_info["xgb"]["prediction"] = float(xgb_prob)
                    except Exception as e:
                        debug_info["xgb"]["error"] = str(e)
            except Exception as e:
                debug_info["xgb"]["error"] = str(e)
    
    # Calculate ensemble probability based on the available models and mode
    if models.get('mode') == 'ensemble' and lr_prob != 0.5 and xgb_prob != 0.5:
        ensemble_prob = (lr_prob + xgb_prob) / 2
        debug_info["ensemble"]["method"] = "ensemble"
        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
        debug_info["ensemble"]["components"] = {"lr": float(lr_prob), "xgb": float(xgb_prob)}
    elif models.get('mode') == 'lr' and lr_prob != 0.5:
        ensemble_prob = lr_prob
        debug_info["ensemble"]["method"] = "lr_only"
        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
    elif models.get('mode') == 'xgb' and xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        debug_info["ensemble"]["method"] = "xgb_only"
        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
    elif lr_prob != 0.5:
        ensemble_prob = lr_prob
        debug_info["ensemble"]["method"] = "lr_fallback"
        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
    elif xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        debug_info["ensemble"]["method"] = "xgb_fallback"
        debug_info["ensemble"]["prediction"] = float(ensemble_prob)
    else:
        # Use points difference as fallback
        if 'points_diff' in matchup_df.columns:
            points_diff = matchup_df['points_diff'].iloc[0]
            ensemble_prob = 1 / (1 + np.exp(-0.05 * points_diff))
            debug_info["ensemble"]["method"] = "points_diff_fallback"
            debug_info["ensemble"]["prediction"] = float(ensemble_prob)
            debug_info["ensemble"]["points_diff"] = float(points_diff)
        else:
            debug_info["ensemble"]["method"] = "default"
            debug_info["ensemble"]["prediction"] = 0.5
    
    return ensemble_prob, lr_prob, xgb_prob, debug_info

"""
Utility functions for creating and managing team matchups.
"""

import pandas as pd
import numpy as np
import streamlit as st
from itertools import combinations
import streamlit_app.utils.model_utils as model_utils
import time

def create_matchup_data(top_seed, bottom_seed, team_data):
    """Create matchup data for model input
    
    Args:
        top_seed: Dictionary with top seed team information
        bottom_seed: Dictionary with bottom seed team information
        team_data: DataFrame with team statistics
        
    Returns:
        DataFrame: Matchup data for prediction
    """
    # Create a single row DataFrame for this matchup
    matchup_data = {}
    
    # Base matchup information
    matchup_data['top_seed_abbrev'] = top_seed.get('teamAbbrev', '')
    matchup_data['bottom_seed_abbrev'] = bottom_seed.get('teamAbbrev', '')
    
    # Get team data for each team
    if 'teamAbbrev' in team_data.columns:
        top_team_filter = team_data['teamAbbrev'] == matchup_data['top_seed_abbrev']
        bottom_team_filter = team_data['teamAbbrev'] == matchup_data['bottom_seed_abbrev']
        
        if sum(top_team_filter) > 0 and sum(bottom_team_filter) > 0:
            top_seed_data = team_data[top_team_filter].iloc[0]
            bottom_seed_data = team_data[bottom_team_filter].iloc[0]
            
            # Get the points difference for basic prediction
            if 'points' in top_seed_data and 'points' in bottom_seed_data:
                matchup_data['points_diff'] = top_seed_data['points'] - bottom_seed_data['points']
            
            # Feature columns to use for prediction
            from streamlit_app.config import CRITICAL_FEATURES
            
            # Log which features are available
            available_features = [col for col in CRITICAL_FEATURES if col in top_seed_data and col in bottom_seed_data]
            print(f"Available features for matchup: {len(available_features)}/{len(CRITICAL_FEATURES)}")
            
            # Add features for each team if available
            for col in CRITICAL_FEATURES:
                if col in top_seed_data and col in bottom_seed_data:
                    # Only add if both values are not NaN
                    if pd.notna(top_seed_data[col]) and pd.notna(bottom_seed_data[col]):
                        matchup_data[f"{col}_top"] = top_seed_data[col]
                        matchup_data[f"{col}_bottom"] = bottom_seed_data[col]
                        matchup_data[f"{col}_diff"] = top_seed_data[col] - bottom_seed_data[col]
    
    matchup_df = pd.DataFrame([matchup_data])
    
    # Validate matchup data before returning
    from streamlit_app.utils.data_validation import validate_and_fix
    
    if isinstance(matchup_df, pd.DataFrame) and not matchup_df.empty:
        matchup_df, validation_report = validate_and_fix(
            matchup_df, 'pre-model', 
            ['top_seed_abbrev', 'bottom_seed_abbrev']
        )
    
    return matchup_df

def generate_all_matchup_combinations(team_data):
    """Generate all possible matchup combinations between teams
    
    Args:
        team_data: DataFrame with team data
        
    Returns:
        dict: Dictionary of all possible matchups
    """
    # Verify team_data has required columns
    if 'teamAbbrev' not in team_data.columns:
        print("Error: team_data missing teamAbbrev column")
        return {}
    
    all_teams = team_data['teamAbbrev'].unique()
    matchup_dict = {}
    
    # Generate all combinations of teams
    team_pairs = list(combinations(all_teams, 2))
    
    for team1, team2 in team_pairs:
        # Create both directions of matchup
        team1_data = team_data[team_data['teamAbbrev'] == team1]
        team2_data = team_data[team_data['teamAbbrev'] == team2]
        
        if team1_data.empty or team2_data.empty:
            continue
            
        # Create team dictionaries
        team1_dict = team1_data.iloc[0].to_dict()
        team2_dict = team2_data.iloc[0].to_dict()
        
        # Create matchup with team1 as top seed
        key1 = f"{team1}_{team2}"
        matchup_dict[key1] = create_matchup_data(team1_dict, team2_dict, team_data)
        
        # Create matchup with team2 as top seed
        key2 = f"{team2}_{team1}"
        matchup_dict[key2] = create_matchup_data(team2_dict, team1_dict, team_data)
    
    return matchup_dict

def get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data=None):
    """Get matchup data from pre-calculated dictionary or create it
    
    Args:
        top_seed: Dictionary or string with top seed team information
        bottom_seed: Dictionary or string with bottom seed team information
        matchup_dict: Dictionary of pre-calculated matchups
        team_data: DataFrame with team data (optional if matchup_dict has the matchup)
        
    Returns:
        DataFrame: Matchup data for prediction
    """
    # Extract team abbreviations
    if isinstance(top_seed, dict):
        top_abbrev = top_seed.get('teamAbbrev', '')
    else:
        top_abbrev = str(top_seed)
        
    if isinstance(bottom_seed, dict):
        bottom_abbrev = bottom_seed.get('teamAbbrev', '')
    else:
        bottom_abbrev = str(bottom_seed)
    
    # Check if matchup exists in pre-calculated dictionary
    key = f"{top_abbrev}_{bottom_abbrev}"
    if key in matchup_dict and not matchup_dict[key].empty:
        return matchup_dict[key]
    
    # If not found and team_data is provided, create the matchup
    if team_data is not None:
        if isinstance(top_seed, str) and isinstance(bottom_seed, str):
            # Convert strings to team dictionaries
            top_team_data = team_data[team_data['teamAbbrev'] == top_abbrev]
            bottom_team_data = team_data[team_data['teamAbbrev'] == bottom_abbrev]
            
            if not top_team_data.empty and not bottom_team_data.empty:
                top_seed = top_team_data.iloc[0].to_dict()
                bottom_seed = bottom_team_data.iloc[0].to_dict()
            else:
                return pd.DataFrame()  # Teams not found
        
        return create_matchup_data(top_seed, bottom_seed, team_data)
    
    # If matchup not found and no team_data provided
    return pd.DataFrame()

def analyze_matchup_data(matchup_dict):
    """Analyze matchup data quality and availability
    
    Args:
        matchup_dict: Dictionary of matchups
        
    Returns:
        dict: Analysis of matchup data
    """
    analysis = {
        "total_matchups": len(matchup_dict),
        "feature_counts": {},
        "nan_counts": {},
        "teams": set()
    }
    
    # Collect all unique features across matchups
    all_features = set()
    for key, matchup_df in matchup_dict.items():
        if matchup_df is not None and not matchup_df.empty:
            diff_features = [col for col in matchup_df.columns if '_diff' in col]
            all_features.update(diff_features)
            
            # Extract team abbreviations
            if 'top_seed_abbrev' in matchup_df.columns:
                analysis["teams"].add(matchup_df['top_seed_abbrev'].iloc[0])
            if 'bottom_seed_abbrev' in matchup_df.columns:
                analysis["teams"].add(matchup_df['bottom_seed_abbrev'].iloc[0])
    
    # Initialize counters for each feature
    for feature in all_features:
        analysis["feature_counts"][feature] = 0
        analysis["nan_counts"][feature] = 0
    
    # Count features and NaN values
    for key, matchup_df in matchup_dict.items():
        if matchup_df is None or matchup_df.empty:
            continue
            
        for feature in all_features:
            if feature in matchup_df.columns:
                analysis["feature_counts"][feature] += 1
                
                # Count NaN values
                if matchup_df[feature].isna().any():
                    analysis["nan_counts"][feature] += 1
    
    # Convert teams to count
    analysis["team_count"] = len(analysis["teams"])
    analysis["teams"] = list(analysis["teams"])
    
    return analysis
