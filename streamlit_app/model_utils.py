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

# Function to load trained models
@st.cache_resource
def load_models(model_folder):
    """Load the trained machine learning models for playoff predictions.
    
    Args:
        model_folder (str): Path to the directory containing model files
        
    Returns:
        dict: Dictionary containing model information and objects
    """
    # Ensure model directory exists
    os.makedirs(model_folder, exist_ok=True)
    
    # Dictionary to return with models
    models = {'models': {}}
    
    # Define model paths
    lr_path = os.path.join(model_folder, 'logistic_regression_model_final.pkl')
    xgb_path = os.path.join(model_folder, 'xgboost_playoff_model_final.pkl')
    default_path = os.path.join(model_folder, 'playoff_model.pkl')
    
    # List available files for debugging
    available_files = os.listdir(model_folder) if os.path.exists(model_folder) else []
    print(f"Available model files: {', '.join(available_files) if available_files else 'None'}")
    
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
            
        # Check if default model exists and load it
        if os.path.exists(default_path):
            try:
                default_model = joblib.load(default_path)
                # If combined is a dict with models inside, merge with our models
                if isinstance(default_model, dict) and 'models' in default_model:
                    for model_key, model_value in default_model['models'].items():
                        models['models'][model_key] = model_value
                    # Copy other keys
                    for key, value in default_model.items():
                        if key != 'models':
                            models[key] = value
                    print("✓ Loaded combined model package")
                else:
                    # Otherwise just add as 'combined'
                    models['models']['combined'] = default_model
                    print("✓ Loaded default model")
            except Exception as default_error:
                print(f"Failed to load default model: {str(default_error)}")
            
        # Set model mode and parameters
        if not models['models']:
            print("No models found - using basic predictions")
            models['mode'] = 'default'
            models['home_ice_boost'] = 0.039
        else:
            # Determine which mode to use based on available models
            if 'lr' in models['models'] and 'xgb' in models['models']:
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
                models['home_ice_boost'] = 0.039
                
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
            'home_ice_boost': 0.039,
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
            
            # Feature columns to use for prediction - expand this list if needed
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
    debug_enabled = True
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
                    # Check for NaN values in features
                    nan_features = [f for f in lr_features if matchup_df[f].isna().any()]
                    if nan_features:
                        print(f"WARNING: NaN values found in LR features: {nan_features}")
                        
                        # Print the actual values of critical features 
                        if debug_enabled:
                            # Check for specific critical features
                            critical_lr_features = ['goalDiff/G_diff', 'adjGoalsScoredAboveX/60_diff', 'PK%_rel_diff']
                            for feature in critical_lr_features:
                                if feature in matchup_df.columns:
                                    value = matchup_df[feature].iloc[0]
                                    print(f"  {feature}: {value}")
                                else:
                                    print(f"  {feature}: MISSING")
                    
                    # Apply emergency scaling fix for known problematic features
                    prediction_data = matchup_df[lr_features].copy()
                    
                    # Check and fix PK%_rel_diff if it seems to be on the wrong scale
                    if 'PK%_rel_diff' in prediction_data.columns:
                        pk_value = prediction_data['PK%_rel_diff'].iloc[0]
                        if abs(pk_value) > 0.2:  # If value is too large (e.g., -4.9 instead of -0.049)
                            print(f"Applying emergency scale fix to PK%_rel_diff: {pk_value} → {pk_value/100}")
                            prediction_data['PK%_rel_diff'] = prediction_data['PK%_rel_diff'] / 100.0
                    
                    # Fill remaining NaN values with 0
                    prediction_data = prediction_data.fillna(0)
                    
                    # Print the prediction data for debugging
                    if debug_enabled:
                        print("LR prediction data:")
                        for feature in lr_features:
                            print(f"  {feature}: {prediction_data[feature].iloc[0]}")
                    
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
                    print(f"XGB model expects {len(expected_xgb_features)} features, found {len(xgb_features)}")
                    missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                    if missing_features:
                        print(f"Missing features for XGB model: {missing_features}")
                elif hasattr(xgb_model, 'feature_names_in_'):
                    expected_xgb_features = xgb_model.feature_names_in_
                    xgb_features = [f for f in expected_xgb_features if f in matchup_df.columns]
                    print(f"XGB model expects {len(expected_xgb_features)} features, found {len(xgb_features)}")
                    missing_features = [f for f in expected_xgb_features if f not in matchup_df.columns]
                    if missing_features:
                        print(f"Missing features for XGB model: {missing_features}")
                else:
                    # Default features - columns ending with '_diff'
                    xgb_features = [col for col in matchup_df.columns if col.endswith('_diff')]
                    print(f"Using {len(xgb_features)} differential features for XGB model")
                
                # Make prediction if we have features and predict_proba method
                if xgb_features and len(xgb_features) > 0 and hasattr(xgb_model, 'predict_proba'):
                    # Check for NaN values in features
                    nan_features = [f for f in xgb_features if matchup_df[f].isna().any()]
                    if nan_features:
                        print(f"WARNING: NaN values found in XGB features: {nan_features}")
                        if debug_enabled:
                            # Print the values of nan features
                            for feature in nan_features:
                                print(f"  {feature}: {matchup_df[feature].iloc[0]}")
                        
                        # Fill NaN values with 0 for prediction
                        prediction_data = matchup_df[xgb_features].fillna(0)
                    else:
                        prediction_data = matchup_df[xgb_features]
                    
                    # Make prediction
                    xgb_prob = xgb_model.predict_proba(prediction_data)[:, 1][0]
                    print(f"XGB model prediction: {xgb_prob:.4f}")
                else:
                    print("XGB model does not have predict_proba method or no features available")
            except Exception as e:
                print(f"Error in XGB prediction: {str(e)}")
                
        # Try fallback model if available
        if 'fallback' in models['models'] and lr_prob == 0.5 and xgb_prob == 0.5:
            try:
                fallback_model = models['models']['fallback']['model']
                if has_points_diff and hasattr(fallback_model, 'predict_proba'):
                    points_diff = matchup_df['points_diff'].iloc[0]
                    fallback_prob = fallback_model.predict_proba(np.array([[points_diff]]))[:, 1][0]
                    print(f"Fallback model prediction: {fallback_prob:.4f} (based on points_diff: {points_diff})")
                    ensemble_prob = fallback_prob
                    return ensemble_prob, lr_prob, xgb_prob
            except Exception as e:
                print(f"Error in fallback model prediction: {str(e)}")
    
    # Calculate ensemble probability based on the available models and mode
    if models.get('mode') == 'ensemble' and lr_prob != 0.5 and xgb_prob != 0.5:
        ensemble_prob = (lr_prob + xgb_prob) / 2
        print(f"Using ensemble mode: {ensemble_prob:.4f} = ({lr_prob:.4f} + {xgb_prob:.4f}) / 2")
    elif models.get('mode') == 'lr' and lr_prob != 0.5:
        ensemble_prob = lr_prob
        print(f"Using LR mode: {ensemble_prob:.4f}")
    elif models.get('mode') == 'xgb' and xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        print(f"Using XGB mode: {ensemble_prob:.4f}")
    elif lr_prob != 0.5:
        ensemble_prob = lr_prob
        print(f"Falling back to LR: {ensemble_prob:.4f}")
    elif xgb_prob != 0.5:
        ensemble_prob = xgb_prob
        print(f"Falling back to XGB: {ensemble_prob:.4f}")
    else:
        # Use points difference as fallback
        if 'points_diff' in matchup_df.columns:
            points_diff = matchup_df['points_diff'].iloc[0]
            ensemble_prob = 1 / (1 + np.exp(-0.05 * points_diff))
            print(f"Using points difference fallback: {ensemble_prob:.4f} (diff: {points_diff})")
        else:
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
        return {'home_win_prob': 0.5, 'model_used': 'Default'}
        
    # Get team names
    home_team = matchup_df['top_seed_abbrev'].iloc[0]
    away_team = matchup_df['bottom_seed_abbrev'].iloc[0]
    
    # Get raw prediction (no home ice advantage)
    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
    
    # Apply home ice advantage here - AFTER ensemble calculation
    home_ice_boost = models.get('home_ice_boost', 0.039)
    win_prob = min(1.0, max(0.0, ensemble_prob + home_ice_boost))
    
    # Determine which model was used for the prediction
    if models.get('mode') == 'ensemble' and lr_prob != 0.5 and xgb_prob != 0.5:
        model_used = 'Ensemble (LR + XGB)'
    elif models.get('mode') == 'lr' and lr_prob != 0.5:
        model_used = 'Logistic Regression'
    elif models.get('mode') == 'xgb' and xgb_prob != 0.5:
        model_used = 'XGBoost'
    else:
        model_used = 'Points-based'
        
        # If we didn't get a valid probability from any model, use default
        if ensemble_prob == 0.5 and 'points_diff' not in matchup_df.columns:
            win_prob = 0.55  # Default to slight home advantage
            model_used = 'Default'
    
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
        'home_ice_boost': home_ice_boost   # The boost value that was applied
    }

def predict_series(matchup_df, models, n_simulations=1000):
    """Predict series outcome using the pre-trained models with updated historical series length distribution"""
    # Extract team abbreviations for display
    top_seed = matchup_df['top_seed_abbrev'].iloc[0]
    bottom_seed = matchup_df['bottom_seed_abbrev'].iloc[0]
    
    # Get basic prediction for the matchup
    base_prediction = predict_matchup(matchup_df, models)
    base_prob = base_prediction['home_win_prob']
    model_used = base_prediction['model_used']
    
    # Updated historical distribution of NHL playoff series outcomes based on provided data
    # 4 games: 14.0%, 5 games: 24.3%, 6 games: 33.6%, 7 games: 28.1%
    # Normalize within each outcome (higher seed wins vs lower seed wins)
    total_percent = 14.0 + 24.3 + 33.6 + 28.1
    
    # Distribution for when higher seed wins (maintain relative proportions)
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