import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import platform
import time
import traceback
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from streamlit_app.utils.data_handlers import *
from streamlit_app.utils.data_validation import *
from streamlit_app.utils.model_utils import *

# Import cache manager functions with centralized time handling
from streamlit_app.utils.cache_manager import get_current_time, should_refresh_data, is_cache_fresh
from streamlit_app.config import (
    REFRESH_TIMEZONE, 
    HOME_ICE_ADVANTAGE, 
    SERIES_LENGTH_DISTRIBUTION, 
    DEBUG_MODE,
    CRITICAL_FEATURES,
    DATA_DIR,
    MODEL_DIR,
    PERCENTAGE_COLUMNS,
    BASE_DIR
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# ===== INTEGRATED DEBUG UTILITY FUNCTIONS =====

def get_data_quality_metrics(team_data):
    """Calculate data quality metrics for team data
    
    Args:
        team_data (DataFrame): Team data to analyze
        
    Returns:
        dict: Dictionary of data quality metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_teams'] = len(team_data)
    metrics['total_columns'] = len(team_data.columns)
    
    # Missing data
    missing_values = team_data.isna().sum().sum()
    total_values = team_data.size
    metrics['missing_values'] = missing_values
    metrics['missing_percentage'] = (missing_values / total_values) * 100
    
    # Define critical features for prediction
    critical_features = CRITICAL_FEATURES
    
    # Check critical features
    critical_features_present = sum([1 for f in critical_features if f in team_data.columns])
    metrics['critical_features_present'] = critical_features_present
    metrics['critical_features_total'] = len(critical_features)
    metrics['critical_features_percentage'] = (critical_features_present / len(critical_features)) * 100
    
    # Check for advanced stats
    advanced_features = ['xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage']
    metrics['has_advanced_stats'] = any([f in team_data.columns for f in advanced_features])
    
    # Calculate data quality score (out of 10)
    # 40% based on critical features, 40% based on missing data, 20% based on advanced stats
    feature_score = min(4.0, (critical_features_present / len(critical_features)) * 4.0)
    missing_score = min(4.0, 4.0 * (1 - (missing_values / total_values)))
    advanced_score = 2.0 if metrics['has_advanced_stats'] else 0.0
    
    metrics['quality_score'] = feature_score + missing_score + advanced_score
    
    # Check validation issues
    # Import locally to avoid circular imports
    try:
        _, validation_report = validate_and_fix(team_data, 'general')
        metrics['validation_issues'] = len(validation_report.get('issues', []))
    except:
        metrics['validation_issues'] = 0
    
    return metrics

def analyze_feature_distributions(team_data, feature_name):
    """Analyze the distribution of a feature across teams
    
    Args:
        team_data (DataFrame): Team data
        feature_name (str): Name of the feature to analyze
        
    Returns:
        tuple: (matplotlib figure, stats dictionary)
    """
    if feature_name not in team_data.columns:
        return None, {"error": f"Feature {feature_name} not found in data"}
    
    # Calculate basic statistics
    stats = {
        "mean": team_data[feature_name].mean(),
        "median": team_data[feature_name].median(),
        "std": team_data[feature_name].std(),
        "min": team_data[feature_name].min(),
        "max": team_data[feature_name].max(),
        "null_count": team_data[feature_name].isna().sum(),
        "null_percentage": team_data[feature_name].isna().mean() * 100
    }
    
    # Create distribution plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Histogram
    sns.histplot(team_data[feature_name].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {feature_name}")
    axes[0].axvline(stats["mean"], color='r', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
    axes[0].axvline(stats["median"], color='g', linestyle='--', label=f"Median: {stats['median']:.3f}")
    axes[0].legend()
    
    # Box plot
    sns.boxplot(x=team_data[feature_name].dropna(), ax=axes[1])
    axes[1].set_title(f"Box Plot of {feature_name}")
    
    plt.tight_layout()
    
    return fig, stats

def check_matchup_features(matchup_df, models):
    """Check if matchup data has all required features for models
    
    Args:
        matchup_df (DataFrame): Matchup data
        models (dict): Models dictionary
        
    Returns:
        dict: Feature check results
    """
    results = {
        "feature_stats": {},
        "missing_features": [],
        "null_features": []
    }
    
    # Define critical features if not provided in models
    critical_features = CRITICAL_FEATURES
    
    # Get all critical feature diffs
    critical_diffs = [f"{feature}_diff" for feature in critical_features]
    
    # Check which features are available and which have null values
    present_critical = [f for f in critical_diffs if f in matchup_df.columns]
    missing_critical = [f for f in critical_diffs if f not in matchup_df.columns]
    null_critical = [f for f in present_critical if matchup_df[f].isna().any()]
    usable_critical = [f for f in present_critical if f not in null_critical]
    
    # Store statistics
    results["feature_stats"] = {
        "total_critical": len(critical_diffs),
        "present_critical": len(present_critical),
        "missing_critical": len(missing_critical),
        "null_critical": len(null_critical),
        "usable_critical": len(usable_critical)
    }
    
    # Store lists of problem features
    results["missing_features"] = missing_critical
    results["null_features"] = null_critical
    
    # Check model-specific features if models are provided
    if models and "models" in models:
        for model_name, model_data in models["models"].items():
            if "features" in model_data:
                model_features = model_data["features"]
                
                # Check for missing features
                model_missing = [f for f in model_features if f not in matchup_df.columns]
                model_null = [f for f in model_features if f in matchup_df.columns and matchup_df[f].isna().any()]
                
                results[f"{model_name}_missing"] = model_missing
                results[f"{model_name}_null"] = model_null
                results[f"{model_name}_usable"] = len(model_features) - len(model_missing) - len(model_null)
    
    return results

def check_simulation_consistency(playoff_matchups, models, n_runs=5, n_sims_per_run=100):
    """Run multiple simulations to check for consistency in results
    
    Args:
        playoff_matchups (dict): Dictionary of playoff matchups
        models (dict): Dictionary of models
        n_runs (int): Number of separate simulation runs to execute
        n_sims_per_run (int): Number of simulations per run
        
    Returns:
        dict: Consistency metrics
    """
    try:
        # Import simulation function
        from streamlit_app.models.simulation import simulate_playoff_bracket
        
        # Load team data
        team_data = st.session_state.get('team_data', None)
        
        if team_data is None or team_data.empty:
            return {"error": "No team data available for simulation"}
        
        # Run multiple simulations
        logger.info(f"Running {n_runs} simulation batches with {n_sims_per_run} simulations each")
        
        simulation_results = []
        for i in range(n_runs):
            # Run simulation
            results = simulate_playoff_bracket(
                playoff_matchups, team_data, models, 
                n_simulations=n_sims_per_run, detailed_tracking=True
            )
            
            # Store results
            if 'team_advancement' in results and not results['team_advancement'].empty:
                simulation_results.append(results['team_advancement'])
            
            logger.info(f"Completed simulation run {i+1}/{n_runs}")
        
        # If we don't have enough results, return error
        if len(simulation_results) < 2:
            return {"error": "Insufficient simulation results for consistency check"}
        
        # Calculate consistency metrics
        logger.info("Calculating consistency metrics")
        
        # Compare the championship probabilities across runs
        champ_probs = {}
        for i, result_df in enumerate(simulation_results):
            for _, row in result_df.iterrows():
                team = row['teamName']
                if team not in champ_probs:
                    champ_probs[team] = []
                
                champ_probs[team].append(row['champion'])
        
        # Calculate standard deviation of probabilities for each team
        consistency = {}
        for team, probs in champ_probs.items():
            if len(probs) >= 2:
                consistency[team] = {
                    "mean_prob": np.mean(probs),
                    "std_dev": np.std(probs),
                    "coefficient_of_variation": np.std(probs) / np.mean(probs) if np.mean(probs) > 0 else 0,
                    "min_prob": min(probs),
                    "max_prob": max(probs),
                    "range": max(probs) - min(probs)
                }
        
        # Sort by coefficient of variation (higher means less consistent)
        sorted_consistency = dict(sorted(consistency.items(), 
                                         key=lambda x: x[1]["coefficient_of_variation"], 
                                         reverse=True))
        
        # Calculate overall consistency metrics
        overall_cv = np.mean([item["coefficient_of_variation"] for item in consistency.values()])
        overall_range = np.mean([item["range"] for item in consistency.values()])
        
        return {
            "overall_metrics": {
                "coefficient_of_variation": overall_cv,
                "average_range": overall_range,
                "stability_score": 10 - min(10, overall_cv * 100),
                "simulation_runs": n_runs,
                "simulations_per_run": n_sims_per_run
            },
            "team_consistency": sorted_consistency
        }
    except Exception as e:
        logger.error(f"Error in simulation consistency check: {str(e)}", exc_info=True)
        return {"error": str(e)}

# ===== MAIN DEBUG PAGE IMPLEMENTATION =====

def display_debug_page(team_data=None, model_data=None):
    """Display debug information and tools for developers"""
    st.title("Debug & Diagnostics")
    
    # Add disclaimer
    st.info("This page is intended for developers to debug model issues and data quality problems.")
    
    # Show cache refresh times
    st.subheader("Cache Status")
    if 'last_data_refresh' in st.session_state:
        refresh_time = st.session_state.last_data_refresh
        st.write(f"Last data refresh: {refresh_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Calculate time since refresh
        current_time = get_current_time()  # Use get_current_time from cache_manager
        
        time_since_refresh = current_time - refresh_time
        hours = time_since_refresh.total_seconds() / 3600
        
        if hours < 24:
            st.write(f"Time since refresh: {hours:.1f} hours")
        else:
            days = hours / 24
            st.write(f"Time since refresh: {days:.1f} days")
    else:
        st.warning("No data refresh recorded in session state")
    
    # Show simulation refresh times
    if 'last_simulation_refresh' in st.session_state:
        sim_time = st.session_state.last_simulation_refresh
        # Add check to ensure sim_time is not None before calling strftime
        if sim_time is not None:
            st.write(f"Last simulation refresh: {sim_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            st.warning("Last simulation refresh time is recorded but has a null value")
    else:
        st.warning("No simulation refresh recorded in session state")
    
    # Create tabs for different debug sections
    debug_tabs = st.tabs([
        "Data Quality", 
        "Model Diagnostics", 
        "Simulation Testing",
        "System Diagnostics"
    ])
    
    # 1. DATA QUALITY TAB
    with debug_tabs[0]:
        st.header("Data Quality Metrics")
        
        # Data refresh status
        if 'last_data_refresh' in st.session_state:
            refresh_time = st.session_state.last_data_refresh
            st.success(f"Data last refreshed: {refresh_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Check if cache is fresh using the updated cache manager
            cache_fresh = is_cache_fresh(refresh_time)
            if cache_fresh:
                st.success("Cache is fresh")
            else:
                st.warning("Cache is stale and needs refresh")
            
            # Calculate time since last refresh
            current_time = get_current_time()
            time_since_refresh = current_time - refresh_time
            hours = time_since_refresh.total_seconds() / 3600
            st.write(f"Time since last refresh: {hours:.1f} hours")
        else:
            st.warning("No data refresh timestamp found. Data may be stale.")
            
        # Force refresh data button
        if st.button("Force Data Refresh"):
            st.write("Refreshing data...")
            
            # Force update with updated function signature
            updated = update_daily_data(DATA_DIR, force=True)
            
            if updated:
                st.success("Data refreshed successfully!")
            else:
                st.error("Data refresh failed.")
        
        # Team data validation section
        st.subheader("Team Data Validation")
        
        # Get team data
        if team_data is not None and not team_data.empty:
            
            # Show basic info
            st.write(f"Team data shape: {team_data.shape}")
            
            # Get data quality metrics
            quality_metrics = get_data_quality_metrics(team_data)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Teams", quality_metrics['total_teams'])
                st.metric("Missing Values %", f"{quality_metrics['missing_percentage']:.1f}%")
            
            with col2:
                st.metric("Critical Features", f"{quality_metrics['critical_features_present']}/{quality_metrics['critical_features_total']}")
                st.metric("Advanced Stats Available", "Yes" if quality_metrics['has_advanced_stats'] else "No")
            
            with col3:
                st.metric("Data Quality Score", f"{quality_metrics['quality_score']:.1f}/10")
                st.metric("Validation Issues", quality_metrics['validation_issues'])
            
            # Show validation report
            if st.checkbox("Show Data Validation Report"):
                _, validation_report = validate_and_fix(team_data, 'general')
                st.json(validation_report)
            
            # Display critical features availability
            st.subheader("Critical Feature Availability")
            
            # Check feature availability
            feature_status = {}
            for feature in CRITICAL_FEATURES:
                if feature in team_data.columns:
                    null_pct = team_data[feature].isna().mean() * 100
                    feature_status[feature] = {
                        'available': True,
                        'null_percentage': null_pct,
                        'status': 'Good' if null_pct < 10 else 'Warning' if null_pct < 30 else 'Critical'
                    }
                else:
                    feature_status[feature] = {
                        'available': False,
                        'null_percentage': 100.0,
                        'status': 'Missing'
                    }
            
            # Create feature availability dataframe
            feature_df = pd.DataFrame.from_dict(feature_status, orient='index')
            feature_df.reset_index(inplace=True)
            feature_df.rename(columns={'index': 'Feature'}, inplace=True)
            
            # Use color coding
            def highlight_status(val):
                if val == 'Good':
                    return 'background-color: #c6efce'
                elif val == 'Warning':
                    return 'background-color: #ffeb9c'
                elif val == 'Critical':
                    return 'background-color: #ffc7ce'
                elif val == 'Missing':
                    return 'background-color: #d9d9d9'
                return ''
            
            # Display styled dataframe
            st.dataframe(feature_df.style.applymap(highlight_status, subset=['status']))
            
            # Feature distribution analysis
            if st.checkbox("Show Feature Distributions"):
                selected_feature = st.selectbox("Select feature to analyze:", CRITICAL_FEATURES)
                
                if selected_feature in team_data.columns:
                    fig, stats = analyze_feature_distributions(team_data, selected_feature)
                    st.pyplot(fig)
                    st.write(stats)
                else:
                    st.warning(f"Feature '{selected_feature}' not available in the data.")
            
            # Team data explorer
            if st.checkbox("Explore Team Data"):
                selected_team = st.selectbox("Select team:", team_data['teamName'].tolist())
                team_row = team_data[team_data['teamName'] == selected_team].iloc[0]
                
                # Format data for display
                display_data = {}
                for col in team_row.index:
                    value = team_row[col]
                    
                    # Format percentages appropriately
                    if any(x in col for x in ['Pct', 'Percentage', '%']):
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            if value <= 1.0:
                                display_data[col] = format_percentage_for_display(value)
                            else:
                                display_data[col] = f"{value:.1f}%"
                    else:
                        display_data[col] = value
                
                # Convert to DataFrame for display
                display_df = pd.DataFrame.from_dict(display_data, orient='index', columns=['Value'])
                # Explicitly convert column to string to avoid Arrow conversion issues
                display_df['Value'] = display_df['Value'].astype(str)
                st.dataframe(display_df)
        else:
            st.warning("No team data available for validation. Try refreshing the data.")
    
    # 2. MODEL DIAGNOSTICS TAB
    with debug_tabs[1]:
        st.header("Model Diagnostics")
        
        # Load models
        models = model_data
        
        # Display model information
        if models and (models.get('models') or models.get('mode')):
            st.subheader("Model Information")
            
            # Show model mode and parameters
            st.write(f"Model mode: {models.get('mode', 'unknown')}")
            
            # Use the HOME_ICE_ADVANTAGE constant from config
            home_ice_value = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
            st.write(f"Home ice advantage: {home_ice_value*100:.1f}%")
            
            # Check which models are available
            model_types = []
            if 'models' in models:
                if 'lr' in models['models']:
                    model_types.append("Logistic Regression")
                if 'xgb' in models['models']:
                    model_types.append("XGBoost")
            
            if model_types:
                st.write(f"Available models: {', '.join(model_types)}")
            else:
                st.warning("No specific model types found in models dictionary.")
            
            # Model features
            model_features = get_model_features(models, 'ensemble')
            if model_features:
                st.subheader("Model Features")
                st.write(f"Number of features: {len(model_features)}")
                
                # Display feature list
                if st.checkbox("Show feature list"):
                    st.write(", ".join(model_features))
            
            # Interactive prediction tester
            st.subheader("Prediction Tester")
            
            # Get team data for matchup creation
            if team_data is not None and not team_data.empty:
                
                # Create two columns for team selection
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Higher Seed (Home Team)")
                    team1 = st.selectbox("Select higher seed:", team_data['teamName'].tolist(), key="team1")
                    team1_row = team_data[team_data['teamName'] == team1].iloc[0]
                    team1_abbrev = team1_row['teamAbbrev']
                    
                with col2:
                    st.write("Lower Seed (Away Team)")
                    team2 = st.selectbox("Select lower seed:", team_data['teamName'].tolist(), key="team2")
                    team2_row = team_data[team_data['teamName'] == team2].iloc[0]
                    team2_abbrev = team2_row['teamAbbrev']
                
                # Create matchup data
                if st.button("Run Prediction"):
                    # Create team dictionaries
                    team1_dict = {
                        'teamName': team1,
                        'teamAbbrev': team1_abbrev,
                        'division_rank': 1  # Assuming higher seed
                    }
                    
                    team2_dict = {
                        'teamName': team2,
                        'teamAbbrev': team2_abbrev,
                        'division_rank': 2  # Assuming lower seed
                    }
                    
                    # Create matchup data
                    matchup_df = create_matchup_data(team1_dict, team2_dict, team_data)
                    
                    # Check features
                    feature_check = check_matchup_features(matchup_df, models)
                    
                    # Display feature quality
                    st.subheader("Feature Quality Check")
                    
                    # Feature statistics
                    if "feature_stats" in feature_check:
                        stats = feature_check["feature_stats"]
                        st.write(f"Total critical features: {stats['total_critical']}")
                        st.write(f"Present critical features: {stats['present_critical']}")
                        st.write(f"Missing critical features: {stats['missing_critical']}")
                        st.write(f"Critical features with null values: {stats['null_critical']}")
                        st.write(f"Usable critical features: {stats['usable_critical']}")
                        
                        # Warning if too many features are missing
                        if stats['usable_critical'] < stats['total_critical'] * 0.7:
                            st.warning(f"Only {stats['usable_critical']}/{stats['total_critical']} critical features are usable. Predictions may be unreliable.")
                    
                    # Run prediction
                    st.subheader("Prediction Results")
                    
                    # Get raw probabilities
                    ensemble_prob, lr_prob, xgb_prob = predict_series_winner(matchup_df, models)
                    
                    # Apply home ice advantage
                    home_ice_boost = models.get('home_ice_boost', HOME_ICE_ADVANTAGE)
                    final_prob = min(1.0, ensemble_prob + home_ice_boost)
                    
                    # Display probabilities
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Raw LR Probability", f"{lr_prob*100:.1f}%")
                    
                    with col2:
                        st.metric("Raw XGB Probability", f"{xgb_prob*100:.1f}%")
                    
                    with col3:
                        st.metric("Raw Ensemble Probability", f"{ensemble_prob*100:.1f}%")
                    
                    # Home ice metrics
                    st.write(f"Home ice advantage: {home_ice_boost*100:.1f}%")
                    st.metric("Final Win Probability", f"{final_prob*100:.1f}%")
                    
                    # Run a full series simulation
                    st.subheader("Series Simulation")
                    series_results = predict_series(matchup_df, models, n_simulations=1000)
                    
                    # Display series outcome distribution
                    outcomes = series_results['win_distribution']
                    total_sims = sum(outcomes.values())
                    
                    # Create a DataFrame for the outcomes
                    outcome_data = []
                    
                    # Higher seed outcomes
                    for outcome in ['4-0', '4-1', '4-2', '4-3']:
                        outcome_data.append({
                            'Team': team1,
                            'Outcome': outcome,
                            'Probability': outcomes[outcome] / total_sims * 100
                        })
                    
                    # Lower seed outcomes
                    for outcome, display_outcome in zip(['0-4', '1-4', '2-4', '3-4'], ['4-0', '4-1', '4-2', '4-3']):
                        outcome_data.append({
                            'Team': team2,
                            'Outcome': display_outcome,
                            'Probability': outcomes[outcome] / total_sims * 100
                        })
                    
                    outcome_df = pd.DataFrame(outcome_data)
                    
                    # Plot the outcomes
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot for team1
                    team1_data = outcome_df[outcome_df['Team'] == team1]
                    ax.bar(
                        team1_data['Outcome'], 
                        team1_data['Probability'],
                        label=team1,
                        alpha=0.7
                    )
                    
                    # Plot for team2
                    team2_data = outcome_df[outcome_df['Team'] == team2]
                    ax.bar(
                        [f"{outcome} ({team2})" for outcome in team2_data['Outcome']], 
                        team2_data['Probability'],
                        label=team2,
                        alpha=0.7
                    )
                    
                    ax.set_xlabel('Series Outcome')
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('Series Outcome Distribution')
                    ax.legend()
                    
                    # Show the plot
                    st.pyplot(fig)
                    
                    # Display raw matchup data for debugging
                    if st.checkbox("Show raw matchup data"):
                        st.dataframe(matchup_df)
            else:
                st.warning("No team data available for prediction testing.")
        else:
            st.error("No models loaded. Check the model files in the models directory.")
            
            # Add diagnostic information to help debug the issue
            from streamlit_app.config import MODEL_DIR
            
            st.subheader("Model Loading Diagnostic Information")
            
            # Check if MODEL_DIR exists
            if os.path.exists(MODEL_DIR):
                st.success(f"Model directory exists: {MODEL_DIR}")
                
                # List files in the model directory
                model_files = os.listdir(MODEL_DIR)
                if model_files:
                    st.write("Files found in model directory:")
                    for file in model_files:
                        file_path = os.path.join(MODEL_DIR, file)
                        file_size = os.path.getsize(file_path) / 1024  # Size in KB
                        st.write(f"- {file} ({file_size:.1f} KB)")
                else:
                    st.warning(f"No files found in model directory: {MODEL_DIR}")
                    st.info("You can create test models using the tools/create_test_models.py script.")
            else:
                st.error(f"Model directory does not exist: {MODEL_DIR}")
                st.info(f"Try creating the directory manually: `mkdir -p {MODEL_DIR}`")
            
            # Option to create test models directly from the UI
            if st.button("Create Test Models"):
                st.write("Creating test models...")
                try:
                    sys.path.append(os.path.join(BASE_DIR, "tools"))
                    from tools.create_test_models import create_test_models
                    create_test_models()
                    st.success("Test models created successfully! Please refresh the page.")
                except Exception as e:
                    st.error(f"Error creating test models: {str(e)}")
                    st.code(traceback.format_exc())

    # 3. SIMULATION TESTING TAB
    with debug_tabs[2]:
        st.header("Simulation Testing")
        
        # Load current playoff matchups
        playoff_matchups = load_current_playoff_matchups(DATA_DIR)
        
        if playoff_matchups:
            st.subheader("Current Playoff Matchups")
            
            # Display matchups by conference
            for conference, matchups in playoff_matchups.items():
                st.write(f"**{conference} Conference**")
                
                for series_id, matchup in matchups.items():
                    top_team = matchup['top_seed']['teamName']
                    bottom_team = matchup['bottom_seed']['teamName']
                    st.write(f"- {series_id}: {top_team} vs {bottom_team}")
            
            # Series length distribution visualization
            st.subheader("Series Length Distribution")
            
            # Use SERIES_LENGTH_DISTRIBUTION from config
            series_lengths = [4, 5, 6, 7]
            percentages = [val * 100 for val in SERIES_LENGTH_DISTRIBUTION]
            
            # Create a DataFrame for plotting
            dist_df = pd.DataFrame({
                'Series Length': series_lengths,
                'Percentage': percentages
            })
            
            # Plot the distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(dist_df['Series Length'], dist_df['Percentage'])
            ax.set_xlabel('Series Length (Games)')
            ax.set_ylabel('Percentage of Series (%)')
            ax.set_title('NHL Playoff Series Length Distribution')
            ax.set_xticks(series_lengths)
            
            # Add percentage labels on top of each bar
            for i, pct in enumerate(percentages):
                ax.text(series_lengths[i], pct + 1, f'{pct:.1f}%', ha='center')
            
            st.pyplot(fig)
            
            # Home ice advantage testing
            st.subheader("Home Ice Advantage Testing")
            
            # Get home ice advantage from config
            from streamlit_app.config import MODEL_DIR
            models = load_models(MODEL_DIR)
            home_ice = models.get('home_ice_boost', HOME_ICE_ADVANTAGE) if models else HOME_ICE_ADVANTAGE
            
            # Allow adjusting home ice advantage
            test_home_ice = st.slider("Test home ice advantage", 
                                      min_value=0.0, 
                                      max_value=0.1, 
                                      value=home_ice,
                                      step=0.001,
                                      format="%.3f")
            
            # Show the effect of home ice advantage
            base_probs = [0.4, 0.45, 0.5, 0.55, 0.6]
            
            # Create a DataFrame with probabilities
            home_ice_df = pd.DataFrame({
                'Base Win Probability': base_probs,
                'With Home Ice': [min(1.0, p + test_home_ice) for p in base_probs]
            })
            
            # Format the probabilities as percentages
            home_ice_df['Base Win Probability'] = home_ice_df['Base Win Probability'] * 100
            home_ice_df['With Home Ice'] = home_ice_df['With Home Ice'] * 100
            home_ice_df['Difference'] = home_ice_df['With Home Ice'] - home_ice_df['Base Win Probability']
            
            # Convert numeric columns to strings with formatted percentages
            # This ensures PyArrow doesn't have conversion issues
            formatted_df = home_ice_df.copy()
            for col in formatted_df.columns:
                if formatted_df[col].dtype in [np.float64, np.int64]:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
            
            # Display the table
            st.dataframe(formatted_df)
            
            # Simulation consistency checker
            if st.checkbox("Run Simulation Consistency Check"):
                with st.spinner("Running simulation consistency checks..."):
                    # Use the integrated function directly
                    consistency_results = check_simulation_consistency(playoff_matchups, models)
                    
                    if consistency_results:
                        st.success("Simulation consistency check completed!")
                        
                        # Display results
                        st.json(consistency_results)
        else:
            st.warning("No playoff matchups available. Try refreshing the data.")
    
    # 4. SYSTEM DIAGNOSTICS TAB
    with debug_tabs[3]:
        st.header("System Diagnostics")
        
        # Add debug mode information
        st.subheader("Debug Settings")
        st.write(f"Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
        st.write(f"Timezone: {REFRESH_TIMEZONE}")
        
        # Check session state
        st.subheader("Session State")
        
        # List key session state variables
        session_vars = {
            'team_data': 'team_data' in st.session_state,
            'playoff_matchups': 'playoff_matchups' in st.session_state,
            'daily_simulations': 'daily_simulations' in st.session_state,
            'last_data_refresh': 'last_data_refresh' in st.session_state
        }
        
        # Display as a table
        session_df = pd.DataFrame({
            'Variable': session_vars.keys(),
            'Available': session_vars.values()
        })
        
        # Convert boolean values to strings to avoid Arrow conversion issues
        session_df['Available'] = session_df['Available'].astype(str)
        st.dataframe(session_df)
        
        # Check file system
        st.subheader("File System Check")
        
        # List files in data folder
        data_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        model_files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Data Files:")
            for file in data_files:
                file_path = os.path.join(DATA_DIR, file)
                size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
                modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(file_path) else "N/A"
                st.write(f"- {file} ({size_kb:.1f} KB, modified: {modified})")
        
        with col2:
            st.write("Model Files:")
            for file in model_files:
                file_path = os.path.join(MODEL_DIR, file)
                size_kb = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
                modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(file_path) else "N/A"
                st.write(f"- {file} ({size_kb:.1f} KB, modified: {modified})")
        
        # Memory usage
        st.subheader("Memory Usage")
        
        if 'team_data' in st.session_state and not st.session_state.team_data.empty:
            team_data_size = st.session_state.team_data.memory_usage(deep=True).sum() / (1024 * 1024)
            st.write(f"Team data memory usage: {team_data_size:.2f} MB")
        
        if 'daily_simulations' in st.session_state:
            # Estimate size based on structure
            sim_size = sys.getsizeof(str(st.session_state.daily_simulations)) / (1024 * 1024)
            st.write(f"Simulation results memory usage: {sim_size:.2f} MB")

        # Display Python and package versions
        st.subheader("Python and Package Versions")
        
        versions = {
            'Python': platform.python_version(),
            'Pandas': pd.__version__,
            'NumPy': np.__version__,
            'Streamlit': st.__version__, # Fix: use matplotlib not plt
            'Matplotlib': matplotlib.__version__,
            'Seaborn': sns.__version__
        }
        
        # Try to get other package versions if available
        try:
            import sklearn
            versions['Scikit-learn'] = sklearn.__version__
        except:
            pass
        
        try:
            import xgboost
            versions['XGBoost'] = xgboost.__version__
        except:
            pass
        
        # Display versions
        for package, version in versions.items():
            st.write(f"- {package}: {version}")

def app():
    """Main entry point for the debug page"""
    # Get necessary data from session state
    models = st.session_state.get('models', None)
    team_data = st.session_state.get('team_data', None)
    
    # Call the display function
    display_debug_page(team_data=team_data, model_data=models)

def show_debug():
    """Entry point for debug page called from app.py"""
    app()

if __name__ == "__main__":
    # This allows the page to be run directly for testing
    display_debug_page()
