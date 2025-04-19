"""
Test script to verify backward compatibility of the consolidated data_handlers module.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test importing from deprecated modules
print("Testing imports from deprecated modules...")

from streamlit_app.utils.data_validation import (
    validate_and_fix,
    get_validation_report,
    validate_matchup_data,
    validate_team_data
)

from streamlit_app.utils.validation_utils import (
    validate_matchup_data_with_ui,
    validate_model_compatibility,
    validate_data_quality
)

# Now test the consolidated module
print("\nTesting imports from consolidated module...")

from streamlit_app.utils.data_handlers import (
    validate_and_fix as new_validate_and_fix,
    validate_matchup_data as new_validate_matchup_data,
    check_data_quality,
    standardize_percentage,
    create_matchup_features,
    merge_team_data
)

# Create sample data for testing
print("\nTesting functionality with sample data...")

sample_team_data = pd.DataFrame({
    'team_name': ['Team A', 'Team B', 'Team C'],
    'wins': [40, 35, 30],
    'losses': [20, 25, 30],
    'win_pct': [0.667, 0.583, 0.5],
    'points': [90, 80, 70]
})

sample_matchup_data = pd.DataFrame({
    'home_team': ['Team A', 'Team B', 'Team C'],
    'away_team': ['Team B', 'Team C', 'Team A'],
    'home_score': [3, 2, 1],
    'away_score': [1, 1, 4]
})

# Test validation functions
print("Testing team data validation...")
fixed_team_data, team_report = validate_team_data(sample_team_data)
print(f"Team validation issues: {sum(r.get('fixes', 0) for r in team_report.values())}")

print("\nTesting matchup data validation...")
fixed_matchup_data, matchup_report = validate_matchup_data(sample_matchup_data)
print(f"Matchup validation issues: {sum(r.get('fixes', 0) for r in matchup_report.values())}")

# Verify data quality checks
print("\nTesting data quality check...")
quality_metrics = check_data_quality(sample_team_data)
print(f"Quality metrics: {len(quality_metrics)} metrics calculated")

print("\nAll tests completed successfully!")
