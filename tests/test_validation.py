"""
Test script for data validation functions.
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit_app.utils.data_validation import validate_data_quality, print_validation_report, validate_and_fix

def create_test_dataframe():
    """Create a test dataframe with known issues for validation testing"""
    # Create a basic DataFrame with some issues
    df = pd.DataFrame({
        'teamName': ['Team A', 'Team B', 'Team C', None, 'Team E'],
        'teamAbbrev': ['TMA', 'TMB', 'TMC', 'TMD', 'TME'],
        'season': ['20232024', '20232024', '20232024', '20232024', '20232024'],
        'points': [100, 90, 80, 70, np.nan],
        'PP%': [0.25, 0.30, 1.5, 110, -0.1],  # Mix of 0-1 and 0-100 scale, plus invalid
        'PK%': [80, 75, 82, 78, 79],  # All on 0-100 scale
        'xGoalsPercentage': [0.52, 0.48, 0.55, np.nan, 0.51],
        'goalDiff/G': [1.2, -0.5, 0.8, 0.1, np.nan]
    })
    return df

def test_validation():
    """Test the validation functions"""
    print("Testing data validation functions...")
    
    # Create test data
    df = create_test_dataframe()
    
    # Test basic validation
    print("\n1. Basic validation:")
    report = validate_data_quality(df)
    print_validation_report(report)
    
    # Test with critical columns
    print("\n2. Validation with critical columns:")
    critical_cols = ['teamName', 'points', 'xGoalsPercentage']
    report = validate_data_quality(df, critical_cols)
    print_validation_report(report)
    
    # Test pre-merge validation
    print("\n3. Pre-merge validation:")
    report = validate_data_quality(df, None, "pre-merge")
    print_validation_report(report)
    
    # Test pre-model validation
    print("\n4. Pre-model validation:")
    report = validate_data_quality(df, None, "pre-model")
    print_validation_report(report)
    
    # Test validate_and_fix
    print("\n5. Testing validate_and_fix:")
    fixed_df, report = validate_and_fix(df)
    
    # Check if fixes were applied
    print("\nOriginal vs. Fixed dataframe:")
    print(f"PP% before: {df['PP%'].tolist()}")
    print(f"PP% after: {fixed_df['PP%'].tolist()}")
    print(f"PK% before: {df['PK%'].tolist()}")
    print(f"PK% after: {fixed_df['PK%'].tolist()}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_validation()
