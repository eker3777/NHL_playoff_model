"""
Stub module for data validation - imports from data_handlers.py
This file is maintained for backward compatibility.
"""

from streamlit_app.utils.data_handlers import (
    validate_and_fix,
    get_validation_report,
    validate_matchup_data,
    display_validation_results,
    validate_team_data,
    check_data_quality,
    standardize_percentage,
    validate_data_quality,
    print_validation_report,
)

__all__ = [
    "validate_and_fix",
    "get_validation_report",
    "validate_matchup_data",
    "display_validation_results",
    "validate_team_data",
    "check_data_quality",
    "standardize_percentage",
    "validate_data_quality",
    "print_validation_report",
]
