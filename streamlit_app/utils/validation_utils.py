"""
Stub module for validation utilities - imports from data_handlers.py
This file is maintained for backward compatibility.
"""

from streamlit_app.utils.data_handlers import (
    check_data_quality,
    display_validation_metrics,
    validate_matchup_data_with_ui,
    validate_model_compatibility,
    get_model_feature_importance,
    validate_data_quality
)

__all__ = [
    'check_data_quality',
    'display_validation_metrics',
    'validate_matchup_data_with_ui',
    'validate_model_compatibility',
    'get_model_feature_importance',
    'validate_data_quality'
]
