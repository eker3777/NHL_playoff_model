# Data Handlers Consolidation Plan

## Overview

This document outlines the plan for consolidating functionality from multiple data-related modules into a single comprehensive `data_handlers.py` module. This approach will simplify the codebase, reduce duplication, and create a single source of truth for data operations.

## Rationale

1. **Reduce Module Count**: Fewer modules means fewer imports to manage across the codebase
2. **Logical Grouping**: Data validation is closely related to data handling
3. **Easier Maintenance**: A single file is easier to maintain than multiple related files
4. **Eliminate Duplication**: Reduce duplicate functionality across multiple files
5. **Standardized Constants**: Use application-wide constants from the config module

## Implementation Approach

### 1. File Structure Analysis

Current files and their responsibilities:
- `data_handlers.py`: Data loading, processing, and transformation
- `data_validation.py`: Core validation functions
- `validation_utils.py`: Helper utilities for validation
- `cache_manager.py`: Data refreshing and caching logic
- `config.py`: Application-wide constants and settings

### 2. Imports to Add

#### From `config.py`:
```python
from streamlit_app.config import (
    DATA_DIR,
    CACHE_DIR,
    SIMULATION_RESULTS_DIR,
    CRITICAL_FEATURES,
    PERCENTAGE_COLUMNS,
    HOME_ICE_ADVANTAGE,
    REFRESH_TIMEZONE,
    CACHE_DURATION,
    REFRESH_HOUR
)
```

#### From `cache_manager.py`:
```python
from streamlit_app.utils.cache_manager import (
    should_refresh_data,
    get_current_time,
    cache_simulation_results,
    load_cached_simulation_results,
    is_cache_fresh
)
```

### 3. Consolidation Plan

We'll organize the consolidated `data_handlers.py` into clear sections:

```python
# Import from config and cache_manager
from streamlit_app.config import (...)
from streamlit_app.utils.cache_manager import (...)

#-----------------------------------------------------------------------------
# DATA VALIDATION FUNCTIONS
#-----------------------------------------------------------------------------
def validate_and_fix(data, validation_type="default", threshold=0.0, verbose=False):
    """Main validation function that identifies and fixes data issues"""
    # Implementation

def get_validation_report(data, validation_type="default", threshold=0.0):
    """Generate validation report without modifying data"""
    # Implementation

def validate_matchup_data(matchup_df, feature_list=None, verbose=False):
    """Validate matchup data specifically for playoff predictions"""
    # Implementation

def display_validation_results(validation_results, container=None):
    """Show validation results in the Streamlit UI"""
    # Implementation

def validate_team_data(team_data, feature_list=None, verbose=False):
    """Validate team data for completeness and consistency"""
    # Implementation

def check_data_quality(df, critical_features=None):
    """Check data quality and return metrics"""
    # Implementation

def standardize_percentage(value):
    """Convert various percentage formats to a standardized decimal (0-1)"""
    # Implementation

def display_validation_metrics(metrics, container=None):
    """Display validation metrics in the UI"""
    # Implementation

def validate_matchup_data_with_ui(matchup_df, feature_list=None):
    """Validate matchup data and show results in the UI"""
    # Implementation

def validate_model_compatibility(models, data):
    """Validate that data is compatible with models"""
    # Implementation

def get_model_feature_importance(model, feature_names=None):
    """Extract feature importance from model"""
    # Implementation

def validate_data_quality(df, feature_list=None, quality_threshold=0.8):
    """General data quality validation"""
    # Implementation

def print_validation_report(report, verbose=False):
    """Print validation report to console"""
    # Implementation

#-----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
#-----------------------------------------------------------------------------
# Existing data loading functions with config references

#-----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
#-----------------------------------------------------------------------------
# Existing data processing functions with config references
```

### 4. Functions to Consolidate

#### From `data_validation.py`:
- `validate_and_fix()`
- `get_validation_report()`
- `validate_matchup_data()`
- `display_validation_results()`
- `validate_team_data()`
- `check_data_quality()`
- `standardize_percentage()`
- `print_validation_report()`

#### From `validation_utils.py`:
- `display_validation_metrics()`
- `validate_matchup_data_with_ui()`
- `validate_model_compatibility()`
- `get_model_feature_importance()`
- `validate_data_quality()`

### 5. Functions to Replace with Imports

- Replace custom `should_refresh_data()` with import from `cache_manager`
- Replace hardcoded constants with imports from `config`

### 6. Backward Compatibility Strategy

We'll maintain backward compatibility by creating stub modules that import from the consolidated file:

```python
# In data_validation.py
from streamlit_app.utils.data_handlers import (
    validate_and_fix, 
    get_validation_report,
    validate_matchup_data,
    display_validation_results,
    validate_team_data,
    check_data_quality,
    standardize_percentage,
    validate_data_quality,
    print_validation_report
)

__all__ = [
    'validate_and_fix',
    'get_validation_report',
    'validate_matchup_data',
    'display_validation_results',
    'validate_team_data',
    'check_data_quality',
    'standardize_percentage',
    'validate_data_quality',
    'print_validation_report'
]
```

```python
# In validation_utils.py
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
```

### 7. Implementation Phases

1. **Preparation Phase**
   - Identify all functions to migrate ✅
   - Identify constants to replace with config imports ✅
   - Identify cache functions to replace with imports ✅

2. **Integration Phase**
   - Add imports from config and cache_manager
   - Add data validation section to data_handlers.py
   - Migrate functions from data_validation.py
   - Migrate functions from validation_utils.py
   - Replace local functions with imported ones
   - Replace hardcoded constants with config references

3. **Testing Phase**
   - Test imported functions
   - Verify config values are correctly used
   - Test backward compatibility through stub modules
   - Run existing UI components to ensure they work with the refactored code

4. **Cleanup Phase**
   - Update documentation
   - Mark original functions as deprecated
   - Communicate changes to the team

## Function Mapping

| Function | Source | Target Section | Dependencies |
|----------|--------|----------------|-------------|
| `validate_and_fix()` | data_validation.py | DATA VALIDATION | `get_validation_report()` |
| `get_validation_report()` | data_validation.py | DATA VALIDATION | `CRITICAL_FEATURES` from config |
| `validate_matchup_data()` | data_validation.py | DATA VALIDATION | - |
| `display_validation_results()` | data_validation.py | DATA VALIDATION | streamlit |
| `validate_team_data()` | data_validation.py | DATA VALIDATION | `CRITICAL_FEATURES` from config |
| `check_data_quality()` | data_validation.py | DATA VALIDATION | - |
| `standardize_percentage()` | data_validation.py | DATA VALIDATION | - |
| `display_validation_metrics()` | validation_utils.py | DATA VALIDATION | streamlit |
| `validate_matchup_data_with_ui()` | validation_utils.py | DATA VALIDATION | streamlit, `validate_matchup_data()` |
| `validate_model_compatibility()` | validation_utils.py | DATA VALIDATION | - |
| `get_model_feature_importance()` | validation_utils.py | DATA VALIDATION | - |
| `validate_data_quality()` | validation_utils.py | DATA VALIDATION | - |
| `print_validation_report()` | validation_utils.py | DATA VALIDATION | - |
| `should_refresh_data()` | data_handlers.py (replace) | Import | From cache_manager |
| `get_standings_data()` | data_handlers.py (modify) | DATA LOADING | API_TIMEOUT from config |
| `get_team_stats_data()` | data_handlers.py (modify) | DATA LOADING | API_TIMEOUT from config |
| `get_advanced_stats_data()` | data_handlers.py (modify) | DATA LOADING | DATA_DIR from config |
| `engineer_features()` | data_handlers.py (modify) | DATA PROCESSING | CRITICAL_FEATURES from config |

## Completion Tracking

| Task | Status | Notes |
|------|--------|-------|
| Identify all functions to consolidate | ✅ Complete | See function mapping above |
| Identify config constants to import | ✅ Complete | See imports section |
| Identify cache functions to import | ✅ Complete | See imports section |
| Create stub modules for compatibility | ✅ Complete | data_validation.py and validation_utils.py |
| Implement consolidated data_handlers.py | ⏳ In Progress | Integration phase |
| Test with existing UI components | 🔄 Not Started | Testing phase |
| Update documentation | 🔄 Not Started | Cleanup phase |

## Next Steps

1. ⏳ Implement the consolidated data_handlers.py:
   - Add imports from config and cache_manager
   - Add validation functions
   - Replace local duplicates with imports
   - Replace hardcoded constants with config references

2. 🔄 Test the consolidated module:
   - Unit tests for validation functions
   - Integration tests with UI components
   - Verify backward compatibility

3. 🔄 Perform cleanup:
   - Documentation updates
   - Code comments for clarity
   - Team communication
