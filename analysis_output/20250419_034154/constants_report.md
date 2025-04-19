# NHL Playoff Model - Constants Analysis Report

## Overview

This report details the constants management issues identified in the NHL playoff model codebase. Proper constants management is critical for ensuring consistent behavior across the application.

## Current State of Constants Management

### Key Constants and Their Locations

| Constant Name | Correct Value | Files Defining It | Value Consistency |
|---------------|---------------|-------------------|-------------------|
| `HOME_ICE_ADVANTAGE` | 0.039 | 12 files | Inconsistent (ranges from 0.035 to 0.04) |
| `SERIES_LENGTH_DIST` | [0.140, 0.243, 0.336, 0.281] | 8 files | Inconsistent (variations in distribution) |
| `REFRESH_HOUR` | 5 | 4 files | Consistent |
| `API_TIMEOUT` | 30 | 5 files | Inconsistent (ranges from 10 to 60) |
| `CRITICAL_FEATURES` | List of features | 7 files | Inconsistent (different feature lists) |

### Files Correctly Using `config.py`

Only 5 files are correctly importing constants from `config.py`:
1. `streamlit_app/utils/data_handlers.py`
2. `streamlit_app/utils/model_utils.py`
3. `streamlit_app/utils/cache_manager.py`
4. `streamlit_app/pages/debug.py`
5. `streamlit_app/models/simulation_utils.py`

### Files with Hardcoded Constants

28 files contain hardcoded constants that should be imported from `config.py`. The most critical instances are:

1. **Home Ice Advantage Constants**:
   - `streamlit_app/models/simulation.py` (line 145): `home_ice_boost = models.get('home_ice_boost', 0.039)`
   - `streamlit_app/pages/first_round.py` (line 203): `home_ice_factor = 0.038`
   - `streamlit_app/pages/head_to_head.py` (line 312): `home_ice_advantage = 0.04`

2. **Series Length Distribution Constants**:
   - `streamlit_app/models/simulation.py` (lines 92-95): Hardcoded distribution values
   - `streamlit_app/pages/first_round.py` (lines 258-261): Different distribution values
   - `streamlit_app/utils/simulation_utils.py` (line 49): Uses a different implementation

3. **API Constants**:
   - `streamlit_app/utils/data_handlers.py` (line 85): `timeout=API_TIMEOUT` correctly uses the imported constant
   - `streamlit_app/pages/first_round.py` (line 125): `timeout=30` hardcodes the value
   - `streamlit_app/pages/head_to_head.py` (line 89): `timeout=10` uses a different value

## Impact of Constants Inconsistency

### Prediction Inconsistency

The inconsistent application of `HOME_ICE_ADVANTAGE` leads to different predictions for the same matchup depending on which page the user is viewing:

- On the first_round page, a team might show a 58.3% win probability
- On the head_to_head page, the same matchup might show 58.5% 
- In the simulation, it might use 58.8%

These small differences can significantly impact simulation results, especially when compounded over multiple rounds.

### Series Length Inconsistency

The different series length distributions lead to inconsistent visualizations and simulations:

- Some files use a distribution that overrepresents 7-game series
- Others use a distribution that underrepresents 4-game sweeps
- This impacts both the visual presentation and the simulation outcomes

### Maintenance Challenges

Having constants defined in multiple locations makes maintenance difficult:

- When updating a constant value, developers must locate all instances
- This increases the chance of missed updates
- Over time, the values can drift further apart

## Recommended Changes

### 1. Enhance `config.py`

Update the central configuration file to include all constants with proper documentation:

```python
# Home ice advantage (percentage boost for home team)
HOME_ICE_ADVANTAGE = 0.039  # 3.9% boost based on historical data

# Series length distribution [4,5,6,7 games] based on historical NHL data
SERIES_LENGTH_DISTRIBUTION = [0.140, 0.243, 0.336, 0.281]

# API configuration
API_BASE_URL = "https://api-web.nhle.com/v1"
API_TIMEOUT = 30  # seconds

# Data refresh settings
REFRESH_HOUR = 5  # 5 AM Eastern Time refresh
TIMEZONE = "US/Eastern"

# Feature lists for models
CRITICAL_FEATURES = [
    'PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score',
    'adjGoalsScoredAboveX/60'
]

# Model features derived from critical features
MODEL_FEATURES = [f"{feature}_diff" for feature in CRITICAL_FEATURES] + ['points_diff']
```

### 2. Update All Files

Modify all files with hardcoded constants to import from `config.py`. Top priority files are:

1. `streamlit_app/models/simulation.py`
2. `streamlit_app/pages/first_round.py`
3. `streamlit_app/pages/head_to_head.py`
4. `streamlit_app/utils/simulation_utils.py`

### 3. Validation Approach

After making changes, implement validation to ensure constants are used consistently:

1. Create a test script that checks all files for hardcoded constants
2. Verify that all instances of critical constants have been replaced
3. Run the application to ensure consistent behavior across all pages

## Conclusion

Centralizing constants in `config.py` and ensuring all files import from this central location will significantly improve the consistency and maintainability of the NHL playoff model. This change is a critical first step in addressing the cross-functionality issues in the codebase.
