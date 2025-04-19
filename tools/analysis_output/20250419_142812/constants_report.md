# NHL Playoff Model - Constants Usage Report

## Overview

Analysis performed on 37 Python files.

## Critical Constants Analysis

Found 31 constants defined in config.py:

- `APP_TITLE` = `NHL Playoff Predictor`
- `APP_VERSION` = `2.0.0`
- `AUTHOR` = `Your Name`
- `GITHUB_URL` = `https://github.com/yourusername/NHL_playoff_model`
- `HOME_ICE_ADVANTAGE` = `0.039`
- `SERIES_LENGTH_DISTRIBUTION` = `[0.14, 0.243, 0.336, 0.281]`
- `API_BASE_URL` = `https://api-web.nhle.com/v1`
- `API_TIMEOUT` = `30`
- `API_RETRIES` = `3`
- `REFRESH_HOUR` = `5`
- `TIMEZONE` = `US/Eastern`
- `CACHE_DURATION` = `86400`
- `CRITICAL_FEATURES` = `['PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score', 'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%', 'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg', 'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60', 'adjGoalsScoredAboveX/60']`
- `MODEL_FEATURES` = `None`
- `PERCENTAGE_COLUMNS` = `['PP%', 'PK%', 'FO%', 'xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage', 'shootingPercentage', 'savePctg', 'homeRegulationWin%', 'roadRegulationWin%']`
- `CONFERENCE_NAMES` = `['Eastern', 'Western']`
- `DIVISION_NAMES` = `['Atlantic', 'Metropolitan', 'Central', 'Pacific']`
- `PLAYOFF_SPOTS_PER_DIVISION` = `3`
- `WILDCARDS_PER_CONFERENCE` = `2`
- `DEFAULT_SIMULATION_COUNT` = `10000`
- `MINIMUM_SIMULATION_COUNT` = `1000`
- `MAXIMUM_SIMULATION_COUNT` = `50000`
- `TEAM_COLORS` = `{'ANA': '#F47A38', 'ARI': '#8C2633', 'BOS': '#FFB81C', 'BUF': '#002654', 'CGY': '#C8102E', 'CAR': '#CC0000', 'CHI': '#CF0A2C', 'COL': '#6F263D', 'CBJ': '#002654', 'DAL': '#006847', 'DET': '#CE1126', 'EDM': '#041E42', 'FLA': '#041E42', 'LAK': '#111111', 'MIN': '#154734', 'MTL': '#AF1E2D', 'NSH': '#FFB81C', 'NJD': '#CE1126', 'NYI': '#00539B', 'NYR': '#0038A8', 'OTT': '#C52032', 'PHI': '#F74902', 'PIT': '#FCB514', 'SJS': '#006D75', 'SEA': '#99D9D9', 'STL': '#002F87', 'TBL': '#002868', 'TOR': '#00205B', 'VAN': '#00205B', 'VGK': '#B4975A', 'WSH': '#C8102E', 'WPG': '#041E42', 'UTA': '#002F87'}`
- `DEBUG_MODE` = `False`
- `LOG_LEVEL` = `INFO`
- `MAX_LOG_FILES` = `5`
- `MAX_LOG_SIZE` = `None`
- `BASE_DIR` = `None`
- `DATA_DIR` = `None`
- `MODEL_DIR` = `None`
- `LOG_DIR` = `None`

11 files import constants from config.py:

### Files NOT importing from config.py:

- tests/test_validation.py
- tests/__init__.py
- streamlit_app/__init__.py
- streamlit_app/components/__init__.py
- streamlit_app/utils/validation_utils.py
- streamlit_app/utils/visualization_utils.py
- streamlit_app/utils/simulation_analysis.py
- streamlit_app/utils/visualization.py
- streamlit_app/utils/__init__.py
- streamlit_app/pages/debug.py
- streamlit_app/pages/simulation_results.py
- streamlit_app/pages/about.py
- streamlit_app/pages/__init__.py
- streamlit_app/models/simulation.py
- streamlit_app/models/__init__.py
- tools/dependency_analyzer.py
- tools/run_analysis.py
- tools/analyze_functions.py
- tools/constants_validator.py
- tools/function_checker.py
- tools/analyze_imports.py
- tools/consistency_verification_script.py
- git/git_pull.py
- git/git_push.py
- git/deploy_streamlit.py

### Critical Constants Defined Outside config.py:

#### `HOME_ICE_ADVANTAGE`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 0.039
- Defined in /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py with value: 0.039
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 0.039

#### `SERIES_LENGTH_DISTRIBUTION`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: [0.14, 0.243, 0.336, 0.281]
- Defined in /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py with value: [0.14, 0.243, 0.336, 0.281]
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: [0.14, 0.243, 0.336, 0.281]

#### `API_BASE_URL`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: https://api-web.nhle.com/v1
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: https://api-web.nhle.com/v1

#### `API_TIMEOUT`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 30
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 30

#### `REFRESH_HOUR`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 5
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: 5

#### `CRITICAL_FEATURES`

- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: ['PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score', 'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%', 'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg', 'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60', 'adjGoalsScoredAboveX/60']
- Defined in /workspaces/NHL_playoff_model/streamlit_app/config.py with value: ['PP%_rel', 'PK%_rel', 'FO%', 'playoff_performance_score', 'xGoalsPercentage', 'homeRegulationWin%', 'roadRegulationWin%', 'possAdjHitsPctg', 'possAdjTakeawaysPctg', 'possTypeAdjGiveawaysPctg', 'reboundxGoalsPctg', 'goalDiff/G', 'adjGoalsSavedAboveX/60', 'adjGoalsScoredAboveX/60']

### Hardcoded Values That Should Use Constants:

#### `HOME_ICE_ADVANTAGE`

- Found in streamlit_app/config.py line 17: 0.039
- Found in streamlit_app/utils/model_utils.py line 21: 0.039
- Found in streamlit_app/models/simulation.py line 1937: 0.039

#### `SERIES_LENGTH_DIST`

- Found in streamlit_app/config.py line 18: [0.14, 0.243, 0.336, 0.281]
- Found in streamlit_app/utils/model_utils.py line 23: [0.14, 0.243, 0.336, 0.281]
- Found in streamlit_app/models/simulation.py line 1938: [0.14, 0.243, 0.336, 0.281]
