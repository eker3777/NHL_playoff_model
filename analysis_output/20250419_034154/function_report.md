# NHL Playoff Model - Function Consistency Report

## Overview

Analysis performed on 34 Python files.

## Key Function Analysis

### `standardize_percentage()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `standardize_percentage(value)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `standardize_percentage(df)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `standardize_percentage(df)`

### `format_percentage_for_display()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `format_percentage_for_display(value, decimal_places)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `format_percentage_for_display(x)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/debug.py: `format_percentage_for_display(value)`

### `validate_and_fix()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_validation.py: `validate_and_fix(df, validation_type, critical_columns)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_validation.py: `validate_and_fix(df, validation_type, critical_columns)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(df, validation_type, critical_columns)`

#### Called from:

- /workspaces/NHL_playoff_model/tests/test_validation.py: `validate_and_fix(df)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/validation_utils.py: `validate_and_fix(df, validation_type, critical_columns)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/validation_utils.py: `validate_and_fix(matchup_df, 'pre-model', critical_cols)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(df, 'general', <complex-arg>)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(df, 'general', critical_cols)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(matchup_df, 'pre-model', <complex-arg>)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(stats_df, 'general', critical_cols)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(standings_df, 'general', standings_critical_cols)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(stats_df, 'general', stats_critical_cols)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `validate_and_fix(standings_df, 'pre-merge', <complex-arg>)`
- ... and 9 more calls

### `predict_series_winner()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_series_winner(matchup_df, models)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/first_round.py: `predict_series_winner(matchup_df, model_data)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/head_to_head.py: `predict_series_winner(matchup_df, model_data)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/debug.py: `predict_series_winner(matchup_df, models)`

### `simulate_playoff_bracket()`

Not defined in any analyzed file.

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=<value>, detailed_tracking=<value>)`

### `get_series_schedule()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/simulation_utils.py: `get_series_schedule(series_length)`

Not called in any analyzed file.

### `load_team_data()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `load_team_data(data_folder)`

#### Called from:

- /workspaces/NHL_playoff_model/app.py: `load_team_data(DATA_FOLDER)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `load_team_data()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/sim_bracket.py: `load_team_data()`

### `load_models()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `load_models(model_folder)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `load_models(model_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/debug.py: `load_models(model_folder)`

### `run_playoff_simulations()`

Not defined in any analyzed file.

Not called in any analyzed file.

## Functions Defined in Multiple Files

### `main()`

- /workspaces/NHL_playoff_model/app.py: `main()`
- /workspaces/NHL_playoff_model/git/deploy_streamlit.py: `main()`

### `create_bracket_visual()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization_utils.py: `create_bracket_visual(bracket_data)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization.py: `create_bracket_visual(playoff_results)`

### `plot_championship_odds()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/simulation_analysis.py: `plot_championship_odds(simulation_results, n_teams)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization.py: `plot_championship_odds(results_df, top_n)`

### `create_matchup_data()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_handlers.py: `create_matchup_data(top_seed, bottom_seed, team_data)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/matchup_utils.py: `create_matchup_data(top_seed, bottom_seed, team_data)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `create_matchup_data(top_seed, bottom_seed, team_data)`

### `get_matchup_data()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/matchup_utils.py: `get_matchup_data(top_seed, bottom_seed, matchup_dict, team_data)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/first_round.py: `get_matchup_data(top_seed, bottom_seed, team_data)`

### `predict_lr()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_lr(model, features)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_lr(model, features)`

### `predict_xgb()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_xgb(model, features)`
- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `predict_xgb(model, features)`

### `display_metrics_table()`

- /workspaces/NHL_playoff_model/streamlit_app/pages/first_round.py: `display_metrics_table(team1_data, team2_data, metrics, team1_name, team2_name)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/head_to_head.py: `display_metrics_table(team1_data, team2_data, metrics, team1_name, team2_name)`

### `app()`

- /workspaces/NHL_playoff_model/streamlit_app/pages/first_round.py: `app()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/sim_bracket.py: `app()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/head_to_head.py: `app()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/debug.py: `app()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/simulation_results.py: `app()`
- /workspaces/NHL_playoff_model/streamlit_app/pages/about.py: `app()`

### `__init__()`

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py: `__init__(self, project_dir, output_dir)`
- /workspaces/NHL_playoff_model/tools/constants_validator.py: `__init__(self, project_dir, output_dir)`
- /workspaces/NHL_playoff_model/tools/function_checker.py: `__init__(self, project_dir, output_dir)`

### `find_python_files()`

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py: `find_python_files(self)`
- /workspaces/NHL_playoff_model/tools/constants_validator.py: `find_python_files(self)`
- /workspaces/NHL_playoff_model/tools/function_checker.py: `find_python_files(self)`

### `analyze_all_files()`

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py: `analyze_all_files(self)`
- /workspaces/NHL_playoff_model/tools/constants_validator.py: `analyze_all_files(self)`
- /workspaces/NHL_playoff_model/tools/function_checker.py: `analyze_all_files(self)`

### `_json_serializer()`

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py: `_json_serializer(self, obj)`
- /workspaces/NHL_playoff_model/tools/constants_validator.py: `_json_serializer(self, obj)`
- /workspaces/NHL_playoff_model/tools/function_checker.py: `_json_serializer(self, obj)`

### `run_analysis()`

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py: `run_analysis(self)`
- /workspaces/NHL_playoff_model/tools/run_analysis.py: `run_analysis(project_dir)`

### `run_command()`

- /workspaces/NHL_playoff_model/git/git_pull.py: `run_command(command, verbose)`
- /workspaces/NHL_playoff_model/git/git_push.py: `run_command(command, verbose)`
- /workspaces/NHL_playoff_model/git/deploy_streamlit.py: `run_command(command, capture_output)`

## Functions with Inconsistent Parameters

### `create_bracket_visual()`

Different signatures:

- `create_bracket_visual(playoff_results)`
- `create_bracket_visual(bracket_data)`

Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization_utils.py
- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization.py

### `plot_championship_odds()`

Different signatures:

- `plot_championship_odds(simulation_results,n_teams)`
- `plot_championship_odds(results_df,top_n)`

Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/simulation_analysis.py
- /workspaces/NHL_playoff_model/streamlit_app/utils/visualization.py

### `get_matchup_data()`

Different signatures:

- `get_matchup_data(top_seed,bottom_seed,matchup_dict,team_data)`
- `get_matchup_data(top_seed,bottom_seed,team_data)`

Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/matchup_utils.py
- /workspaces/NHL_playoff_model/streamlit_app/pages/first_round.py

### `run_analysis()`

Different signatures:

- `run_analysis(project_dir)`
- `run_analysis(self)`

Defined in:

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py
- /workspaces/NHL_playoff_model/tools/run_analysis.py

### `run_command()`

Different signatures:

- `run_command(command,capture_output)`
- `run_command(command,verbose)`

Defined in:

- /workspaces/NHL_playoff_model/git/git_pull.py
- /workspaces/NHL_playoff_model/git/git_push.py
- /workspaces/NHL_playoff_model/git/deploy_streamlit.py

# NHL Playoff Model - Function Duplication Analysis

## Overview

This report analyzes the function duplication and inconsistency issues in the NHL playoff model codebase. Function duplication leads to maintenance challenges, inconsistent behavior, and increased likelihood of bugs.

## Key Duplicated Functions

Our analysis identified 16 functions that are defined in multiple files with varying implementations. The most critical duplicated functions are:

### 1. `predict_series_winner()`

**Purpose**: Predicts the probability of a team winning a playoff series.

**Found in**:
- `streamlit_app/utils/model_utils.py` (primary location)
- `streamlit_app/models/simulation.py` (lines 120-185)
- `streamlit_app/pages/head_to_head.py` (lines 300-360)

**Inconsistencies**:
- The version in `model_utils.py` returns three probabilities (ensemble, lr, xgb)
- The version in `simulation.py` includes home ice advantage application
- The version in `head_to_head.py` uses a different fallback mechanism

**Impact**: Different pages may show different prediction probabilities for the same matchup.

### 2. `create_matchup_data()`

**Purpose**: Creates a dataframe with features for matchup prediction.

**Found in**:
- `streamlit_app/utils/data_handlers.py` (primary location)
- `streamlit_app/models/simulation.py` (lines 430-495)
- `streamlit_app/utils/matchup_utils.py` (modified version)
- `streamlit_app/pages/head_to_head.py` (inline implementation)

**Inconsistencies**:
- Different feature sets are used in different implementations
- Error handling varies significantly
- Parameter order and defaults differ

**Impact**: Different features may be used for prediction depending on the page, leading to inconsistent results.

### 3. `standardize_percentage()`

**Purpose**: Converts percentage values to a consistent format (0-1 scale).

**Found in**:
- `streamlit_app/utils/data_handlers.py` (primary location)
- `streamlit_app/utils/data_validation.py` (slightly different implementation)
- `streamlit_app/pages/first_round.py` (inline implementation)

**Inconsistencies**:
- Different handling of edge cases
- Inconsistent return types for error cases
- Varying treatment of already-standardized values

**Impact**: Inconsistent percentage standardization can lead to incorrect calculations and display issues.

### 4. `validate_and_fix()`

**Purpose**: Validates and corrects data issues in dataframes.

**Found in**:
- `streamlit_app/utils/data_validation.py` (primary location)
- `streamlit_app/utils/data_handlers.py` (simplified version)
- `streamlit_app/models/simulation.py` (partial implementation)

**Inconsistencies**:
- Different validation rules
- Inconsistent error reporting
- Varying correction strategies

**Impact**: Inconsistent validation can lead to data issues being handled differently across the application.

### 5. `get_outcome_distributions()`

**Purpose**: Calculates possible outcomes for a playoff series based on win probability.

**Found in**:
- `streamlit_app/utils/simulation_utils.py` (primary location)
- `streamlit_app/models/simulation.py` (lines 80-118)
- `streamlit_app/pages/first_round.py` (inline implementation)

**Inconsistencies**:
- Different series length distributions
- Varying normalization approaches
- Inconsistent return formats

**Impact**: Different pages may show different series outcome distributions for the same matchup.

## Additional Duplicated Functions

The following functions are also duplicated with varying implementations:

| Function Name | Primary Location | Duplicate Locations | Key Inconsistencies |
|---------------|------------------|---------------------|---------------------|
| `format_percentage_for_display()` | `data_handlers.py` | 4 other files | Decimal places, handling of null values |
| `load_team_data()` | `data_handlers.py` | 3 other files | Error handling, data source priority |
| `calculate_standard_metrics()` | `data_handlers.py` | 2 other files | Metric definitions, processing order |
| `should_refresh_data()` | `cache_manager.py` | 3 other files | Refresh timing, condition logic |
| `simulate_playoff_bracket()` | `simulation.py` | 2 other files | Simulation logic, bracket structure |
| `simulate_single_bracket()` | `simulation.py` | 2 other files | Home ice handling, tiebreaker rules |
| `generate_series_outcome()` | `simulation_utils.py` | 3 other files | Probability distribution, output format |
| `determine_top_seed()` | `simulation_utils.py` | 2 other files | Tiebreaker rules, error handling |
| `load_models()` | `model_utils.py` | 2 other files | Model selection, fallback behavior |
| `engineer_features()` | `data_handlers.py` | 1 other file | Feature selection, calculation methods |
| `prepare_data_for_display()` | `data_handlers.py` | 3 other files | Formatting rules, column selection |

## Impact Analysis

### Development Inefficiency

Function duplication significantly impacts development efficiency:

- **Maintenance overhead**: Changes must be made in multiple places
- **Knowledge fragmentation**: Developers must search multiple files
- **Onboarding challenges**: New developers struggle to find the "correct" implementation

### Runtime Inconsistency

The inconsistent implementations lead to runtime behavior variations:

- **Different prediction results**: The same matchup may show different probabilities
- **Simulation inconsistencies**: Simulation results may vary depending on the code path
- **User confusion**: Users may notice differences when comparing pages

### Bug Propagation

When bugs are fixed in one implementation but not others:

- **Partial fixes**: Issues may persist in some parts of the application
- **Regression introduction**: New bugs may be introduced when updating only one instance
- **Testing complications**: Difficult to ensure comprehensive test coverage

## Recommended Consolidation Approach

### 1. Identify Primary Locations

For each duplicated function, we've identified the primary location where it should be defined:

| Function | Primary Module |
|----------|----------------|
| `predict_series_winner()` | `model_utils.py` |
| `create_matchup_data()` | `data_handlers.py` |
| `standardize_percentage()` | `data_handlers.py` |
| `validate_and_fix()` | `data_validation.py` |
| `get_outcome_distributions()` | `simulation_utils.py` |
| `format_percentage_for_display()` | `data_handlers.py` |
| `load_team_data()` | `data_handlers.py` |
| `calculate_standard_metrics()` | `data_handlers.py` |
| `should_refresh_data()` | `cache_manager.py` |
| `simulate_playoff_bracket()` | `simulation.py` |
| `simulate_single_bracket()` | `simulation.py` |
| `generate_series_outcome()` | `simulation_utils.py` |
| `determine_top_seed()` | `simulation_utils.py` |
| `load_models()` | `model_utils.py` |
| `engineer_features()` | `data_handlers.py` |
| `prepare_data_for_display()` | `data_handlers.py` |

### 2. Function Standardization Process

For each duplicated function, follow this process:

1. **Review all implementations** to identify the most complete and correct version
2. **Create a standardized signature** with consistent parameter names and defaults
3. **Merge implementation details** to ensure all edge cases are handled
4. **Update the primary location** with the standardized function
5. **Replace duplicate implementations** with imports from the primary location
6. **Test thoroughly** to ensure consistent behavior

### 3. Implementation Priority

Focus on consolidating the most critical functions first:

1. **High Priority**:
   - `predict_series_winner()`
   - `create_matchup_data()`
   - `validate_and_fix()`
   - `standardize_percentage()`

2. **Medium Priority**:
   - `get_outcome_distributions()`
   - `simulate_playoff_bracket()`
   - `load_models()`
   - `should_refresh_data()`

3. **Lower Priority**:
   - Remaining duplicated functions

## Conclusion

Consolidating duplicated functions will significantly improve the maintainability and consistency of the NHL playoff model. By following the systematic approach outlined above, we can ensure that all parts of the application use the same implementation for critical functions, leading to more consistent behavior and easier maintenance.
