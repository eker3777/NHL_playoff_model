# NHL Playoff Model - Function Consistency Report

## Overview

Analysis performed on 34 Python files.

## Key Function Analysis

### `standardize_percentage()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/data_validation.py: `standardize_percentage(value)`
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
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `predict_series_winner(matchup_df, models)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `predict_series_winner(matchup_df, models)`
- ... and 12 more calls

### `simulate_playoff_bracket()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations, detailed_tracking)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations, detailed_tracking)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=<value>, detailed_tracking=<value>)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_playoff_bracket(playoff_matchups, team_data, models, n_simulations=<value>)`

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
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `load_team_data(data_folder)`

### `load_models()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/utils/model_utils.py: `load_models(model_folder)`

#### Called from:

- /workspaces/NHL_playoff_model/streamlit_app/utils/debug_utils.py: `load_models(model_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/pages/debug.py: `load_models(model_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `load_models(model_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `load_models(model_folder)`

### `run_playoff_simulations()`

#### Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `run_playoff_simulations(model_folder, data_folder, force_refresh)`

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

### `get_eastern_time()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/cache_manager.py: `get_eastern_time()`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `get_eastern_time()`

### `should_refresh_data()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/cache_manager.py: `should_refresh_data()`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `should_refresh_data()`

### `cache_simulation_results()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/cache_manager.py: `cache_simulation_results(results, data_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `cache_simulation_results(results, data_folder)`

### `load_cached_simulation_results()`

- /workspaces/NHL_playoff_model/streamlit_app/utils/cache_manager.py: `load_cached_simulation_results(data_folder)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `load_cached_simulation_results(data_folder)`

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

### `simulate_single_bracket()`

- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_single_bracket(playoff_matchups, team_data, models)`
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py: `simulate_single_bracket(playoff_matchups, team_data, models)`

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

### `simulate_playoff_bracket()`

Different signatures:

- `simulate_playoff_bracket(playoff_matchups,team_data,models,n_simulations)`
- `simulate_playoff_bracket(playoff_matchups,team_data,models,n_simulations,detailed_tracking)`

Defined in:

- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py
- /workspaces/NHL_playoff_model/streamlit_app/models/simulation.py

### `run_analysis()`

Different signatures:

- `run_analysis(self)`
- `run_analysis(project_dir)`

Defined in:

- /workspaces/NHL_playoff_model/tools/dependency_analyzer.py
- /workspaces/NHL_playoff_model/tools/run_analysis.py

### `run_command()`

Different signatures:

- `run_command(command,verbose)`
- `run_command(command,capture_output)`

Defined in:

- /workspaces/NHL_playoff_model/git/git_pull.py
- /workspaces/NHL_playoff_model/git/git_push.py
- /workspaces/NHL_playoff_model/git/deploy_streamlit.py
