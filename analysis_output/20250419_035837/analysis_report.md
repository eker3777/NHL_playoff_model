# NHL Playoff Model - Cross-Functionality Analysis Report

Analysis performed on 34 Python files.

## Key Findings

### Constants Analysis

#### Duplicate Constants

- `HOME_ICE_ADVANTAGE` is defined in multiple files: streamlit_app/config.py, streamlit_app/utils/model_utils.py, streamlit_app/models/simulation.py
- `SERIES_LENGTH_DISTRIBUTION` is defined in multiple files: streamlit_app/config.py, streamlit_app/utils/model_utils.py, streamlit_app/models/simulation.py

### Function Usage Analysis

#### Duplicate Function Definitions

- `main()` is defined in multiple files: app.py, git/deploy_streamlit.py
- `create_bracket_visual()` is defined in multiple files: streamlit_app/utils/visualization_utils.py, streamlit_app/utils/visualization.py
- `validate_and_fix()` is defined in multiple files: streamlit_app/utils/data_validation.py, streamlit_app/utils/data_validation.py, streamlit_app/utils/data_handlers.py
- `standardize_percentage()` is defined in multiple files: streamlit_app/utils/data_validation.py, streamlit_app/utils/data_handlers.py
- `plot_championship_odds()` is defined in multiple files: streamlit_app/utils/simulation_analysis.py, streamlit_app/utils/visualization.py
- `create_matchup_data()` is defined in multiple files: streamlit_app/utils/data_handlers.py, streamlit_app/utils/matchup_utils.py, streamlit_app/utils/model_utils.py
- `get_matchup_data()` is defined in multiple files: streamlit_app/utils/matchup_utils.py, streamlit_app/pages/first_round.py
- `predict_lr()` is defined in multiple files: streamlit_app/utils/model_utils.py, streamlit_app/utils/model_utils.py
- `predict_xgb()` is defined in multiple files: streamlit_app/utils/model_utils.py, streamlit_app/utils/model_utils.py
- `get_eastern_time()` is defined in multiple files: streamlit_app/utils/cache_manager.py, streamlit_app/models/simulation.py
- `should_refresh_data()` is defined in multiple files: streamlit_app/utils/cache_manager.py, streamlit_app/models/simulation.py
- `cache_simulation_results()` is defined in multiple files: streamlit_app/utils/cache_manager.py, streamlit_app/models/simulation.py
- `load_cached_simulation_results()` is defined in multiple files: streamlit_app/utils/cache_manager.py, streamlit_app/models/simulation.py
- `display_metrics_table()` is defined in multiple files: streamlit_app/pages/first_round.py, streamlit_app/pages/head_to_head.py
- `app()` is defined in multiple files: streamlit_app/pages/first_round.py, streamlit_app/pages/sim_bracket.py, streamlit_app/pages/head_to_head.py, streamlit_app/pages/debug.py, streamlit_app/pages/simulation_results.py, streamlit_app/pages/about.py
- `simulate_playoff_bracket()` is defined in multiple files: streamlit_app/models/simulation.py, streamlit_app/models/simulation.py, streamlit_app/models/simulation.py
- `simulate_single_bracket()` is defined in multiple files: streamlit_app/models/simulation.py, streamlit_app/models/simulation.py
- `__init__()` is defined in multiple files: tools/dependency_analyzer.py, tools/constants_validator.py, tools/function_checker.py
- `find_python_files()` is defined in multiple files: tools/dependency_analyzer.py, tools/constants_validator.py, tools/function_checker.py
- `analyze_all_files()` is defined in multiple files: tools/dependency_analyzer.py, tools/constants_validator.py, tools/function_checker.py
- `_json_serializer()` is defined in multiple files: tools/dependency_analyzer.py, tools/constants_validator.py, tools/function_checker.py
- `run_analysis()` is defined in multiple files: tools/dependency_analyzer.py, tools/run_analysis.py
- `run_command()` is defined in multiple files: git/git_pull.py, git/git_push.py, git/deploy_streamlit.py

#### Key Function Usage

- `validate_and_fix()` is used in 19 files
- `format_percentage_for_display()` is used in 2 files
- `predict_series_winner()` is used in 21 files
- `simulate_playoff_bracket()` is used in 3 files

### Import Analysis

#### Most Complex Import Patterns

- streamlit_app/models/simulation.py: 40 total imports (9 stdlib, 9 third-party, 22 local)
- streamlit_app/utils/debug_utils.py: 30 total imports (6 stdlib, 4 third-party, 20 local)
- streamlit_app/utils/data_handlers.py: 27 total imports (6 stdlib, 3 third-party, 18 local)
- app.py: 26 total imports (2 stdlib, 2 third-party, 22 local)
- streamlit_app/utils/visualization_utils.py: 26 total imports (0 stdlib, 7 third-party, 19 local)
