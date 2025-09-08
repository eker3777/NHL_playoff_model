# Branch Comparison Report

**Generated:** 2025-09-08 22:36:22
**Repository:** /home/runner/work/NHL_playoff_model/NHL_playoff_model
**Base Branch:** main
**Selected Branch:** copilot/compare-main-to-selected-branch

## Branch Information

- **Main Branch Commit:** e59cc9744ee0be338679e0cf95f0ed4021e4ce33
- **Selected Branch Commit:** b917a150c6380ecfc74b9beb34f259fe0c31da87
- **Commits Behind Main:** 19
- **Commits Ahead of Main:** 2

## Change Summary

| Type | Count |
|------|-------|
| Files Added | 106 |
| Files Modified | 22 |
| Files Deleted | 26 |
| Files Renamed | 11 |
| Lines Added | 116907 |
| Lines Removed | 6751 |

## Changes by File Type

| Extension | Count | Files |
|-----------|-------|-------|
| .pyc | 41 | streamlit_app/__pycache__/__init__.cpython-311.pyc, streamlit_app/__pycache__/cache_manager.cpython-311.pyc, streamlit_app/__pycache__/config.cpython-311.pyc (and 38 more) |
| .py | 36 | app.py, git/deploy_streamlit.py, git/git_pull.py (and 33 more) |
| .md | 25 | CODEBASE_ANALYSIS.md, CORE_NARRATIVE.md, CROSS_FILE_CHECKLIST.md (and 22 more) |
| .json | 22 | analysis_output/20250419_034154/config_analysis.json, analysis_output/20250419_034154/constants_analysis.json, analysis_output/20250419_034154/constants_usage.json (and 19 more) |
| .csv | 19 | streamlit_app/utils/data/standings_raw_20242025.csv, data/playoff_conf_finals_matchups_20242025.csv, data/playoff_round2_matchups_20242025.csv (and 16 more) |
| .dot | 4 | analysis_output/20250419_034154/dependency_graph.dot, analysis_output/20250419_035837/dependency_graph.dot, tools/analysis_output/20250419_141727/dependency_graph.dot (and 1 more) |
| .txt | 3 | data/error_log.txt, data/refresh_log.txt, streamlit_app/data/refresh_log.txt |
| .pkl | 2 | models/playoff_model.pkl, streamlit_app/models/logistic_regression_model_final.pkl |
| .ipynb | 1 | data_processing_notebook.ipynb |
| .toml | 1 | streamlit_app/secrets.toml |

## Changes by Directory

| Directory | Files Changed |
|-----------|---------------|
| root | 17 |
| streamlit_app/data | 12 |
| analysis_output/20250419_034154 | 10 |
| analysis_output/20250419_035837 | 10 |
| data | 10 |
| streamlit_app/__pycache__ | 10 |
| streamlit_app/pages/__pycache__ | 10 |
| streamlit_app/utils/__pycache__ | 10 |
| tools/analysis_output/20250419_141727 | 9 |
| tools/analysis_output/20250419_142812 | 9 |
| streamlit_app/utils | 8 |
| tools | 8 |
| streamlit_app | 7 |
| streamlit_app/app_pages/__pycache__ | 6 |
| git | 3 |
| streamlit_app/models | 3 |
| tests | 3 |
| tools/__pycache__ | 3 |
| streamlit_app/models/__pycache__ | 2 |
| streamlit_app/components | 1 |
| streamlit_app/pages | 1 |
| streamlit_app/utils/data | 1 |
| models | 1 |

## Added Files

- CODEBASE_ANALYSIS.md
- CORE_NARRATIVE.md
- CROSS_FILE_CHECKLIST.md
- DATA_CONSOLIDATION_PLAN.md
- DEPLOYMENT_ROADMAP.md
- IMPROVEMENTS.md
- PAGE_STRUCTURE.md
- analysis_output/20250419_034154/analysis_report.md
- analysis_output/20250419_034154/config_analysis.json
- analysis_output/20250419_034154/constants_analysis.json
- analysis_output/20250419_034154/constants_report.md
- analysis_output/20250419_034154/constants_usage.json
- analysis_output/20250419_034154/dependency_graph.dot
- analysis_output/20250419_034154/file_analysis.json
- analysis_output/20250419_034154/function_analysis.json
- analysis_output/20250419_034154/function_report.md
- analysis_output/20250419_034154/summary_report.md
- analysis_output/20250419_035837/analysis_report.md
- analysis_output/20250419_035837/config_analysis.json
- analysis_output/20250419_035837/constants_analysis.json
- analysis_output/20250419_035837/constants_report.md
- analysis_output/20250419_035837/constants_usage.json
- analysis_output/20250419_035837/dependency_graph.dot
- analysis_output/20250419_035837/file_analysis.json
- analysis_output/20250419_035837/function_analysis.json
- analysis_output/20250419_035837/function_report.md
- analysis_output/20250419_035837/summary_report.md
- app.py
- data/error_log.txt
- data/playoff_matchups.json
- data/refresh_log.txt
- git/deploy_streamlit.py
- git/git_pull.py
- git/git_push.py
- streamlit_app/__init__.py
- streamlit_app/__pycache__/__init__.cpython-311.pyc
- streamlit_app/__pycache__/cache_manager.cpython-311.pyc
- streamlit_app/__pycache__/config.cpython-311.pyc
- streamlit_app/__pycache__/data_validation.cpython-311.pyc
- streamlit_app/__pycache__/debug_utils.cpython-311.pyc
- streamlit_app/__pycache__/simulation_utils.cpython-311.pyc
- streamlit_app/__pycache__/validation_utils.cpython-311.pyc
- streamlit_app/components/__init__.py
- streamlit_app/config.py
- streamlit_app/data/refresh_log.txt
- streamlit_app/models/__init__.py
- streamlit_app/models/__pycache__/__init__.cpython-311.pyc
- streamlit_app/models/__pycache__/simulation.cpython-311.pyc
- streamlit_app/models/simulation.py
- streamlit_app/pages/__pycache__/bracket.cpython-311.pyc
- streamlit_app/pages/__pycache__/debug.cpython-311.pyc
- streamlit_app/pages/__pycache__/overview.cpython-311.pyc
- streamlit_app/pages/__pycache__/team_odds.cpython-311.pyc
- streamlit_app/pages/debug.py
- streamlit_app/utils/__init__.py
- streamlit_app/utils/__pycache__/__init__.cpython-311.pyc
- streamlit_app/utils/__pycache__/cache_manager.cpython-311.pyc
- streamlit_app/utils/__pycache__/data_handlers.cpython-311.pyc
- streamlit_app/utils/__pycache__/data_validation.cpython-311.pyc
- streamlit_app/utils/__pycache__/debug_utils.cpython-311.pyc
- streamlit_app/utils/__pycache__/matchup_utils.cpython-311.pyc
- streamlit_app/utils/__pycache__/model_utils.cpython-311.pyc
- streamlit_app/utils/__pycache__/simulation_utils.cpython-311.pyc
- streamlit_app/utils/__pycache__/validation_utils.cpython-311.pyc
- streamlit_app/utils/__pycache__/visualization.cpython-311.pyc
- streamlit_app/utils/cache_manager.py
- streamlit_app/utils/data/standings_raw_20242025.csv
- streamlit_app/utils/data_validation.py
- streamlit_app/utils/matchup_utils.py
- streamlit_app/utils/model_utils.py
- streamlit_app/utils/simulation_analysis.py
- streamlit_app/utils/simulation_utils.py
- streamlit_app/utils/validation_utils.py
- test.py
- tests/__init__.py
- tests/test_validation.py
- tests/test_validation_imports.py
- tools/__pycache__/constants_validator.cpython-311.pyc
- tools/__pycache__/dependency_analyzer.cpython-311.pyc
- tools/__pycache__/function_checker.cpython-311.pyc
- tools/analysis_output/20250419_141727/analysis_report.md
- tools/analysis_output/20250419_141727/config_analysis.json
- tools/analysis_output/20250419_141727/constants_analysis.json
- tools/analysis_output/20250419_141727/constants_report.md
- tools/analysis_output/20250419_141727/constants_usage.json
- tools/analysis_output/20250419_141727/dependency_graph.dot
- tools/analysis_output/20250419_141727/file_analysis.json
- tools/analysis_output/20250419_141727/function_analysis.json
- tools/analysis_output/20250419_141727/summary_report.md
- tools/analysis_output/20250419_142812/analysis_report.md
- tools/analysis_output/20250419_142812/config_analysis.json
- tools/analysis_output/20250419_142812/constants_analysis.json
- tools/analysis_output/20250419_142812/constants_report.md
- tools/analysis_output/20250419_142812/constants_usage.json
- tools/analysis_output/20250419_142812/dependency_graph.dot
- tools/analysis_output/20250419_142812/file_analysis.json
- tools/analysis_output/20250419_142812/function_analysis.json
- tools/analysis_output/20250419_142812/summary_report.md
- tools/analyze_functions.py
- tools/analyze_imports.py
- tools/consistency_verification_script.py
- tools/constants_validator.py
- tools/create_test_models.py
- tools/dependency_analyzer.py
- tools/function_checker.py
- tools/run_analysis.py

## Modified Files

- data/playoff_conf_finals_matchups_20242025.csv
- data/playoff_round2_matchups_20242025.csv
- data/playoff_sim_results_20242025.csv
- data/playoff_stanley_cup_matchups_20242025.csv
- data/standings_20242025.csv
- data/stats_20242025.csv
- data/team_data_20242025.csv
- data_processing_notebook.ipynb
- streamlit_app/__pycache__/data_handlers.cpython-311.pyc
- streamlit_app/__pycache__/model_utils.cpython-311.pyc
- streamlit_app/__pycache__/simulation.cpython-311.pyc
- streamlit_app/data/moneypuck_regular_2024.csv
- streamlit_app/data/nhl_playoff_wins_2005_present.csv
- streamlit_app/data/playoff_matchups.json
- streamlit_app/data/team_data_20242025.csv
- streamlit_app/data/temp_data.csv
- streamlit_app/pages/__pycache__/__init__.cpython-311.pyc
- streamlit_app/pages/__pycache__/about.cpython-311.pyc
- streamlit_app/pages/__pycache__/first_round.cpython-311.pyc
- streamlit_app/pages/__pycache__/head_to_head.cpython-311.pyc
- streamlit_app/pages/__pycache__/sim_bracket.cpython-311.pyc
- streamlit_app/pages/__pycache__/simulation_results.cpython-311.pyc

## Deleted Files

- CODEBASE_REVIEW.md
- CODE_QUALITY_EXAMPLES.md
- EXECUTIVE_SUMMARY.md
- IMPLEMENTATION_ROADMAP.md
- check_models.py
- git_push.py
- models/playoff_model.pkl
- streamlit_app/app_pages/__pycache__/__init__.cpython-311.pyc
- streamlit_app/app_pages/__pycache__/about.cpython-311.pyc
- streamlit_app/app_pages/__pycache__/first_round.cpython-311.pyc
- streamlit_app/app_pages/__pycache__/head_to_head.cpython-311.pyc
- streamlit_app/app_pages/__pycache__/sim_bracket.cpython-311.pyc
- streamlit_app/app_pages/__pycache__/simulation_results.cpython-311.pyc
- streamlit_app/data/playoff_conf_finals_matchups_20242025.csv
- streamlit_app/data/playoff_round2_matchups_20242025.csv
- streamlit_app/data/playoff_sim_results_20242025.csv
- streamlit_app/data/playoff_stanley_cup_matchups_20242025.csv
- streamlit_app/data/standings_20242025.csv
- streamlit_app/data/standings_raw_20242025.csv
- streamlit_app/main.py
- streamlit_app/matchup_utils.py
- streamlit_app/model_utils.py
- streamlit_app/models/logistic_regression_model_final.pkl
- streamlit_app/secrets.toml
- streamlit_app/simulation.py
- temp_data.csv

## Renamed Files

- streamlit_app/app.py -> debug.log
- streamlit_app/app_pages/__init__.py -> streamlit_app/pages/__init__.py
- streamlit_app/app_pages/about.py -> streamlit_app/pages/about.py
- streamlit_app/app_pages/first_round.py -> streamlit_app/pages/first_round.py
- streamlit_app/app_pages/head_to_head.py -> streamlit_app/pages/head_to_head.py
- streamlit_app/app_pages/sim_bracket.py -> streamlit_app/pages/sim_bracket.py
- streamlit_app/app_pages/simulation_results.py -> streamlit_app/pages/simulation_results.py
- streamlit_app/data_handlers.py -> streamlit_app/utils/data_handlers.py
- streamlit_app/models/xgboost_playoff_model_final.pkl -> models/ensemble_model.pkl
- streamlit_app/visualization.py -> streamlit_app/utils/visualization.py
- streamlit_app/visualization_utils.py -> streamlit_app/utils/visualization_utils.py
