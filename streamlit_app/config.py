"""
Central configuration module for NHL Playoff Predictor
All global constants should be defined here and imported elsewhere
"""

import os
import pytz
from datetime import time

# Application Information
APP_TITLE = "NHL Playoff Predictor"
APP_VERSION = "2.0.0"
AUTHOR = "Elliott Kervin"
GITHUB_URL = "https://github.com/yourusername/NHL_playoff_model"
THEME_COLOR = "#FF5733"  # Example color, change as needed

# Core Prediction Parameters
HOME_ICE_ADVANTAGE = 0.039  # 3.9% boost for home ice advantage
SERIES_LENGTH_DISTRIBUTION = [0.14, 0.243, 0.336, 0.281]  # 4, 5, 6, 7 games

# API Configuration
API_BASE_URL = "https://api-web.nhle.com/v1"
API_TIMEOUT = 30
API_RETRIES = 3

# Refresh Settings
REFRESH_HOUR = 5  # 5 AM in TIMEZONE
TIMEZONE = "US/Eastern"
REFRESH_TIMEZONE = pytz.timezone(TIMEZONE)
REFRESH_TIME = time(hour=REFRESH_HOUR, minute=0, second=0)
REFRESH_INTERVAL = 86400  # 24 hour in seconds
CACHE_DURATION = 86400  # 24 hours in seconds

# Feature Lists
CRITICAL_FEATURES = [
    "PP%_rel",
    "PK%_rel",
    "FO%",
    "playoff_performance_score",
    "xGoalsPercentage",
    "homeRegulationWin%",
    "roadRegulationWin%",
    "possAdjHitsPctg",
    "possAdjTakeawaysPctg",
    "possTypeAdjGiveawaysPctg",
    "reboundxGoalsPctg",
    "goalDiff/G",
    "adjGoalsSavedAboveX/60",
    "adjGoalsScoredAboveX/60",
]

MODEL_FEATURES = [f"{feature}_diff" for feature in CRITICAL_FEATURES]


PERCENTAGE_COLUMNS = [
    "PP%",
    "PK%",
    "FO%",
    "xGoalsPercentage",
    "corsiPercentage",
    "fenwickPercentage",
    "shootingPercentage",
    "savePctg",
    "homeRegulationWin%",
    "roadRegulationWin%",
]

# NHL Structure Constants
CONFERENCE_NAMES = ["Eastern", "Western"]
DIVISION_NAMES = ["Atlantic", "Metropolitan", "Central", "Pacific"]
PLAYOFF_SPOTS_PER_DIVISION = 3
WILDCARDS_PER_CONFERENCE = 2

# Simulation Constants
DEFAULT_SIMULATION_COUNT = 1000
MINIMUM_SIMULATION_COUNT = 100
MAXIMUM_SIMULATION_COUNT = 10000

# Team Colors (for visualizations)
TEAM_COLORS = {
    "ANA": "#F47A38",
    "ARI": "#8C2633",
    "BOS": "#FFB81C",
    "BUF": "#002654",
    "CGY": "#C8102E",
    "CAR": "#CC0000",
    "CHI": "#CF0A2C",
    "COL": "#6F263D",
    "CBJ": "#002654",
    "DAL": "#006847",
    "DET": "#CE1126",
    "EDM": "#041E42",
    "FLA": "#041E42",
    "LAK": "#111111",
    "MIN": "#154734",
    "MTL": "#AF1E2D",
    "NSH": "#FFB81C",
    "NJD": "#CE1126",
    "NYI": "#00539B",
    "NYR": "#0038A8",
    "OTT": "#C52032",
    "PHI": "#F74902",
    "PIT": "#FCB514",
    "SJS": "#006D75",
    "SEA": "#99D9D9",
    "STL": "#002F87",
    "TBL": "#002868",
    "TOR": "#00205B",
    "VAN": "#00205B",
    "VGK": "#B4975A",
    "WSH": "#C8102E",
    "WPG": "#041E42",
    "UTA": "#002F87",
}

# Debug Settings
DEBUG_MODE = True
LOG_LEVEL = "INFO"
MAX_LOG_FILES = 5
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
ENABLE_DEBUG_UI = True

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SIMULATION_RESULTS_DIR = os.path.join(DATA_DIR, "simulation_results")
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Directory for trained ML models
LOG_DIR = os.path.join(BASE_DIR, "logs")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
COMPONENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components")
APP_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)  # For model code

# Make sure essential directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SIMULATION_RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(COMPONENTS_DIR, exist_ok=True)
os.makedirs(APP_MODELS_DIR, exist_ok=True)
os.makedirs(TESTS_DIR, exist_ok=True)

# Model Configuration
MODEL_MODES = ["ensemble", "lr", "xgb", "default"]
DEFAULT_MODEL_MODE = "ensemble"
FEATURE_VALIDATION_THRESHOLD = 0.0

# Playoff Advancement Logic
PLAYOFF_SERIES_WIN_THRESHOLD = 4  # First to 4 wins
ROUND_NAMES = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Finals"]
PLAYOFF_FORMAT = "division"  # 'division' or 'conference' based format

# Performance Metrics
SLOW_OPERATION_THRESHOLD = 1.0  # seconds
MEMORY_WARNING_THRESHOLD = 1000  # MB
