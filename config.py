"""
Configuration settings for NHL Playoff Prediction System
"""

import os
from datetime import datetime


class Config:
    """Configuration class for NHL playoff prediction system"""
    
    # Season configuration
    @property
    def current_season(self):
        """Calculate current NHL season based on date"""
        now = datetime.now()
        return now.year if now.month >= 9 else now.year - 1
    
    @property
    def season_str(self):
        """Get season string in format YYYYYYY"""
        return f"{self.current_season}{self.current_season + 1}"
    
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    
    # Model files
    LOGISTIC_MODEL_FILE = "logistic_regression_model_final.pkl"
    XGBOOST_MODEL_FILE = "xgboost_playoff_model_final.pkl"
    ENSEMBLE_MODEL_FILE = "playoff_model.pkl"
    
    # API settings
    NHL_API_BASE_URL = "https://api.nhle.com"
    MONEYPUCK_API_BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    
    # Data refresh settings
    DATA_REFRESH_HOURS = 24
    
    # Simulation settings
    DEFAULT_SIMULATIONS = 10000
    
    # Home ice advantage
    HOME_ICE_BOOST = 0.039
    
    def __init__(self):
        """Initialize configuration and create necessary directories"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.REPORTS_DIR, exist_ok=True)


# Global configuration instance
config = Config()