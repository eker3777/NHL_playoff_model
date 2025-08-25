# Code Quality Analysis - Specific Examples

## 1. Print Statement Issues

### Current Code Examples:
```python
# From check_models.py
print("Checking for model files...")
print("Creating placeholder models...")
print(f"- Created LR model at {lr_path}")

# From streamlit_app/data_handlers.py
print(f"Converting percentage column: {col}")
print(f"Filling {null_count} NaN values in {col} with column mean")
print(f"WARNING: {col} still has {null_count} NaN values after processing")

# From streamlit_app/model_utils.py
print(f"Available model files: {', '.join(available_files) if available_files else 'None'}")
print(f"Missing features for LR model: {missing_features}")
```

### Recommended Logging Implementation:
```python
import logging
logger = logging.getLogger(__name__)

# Replace print statements with:
logger.info("Checking for model files")
logger.info("Creating placeholder models")
logger.info("Created LR model at %s", lr_path)

logger.debug("Converting percentage column: %s", col)
logger.warning("Filling %d NaN values in %s with column mean", null_count, col)
logger.error("%s still has %d NaN values after processing", col, null_count)

logger.debug("Available model files: %s", ', '.join(available_files) if available_files else 'None')
logger.warning("Missing features for LR model: %s", missing_features)
```

## 2. Exception Handling Issues

### Current Problematic Patterns:
```python
# From check_models.py - Too broad
try:
    lr_model = LogisticRegression(random_state=42)
    X_lr = np.random.random((100, len(lr_feature_columns)))
    y_lr = np.random.randint(0, 2, 100)
    lr_model.fit(X_lr, y_lr)
    # ... more code
except Exception as e:
    print(f"Error creating LR model: {str(e)}")

# From data_handlers.py - Masks specific errors
try:
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        return data
except Exception as e:
    st.warning(f"Error loading data: {str(e)}")
    return None
```

### Recommended Specific Exception Handling:
```python
# Specific exception handling
try:
    lr_model = LogisticRegression(random_state=42)
    X_lr = np.random.random((100, len(lr_feature_columns)))
    y_lr = np.random.randint(0, 2, 100)
    lr_model.fit(X_lr, y_lr)
except (ValueError, MemoryError) as e:
    logger.error("Failed to create/train LR model: %s", e)
    raise ModelCreationError(f"LR model creation failed: {e}")
except ImportError as e:
    logger.error("Missing dependencies for LR model: %s", e)
    raise DependencyError(f"Required packages not available: {e}")

# Better file handling
try:
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        return data
    else:
        logger.warning("File not found: %s", filepath)
        return None
except pd.errors.EmptyDataError:
    logger.warning("Empty data file: %s", filepath)
    return None
except pd.errors.ParserError as e:
    logger.error("Failed to parse CSV file %s: %s", filepath, e)
    return None
except PermissionError:
    logger.error("Permission denied accessing file: %s", filepath)
    return None
except OSError as e:
    logger.error("OS error reading file %s: %s", filepath, e)
    return None
```

## 3. Function Complexity Issues

### Current Large Function Example:
```python
# From data_handlers.py - engineer_features() is ~200 lines
def engineer_features(combined_data):
    """Engineer features for the prediction model"""
    # Create a copy of the data to avoid modifying original
    df = combined_data.copy()
    
    # Convert percentage strings to floats where needed
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # Check if this column contains percentage strings
                if df[col].astype(str).str.contains('%').any():
                    print(f"Converting percentage column: {col}")
                    df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
    
    # ... 180+ more lines of feature engineering
    return df
```

### Recommended Decomposition:
```python
class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        df = data.copy()
        
        df = self._convert_percentage_columns(df)
        df = self._calculate_basic_metrics(df)
        df = self._calculate_advanced_metrics(df)
        df = self._handle_missing_values(df)
        df = self._validate_features(df)
        
        return df
    
    def _convert_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert percentage string columns to float."""
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    if df[col].astype(str).str.contains('%').any():
                        self.logger.debug("Converting percentage column: %s", col)
                        df[col] = df[col].astype(str).str.rstrip('%').astype(float) / 100
                except (ValueError, TypeError) as e:
                    self.logger.warning("Failed to convert percentage column %s: %s", col, e)
        return df
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic derived metrics."""
        if all(col in df.columns for col in ['wins', 'losses', 'overtimeLosses']):
            df['totalGames'] = df['wins'] + df['losses'] + df['overtimeLosses']
            df['winPercentage'] = df['wins'] / df['totalGames'].clip(lower=1)
        return df
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced analytics metrics."""
        # Implementation for advanced metrics
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in critical features."""
        critical_features = ['PP%_rel', 'PK%_rel', 'FO%']
        
        for col in critical_features:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                self.logger.warning("Filling %d NaN values in %s with column mean", null_count, col)
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate engineered features."""
        critical_features = ['PP%_rel', 'PK%_rel', 'FO%']
        
        for col in critical_features:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    self.logger.error("%s still has %d NaN values after processing", col, null_count)
        
        return df
```

## 4. Import Organization Issues

### Current Scattered Imports:
```python
# From various files - imports scattered throughout
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO
from datetime import datetime, timedelta
import time
import json
try:
    # Try importing the module (package is installed as nhl-api-py, but imported as nhlpy)
    import nhlpy
    from nhlpy.nhl_client import NHLClient
except ImportError:
    st.write("Installing NHL API package...")
    # ... installation code
```

### Recommended Import Organization:
```python
# Standard library imports
import json
import logging
import os
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Local imports
from config.settings import settings
from utils.exceptions import APIError, DataValidationError
from utils.logging_config import setup_logging

# Optional imports with graceful handling
try:
    import nhlpy
    from nhlpy.nhl_client import NHLClient
    NHL_API_AVAILABLE = True
except ImportError as e:
    logging.warning("NHL API not available: %s", e)
    NHLClient = None
    NHL_API_AVAILABLE = False
```

## 5. Configuration Issues

### Current Hard-coded Values:
```python
# Scattered throughout codebase
home_ice_boost = 0.039
n_simulations = 10000
cache_ttl = 86400  # 24 hours
timeout = 30
retry_attempts = 3
```

### Recommended Configuration Management:
```python
# config/settings.py
from dataclasses import dataclass
from typing import Dict
import os

@dataclass
class ModelSettings:
    home_ice_advantage: float = 0.039
    simulation_count: int = 10000
    feature_importance_threshold: float = 0.01
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {'lr': 0.4, 'xgb': 0.6}

@dataclass
class APISettings:
    base_url: str = "https://api.nhle.com"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 60

@dataclass
class CacheSettings:
    data_ttl: int = 86400  # 24 hours
    model_ttl: int = 604800  # 1 week
    max_cache_size_mb: int = 500

class Settings:
    def __init__(self):
        self.model = ModelSettings()
        self.api = APISettings()
        self.cache = CacheSettings()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Override defaults with environment variables."""
        self.model.simulation_count = int(
            os.getenv('SIMULATION_COUNT', self.model.simulation_count)
        )
        self.api.timeout = int(
            os.getenv('API_TIMEOUT', self.api.timeout)
        )

# Global settings instance
settings = Settings()
```

## 6. Data Validation Issues

### Current Minimal Validation:
```python
# Weak validation
if data is None or data.empty:
    return False
```

### Recommended Comprehensive Validation:
```python
class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_team_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Comprehensive team data validation."""
        errors = []
        
        # Check required columns
        required_columns = ['teamName', 'gamesPlayed', 'wins', 'losses']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'gamesPlayed' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['gamesPlayed']):
                errors.append("'gamesPlayed' must be numeric")
        
        # Check value ranges
        if 'wins' in data.columns and 'gamesPlayed' in data.columns:
            invalid_wins = data['wins'] > data['gamesPlayed']
            if invalid_wins.any():
                errors.append("Wins cannot exceed games played")
        
        # Check for suspicious values
        if 'gamesPlayed' in data.columns:
            if (data['gamesPlayed'] < 10).any():
                self.logger.warning("Some teams have very few games played")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.error("Data validation failed: %s", "; ".join(errors))
        
        return is_valid, errors
```

These examples demonstrate specific improvements that would significantly enhance code quality, maintainability, and reliability of the NHL Playoff Model application.