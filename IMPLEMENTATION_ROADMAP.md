# Development Roadmap & Implementation Guide

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Add Logging System

Create a centralized logging configuration:

```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level='INFO', log_file=None):
    """Configure application logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or 'nhl_predictor.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger('nhl_predictor')
```

### 1.2 Basic Test Framework

```python
# tests/test_data_handlers.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from streamlit_app.data_handlers import process_standings_data, engineer_features

class TestDataHandlers:
    
    def test_process_standings_data_valid_input(self):
        """Test standings data processing with valid input."""
        sample_data = {
            'standings': [
                {'teamName': 'Boston Bruins', 'wins': 50, 'losses': 20},
                {'teamName': 'Toronto Maple Leafs', 'wins': 45, 'losses': 25}
            ]
        }
        
        result = process_standings_data(sample_data, '20242025')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'teamName' in result.columns
        assert 'wins' in result.columns
    
    def test_engineer_features_handles_missing_columns(self):
        """Test feature engineering with missing columns."""
        df = pd.DataFrame({
            'teamName': ['Team A', 'Team B'],
            'gamesPlayed': [70, 70]
        })
        
        result = engineer_features(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Should handle missing columns gracefully
    
    @patch('streamlit_app.data_handlers.requests.get')
    def test_api_retry_mechanism(self, mock_get):
        """Test API retry logic."""
        mock_get.side_effect = [
            Exception("Connection error"),
            Mock(status_code=200, json=lambda: {'data': 'success'})
        ]
        
        # Test that retry mechanism works
        # (Implementation would depend on actual retry logic)
```

### 1.3 Code Quality Tools Setup

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

```ini
# setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist

[isort]
profile = black
multi_line_output = 3

[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

## Phase 2: Architecture Improvements (Weeks 3-4)

### 2.1 Configuration Management

```python
# config/settings.py
from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class APIConfig:
    base_url: str = "https://api.nhle.com"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class ModelConfig:
    home_ice_advantage: float = 0.039
    simulation_count: int = 10000
    model_cache_ttl: int = 86400
    feature_importance_threshold: float = 0.01

@dataclass
class AppConfig:
    debug: bool = False
    log_level: str = "INFO"
    data_refresh_hours: int = 24
    max_cache_size_mb: int = 500

class Settings:
    def __init__(self):
        self.api = APIConfig()
        self.model = ModelConfig()
        self.app = AppConfig()
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        self.app.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.app.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.model.simulation_count = int(os.getenv('SIMULATION_COUNT', '10000'))

settings = Settings()
```

### 2.2 Data Management Classes

```python
# data/managers.py
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class DataManager(ABC):
    """Abstract base class for data management."""
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> Optional[pd.DataFrame]:
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        pass

class NHLAPIManager(DataManager):
    """Manages NHL API data fetching and caching."""
    
    def __init__(self, api_client):
        self.client = api_client
        self.cache = {}
        
    def fetch_team_stats(self, season: str) -> Optional[pd.DataFrame]:
        """Fetch team statistics for a given season."""
        cache_key = f"team_stats_{season}"
        
        if cache_key in self.cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"Fetching team stats for season {season}")
            data = self._fetch_from_api(season)
            
            if self.validate_data(data):
                self.cache[cache_key] = data
                return data
            else:
                logger.warning(f"Data validation failed for {season}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data."""
        if data is None or data.empty:
            return False
        
        required_columns = ['teamName', 'gamesPlayed', 'wins', 'losses']
        return all(col in data.columns for col in required_columns)
    
    def _fetch_from_api(self, season: str) -> pd.DataFrame:
        """Internal method to fetch data from API."""
        # Implementation would call actual NHL API
        pass

class FeatureEngineer:
    """Handles feature engineering and data transformation."""
    
    def __init__(self):
        self.feature_configs = self._load_feature_configs()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        logger.info("Starting feature engineering")
        
        data = data.copy()
        
        # Apply each feature engineering step
        for feature_name, config in self.feature_configs.items():
            try:
                data = self._apply_feature(data, feature_name, config)
            except Exception as e:
                logger.warning(f"Failed to create feature {feature_name}: {e}")
        
        return data
    
    def _load_feature_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load feature engineering configurations."""
        return {
            'win_percentage': {
                'columns': ['wins', 'gamesPlayed'],
                'formula': lambda df: df['wins'] / df['gamesPlayed'].clip(lower=1)
            },
            'goal_differential': {
                'columns': ['goalsFor', 'goalsAgainst'],
                'formula': lambda df: df['goalsFor'] - df['goalsAgainst']
            }
        }
    
    def _apply_feature(self, data: pd.DataFrame, feature_name: str, config: Dict) -> pd.DataFrame:
        """Apply a single feature engineering step."""
        required_cols = config['columns']
        
        if all(col in data.columns for col in required_cols):
            data[feature_name] = config['formula'](data)
            logger.debug(f"Created feature: {feature_name}")
        else:
            missing = [col for col in required_cols if col not in data.columns]
            logger.warning(f"Cannot create {feature_name}: missing columns {missing}")
        
        return data
```

### 2.3 Model Management System

```python
# models/manager.py
from typing import Dict, Any, Optional, Tuple
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages machine learning models for playoff prediction."""
    
    def __init__(self, model_directory: str):
        self.model_dir = Path(model_directory)
        self.models = {}
        self.metadata = {}
        
    def load_models(self) -> Dict[str, Any]:
        """Load all available models."""
        logger.info("Loading machine learning models")
        
        model_configs = {
            'logistic_regression': {
                'path': 'logistic_regression_model_final.pkl',
                'type': 'sklearn'
            },
            'xgboost': {
                'path': 'xgboost_playoff_model_final.pkl',
                'type': 'xgboost'
            },
            'ensemble': {
                'path': 'ensemble_model.pkl',
                'type': 'custom'
            }
        }
        
        loaded_models = {}
        
        for model_name, config in model_configs.items():
            model_path = self.model_dir / config['path']
            
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    loaded_models[model_name] = self._prepare_model(model_data, config)
                    logger.info(f"✓ Loaded {model_name} model")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        self.models = loaded_models
        return self._create_model_package()
    
    def _prepare_model(self, model_data: Any, config: Dict) -> Dict[str, Any]:
        """Prepare model data with metadata."""
        if isinstance(model_data, dict):
            return {
                'model': model_data.get('model'),
                'features': model_data.get('features', []),
                'type': config['type'],
                'metadata': model_data.get('metadata', {})
            }
        else:
            return {
                'model': model_data,
                'features': [],
                'type': config['type'],
                'metadata': {}
            }
    
    def _create_model_package(self) -> Dict[str, Any]:
        """Create standardized model package."""
        return {
            'models': self.models,
            'mode': self._determine_mode(),
            'home_ice_boost': settings.model.home_ice_advantage,
            'feature_importance': self._get_feature_importance()
        }
    
    def _determine_mode(self) -> str:
        """Determine the best mode based on available models."""
        if 'logistic_regression' in self.models and 'xgboost' in self.models:
            return 'ensemble'
        elif 'xgboost' in self.models:
            return 'xgboost'
        elif 'logistic_regression' in self.models:
            return 'logistic_regression'
        else:
            return 'default'
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from models."""
        importance = {}
        
        for model_name, model_data in self.models.items():
            model = model_data.get('model')
            if hasattr(model, 'feature_importances_'):
                features = model_data.get('features', [])
                if features:
                    importance[model_name] = dict(zip(features, model.feature_importances_))
        
        return importance

class PredictionEngine:
    """Handles playoff series predictions."""
    
    def __init__(self, model_package: Dict[str, Any]):
        self.models = model_package.get('models', {})
        self.mode = model_package.get('mode', 'default')
        self.home_ice_boost = model_package.get('home_ice_boost', 0.039)
        
    def predict_series_winner(self, matchup_data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Predict the probability of the higher seed winning."""
        if self.mode == 'ensemble':
            return self._ensemble_prediction(matchup_data)
        elif self.mode == 'xgboost':
            return self._single_model_prediction(matchup_data, 'xgboost')
        elif self.mode == 'logistic_regression':
            return self._single_model_prediction(matchup_data, 'logistic_regression')
        else:
            return self._default_prediction(matchup_data)
    
    def _ensemble_prediction(self, matchup_data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Combine predictions from multiple models."""
        predictions = {}
        weights = {'logistic_regression': 0.4, 'xgboost': 0.6}
        
        for model_name, weight in weights.items():
            if model_name in self.models:
                prob, _ = self._single_model_prediction(matchup_data, model_name)
                predictions[model_name] = {'probability': prob, 'weight': weight}
        
        if predictions:
            weighted_prob = sum(
                pred['probability'] * pred['weight']
                for pred in predictions.values()
            )
            return weighted_prob, {'individual_predictions': predictions}
        else:
            return self._default_prediction(matchup_data)
    
    def _single_model_prediction(self, matchup_data: pd.DataFrame, model_name: str) -> Tuple[float, Dict[str, Any]]:
        """Make prediction using a single model."""
        model_data = self.models.get(model_name)
        if not model_data:
            return self._default_prediction(matchup_data)
        
        model = model_data['model']
        features = model_data['features']
        
        try:
            # Prepare features
            X = self._prepare_features(matchup_data, features)
            
            # Make prediction
            prob = model.predict_proba(X)[0][1]  # Probability of class 1
            
            return prob, {'model_used': model_name, 'features_used': len(features)}
        
        except Exception as e:
            logger.error(f"Prediction failed with {model_name}: {e}")
            return self._default_prediction(matchup_data)
    
    def _prepare_features(self, matchup_data: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """Prepare features for model prediction."""
        # Implementation would prepare feature vector
        pass
    
    def _default_prediction(self, matchup_data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Default prediction based on simple metrics."""
        base_prob = 0.55  # Higher seed advantage
        return base_prob + self.home_ice_boost, {'model_used': 'default'}
```

## Phase 3: Testing & Validation (Weeks 5-6)

### 3.1 Integration Tests

```python
# tests/integration/test_prediction_pipeline.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch

class TestPredictionPipeline:
    
    @pytest.fixture
    def sample_team_data(self):
        return pd.DataFrame({
            'teamName': ['Boston Bruins', 'Toronto Maple Leafs'],
            'gamesPlayed': [82, 82],
            'wins': [50, 45],
            'losses': [25, 30],
            'goalsFor': [250, 240],
            'goalsAgainst': [200, 220]
        })
    
    @pytest.fixture
    def mock_model_package(self):
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        return {
            'models': {
                'xgboost': {
                    'model': mock_model,
                    'features': ['win_pct_diff', 'goal_diff_per_game'],
                    'type': 'xgboost'
                }
            },
            'mode': 'xgboost',
            'home_ice_boost': 0.039
        }
    
    def test_end_to_end_prediction(self, sample_team_data, mock_model_package):
        """Test complete prediction pipeline."""
        # Test data fetching -> feature engineering -> prediction
        from data.managers import FeatureEngineer
        from models.manager import PredictionEngine
        
        # Engineer features
        feature_engineer = FeatureEngineer()
        processed_data = feature_engineer.create_features(sample_team_data)
        
        # Create matchup data
        # (Implementation would create head-to-head comparison)
        
        # Make prediction
        prediction_engine = PredictionEngine(mock_model_package)
        prob, metadata = prediction_engine.predict_series_winner(processed_data)
        
        assert 0 <= prob <= 1
        assert 'model_used' in metadata
```

### 3.2 Performance Tests

```python
# tests/performance/test_performance.py
import time
import pytest
import pandas as pd
import numpy as np

class TestPerformance:
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance with large dataset."""
        # Create large test dataset
        large_data = pd.DataFrame({
            'teamName': [f'Team_{i}' for i in range(1000)],
            'gamesPlayed': np.random.randint(70, 82, 1000),
            'wins': np.random.randint(20, 60, 1000),
            'goalsFor': np.random.randint(180, 300, 1000),
            'goalsAgainst': np.random.randint(180, 300, 1000)
        })
        
        from data.managers import FeatureEngineer
        
        start_time = time.time()
        engineer = FeatureEngineer()
        result = engineer.create_features(large_data)
        end_time = time.time()
        
        # Should process 1000 teams in under 1 second
        assert end_time - start_time < 1.0
        assert len(result) == 1000
    
    def test_prediction_latency(self):
        """Test prediction response time."""
        # Test should ensure predictions are made quickly
        pass
```

## Phase 4: Documentation & Deployment (Weeks 7-8)

### 4.1 API Documentation

```python
# docs/api_reference.py
"""
NHL Playoff Predictor API Reference

This module provides comprehensive documentation for all public APIs
in the NHL Playoff Predictor application.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

def predict_series_outcome(
    team1: str, 
    team2: str, 
    season: str = "20242025",
    home_team: Optional[str] = None
) -> Dict[str, float]:
    """
    Predict the outcome of a playoff series between two teams.
    
    Args:
        team1: Name of the first team (e.g., "Boston Bruins")
        team2: Name of the second team (e.g., "Toronto Maple Leafs")
        season: Season string in format "YYYYYYY" (default: current season)
        home_team: Which team has home ice advantage (optional)
    
    Returns:
        Dictionary containing:
        - 'team1_win_probability': Probability team1 wins the series
        - 'team2_win_probability': Probability team2 wins the series
        - 'confidence': Model confidence in the prediction
        - 'factors': Key factors influencing the prediction
    
    Example:
        >>> result = predict_series_outcome("Boston Bruins", "Toronto Maple Leafs")
        >>> print(f"Boston wins: {result['team1_win_probability']:.1%}")
        Boston wins: 64.2%
    
    Raises:
        ValueError: If team names are invalid or data is unavailable
        APIError: If NHL API is unavailable
    """
    pass

def simulate_full_playoffs(
    season: str = "20242025",
    num_simulations: int = 10000
) -> Dict[str, Dict[str, float]]:
    """
    Simulate the complete NHL playoff tournament.
    
    Args:
        season: Season to simulate
        num_simulations: Number of Monte Carlo simulations to run
    
    Returns:
        Dictionary with simulation results:
        - 'championship_odds': Probability each team wins the Stanley Cup
        - 'round_advancement': Probability of reaching each round
        - 'expected_matchups': Most likely matchups in each round
    
    Example:
        >>> results = simulate_full_playoffs(num_simulations=1000)
        >>> for team, prob in results['championship_odds'].items():
        ...     print(f"{team}: {prob:.1%}")
    """
    pass
```

This roadmap provides a structured approach to improving the codebase while maintaining functionality. Each phase builds upon the previous one, ensuring steady progress toward a more maintainable and robust application.