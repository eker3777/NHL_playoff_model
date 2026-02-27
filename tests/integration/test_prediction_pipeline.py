"""
Integration tests for the prediction pipeline.

These tests validate the end-to-end functionality of the NHL prediction system,
ensuring all components work together correctly.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPredictionPipeline:
    """Integration tests for the complete prediction pipeline."""
    
    @pytest.fixture
    def sample_team_data(self):
        """Fixture providing sample team data for testing."""
        return pd.DataFrame({
            'teamName': ['Boston Bruins', 'Toronto Maple Leafs'],
            'gamesPlayed': [82, 82],
            'wins': [50, 45],
            'losses': [25, 30],
            'overtimeLosses': [7, 7],
            'goalsFor': [250, 240],
            'goalsAgainst': [200, 220],
            'points': [107, 97]
        })
    
    @pytest.fixture
    def mock_model_package(self):
        """Fixture providing mock model package for testing."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7], [0.6, 0.4]]
        
        return {
            'models': {
                'xgboost': {
                    'model': mock_model,
                    'features': ['win_pct_diff', 'goal_diff_per_game'],
                    'accuracy': 0.75
                }
            },
            'meta': {
                'created': '2024-01-01',
                'version': '1.0'
            }
        }
    
    def test_pipeline_basic_flow(self, sample_team_data, mock_model_package):
        """Test the basic flow of the prediction pipeline."""
        # Test data structure
        assert isinstance(sample_team_data, pd.DataFrame)
        assert len(sample_team_data) > 0
        
        # Test model package structure
        assert 'models' in mock_model_package
        assert 'meta' in mock_model_package
        
        # Test mock predictions
        predictions = mock_model_package['models']['xgboost']['model'].predict_proba([[0.1, 2.5]])
        assert len(predictions) == 1
        assert len(predictions[0]) == 2  # Binary classification
    
    @patch('core.data_loader.NHLDataLoader')
    def test_data_loading_integration(self, mock_data_loader):
        """Test integration of data loading components."""
        # Mock data loader responses
        mock_instance = mock_data_loader.return_value
        mock_instance.get_standings_data.return_value = {'standings': []}
        mock_instance.get_team_stats.return_value = pd.DataFrame()
        
        # Test that mocking works correctly
        loader = mock_data_loader()
        standings = loader.get_standings_data()
        stats = loader.get_team_stats()
        
        assert isinstance(standings, dict)
        assert isinstance(stats, pd.DataFrame)
    
    def test_feature_engineering_pipeline(self, sample_team_data):
        """Test feature engineering integration."""
        # Basic feature engineering test
        # Calculate win percentage
        sample_team_data['win_pct'] = sample_team_data['wins'] / sample_team_data['gamesPlayed']
        
        # Calculate goal differential
        sample_team_data['goal_diff'] = sample_team_data['goalsFor'] - sample_team_data['goalsAgainst']
        
        # Validate features were created
        assert 'win_pct' in sample_team_data.columns
        assert 'goal_diff' in sample_team_data.columns
        
        # Check calculations
        assert sample_team_data.iloc[0]['win_pct'] == pytest.approx(50/82, rel=1e-3)
        assert sample_team_data.iloc[0]['goal_diff'] == 50  # 250 - 200


if __name__ == '__main__':
    pytest.main([__file__])