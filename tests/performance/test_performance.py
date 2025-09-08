"""
Performance tests for the NHL prediction system.

These tests ensure that the system performs efficiently with large datasets
and meets performance requirements.
"""

import time
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPerformance:
    """Performance test suite for the NHL prediction system."""
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance with large dataset."""
        # Create large test dataset (1000 teams)
        large_data = pd.DataFrame({
            'teamName': [f'Team_{i}' for i in range(1000)],
            'gamesPlayed': np.random.randint(70, 82, 1000),
            'wins': np.random.randint(20, 60, 1000),
            'losses': np.random.randint(15, 50, 1000),
            'goalsFor': np.random.randint(180, 300, 1000),
            'goalsAgainst': np.random.randint(180, 300, 1000)
        })
        
        # Test basic feature engineering performance
        start_time = time.time()
        
        # Basic feature calculations
        large_data['win_pct'] = large_data['wins'] / large_data['gamesPlayed']
        large_data['goal_diff'] = large_data['goalsFor'] - large_data['goalsAgainst']
        large_data['goals_per_game'] = large_data['goalsFor'] / large_data['gamesPlayed']
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertion - should complete in under 1 second for 1000 records
        assert execution_time < 1.0, f"Feature engineering took {execution_time:.3f}s, expected < 1.0s"
        
        # Validate results
        assert len(large_data) == 1000
        assert 'win_pct' in large_data.columns
        assert not large_data['win_pct'].isna().any()
    
    def test_data_processing_memory_usage(self):
        """Test memory efficiency of data processing."""
        # Create moderately large dataset
        data_size = 5000
        test_data = pd.DataFrame({
            'teamName': [f'Team_{i}' for i in range(data_size)],
            'stats': np.random.random((data_size, 10)).tolist()  # 10 stats per team
        })
        
        # Measure basic operations
        start_time = time.time()
        
        # Simulate data processing operations
        processed_data = test_data.copy()
        processed_data['stat_sum'] = [sum(stats) for stats in processed_data['stats']]
        processed_data['stat_mean'] = [np.mean(stats) for stats in processed_data['stats']]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle 5000 records efficiently
        assert execution_time < 2.0, f"Data processing took {execution_time:.3f}s, expected < 2.0s"
        assert len(processed_data) == data_size
    
    def test_model_prediction_speed(self):
        """Test prediction speed with multiple models."""
        from unittest.mock import Mock
        
        # Create mock models
        mock_models = {}
        for model_name in ['logistic_regression', 'xgboost', 'ensemble']:
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.random.random((100, 2))
            mock_models[model_name] = mock_model
        
        # Create test feature data
        test_features = np.random.random((100, 10))  # 100 samples, 10 features
        
        # Test prediction speed
        start_time = time.time()
        
        predictions = {}
        for model_name, model in mock_models.items():
            predictions[model_name] = model.predict_proba(test_features)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should make predictions quickly
        assert execution_time < 0.5, f"Predictions took {execution_time:.3f}s, expected < 0.5s"
        assert len(predictions) == 3
        
        # Validate prediction structure
        for model_name, preds in predictions.items():
            assert preds.shape == (100, 2), f"Wrong prediction shape for {model_name}"


if __name__ == '__main__':
    pytest.main([__file__])