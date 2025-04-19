import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Add the parent directory to the path to import from streamlit_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config to get MODEL_DIR
from streamlit_app.config import MODEL_DIR, CRITICAL_FEATURES

# Define features based on critical features from config
features = [f"{feature}_diff" for feature in CRITICAL_FEATURES]

def create_test_lr_model():
    """Create a simple test logistic regression model"""
    print("Creating test logistic regression model...")
    
    # Create a simple model with random coefficients
    lr_model = LogisticRegression()
    
    # Simulate fitting by directly setting coefficients
    n_features = len(features)
    lr_model.classes_ = np.array([0, 1])
    lr_model.coef_ = np.random.randn(1, n_features)
    lr_model.intercept_ = np.array([0.0])
    
    # Add feature names (needed by some functions)
    lr_model.feature_names_ = features
    
    # Set up prediction methods (placeholder implementations)
    lr_model._predict_proba_lr = lambda X: np.hstack([1-np.random.random((X.shape[0], 1)), 
                                                      np.random.random((X.shape[0], 1))])
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, 'lr_model.pkl')
    joblib.dump(lr_model, model_path)
    print(f"Saved test LR model to {model_path}")
    
    # Create a feature information file
    features_path = os.path.join(MODEL_DIR, 'lr_model_features.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    print(f"Saved feature list to {features_path}")
    
    return model_path

def create_test_xgb_model():
    """Create a simple test XGBoost model"""
    print("Creating test XGBoost model...")
    
    # Create a simple model
    xgb_model = xgb.XGBClassifier()
    
    # Set up minimum required attributes
    xgb_model.classes_ = np.array([0, 1])
    xgb_model.n_classes_ = 2
    xgb_model.n_features_in_ = len(features)
    xgb_model.feature_names_ = features
    
    # Create feature importance (needed by the app)
    xgb_model.feature_importances_ = np.random.random(len(features))
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    joblib.dump(xgb_model, model_path)
    print(f"Saved test XGB model to {model_path}")
    
    # Create a feature information file
    features_path = os.path.join(MODEL_DIR, 'xgb_model_features.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    print(f"Saved feature list to {features_path}")
    
    return model_path

def create_test_models():
    """Create all test models needed for the app"""
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if models already exist
    lr_path = os.path.join(MODEL_DIR, 'lr_model.pkl')
    xgb_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    
    if os.path.exists(lr_path) and os.path.exists(xgb_path):
        print(f"Test models already exist in {MODEL_DIR}")
        return
    
    # Create test models
    create_test_lr_model()
    create_test_xgb_model()
    
    print("Test models created successfully!")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"Available files: {os.listdir(MODEL_DIR)}")

if __name__ == "__main__":
    create_test_models()
