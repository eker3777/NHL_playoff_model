#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Create model directory if it doesn't exist
mkdir -p models

# Create placeholder models if needed
if [ ! -f models/logistic_regression_model_final.pkl ] || [ ! -f models/xgboost_playoff_model_final.pkl ]; then
    python models/create_placeholder_models.py
fi

# Run the Streamlit app
streamlit run playoff_model_streamlit.py
