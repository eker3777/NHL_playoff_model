import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def display_about_page(model_data=None):
    """Display information about the application and models."""
    st.header("About NHL Playoff Predictor")
    
    st.markdown("""
    ### Overview
    This application predicts NHL playoff outcomes using machine learning models 
    trained on historical playoff data. It provides probabilities for each playoff 
    series and simulates entire playoff tournaments to estimate championship odds.
    
    ### Models
    The application uses three primary models:
    - **Logistic Regression**: A simple model that captures linear relationships
    - **XGBoost**: A gradient boosting model that captures non-linear relationships
    - **Ensemble Model**: A weighted average of the above models
    
    The models use team statistics from the regular season to predict playoff performance.
    """)
    
    # Model details if available
    if model_data:
        st.subheader("Model Details")
        
        # Show mode and parameters
        st.write(f"**Model mode:** {model_data.get('mode', 'default')}")
        st.write(f"**Home ice advantage:** {model_data.get('home_ice_boost', 0.039)*100:.1f}%")
        
        # Show features if available
        if 'features' in model_data:
            st.write("**Key features:**")
            for feature in model_data['features'][:10]:  # Show top 10 features
                st.write(f"- {feature}")
    
    # Data sources
    st.subheader("Data Sources")
    st.markdown("""
    - Team statistics are sourced from NHL API
    - Advanced metrics from Natural Stat Trick and Money Puck
    - Historical playoff results from Hockey Reference
    """)
    
    # Credits
    st.subheader("Credits")
    st.markdown("""
    Developed by an NHL enthusiast and data scientist.
    
    This application is for entertainment purposes only.
    """)
    
    # Contact info
    st.subheader("Contact")
    st.markdown("""
    If you have questions or suggestions, please reach out on GitHub.
    """)

def app():
    """Main entry point for the about page"""
    # Call the display function
    display_about_page()

def show_about():
    """Entry point for about page called from app.py"""
    app()

if __name__ == "__main__":
    # This allows the page to be run directly for testing
    display_about_page()
