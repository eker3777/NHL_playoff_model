# NHL Playoff Prediction System

A comprehensive machine learning system for predicting NHL playoff outcomes using advanced statistical analysis and ensemble modeling. This system generates detailed markdown reports with visualizations for professional analysis and presentation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides a complete data science pipeline for NHL playoff predictions, combining:

- **Data Integration**: Automated fetching from NHL API and MoneyPuck advanced statistics
- **Machine Learning**: Ensemble models using Logistic Regression and XGBoost
- **Feature Engineering**: Advanced hockey metrics and historical performance analysis  
- **Visualization**: Professional charts and statistical analysis
- **Automated Reporting**: Comprehensive markdown reports for presentation

## Key Features

### 🤖 Advanced Machine Learning
- **Ensemble Modeling**: Combines Logistic Regression and XGBoost algorithms
- **Feature Engineering**: 20+ advanced hockey metrics including Expected Goals, Corsi, and playoff history
- **Cross-Validation**: Historical season validation for model reliability
- **Confidence Scoring**: Prediction confidence and model agreement analysis

### 📊 Professional Visualizations
- **Interactive Charts**: Team comparisons, standings analysis, and performance metrics
- **Playoff Brackets**: Predicted tournament brackets with win probabilities
- **Statistical Analysis**: Feature importance and model performance visualizations
- **Publication-Ready**: High-resolution charts suitable for professional presentation

### 📈 Comprehensive Reporting
- **Automated Reports**: Generated markdown documents with complete analysis
- **Executive Summary**: Key findings and prediction highlights
- **Methodology**: Detailed explanation of models and data sources
- **Seasonal Analysis**: Historical context and team performance trends

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/eker3777/NHL_playoff_model.git
cd NHL_playoff_model

# Install dependencies
pip install -r requirements.txt

# Run the prediction pipeline
python main.py
```

### Optional: Streamlit Application

```bash
# Run the interactive web application
streamlit run streamlit_app/main.py
```

## Usage

### Basic Prediction Pipeline

Generate predictions for the current NHL season:

```bash
python main.py
```

Generate predictions for a specific season:

```bash
python main.py 2023  # For 2023-2024 season
```

### Output

The system generates:

1. **Markdown Report**: `reports/NHL_Playoff_Report_YYYYZZZZ.md`
2. **Visualizations**: PNG charts in the `reports/` directory
3. **Data Cache**: Processed team data in `data/` directory
4. **Logs**: Execution logs in `nhl_prediction.log`

### Sample Output Structure

```
reports/
├── NHL_Playoff_Report_20242025.md
├── standings_chart.png
├── season_performance.png
├── playoff_bracket.png
├── prediction_confidence.png
└── feature_importance.png
```

## Architecture

### Core Components

```
core/
├── data_loader.py       # NHL API and data processing
├── model_predictor.py   # Machine learning models
├── visualizations.py    # Chart generation
└── report_generator.py  # Markdown report creation
```

### Data Flow

1. **Data Collection**: Fetch current standings, team statistics, and advanced metrics
2. **Feature Engineering**: Calculate differential metrics and relative performance indicators
3. **Model Prediction**: Generate ensemble predictions with confidence scoring
4. **Visualization**: Create professional charts and statistical visualizations
5. **Report Generation**: Compile comprehensive markdown analysis

## Models and Methodology

### Machine Learning Approach

- **Logistic Regression**: Linear baseline model with engineered features
- **XGBoost**: Gradient boosting capturing non-linear relationships
- **Ensemble Method**: Weighted combination (40% LR, 60% XGB) for optimal performance

### Key Features

- **Expected Goals Percentage (xG%)**: Shot quality and finishing ability
- **Special Teams**: Power play and penalty kill effectiveness relative to league average
- **Possession Metrics**: Corsi, Fenwick, and shot attempt differentials
- **Playoff History**: Weighted historical playoff performance (2+ seasons)
- **Situational Performance**: Home/road regulation win percentages
- **Advanced Defense**: Goals saved above expected and venue-adjusted metrics

### Validation Approach

- **Historical Cross-Validation**: Models tested on previous playoff seasons
- **Feature Importance Analysis**: Statistical significance testing
- **Ensemble Optimization**: Weight tuning based on historical accuracy
- **Confidence Calibration**: Prediction reliability scoring

## Data Sources

### Primary Sources
- **NHL Official API**: Real-time standings, game results, and team statistics
- **MoneyPuck.com**: Advanced analytics including Expected Goals and possession metrics
- **Historical Archives**: Multi-season playoff results for model training

### Data Quality
- **Automated Validation**: Data integrity checks and outlier detection
- **Caching Strategy**: Local storage for performance and API rate limiting
- **Fallback Methods**: Multiple data source endpoints for reliability

## Project Structure

```
NHL_playoff_model/
├── main.py                 # Main pipeline script
├── config.py              # Configuration and settings
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── core/                  # Core functionality modules
│   ├── data_loader.py
│   ├── model_predictor.py
│   ├── visualizations.py
│   └── report_generator.py
├── streamlit_app/         # Interactive web application
├── data/                  # Cached NHL data
├── models/                # Trained ML models
├── reports/               # Generated reports and charts
└── logs/                  # Execution logs
```

## Configuration

### Season Settings

The system automatically detects the current NHL season but can be configured:

```python
# config.py
CURRENT_SEASON = 2024  # Override automatic detection
DEFAULT_SIMULATIONS = 10000  # Simulation count
HOME_ICE_BOOST = 0.039  # Home advantage factor
```

### Customization

- **Feature Selection**: Modify `core/model_predictor.py` to adjust model features
- **Visualization Themes**: Update team colors and chart styling in `core/visualizations.py`
- **Report Templates**: Customize markdown generation in `core/report_generator.py`

## Performance

### System Requirements
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 500MB for data cache and models
- **Network**: Internet connection for API data fetching

### Execution Time
- **Full Pipeline**: 2-5 minutes depending on data freshness
- **Incremental Updates**: 30-60 seconds for cached data
- **Report Generation**: 10-20 seconds

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black core/ main.py
flake8 core/ main.py
```

### Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Validate with historical data

## Future Enhancements

### Planned Features
- **Real-time Updates**: Live game result integration
- **Injury Impact**: Player availability modeling
- **Betting Odds Integration**: Market prediction comparison
- **Interactive Dashboard**: Enhanced web interface
- **API Endpoint**: RESTful service for predictions

### Research Opportunities
- **Deep Learning**: Neural network architectures for hockey analytics
- **Bayesian Methods**: Uncertainty quantification in predictions
- **Simulation Engine**: Monte Carlo playoff simulations
- **Market Analysis**: Prediction accuracy vs. betting markets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This system is designed for analytical and educational purposes. Predictions should not be used for gambling or wagering activities. The authors are not responsible for any financial decisions based on these predictions.

## Contact

For questions, suggestions, or collaboration opportunities:

- **GitHub Issues**: [Report bugs or request features](https://github.com/eker3777/NHL_playoff_model/issues)
- **Discussions**: [Join the community discussion](https://github.com/eker3777/NHL_playoff_model/discussions)

---

*Built with ❤️ for hockey analytics and data science*