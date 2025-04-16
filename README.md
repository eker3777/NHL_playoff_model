# NHL Playoff Predictor

A Streamlit application that predicts NHL playoff outcomes based on team statistics and advanced metrics.

## Installation

1. Clone this repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run playoff_model_streamlit.py
```

## Features

- Current NHL playoff bracket visualization
- Playoff series predictions
- Team comparison tool
- Full playoff simulation
- Advanced metrics analysis

## Data Sources

- NHL API (via nhl-api-py package)
- Advanced metrics (when available)
- Historical playoff data

## Directory Structure

- `data/`: Cache directory for NHL data
- `models/`: Directory for machine learning models