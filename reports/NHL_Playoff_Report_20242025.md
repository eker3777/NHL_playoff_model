# NHL Playoff Prediction Report
## Season 20242025

**Generated:** 2025-09-08 19:17:35
**Analysis Type:** Machine Learning Playoff Predictions
**Data Sources:** NHL API, MoneyPuck Advanced Statistics

---

## Executive Summary

This report provides comprehensive NHL playoff predictions using advanced machine learning models and statistical analysis. Our ensemble approach combines multiple algorithms to predict playoff matchup outcomes with high accuracy.

### Key Findings

- **Total Predictions Generated:** 8
- **Average Prediction Confidence:** 5.9%
- **High Confidence Predictions:** 0
- **Models Utilized:** LR, XGB, COMBINED
- **Prediction Mode:** Ensemble

## Season Overview

### Current Standings Summary

#### Eastern Conference Playoff Picture

| Rank | Team | Points | Wins | Losses | OT/SO |
|------|------|--------|------|--------|-------|
| 1 | WSH | 111 | 51 | 21 | 9 |
| 2 | TOR | 106 | 51 | 26 | 4 |
| 3 | TBL | 102 | 47 | 26 | 8 |
| 4 | CAR | 99 | 47 | 29 | 5 |
| 5 | FLA | 98 | 47 | 31 | 4 |
| 6 | OTT | 95 | 44 | 30 | 7 |
| 7 | NJD | 91 | 42 | 33 | 7 |
| 8 | MTL | 91 | 40 | 31 | 11 |

#### Western Conference Playoff Picture

| Rank | Team | Points | Wins | Losses | OT/SO |
|------|------|--------|------|--------|-------|
| 1 | WPG | 116 | 56 | 22 | 4 |
| 2 | VGK | 110 | 50 | 22 | 10 |
| 3 | DAL | 106 | 50 | 26 | 6 |
| 4 | LAK | 105 | 48 | 24 | 9 |
| 5 | COL | 102 | 49 | 29 | 4 |
| 6 | EDM | 101 | 48 | 29 | 5 |
| 7 | MIN | 97 | 45 | 30 | 7 |
| 8 | STL | 96 | 44 | 30 | 8 |

### Season Statistics Overview

- **League Leader (Points):** WPG (116 points)
- **Top Offense:** CBJ (193.0 goals for)
- **Best Defense:** WPG (118.0 goals against)

## Playoff Predictions

### First Round Matchup Predictions

| Matchup | Higher Seed Win Probability | Confidence | Model Consensus |
|---------|----------------------------|------------|------------------|
| WPG vs COL | 57.0% | 14.0% | High |
| WSH vs TBL | 54.5% | 9.0% | High |
| VGK vs LAK | 52.5% | 5.0% | High |
| TOR vs DAL | 50.0% | 0.0% | High |
| EDM vs NJD | 55.0% | 10.0% | High |
| CAR vs CGY | 52.5% | 5.0% | High |
| FLA vs OTT | 51.5% | 3.0% | High |
| MIN vs STL | 50.5% | 1.0% | High |

### Upset Alerts

Teams with upset potential (lower seed win probability > 40%):

- **COL** has a 43.0% chance against **WPG**
- **TBL** has a 45.5% chance against **WSH**
- **LAK** has a 47.5% chance against **VGK**
- **DAL** has a 50.0% chance against **TOR**
- **NJD** has a 45.0% chance against **EDM**
- **CGY** has a 47.5% chance against **CAR**
- **OTT** has a 48.5% chance against **FLA**
- **STL** has a 49.5% chance against **MIN**

## Model Analysis

### Model Performance Overview

**Prediction Mode:** Ensemble

#### Available Models

- **LR:** 3 features
- **XGB:** 6 features
- **COMBINED:** 9 features

**Home Ice Advantage:** +3.9%

### Key Predictive Features

Our models utilize advanced hockey analytics including:

- **Expected Goals Percentage (xG%):** Advanced metric measuring shot quality
- **Special Teams Performance:** Power play and penalty kill effectiveness
- **Possession Metrics:** Corsi, Fenwick, and shot attempt percentages
- **Playoff History:** Historical playoff performance weighting
- **Regulation Win Percentage:** Home and road regulation win rates
- **Advanced Defensive Metrics:** Goals saved above expected

## Data Visualizations

The following charts provide visual analysis of the current season and predictions:

### Current Standings

![Standings Chart](standings_chart.png)

*Current NHL standings by conference showing point totals and playoff positioning.*

### Season Performance Analysis

![Season Performance](season_performance.png)

*Comprehensive analysis of team performance across multiple statistical categories.*

### Predicted Playoff Bracket

![Playoff Bracket](playoff_bracket.png)

*Predicted playoff bracket with win probabilities for each matchup.*

### Prediction Confidence Analysis

![Prediction Confidence](prediction_confidence.png)

*Analysis of prediction confidence levels and model agreement.*

## Methodology

### Data Sources

1. **NHL Official API:** Team standings, basic statistics, and game results
2. **MoneyPuck:** Advanced analytics including expected goals, Corsi, and possession metrics
3. **Historical Playoff Data:** Multi-year playoff performance weighting

### Model Approach

Our ensemble approach combines multiple machine learning algorithms:

1. **Logistic Regression:** Baseline linear model with feature engineering
2. **XGBoost:** Gradient boosting model capturing non-linear relationships
3. **Ensemble Weighting:** Combines predictions with optimized weights (40% LR, 60% XGB)

### Feature Engineering

Key engineered features include:

- **Differential Metrics:** Head-to-head statistical comparisons
- **Relative Percentages:** Performance relative to league average
- **Playoff Performance Score:** Weighted historical playoff success
- **Advanced Possession Metrics:** Venue-adjusted advanced statistics

### Validation

Models are validated using:

- **Historical Cross-Validation:** Testing on previous seasons
- **Feature Importance Analysis:** Identifying key predictive variables
- **Confidence Scoring:** Measuring prediction reliability

---

## Appendix

### Complete Team Statistics

| teamAbbrev   |   points |   wins |   losses |   otLosses |   goalsFor |   goalsAgainst |   PP% |   PK% |
|:-------------|---------:|-------:|---------:|-----------:|-----------:|---------------:|------:|------:|
| WPG          |      116 |     56 |       22 |          4 |        169 |            118 | 0.289 | 0.794 |
| WSH          |      111 |     51 |       21 |          9 |        188 |            156 | 0.233 | 0.823 |
| VGK          |      110 |     50 |       22 |         10 |        177 |            152 | 0.283 | 0.757 |
| TOR          |      106 |     51 |       26 |          4 |        164 |            134 | 0.25  | 0.781 |
| DAL          |      106 |     50 |       26 |          6 |        178 |            148 | 0.22  | 0.82  |
| LAK          |      105 |     48 |       24 |          9 |        170 |            119 | 0.176 | 0.813 |
| TBL          |      102 |     47 |       26 |          8 |        183 |            140 | 0.262 | 0.815 |
| COL          |      102 |     49 |       29 |          4 |        165 |            155 | 0.248 | 0.798 |
| EDM          |      101 |     48 |       29 |          5 |        166 |            168 | 0.237 | 0.782 |
| CAR          |       99 |     47 |       29 |          5 |        170 |            160 | 0.185 | 0.843 |
| FLA          |       98 |     47 |       31 |          4 |        150 |            132 | 0.235 | 0.807 |
| MIN          |       97 |     45 |       30 |          7 |        140 |            146 | 0.209 | 0.724 |
| STL          |       96 |     44 |       30 |          8 |        172 |            147 | 0.221 | 0.742 |
| OTT          |       95 |     44 |       30 |          7 |        135 |            151 | 0.235 | 0.779 |
| CGY          |       94 |     40 |       27 |         14 |        133 |            138 | 0.212 | 0.762 |
| NJD          |       91 |     42 |       33 |          7 |        146 |            146 | 0.282 | 0.827 |
| MTL          |       91 |     40 |       31 |         11 |        153 |            179 | 0.201 | 0.809 |
| VAN          |       90 |     38 |       30 |         14 |        150 |            159 | 0.225 | 0.826 |
| UTA          |       89 |     38 |       31 |         13 |        153 |            146 | 0.242 | 0.793 |
| CBJ          |       87 |     39 |       33 |          9 |        193 |            178 | 0.195 | 0.769 |
| DET          |       85 |     39 |       35 |          7 |        134 |            156 | 0.268 | 0.698 |
| NYR          |       83 |     38 |       36 |          7 |        173 |            170 | 0.177 | 0.801 |
| NYI          |       82 |     35 |       34 |         12 |        160 |            154 | 0.126 | 0.722 |
| ANA          |       80 |     35 |       37 |         10 |        154 |            162 | 0.118 | 0.742 |
| PIT          |       78 |     33 |       36 |         12 |        154 |            194 | 0.256 | 0.78  |
| BUF          |       77 |     35 |       39 |          7 |        182 |            179 | 0.189 | 0.766 |
| PHI          |       76 |     33 |       38 |         10 |        164 |            190 | 0.146 | 0.775 |
| SEA          |       76 |     35 |       41 |          6 |        167 |            179 | 0.189 | 0.772 |
| BOS          |       76 |     33 |       39 |         10 |        150 |            163 | 0.152 | 0.763 |
| NSH          |       68 |     30 |       44 |          8 |        124 |            184 | 0.219 | 0.815 |
| CHI          |       61 |     25 |       46 |         11 |        148 |            204 | 0.249 | 0.793 |
| SJS          |       52 |     20 |       50 |         12 |        143 |            201 | 0.186 | 0.742 |

### Glossary

- **GP:** Games Played
- **W:** Wins
- **L:** Losses  
- **OTL:** Overtime/Shootout Losses
- **PTS:** Points (2 for win, 1 for OT/SO loss)
- **GF:** Goals For
- **GA:** Goals Against
- **PP%:** Power Play Percentage
- **PK%:** Penalty Kill Percentage
- **xG%:** Expected Goals Percentage
- **CF%:** Corsi For Percentage
- **FF%:** Fenwick For Percentage

---

*This report was generated using the NHL Playoff Prediction System - an advanced machine learning platform for hockey analytics.*

**Contact:** For questions about this analysis or the underlying methodology, please refer to the project documentation.

**Disclaimer:** These predictions are for analytical purposes only and should not be used for gambling or wagering activities.
