# NFL-Big-data-bowl-2025

# NFL Big Data Bowl 2025: Predictive Analysis of Play Results

## Overview
This repository contains a detailed data analysis and predictive modeling project for the NFL Big Data Bowl 2025 competition. The project aims to predict play outcomes by using player tracking data, plays metadata, and player statistics.

## Objective
The main objective is to provide insightful analysis and predictions that can be used by NFL teams or the league office on a week-to-week basis. The project takes into account complex variables, like player speed, distance from the line of scrimmage, and coverage strategies, to predict different play results (e.g., complete pass, interception, rush, etc).

## Datasets
- **Plays**: This dataset includes metadata about each play, such as game ID, play ID, and pass result.
- **Players**: Information about each player, including height, weight, and position.
- **Tracking Data**: Player movement data, including speed, acceleration, and location throughout each play.
- https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data

All data was sourced from the official NFL Big Data Bowl dataset.

## Methodology
### Data Processing
- **Memory Optimization**: Reduced the memory usage of datasets for efficient processing.
- **Handling Missing Values**: Used SimpleImputer for numeric columns and `fillna` for categorical columns to handle missing values.
- **Feature Engineering**: Created new features based on player speed, acceleration, and distance from the line of scrimmage, as well as encoding offensive and defensive formations.

### Exploratory Data Analysis (EDA)
- **Missing Value Analysis**: Visualized missing data using heatmaps.
- **Play & Player Analysis**: Examined the distribution of different play types and player positions.
- **Speed Analysis**: Analyzed the speed distribution of players during plays.

### Modeling
- **Logistic Regression**: Used to classify the outcome of plays (e.g., pass complete, interception). The features were standardized using `StandardScaler` to improve model performance.
- **Evaluation Metrics**: The model achieved an accuracy of 95%, precision of 95%, recall of 95%, and F1 score of 94%, indicating strong performance.

## Results
- The model predicts different play outcomes with high accuracy.
- Detailed analysis and feature engineering focused on critical factors that influence play results, such as player movement before and after the snap.

## Football Score Evaluation
- **Practical Application**: The model can be used by NFL teams or the league office to evaluate play strategies, predict opposing team decisions, and make data-driven improvements.
- **Complexity Consideration**: The analysis takes into account several key variables like acceleration, speed, formations, and coverage types to make accurate predictions.
- **Unique Approach**: The feature engineering and focus on player positioning pre- and post-snap provide a unique view of play dynamics.

## Data Science Evaluation
- **Correctness**: The model and analysis were validated using appropriate metrics and data validation steps.
- **Claims and Backups**: All insights were backed up by exploratory data analysis and model evaluations.
- **Innovation**: Feature engineering introduced innovative metrics like maximum and average speed/acceleration pre-snap, giving additional insights.

## Report and Visualization
- The analysis and model results were clearly documented.
- Data visualizations, including player speed distributions, missing value heatmaps, and play type distributions, were provided for better understanding of the data and model outcomes.

## Running the Project
1. **Clone the Repository**: `git clone <repository_url>`
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install the required Python packages.
3. **Run the Analysis**: Open `nfl_data_analysis.ipynb` in Jupyter Notebook or any compatible environment to see the detailed analysis.

## Requirements
- Python 3.8+
- Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## Author
Vijeeth Kumar

## License
This project is licensed under the MIT License.
