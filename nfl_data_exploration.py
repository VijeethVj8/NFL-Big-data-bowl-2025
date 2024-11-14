import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer

# Function to reduce memory usage by optimizing data types
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)    
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

# Load plays and players datasets with optimized memory usage
plays = pd.read_csv("plays.csv")
players = pd.read_csv("players.csv")

# Reduce memory usage for each dataset
plays = reduce_memory_usage(plays)
players = reduce_memory_usage(players)

# Load tracking data in chunks to avoid memory overload
chunk_size = 500000
tracking_data_chunks = []

for chunk in pd.read_csv("tracking_week_1.csv", chunksize=chunk_size):
    chunk = reduce_memory_usage(chunk)
    tracking_data_chunks.append(chunk)

# Concatenate chunks into a single dataframe
tracking_week_1 = pd.concat(tracking_data_chunks, axis=0)

# Sampling data to work with a smaller subset (e.g., 5% of the data)
tracking_week_1_sample = tracking_week_1.sample(frac=0.05, random_state=42)

# Display first few rows of each dataset to get a sense of the structure
print("Plays Dataset:
", plays.head())
print("
Players Dataset:
", players.head())
print("
Tracking Week 1 Sample Dataset:
", tracking_week_1_sample.head())

# Basic info to understand the size and column types of each dataframe
print("
Plays Info:
")
print(plays.info())
print("
Players Info:
")
print(players.info())
print("
Tracking Week 1 Sample Info:
")
print(tracking_week_1_sample.info())

# Check for missing values in each dataset
print("
Missing Values in Plays Dataset:
", plays.isnull().sum())
print("
Missing Values in Players Dataset:
", players.isnull().sum())
print("
Missing Values in Tracking Week 1 Sample Dataset:
", tracking_week_1_sample.isnull().sum())

# Visualize missing values using a heatmap for better understanding
plt.figure(figsize=(16, 9))
sns.heatmap(plays.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Plays Dataset")
plt.show()

plt.figure(figsize=(16, 9))
sns.heatmap(players.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Players Dataset")
plt.show()

plt.figure(figsize=(16, 9))
sns.heatmap(tracking_week_1_sample.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Tracking Week 1 Sample Dataset")
plt.show()

# Initial exploration of key columns
# Distribution of play types (run vs pass) using 'passResult' and 'isDropback'
plays['passResult'].value_counts(dropna=False).plot(kind='bar', figsize=(10, 6))
plt.title("Distribution of Pass Results")
plt.xlabel("Pass Result")
plt.ylabel("Count")
plt.show()

plays['isDropback'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title("Dropback Play Distribution")
plt.xlabel("Is Dropback")
plt.ylabel("Count")
plt.show()

# Exploring player positions
players['position'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title("Distribution of Player Positions")
plt.xlabel("Position")
plt.ylabel("Count")
plt.show()

# Exploring tracking data (Player speed distribution)
sns.histplot(tracking_week_1_sample['s'], kde=True, bins=30)
plt.title("Distribution of Player Speeds in Week 1 Tracking Data (Sample)")
plt.xlabel("Speed (yards/second)")
plt.ylabel("Frequency")
plt.show()

# Feature Engineering
# Merging plays and tracking data on gameId and playId
plays_tracking_merged = pd.merge(tracking_week_1_sample, plays, on=['gameId', 'playId'], how='left')

# Adding player information to the merged dataframe
full_merged = pd.merge(plays_tracking_merged, players, on='nflId', how='left')

# Creating new features
# 1. Average, Maximum, and Minimum speed and acceleration of each player before the snap
avg_speed_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['s'].mean().reset_index()
avg_speed_before_snap.rename(columns={'s': 'avg_speed_before_snap'}, inplace=True)

max_speed_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['s'].max().reset_index()
max_speed_before_snap.rename(columns={'s': 'max_speed_before_snap'}, inplace=True)

min_speed_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['s'].min().reset_index()
min_speed_before_snap.rename(columns={'s': 'min_speed_before_snap'}, inplace=True)

avg_acceleration_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['a'].mean().reset_index()
avg_acceleration_before_snap.rename(columns={'a': 'avg_acceleration_before_snap'}, inplace=True)

max_acceleration_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['a'].max().reset_index()
max_acceleration_before_snap.rename(columns={'a': 'max_acceleration_before_snap'}, inplace=True)

min_acceleration_before_snap = full_merged[full_merged['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'playId', 'nflId'])['a'].min().reset_index()
min_acceleration_before_snap.rename(columns={'a': 'min_acceleration_before_snap'}, inplace=True)

# Merging the new features back to the main dataframe
full_merged = pd.merge(full_merged, avg_speed_before_snap, on=['gameId', 'playId', 'nflId'], how='left')
full_merged = pd.merge(full_merged, max_speed_before_snap, on=['gameId', 'playId', 'nflId'], how='left')
full_merged = pd.merge(full_merged, min_speed_before_snap, on=['gameId', 'playId', 'nflId'], how='left')

full_merged = pd.merge(full_merged, avg_acceleration_before_snap, on=['gameId', 'playId', 'nflId'], how='left')
full_merged = pd.merge(full_merged, max_acceleration_before_snap, on=['gameId', 'playId', 'nflId'], how='left')
full_merged = pd.merge(full_merged, min_acceleration_before_snap, on=['gameId', 'playId', 'nflId'], how='left')

# 2. Encoding offense and defense formations
full_merged = pd.get_dummies(full_merged, columns=['offenseFormation', 'pff_passCoverage'], drop_first=True)

# 3. Calculating distance of each player from the line of scrimmage
full_merged['distance_from_los'] = abs(full_merged['x'] - full_merged['yardlineNumber'])

# Handling missing values using SimpleImputer for numeric columns and fillna for categorical columns
numeric_cols = full_merged.select_dtypes(include=['number']).columns
categorical_cols = full_merged.select_dtypes(exclude=['number']).columns

imputer = SimpleImputer(strategy='mean')
full_merged[numeric_cols] = imputer.fit_transform(full_merged[numeric_cols])
full_merged[categorical_cols] = full_merged[categorical_cols].fillna('Unknown')

# Check for any remaining NaNs
print("Remaining NaNs after imputation:", full_merged.isnull().sum().sum())

# Preparing data for model building
features = full_merged.drop(['passResult'], axis=1, errors='ignore')
labels = full_merged['passResult'].fillna(0)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['number']))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=['number']))

# Model Building
# Train a Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Classification report
print("
Classification Report:
")
print(classification_report(y_test, y_pred, zero_division=1))