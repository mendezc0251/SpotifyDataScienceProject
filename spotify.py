# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('spotify-2023.csv', encoding='latin1')

# Rename columns
re_col = {
    'danceability_%': 'danceability',
    'valence_%': 'valence',
    'energy_%': 'energy',
    'acousticness_%': 'acousticness',
    'instrumentalness_%': 'instrumentalness',
    'liveness_%': 'liveness',
    'speechiness_%': 'speechiness' 
}
df.rename(columns=re_col, inplace=True)

# Display basic information about the DataFrame
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Handle missing values more robustly
df = df.dropna(subset=['streams', 'bpm', 'danceability', 'valence', 'energy', 'acousticness'])

# Convert 'streams' column to numeric
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Explore correlations between numeric features
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Convert 'streams' column to numeric
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Initialize a scaler
scaler = MinMaxScaler()

# Fit the scaler to the 'streams' data and transform it
df['streams'] = scaler.fit_transform(df[['streams']])
# Select relevant features for the model
features = ['bpm', 'danceability', 'valence', 'energy', 'acousticness']
target_variable = 'streams'

X = df[features]
y = df[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the first few rows of X_train and y_train for inspection
print("First few rows of X_train:")
print(X_train.head())

print("\nFirst few rows of y_train:")
print(y_train.head())

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Apply inverse transformation to y_pred
y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Apply inverse transformation to y_test
y_test_orig = scaler.inverse_transform(y_test.values.reshape(-1, 1))

# Now, calculate the MSE
mse = mean_squared_error(y_test_orig, y_pred_orig)
print(f'Mean Squared Error: {mse}')

# Visualize predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred_orig)
plt.xlabel('Actual Streams')
plt.ylabel('Predicted Streams')
plt.title('Actual vs Predicted Streams')
plt.show()
# Exclude non-numeric columns before correlation analysis
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

# Display the correlation matrix
correlation_matrix

# 4. Predictive Modeling
# Assume 'X' contains your independent variables and 'y' is the target variable (streams)
X = df.drop('streams', axis=1)
y = df['streams']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Drop non-numeric columns and handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numeric_columns].drop(['streams'], axis=1)
y = df['streams']

# Handle missing values
X = X.fillna(0)  # You might need a more sophisticated approach based on your data

# Encode categorical variables if there are any
X = pd.get_dummies(X)

# Split the data and fit the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Check the coefficients
coefficients = model.coef_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance

# 6. Insights and Conclusions
# Add your insights and conclusions here based on the analysis
