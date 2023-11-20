import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/mendezc/Desktop/Spotify Project/spotify-2023.csv', encoding='latin-1')

# Check the first few rows
print(df.head())

# Check summary statistics
print(df.describe())

# Check data types and missing values
print(df.info())
# Dropping in_shazam_charts and key as they did not have enough non-null data
df = df.drop('in_shazam_charts', axis=1)
df = df.drop('key', axis=1)
# Ensuring data in streams is numeric, if not replacing with blank
df['streams'] = pd.to_numeric(df['streams'].str.replace(r'\D', ''), errors='coerce')
print(df.isnull().sum())

# Scatter plot of artist_count vs streams
plt.figure(figsize=(10,6))
sns.scatterplot(x='artist_count', y='streams', data=df)
plt.title('Artist count vs. Streams (in billions)')
plt.tight_layout()
# Scatter plot of 

plt.show()