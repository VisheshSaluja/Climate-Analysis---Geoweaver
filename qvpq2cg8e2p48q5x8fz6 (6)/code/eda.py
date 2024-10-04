import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/vishesh/Desktop/geo_project/data.csv'
data = pd.read_csv(file_path, skiprows=3)

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')

# Plot temperature values over time
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Value'], label='Temperature Values')
plt.title('Temperature Values Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (Degrees Fahrenheit)')
plt.grid(True)
plt.legend()
plt.show()

# Decompose the time series to identify trends and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['Value'], model='additive', period=12)
result.plot()
plt.show()

# Plot a correlation matrix
sns.heatmap(data.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

# Save summary statistics to a file
summary_stats = data.describe()
summary_stats.to_csv('eda_summary.csv')

