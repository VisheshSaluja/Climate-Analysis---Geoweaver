import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = '/Users/vishesh/Desktop/geo_project/data.csv'
data = pd.read_csv(file_path, skiprows=3)

# Step 2: Preprocess the data
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')
values = data['Value'].values

# Step 3: Normalize the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))

# Save the scaler for inverse transformation later
joblib.dump(scaler, 'scaler.pkl')

# Step 4: Create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

sequence_length = 12  # 12 months
X, y = create_sequences(scaled_data, sequence_length)

# Step 5: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 6: Save preprocessed data for later use
np.savez('preprocessed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Data preprocessing completed and saved!")



plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Value'], label='Temperature Values')
plt.title('Temperature Values Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (Degrees Fahrenheit)')
plt.grid(True)
plt.legend()
plt.show()

