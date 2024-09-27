# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def visualize(file_name, predictions_file, anomalies_file):
#     data = pd.read_csv(file_name)
#     predictions = pd.read_csv(predictions_file)
#     anomalies = pd.read_csv(anomalies_file)
    
#     plt.figure(figsize=(12, 8))
    
#     # Plotting temperature trends with predictions
#     plt.subplot(2, 1, 1)
#     plt.plot(data['Year'], data['Temp_Value'], label='Observed Temperature')
#     plt.plot(predictions['Year'], predictions['Temp_Prediction'], label='Predicted Temperature', linestyle='--')
#     plt.title('Temperature Trends and Predictions')
#     plt.xlabel('Year')
#     plt.ylabel('Temperature (°C)')
#     plt.legend()
    
#     # Plotting precipitation trends with predictions
#     plt.subplot(2, 1, 2)
#     plt.plot(data['Year'], data['Precipitation'], label='Observed Precipitation')
#     plt.plot(predictions['Year'], predictions['Precip_Prediction'], label='Predicted Precipitation', linestyle='--')
#     plt.title('Precipitation Trends and Predictions')
#     plt.xlabel('Year')
#     plt.ylabel('Precipitation (mm)')
#     plt.legend()
    
#     # Highlighting anomalies
#     sns.scatterplot(data=anomalies, x='Year', y='Temp_Value', hue='Cluster', palette='deep', legend='full', s=100)
    
#     plt.tight_layout()
#     plt.savefig('climate_trends_and_anomalies.png')
#     plt.show()

# if __name__ == "__main__":
#     visualize('/Users/vishesh/Desktop/geo_project/analyzed_climate_data.csv', '/Users/vishesh/Desktop/geo_project/predicted_climate_data.csv', '/Users/vishesh/Desktop/geo_project/anomaly_detected_data.csv')






import numpy as np
import matplotlib.pyplot as plt
import joblib

# Step 1: Load predictions, actual values, and scaler
predictions = np.load('add path to predictions .npy file')
data = np.load('add path to .npz file')
y_test = data['y_test']

# Load the scaler for inverse transformation
scaler = joblib.load('add path to/scaler.pkl')

# Step 2: Inverse scale the predictions and real values
predicted_values = scaler.inverse_transform(predictions)
real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 3: Visualize the loss over epochs
history = np.load('add path to/training_history.npy', allow_pickle=True).item()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 4: Visualize the predicted vs actual values
plt.plot(real_values, color='blue', label='Real Temperature Values')
plt.plot(predicted_values, color='red', label='Predicted Temperature Values')
plt.title('Real vs Predicted Temperature Values')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()