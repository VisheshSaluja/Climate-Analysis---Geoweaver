import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load predictions, actual values, and scaler
predictions = np.load('/path/to/your/predictions.npy')
data = np.load('/path/to/your/preprocessed_data.npz')
y_test = data['y_test']

# Load the scaler for inverse transformation
scaler = joblib.load('/path/to/your/scaler.pkl')

# Inverse scale the predictions and real values
predicted_values = scaler.inverse_transform(predictions)
real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Scientific Objective: Analyze Heatwave Events
# Assuming a heatwave is defined as a temperature above 90°F

# Identify predicted heatwave events
heatwave_threshold = 90.0
predicted_heatwaves = predicted_values > heatwave_threshold
real_heatwaves = real_values > heatwave_threshold

# Count predicted vs. actual heatwave events
num_predicted_heatwaves = np.sum(predicted_heatwaves)
num_actual_heatwaves = np.sum(real_heatwaves)

print(f"Number of predicted heatwave events: {num_predicted_heatwaves}")
print(f"Number of actual heatwave events: {num_actual_heatwaves}")

# Calculate the prediction accuracy for heatwave events
heatwave_accuracy = np.sum(predicted_heatwaves == real_heatwaves) / len(real_heatwaves)
print(f"Heatwave Prediction Accuracy: {heatwave_accuracy * 100:.2f}%")

# Visualize heatwave predictions vs actual values
plt.figure(figsize=(14, 6))
plt.plot(real_values, label='Actual Temperature', color='blue')
plt.plot(predicted_values, label='Predicted Temperature', color='red')
plt.axhline(y=heatwave_threshold, color='green', linestyle='--', label='Heatwave Threshold (90°F)')
plt.title('Heatwave Events: Real vs Predicted Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)
plt.show()

# Save heatwave analysis to a file
with open('heatwave_analysis.txt', 'w') as f:
    f.write(f"Number of predicted heatwave events: {num_predicted_heatwaves}\n")
    f.write(f"Number of actual heatwave events: {num_actual_heatwaves}\n")
    f.write(f"Heatwave Prediction Accuracy: {heatwave_accuracy * 100:.2f}%\n")

