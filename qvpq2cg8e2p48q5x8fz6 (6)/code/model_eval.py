# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib

# # Load predictions, actual values, and scaler
# predictions = np.load('/Users/vishesh/gw-workspace/9I3eI9y6wb9B/predictions.npy')
# data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
# y_test = data['y_test']

# # Load the scaler for inverse transformation
# scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# # Step 1: Inverse scale the predictions and real values
# predicted_values = scaler.inverse_transform(predictions)
# real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Step 2: Evaluation Metrics
# mae = mean_absolute_error(real_values, predicted_values)
# mse = mean_squared_error(real_values, predicted_values)
# rmse = np.sqrt(mse)
# r2 = r2_score(real_values, predicted_values)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R-squared (R²): {r2}")

# # Step 3: Error Distribution Plot
# errors = real_values - predicted_values
# plt.figure(figsize=(10, 6))
# plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
# plt.title('Error Distribution')
# plt.xlabel('Error')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Step 4: Seasonal Error Analysis (if Date information is available)
# date_series = pd.to_datetime(data['Date'], format='%Y%m')
# seasonal_errors = pd.DataFrame({'Date': date_series, 'Error': errors.flatten()})
# seasonal_errors['Month'] = seasonal_errors['Date'].dt.month

# # Calculate average error for each month
# monthly_error = seasonal_errors.groupby('Month')['Error'].mean()
# plt.figure(figsize=(10, 6))
# monthly_error.plot(kind='bar', color='coral', edgecolor='black')
# plt.title('Average Error by Month')
# plt.xlabel('Month')
# plt.ylabel('Average Error')
# plt.grid(True)
# plt.show()

# # Step 5: Backtesting
# # Here, a rolling forecast is performed by retraining the model iteratively. This is a placeholder implementation.
# # Replace this with your specific model training code.
# backtest_steps = 12  # Example: 12 months
# backtest_real = []
# backtest_pred = []

# for i in range(backtest_steps, len(real_values)):
#     # Perform training using historical data up to the current point
#     current_train_data = real_values[:i]
#     # (Insert model training code here)
#     # Predict the next step
#     current_prediction = predicted_values[i]  # Replace with model's predicted output
#     backtest_real.append(real_values[i])
#     backtest_pred.append(current_prediction)

# # Visualizing backtest results
# plt.figure(figsize=(10, 6))
# plt.plot(backtest_real, color='blue', label='Real Values')
# plt.plot(backtest_pred, color='red', label='Backtest Predictions')
# plt.title('Backtesting Real vs Predicted Values')
# plt.xlabel('Time')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load predictions, actual values, and scaler
predictions = np.load('/Users/vishesh/gw-workspace/9I3eI9y6wb9B/predictions.npy')
data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
y_test = data['y_test']
dates = pd.to_datetime(data['dates'], format='%Y-%m-%d')  # Adjust format based on your date column

# Load the scaler for inverse transformation
scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# Inverse scale the predictions and real values
predicted_values = scaler.inverse_transform(predictions)
real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Extensive Model Evaluation

# Calculate evaluation metrics
mse = mean_squared_error(real_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_values, predicted_values)
r2 = r2_score(real_values, predicted_values)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-Squared (R²): {r2}")

# Backtesting: Plotting Real vs Predicted with Dates
plt.figure(figsize=(14, 6))
plt.plot(dates, real_values, label='Actual Temperature', color='blue')
plt.plot(dates, predicted_values, label='Predicted Temperature', color='red')
plt.title('Model Backtesting: Real vs Predicted Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)
plt.show()

# Seasonal Error Analysis
months = dates.month  # Extract month information from dates
seasonal_mae = {}

for month in range(1, 13):
    indices = np.where(months == month)[0]
    seasonal_mae[month] = mean_absolute_error(real_values[indices], predicted_values[indices])

# Plot seasonal errors
plt.figure(figsize=(10, 5))
plt.bar(seasonal_mae.keys(), seasonal_mae.values(), color='skyblue')
plt.title('Seasonal Mean Absolute Error (MAE)')
plt.xlabel('Month')
plt.ylabel('Mean Absolute Error')
plt.grid(True)
plt.show()

# Save metrics to a file
with open('model_evaluation_metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"R-Squared (R²): {r2}\n")
    f.write(f"Seasonal MAE: {seasonal_mae}\n")

