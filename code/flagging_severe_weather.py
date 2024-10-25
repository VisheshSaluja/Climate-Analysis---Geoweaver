# import joblib
# import numpy as np

# # Load the scaler for inverse transformation
# scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# # Load the predictions for the next week
# next_week_predictions = np.load('/Users/vishesh/gw-workspace/8ZZWYYcx2NtS/next_week_predictions.npy')

# # Inverse scale the predictions for the next week
# predicted_next_week_temp = scaler.inverse_transform(next_week_predictions)

# # Define thresholds for severe weather
# heat_wave_threshold = 53  # Example: temperatures above 95°F indicate a heat wave
# cold_wave_threshold = 32  # Example: temperatures below 32°F indicate a cold spell

# # Flag severe weather conditions
# def flag_severe_weather(predictions, heat_wave_thresh, cold_wave_thresh):
#     severe_weather_flags = []
#     for temp in predictions:
#         if temp > heat_wave_thresh:
#             severe_weather_flags.append('Heat Wave')
#         elif temp < cold_wave_thresh:
#             severe_weather_flags.append('Cold Spell')
#         else:
#             severe_weather_flags.append('Normal')
#     return severe_weather_flags

# # Flag the predictions
# severe_weather_flags = flag_severe_weather(predicted_next_week_temp, heat_wave_threshold, cold_wave_threshold)

# # Display the results
# for day, temp in enumerate(predicted_next_week_temp, 1):
#     print(f"Day {day}: Predicted Temp = {temp[0]:.2f}°F, Condition = {severe_weather_flags[day-1]}")

#################################################

# import tensorflow as tf
# import numpy as np
# import joblib
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import tensorflow as tf
# import numpy as np
# import joblib
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Import the missing metrics

# # Load the scaler and model
# scaler = joblib.load('/Users/vishesh/gw-workspace/zfro02zn1ht/scaler.pkl')
# model = tf.keras.models.load_model('/Users/vishesh/gw-workspace/8wihud1jiul/improved_lstm_model.h5')

# # Load the preprocessed data
# data = np.load('/Users/vishesh/gw-workspace/xik1nlomsdk/preprocessed_data.npz')
# X_test = data['X_test']
# y_test = data['y_test']  # Ensure y_test is loaded

# # Prepare the last test sequence for next week prediction
# last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# # Function for rolling forecast predictions
# def rolling_predict_next_week(model, input_seq, num_days=7):
#     predictions = []
#     current_input = input_seq
#     for _ in range(num_days):
#         pred = model.predict(current_input)
#         predictions.append(pred[0])
#         current_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
#     return np.array(predictions)

# # Make predictions for the next week
# next_week_predictions = rolling_predict_next_week(model, last_sequence)

# # Inverse transform the predictions
# predicted_temps = scaler.inverse_transform(next_week_predictions)

# # Define thresholds for heat wave and cold wave
# heat_wave_threshold = 95  # °F
# cold_wave_threshold = 32  # °F

# # Function to flag severe weather conditions
# def flag_severe_weather(predictions, heat_thresh, cold_thresh):
#     return [
#         'Heat Wave' if temp > heat_thresh else 'Cold Spell' if temp < cold_thresh else 'Normal'
#         for temp in predictions
#     ]

# # Flag the predictions with weather conditions
# conditions = flag_severe_weather(predicted_temps, heat_wave_threshold, cold_wave_threshold)

# # Display the predictions and corresponding weather conditions
# for day, (temp, condition) in enumerate(zip(predicted_temps, conditions), 1):
#     print(f"Day {day}: Predicted Temp = {temp[0]:.2f}°F, Condition = {condition}")

# # Save the predictions
# np.save('next_week_predictions.npy', next_week_predictions)

# # Evaluate the model performance
# y_test_inv = scaler.inverse_transform(y_test)  # Inverse transform y_test
# y_pred = scaler.inverse_transform(model.predict(X_test))  # Inverse transform predictions

# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test_inv, y_pred)
# mse = mean_squared_error(y_test_inv, y_pred)
# r2 = r2_score(y_test_inv, y_pred)

# print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")


###################################################

import tensorflow as tf
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the scaler and model
scaler = joblib.load('/Users/vishesh/gw-workspace/zfro02zn1ht/scaler.pkl')
model = tf.keras.models.load_model('/Users/vishesh/gw-workspace/8wihud1jiul/improved_lstm_model.h5')

# Load the preprocessed data
data = np.load('/Users/vishesh/gw-workspace/xik1nlomsdk/preprocessed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

# Prepare the last sequence for prediction
last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# Function to perform rolling predictions for the next 7 days
def rolling_predict(model, input_seq, days=7):
    predictions = []
    for _ in range(days):
        pred = model.predict(input_seq)
        predictions.append(pred[0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    return np.array(predictions)

# Make predictions for the next week
next_week_predictions = rolling_predict(model, last_sequence)

# Ensure predictions have the correct shape for inverse transformation
print(f"Predictions shape: {next_week_predictions.shape}")
if next_week_predictions.ndim == 1:
    next_week_predictions = next_week_predictions.reshape(-1, 1)

# Apply inverse transformation to get temperatures in Fahrenheit
predicted_temps = scaler.inverse_transform(next_week_predictions)

print("Predicted Temperatures (°F):")
print(predicted_temps)

# Evaluate the model
y_test_inv = scaler.inverse_transform(y_test)  # Inverse transform y_test
y_pred = scaler.inverse_transform(model.predict(X_test))  # Inverse transform predictions

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_inv, y_pred)
mse = mean_squared_error(y_test_inv, y_pred)
r2 = r2_score(y_test_inv, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

# Save the predictions for visualization
np.save('next_week_predictions.npy', next_week_predictions)

