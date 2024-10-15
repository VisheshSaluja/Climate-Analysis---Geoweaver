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




import joblib
import numpy as np

# Step 1: Load the scaler for inverse transformation
scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# Step 2: Load the predictions for the next week from the improved LSTM model
next_week_predictions = np.load('/Users/vishesh/gw-workspace/next_week_predictions_improved.npy')

# Step 3: Inverse scale the predictions for the next week
predicted_next_week_temp = scaler.inverse_transform(next_week_predictions)

# Step 4: Define thresholds for severe weather
heat_wave_threshold = 53  # Example: temperatures above 53°F indicate a heat wave
cold_wave_threshold = 32  # Example: temperatures below 32°F indicate a cold spell

# Step 5: Flag severe weather conditions
def flag_severe_weather(predictions, heat_wave_thresh, cold_wave_thresh):
    severe_weather_flags = []
    for temp in predictions:
        if temp > heat_wave_thresh:
            severe_weather_flags.append('Heat Wave')
        elif temp < cold_wave_thresh:
            severe_weather_flags.append('Cold Spell')
        else:
            severe_weather_flags.append('Normal')
    return severe_weather_flags

# Step 6: Flag the predictions
severe_weather_flags = flag_severe_weather(predicted_next_week_temp, heat_wave_threshold, cold_wave_threshold)

# Step 7: Display the results
for day, temp in enumerate(predicted_next_week_temp, 1):
    print(f"Day {day}: Predicted Temp = {temp[0]:.2f}°F, Condition = {severe_weather_flags[day-1]}")

