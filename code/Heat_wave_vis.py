# import matplotlib.pyplot as plt
# import numpy as np
# import joblib
# import pandas as pd

# # Step 1: Load the original dataset to get the dates
# file_path = '/Users/vishesh/Desktop/geo_project/data.csv'
# data = pd.read_csv(file_path, skiprows=3)

# # Convert the 'Date' column to datetime format
# data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')

# # Step 2: Load predictions for the next week
# next_week_predictions = np.load('/Users/vishesh/gw-workspace/8ZZWYYcx2NtS/next_week_predictions.npy')

# # Step 3: Load the scaler for inverse transformation
# scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# # Step 4: Inverse scale the next week's predicted values
# predicted_next_week_temp = scaler.inverse_transform(next_week_predictions)

# # Step 5: Generate dates for the next 7 days based on the last date in the dataset
# last_date = data['Date'].iloc[-1]
# next_week_dates = pd.date_range(start=last_date, periods=7 + 1, inclusive='right')

# # Step 6: Plot the predicted values
# plt.figure(figsize=(10, 6))
# plt.plot(next_week_dates, predicted_next_week_temp, color='red', label='Predicted Temperatures')

# # Customize the plot
# plt.title('Predicted Temperature for the Next Week')
# plt.xlabel('Date')
# plt.ylabel('Temperature (Degrees Fahrenheit)')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()

# # Show the plot
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import joblib
# import pandas as pd

# # Step 1: Load the original dataset to get the dates
# file_path = '/Users/vishesh/Desktop/geo_project/data.csv'
# data = pd.read_csv(file_path, skiprows=3)

# # Convert the 'Date' column to datetime format
# data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')

# # Step 2: Load predictions for the next week
# next_week_predictions = np.load('/Users/vishesh/gw-workspace/8ZZWYYcx2NtS/next_week_predictions.npy')

# # Step 3: Load the scaler for inverse transformation
# scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# # Step 4: Inverse scale the next week's predicted values
# predicted_next_week_temp = scaler.inverse_transform(next_week_predictions)

# # Step 5: Generate dates for the next 7 days based on the last date in the dataset
# last_date = data['Date'].iloc[-1]
# next_week_dates = pd.date_range(start=last_date, periods=7 + 1, inclusive='right')

# # Step 6: Define thresholds for heat wave and cold wave
# heat_wave_threshold = 53  # Temperatures above 95°F indicate a heat wave
# cold_wave_threshold = 32  # Temperatures below 32°F indicate a cold spell

# # Step 7: Identify days with heat wave or cold wave conditions
# heat_wave_days = predicted_next_week_temp >= heat_wave_threshold
# cold_wave_days = predicted_next_week_temp <= cold_wave_threshold

# # Step 8: Create a better visualization
# plt.figure(figsize=(12, 8))

# # Plot the predicted temperatures with a smooth line
# plt.plot(next_week_dates, predicted_next_week_temp, color='black', marker='o', markersize=8, linewidth=2, label='Predicted Temperatures')

# # Highlight heat wave days with red shading
# plt.fill_between(next_week_dates, predicted_next_week_temp.flatten(), heat_wave_threshold,
#                  where=heat_wave_days.flatten(), color='salmon', alpha=0.6, label='Heat Wave')

# # Highlight cold wave days with blue shading
# plt.fill_between(next_week_dates, predicted_next_week_temp.flatten(), cold_wave_threshold,
#                  where=cold_wave_days.flatten(), color='lightblue', alpha=0.6, label='Cold Wave')

# # Customize the plot
# plt.title('Predicted Temperature for the Next Week with Heat/Cold Wave Alerts', fontsize=16, weight='bold')
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Temperature (Degrees Fahrenheit)', fontsize=14)
# plt.axhline(y=heat_wave_threshold, color='red', linestyle='--', label='Heat Wave Threshold (95°F)', linewidth=2)
# plt.axhline(y=cold_wave_threshold, color='blue', linestyle='--', label='Cold Wave Threshold (32°F)', linewidth=2)

# # Adding grid, tick marks, and rotation for better readability
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(rotation=45, fontsize=12)
# plt.yticks(fontsize=12)

# # Adding legend
# plt.legend(fontsize=12, loc='upper left')

# # Adding annotations for heat and cold waves
# for i, temp in enumerate(predicted_next_week_temp):
#     if heat_wave_days[i]:
#         plt.annotate('Heat Wave', (next_week_dates[i], predicted_next_week_temp[i]), textcoords="offset points", xytext=(0,5), ha='center', color='red', fontsize=10, fontweight='bold')
#     elif cold_wave_days[i]:
#         plt.annotate('Cold Wave', (next_week_dates[i], predicted_next_week_temp[i]), textcoords="offset points", xytext=(0,5), ha='center', color='blue', fontsize=10, fontweight='bold')

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()




import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd

# Step 1: Load the original dataset to get the dates
file_path = '/Users/vishesh/Desktop/geo_project/data.csv'
data = pd.read_csv(file_path, skiprows=3)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m')

# Step 2: Load predictions for the next week from the improved LSTM model
next_week_predictions = np.load('/Users/vishesh/gw-workspace/next_week_predictions_improved.npy')

# Step 3: Load the scaler for inverse transformation (same scaler used during training)
scaler = joblib.load('/Users/vishesh/gw-workspace/f7bNu3SQ6blN/scaler.pkl')

# Step 4: Inverse scale the next week's predicted values
predicted_next_week_temp = scaler.inverse_transform(next_week_predictions)

# Step 5: Generate dates for the next 7 days based on the last date in the dataset
last_date = data['Date'].iloc[-1]
next_week_dates = pd.date_range(start=last_date, periods=7 + 1, inclusive='right')

# Step 6: Define thresholds for heat wave and cold wave
heat_wave_threshold = 53  # Adjusted temperature threshold
cold_wave_threshold = 32  # Adjusted temperature threshold

# Step 7: Identify days with heat wave or cold wave conditions
heat_wave_days = predicted_next_week_temp >= heat_wave_threshold
cold_wave_days = predicted_next_week_temp <= cold_wave_threshold

# Step 8: Create a better visualization
plt.figure(figsize=(12, 8))

# Plot the predicted temperatures with a smooth line
plt.plot(next_week_dates, predicted_next_week_temp, color='black', marker='o', markersize=8, linewidth=2, label='Predicted Temperatures')

# Highlight heat wave days with red shading
plt.fill_between(next_week_dates, predicted_next_week_temp.flatten(), heat_wave_threshold,
                 where=heat_wave_days.flatten(), color='salmon', alpha=0.6, label='Heat Wave')

# Highlight cold wave days with blue shading
plt.fill_between(next_week_dates, predicted_next_week_temp.flatten(), cold_wave_threshold,
                 where=cold_wave_days.flatten(), color='lightblue', alpha=0.6, label='Cold Wave')

# Customize the plot
plt.title('Predicted Temperature for the Next Week with Heat/Cold Wave Alerts', fontsize=16, weight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (Degrees Fahrenheit)', fontsize=14)
plt.axhline(y=heat_wave_threshold, color='red', linestyle='--', label=f'Heat Wave Threshold ({heat_wave_threshold}°F)', linewidth=2)
plt.axhline(y=cold_wave_threshold, color='blue', linestyle='--', label=f'Cold Wave Threshold ({cold_wave_threshold}°F)', linewidth=2)

# Adding grid, tick marks, and rotation for better readability
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Adding legend
plt.legend(fontsize=12, loc='upper left')

# Adding annotations for heat and cold waves
for i, temp in enumerate(predicted_next_week_temp):
    if heat_wave_days[i]:
        plt.annotate('Heat Wave', (next_week_dates[i], predicted_next_week_temp[i]), textcoords="offset points", xytext=(0,5), ha='center', color='red', fontsize=10, fontweight='bold')
    elif cold_wave_days[i]:
        plt.annotate('Cold Wave', (next_week_dates[i], predicted_next_week_temp[i]), textcoords="offset points", xytext=(0,5), ha='center', color='blue', fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

