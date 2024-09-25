# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error

# def train_predict(file_name, degree=3):
#     data = pd.read_csv(file_name)
    
#     # Use Year as input, Temp_Value and Precipitation as targets
#     X = data[['Year']]
#     y_temp = data['Temp_Value']
#     y_precip = data['Precipitation']
    
#     # Save original X_test for use in the final output
#     X_train_orig, X_test_orig, y_temp_train, y_temp_test, y_precip_train, y_precip_test = train_test_split(
#         X, y_temp, y_precip, test_size=0.2, random_state=42)
    
#     # Polynomial Features Transformation
#     poly = PolynomialFeatures(degree=degree)
#     X_poly = poly.fit_transform(X)
    
#     # Perform the same train-test split on the transformed data
#     X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test = train_test_split(
#         X_poly, y_temp, y_precip, test_size=0.2, random_state=42)
    
#     # Train Polynomial Regression models for temperature and precipitation
#     model_temp = LinearRegression().fit(X_train, y_temp_train)
#     model_precip = LinearRegression().fit(X_train, y_precip_train)
    
#     # Make predictions
#     temp_predictions = model_temp.predict(X_test)
#     precip_predictions = model_precip.predict(X_test)
    
#     # Evaluate the models
#     temp_mse = mean_squared_error(y_temp_test, temp_predictions)
#     precip_mse = mean_squared_error(y_precip_test, precip_predictions)
#     print(f"Temperature Model MSE: {temp_mse}")
#     print(f"Precipitation Model MSE: {precip_mse}")
    
#     # Save predictions using the original Year from X_test_orig
#     predictions = pd.DataFrame({'Year': X_test_orig['Year'], 'Temp_Prediction': temp_predictions, 
#                                 'Precip_Prediction': precip_predictions})
#     predictions.to_csv('/Users/vishesh/Desktop/geo_project/predicted_climate_data.csv', index=False)
#     print("Predictions saved as predicted_climate_data.csv")

# if __name__ == "__main__":
#     train_predict('/Users/vishesh/Desktop/geo_project/analyzed_climate_data.csv', degree=3)
##############################


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_squared_error

# def prepare_lstm_data(X, y, time_steps=30):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         Xs.append(X[i:(i + time_steps)])
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

# def train_predict(file_name, time_steps=30, learning_rate=0.0001):
#     data = pd.read_csv(file_name)
    
#     # Use Year as input, Temp_Value and Precipitation as targets
#     X = data[['Year']]
#     y_temp = data['Temp_Value'].values
#     y_precip = data['Precipitation'].values
    
#     # Normalize the data using MinMaxScaler
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Prepare data for LSTM (look back by time_steps)
#     X_temp, y_temp = prepare_lstm_data(X_scaled, y_temp, time_steps)
#     X_precip, y_precip = prepare_lstm_data(X_scaled, y_precip, time_steps)
    
#     # Split data into training and test sets
#     X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
#         X_temp, y_temp, test_size=0.2, random_state=42)
    
#     X_precip_train, X_precip_test, y_precip_train, y_precip_test = train_test_split(
#         X_precip, y_precip, test_size=0.2, random_state=42)
    
#     # Define the LSTM model with more layers, dropout, and lower learning rate
#     def build_lstm_model():
#         model = Sequential()
#         model.add(LSTM(100, return_sequences=True, input_shape=(time_steps, 1)))
#         model.add(LSTM(100, return_sequences=False))
#         model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
#         model.add(Dense(50))
#         model.add(Dense(1))
        
#         # Compile model with a lower learning rate
#         optimizer = Adam(learning_rate=learning_rate)
#         model.compile(optimizer=optimizer, loss='mean_squared_error')
#         return model
    
#     # Reshape the data for LSTM (LSTM expects 3D input: [samples, time_steps, features])
#     X_temp_train = np.reshape(X_temp_train, (X_temp_train.shape[0], X_temp_train.shape[1], 1))
#     X_temp_test = np.reshape(X_temp_test, (X_temp_test.shape[0], X_temp_test.shape[1], 1))
    
#     X_precip_train = np.reshape(X_precip_train, (X_precip_train.shape[0], X_precip_train.shape[1], 1))
#     X_precip_test = np.reshape(X_precip_test, (X_precip_test.shape[0], X_precip_test.shape[1], 1))
    
#     # Early stopping to prevent overfitting
#     early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    
#     # Train LSTM model for temperature prediction
#     model_temp = build_lstm_model()
#     model_temp.fit(X_temp_train, y_temp_train, batch_size=64, epochs=100, callbacks=[early_stop], verbose=1)
    
#     # Train LSTM model for precipitation prediction
#     model_precip = build_lstm_model()
#     model_precip.fit(X_precip_train, y_precip_train, batch_size=64, epochs=100, callbacks=[early_stop], verbose=1)
    
#     # Make predictions
#     temp_predictions = model_temp.predict(X_temp_test)
#     precip_predictions = model_precip.predict(X_precip_test)
    
#     # Evaluate the models
#     temp_mse = mean_squared_error(y_temp_test, temp_predictions)
#     precip_mse = mean_squared_error(y_precip_test, precip_predictions)
#     print(f"Temperature Model MSE: {temp_mse}")
#     print(f"Precipitation Model MSE: {precip_mse}")
    
#     # Save predictions using the original Year from X_temp_test
#     predictions = pd.DataFrame({
#         'Year': scaler.inverse_transform(X_temp_test.reshape(X_temp_test.shape[0], X_temp_test.shape[1]))[:, -1],
#         'Temp_Prediction': temp_predictions.flatten(),
#         'Precip_Prediction': precip_predictions.flatten()
#     })
    
#     predictions.to_csv('/Users/vishesh/Desktop/geo_project/predicted_climate_data.csv', index=False)
#     print("Predictions saved as predicted_climate_data.csv")

# if __name__ == "__main__":
#     train_predict('/Users/vishesh/Desktop/geo_project/analyzed_climate_data.csv', time_steps=30, learning_rate=0.0001)



########################

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Step 1: Load the preprocessed data
data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Step 2: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Step 3: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Save the model and the training history
model.save('lstm_model.h5')
np.save('training_history.npy', history.history)

# Step 6: Make predictions on the test data
predictions = model.predict(X_test)

# Save predictions for later use
np.save('predictions.npy', predictions)

print("Model training completed, predictions made and saved!")