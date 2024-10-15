# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# import joblib

# # Load the preprocessed data
# data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
# X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.3))
# model.add(Dense(units=1))

# # Compile the model with reduced learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
# model.compile(optimizer=optimizer, loss='mean_squared_error')

# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# # Save the model and the training history
# model.save('lstm_model.h5')
# np.save('training_history.npy', history.history)

# # Making future predictions for next week (7 days)
# def predict_next_week(model, input_sequence, num_days=7):
#     predictions = []
#     current_input = input_sequence

#     for day in range(num_days):
#         pred = model.predict(current_input)
#         predictions.append(pred[0])
#         # Update input with the prediction (reshape pred to match current_input shape)
#         pred = np.reshape(pred, (1, 1, 1))  # Reshape to (1, 1, 1) to match LSTM input format
#         current_input = np.append(current_input[:, 1:, :], pred, axis=1)

#     return np.array(predictions)

# # Take the last sequence from the test data as the input for future predictions
# last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# # Predict for the next 7 days
# next_week_predictions = predict_next_week(model, last_sequence)

# # Save predictions for the next week
# np.save('next_week_predictions.npy', next_week_predictions)







import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib

# Step 1: Load the preprocessed data
data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Step 2: Build the improved LSTM model
model = Sequential()

# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001))))
model.add(Dropout(0.1))  # Reduced dropout rate to retain more information

# Add another LSTM layer with more units
model.add(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.1))

# Final dense layer for output
model.add(Dense(units=1))

# Step 3: Compile the model with an optimized learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Experiment with a slightly higher learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Step 4: Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Step 6: Save the model and the training history
model.save('improved_lstm_model.h5')
np.save('improved_training_history.npy', history.history)

# Step 7: Making future predictions for next week (7 days)
def predict_next_week(model, input_sequence, num_days=7):
    predictions = []
    current_input = input_sequence

    for day in range(num_days):
        pred = model.predict(current_input)
        predictions.append(pred[0])
        # Update input with the prediction (reshape pred to match current_input shape)
        pred = np.reshape(pred, (1, 1, 1))  # Reshape to (1, 1, 1) to match LSTM input format
        current_input = np.append(current_input[:, 1:, :], pred, axis=1)

    return np.array(predictions)

# Step 8: Take the last sequence from the test data as the input for future predictions
last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# Step 9: Predict for the next 7 days
next_week_predictions = predict_next_week(model, last_sequence)

# Step 10: Save predictions for the next week
np.save('next_week_predictions_improved.npy', next_week_predictions)

# Optionally, you can load these predictions for visualization later

