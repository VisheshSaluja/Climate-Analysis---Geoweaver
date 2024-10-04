

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib

# Step 1: Load the preprocessed data
data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Step 2: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), 
               kernel_regularizer=l2(0.001)))  # L2 regularization
model.add(Dropout(0.2))  # Increased dropout rate
model.add(LSTM(units=50, return_sequences=False, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# Step 3: Compile the model with reduced learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Step 4: Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])

# Step 5: Save the model and the training history
model.save('lstm_model.h5')
np.save('training_history.npy', history.history)

# Step 6: Make predictions on the test data
predictions = model.predict(X_test)

# Save predictions for later use
np.save('predictions.npy', predictions)

print("Model training completed, predictions made and saved!")


