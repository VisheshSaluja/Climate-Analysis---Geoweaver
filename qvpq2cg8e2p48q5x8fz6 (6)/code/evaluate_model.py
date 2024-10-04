import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Step 1: Load the saved model
model = tf.keras.models.load_model('lstm_model.h5')

# Step 2: Load the preprocessed data (X_test and y_test)
data = np.load('preprocessed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

# Step 3: Load the scaler for inverse transformation
scaler = joblib.load('scaler.pkl')

# Step 4: Make predictions on the test data
predictions = model.predict(X_test)

# Step 5: Inverse transform the predictions and the actual values (y_test)
predicted_values = scaler.inverse_transform(predictions)
real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Calculate evaluation metrics
mse = mean_squared_error(real_values, predicted_values)
mae = mean_absolute_error(real_values, predicted_values)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Step 7: Plot the predicted vs actual values
plt.figure(figsize=(10,6))
plt.plot(real_values, color='blue', label='Actual Values')
plt.plot(predicted_values, color='red', label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

