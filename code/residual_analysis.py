import numpy as np
import matplotlib.pyplot as plt

# Load predictions and true values
ensemble_predictions = np.load('ensemble_predictions.npy')
data = np.load('/path/to/preprocessed_data.npz')
y_test = data['y_test']

# Calculate residuals
residuals = y_test - ensemble_predictions

# Plot residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Analysis')
plt.xlabel('Data Points')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Histogram of residuals
plt.hist(residuals, bins=50)
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

