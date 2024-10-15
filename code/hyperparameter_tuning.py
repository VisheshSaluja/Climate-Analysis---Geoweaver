import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

# Load preprocessed data
data = np.load('/Users/vishesh/gw-workspace/L6gDfkNJofh8/preprocessed_data.npz')
X_train = data['X_train']
y_train = data['y_train']

# Define the model creation function for Optuna
from tensorflow.keras.layers import Input

def create_model(trial):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))  # Input layer added here
    model.add(LSTM(units=trial.suggest_int('units', 16, 128, log=True), return_sequences=True))
    model.add(Dropout(trial.suggest_float('dropout_rate', 0.1, 0.5)))
    model.add(LSTM(units=trial.suggest_int('units_l2', 16, 128, log=True), return_sequences=False))
    model.add(Dropout(trial.suggest_float('dropout_rate_l2', 0.1, 0.5)))
    model.add(Dense(1, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


# Cross-validation for Optuna
def objective(trial):
    kfold = KFold(n_splits=5)
    fold_mae = []

    # Create model once for the trial, then use for each fold
    model = create_model(trial)
    
    for train_index, val_index in kfold.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        model.fit(X_fold_train, y_fold_train, epochs=10, batch_size=32, verbose=0)
        predictions = model.predict(X_fold_val)
        mae = mean_absolute_error(y_fold_val, predictions)
        fold_mae.append(mae)

    return np.mean(fold_mae)


# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

print(f"Best Hyperparameters: {best_params}")




import matplotlib.pyplot as plt
import optuna

# Assuming `study` is the Optuna study object obtained after optimization
# Create a plot of the optimization history
fig = optuna.visualization.plot_optimization_history(study)
plt.title('Optuna Optimization History')
plt.xlabel('Trial')
plt.ylabel('Objective Value (Mean Absolute Error)')
plt.show()

# Create a plot for the importance of hyperparameters
fig = optuna.visualization.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.show()

