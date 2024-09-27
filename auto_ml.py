import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from pytorch_tabnet.tab_model import TabNetRegressor

# Load preprocessed data
data_path = 'add path to .npz file'
data = np.load(data_path)

# List all variable names in the .npz file to identify contents
print("Variables in the .npz file:", data.files)

# Load data based on contents of the .npz file
X_train = data['X_train']
y_train = data['y_train']

# Check if 'X_val' and 'y_val' exist; if not, create validation split
if 'X_val' in data.files and 'y_val' in data.files:
    X_val = data['X_val']
    y_val = data['y_val']
else:
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize a dictionary to store model performance
model_performance = {}

# Define the function to create various models
def create_model(trial):
    input_shape = X_train.shape[1]
    model_type = trial.suggest_categorical('model_type', ['dense', 'cnn', 'lstm', 'transformer', 'tabnet'])

    if model_type == 'dense':
        model = Sequential()
        model.add(Input(shape=(input_shape,)))  # Use Input layer to specify input shape
        num_layers = trial.suggest_int('num_layers', 1, 5)
        for i in range(num_layers):
            num_units = trial.suggest_int(f'num_units_l{i}', 16, 128, log=True)
            model.add(Dense(num_units, activation='relu'))
            dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.1, 0.5)
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'cnn':
        model = Sequential()
        model.add(Input(shape=(input_shape, 1)))  # Use Input layer for CNN input shape
        model.add(Conv1D(filters=trial.suggest_int('filters', 16, 64, log=True),
                         kernel_size=trial.suggest_int('kernel_size', 3, 5),
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'lstm':
        model = Sequential()
        model.add(Input(shape=(input_shape, 1)))  # Use Input layer for LSTM input shape
        model.add(LSTM(units=trial.suggest_int('lstm_units', 16, 64, log=True)))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'transformer':
        inputs = Input(shape=(input_shape, 1))
        attention = MultiHeadAttention(num_heads=trial.suggest_int('num_heads', 2, 8), key_dim=trial.suggest_int('key_dim', 16, 64))(inputs, inputs)
        attention = Add()([inputs, attention])
        attention = LayerNormalization(epsilon=1e-6)(attention)
        outputs = Flatten()(attention)
        outputs = Dense(1, activation='linear')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
    
    elif model_type == 'tabnet':
        tabnet_model = TabNetRegressor(
            n_d=trial.suggest_int('n_d', 8, 64),
            n_a=trial.suggest_int('n_a', 8, 64),
            n_steps=trial.suggest_int('n_steps', 3, 10),
            gamma=trial.suggest_float('gamma', 1.0, 2.0),
            lambda_sparse=trial.suggest_float('lambda_sparse', 1e-6, 1e-3)
        )
        tabnet_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )
        return tabnet_model

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

# Define the objective function for Optuna
def objective(trial):
    model = create_model(trial)

    if isinstance(model, TabNetRegressor):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
    else:
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
        _, mae = model.evaluate(X_val, y_val, verbose=0)

    model_type = trial.params['model_type']
    model_performance[model_type] = mae
    return mae

# Run the Bayesian optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Set n_trials as needed

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Display performance of each model
print("Model performance (MAE):")
for model_type, mae in model_performance.items():
    print(f"{model_type}: {mae}")

# Plot the model performance
plt.figure(figsize=(10, 5))
plt.bar(model_performance.keys(), model_performance.values(), color='skyblue')
plt.xlabel('Model Type')
plt.ylabel('Mean Absolute Error')
plt.title('Model Performance Comparison')
plt.show()
