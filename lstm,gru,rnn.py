import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
n_samples = 1000
time_steps = 20

# Generate a simple sinusoidal time series
t = np.linspace(0, 10, n_samples, endpoint=False)
data = np.sin(t) + 0.1 * np.random.randn(n_samples)

# Create sequences of data with corresponding targets
sequences = []
targets = []
for i in range(n_samples - time_steps):
    seq = data[i : i + time_steps]
    target = data[i + time_steps]
    sequences.append(seq)
    targets.append(target)

# Convert to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Reshape the input data for RNNs
sequences = sequences.reshape(-1, time_steps, 1)

# Split the data into training and testing sets
split = int(0.8 * n_samples)
X_train, X_test = sequences[:split], sequences[split:]
y_train, y_test = targets[:split], targets[split:]

# Function to build and train the model
def build_and_train_model(model_type):
    model = Sequential()
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(50, activation="relu", input_shape=(time_steps, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(50, activation="relu", input_shape=(time_steps, 1)))
    elif model_type == "GRU":
        model.add(GRU(50, activation="relu", input_shape=(time_steps, 1)))
    else:
        raise ValueError("Invalid model type")

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    return model, history

# Train models
rnn_model, rnn_history = build_and_train_model("SimpleRNN")
lstm_model, lstm_history = build_and_train_model("LSTM")
gru_model, gru_history = build_and_train_model("GRU")

# Evaluate models on the test set
rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)
gru_pred = gru_model.predict(X_test)

# Calculate Mean Squared Error
rnn_mse = mean_squared_error(y_test, rnn_pred)
lstm_mse = mean_squared_error(y_test, lstm_pred)
gru_mse = mean_squared_error(y_test, gru_pred)

# Plot performance comparison
plt.plot(rnn_history.history["loss"], label="SimpleRNN Training Loss")
plt.plot(lstm_history.history["loss"], label="LSTM Training Loss")
plt.plot(gru_history.history["loss"], label="GRU Training Loss")
plt.legend()
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()

print(f"Mean Squared Error on Test Set:")
print(f"SimpleRNN: {rnn_mse}")
print(f"LSTM: {lstm_mse}")
print(f"GRU: {gru_mse}")
