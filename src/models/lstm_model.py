import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm_data(series, lookback=5):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)

def train_and_forecast_lstm(series, forecast_steps=30, lookback=5):
    series = np.array(series, dtype=float)
    X, y = prepare_lstm_data(series, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, activation='relu', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train quietly
    model.fit(X, y, epochs=20, batch_size=8, verbose=0)

    # Recursive forecasting
    input_seq = series[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(forecast_steps):
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(next_pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

    return np.array(preds)
