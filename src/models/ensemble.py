import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from models.lstm_model import train_and_forecast_lstm

# -----------------------------
# SIR MODEL
# -----------------------------
def sir_simulate(confirmed, beta=0.3, gamma=0.1, population=1_000_000):
    I0 = confirmed[0]
    R0 = 0
    S0 = population - I0 - R0
    S, I, R = [S0], [I0], [R0]
    N = population

    for _ in range(len(confirmed)):
        new_infected = beta * S[-1] * I[-1] / N
        new_recovered = gamma * I[-1]
        S_next = S[-1] - new_infected
        I_next = I[-1] + new_infected - new_recovered
        R_next = R[-1] + new_recovered
        S.append(S_next)
        I.append(I_next)
        R.append(R_next)

    return np.array(I[1:])

# -----------------------------
# MAIN ENSEMBLE FORECAST FUNCTION
# -----------------------------
def get_ensemble_forecast(country_data, forecast_days=30):
    series = country_data['Confirmed'].values.astype(float)
    dates = pd.to_datetime(country_data['Date'])

    # --- ARIMA ---
    try:
        arima_model = ARIMA(series, order=(2, 1, 2)).fit()
        arima_forecast = arima_model.forecast(steps=forecast_days)
    except Exception as e:
        print("ARIMA failed:", e)
        arima_forecast = np.zeros(forecast_days)

    # --- Prophet ---
    try:
        df = pd.DataFrame({'ds': dates, 'y': series})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(df)
        future = prophet_model.make_future_dataframe(periods=forecast_days)
        prophet_forecast = prophet_model.predict(future)['yhat'][-forecast_days:].values
    except Exception as e:
        print("Prophet failed:", e)
        prophet_forecast = np.zeros(forecast_days)

    # --- SIR ---
    try:
        sir_result = sir_simulate(series, beta=0.3, gamma=0.1)
        sir_forecast = np.pad(
            sir_result[-forecast_days:],
            (max(0, forecast_days - len(sir_result)), 0)
        )
    except Exception as e:
        print("SIR failed:", e)
        sir_forecast = np.zeros(forecast_days)

    # --- LSTM ---
    try:
        lstm_forecast = train_and_forecast_lstm(series, forecast_steps=forecast_days)
    except Exception as e:
        print("LSTM failed:", e)
        lstm_forecast = np.zeros(forecast_days)

    # --- Replace NaN values ---
    preds = [np.nan_to_num(p, nan=0.0) for p in [arima_forecast, prophet_forecast, sir_forecast, lstm_forecast]]
    model_names = ["ARIMA", "Prophet", "SIR", "LSTM"]

    # --- RMSE + Weighted Ensemble ---
    def safe_rmse(y_true, y_pred):
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            return 1e6
        return np.sqrt(mean_squared_error(y_true[-min_len:], y_pred[-min_len:]))

    rmses = [safe_rmse(series, p) for p in preds]
    weights = np.array([1 / (r + 1e-6) for r in rmses])
    weights /= weights.sum()
    ensemble = np.average(preds, axis=0, weights=weights)

    # --- Forecast Dates ---
    forecast_dates = pd.date_range(
        start=dates.iloc[-1],
        periods=forecast_days + 1,
        freq='D'
    )[1:]

    # --- Peak detection ---
    peak_idx = int(np.argmax(ensemble))
    peak_date = forecast_dates[peak_idx]
    peak_value = ensemble[peak_idx]

    # --- Final return ---
    return {
        "Predicted": ensemble,                # for plotting
        "Dates": forecast_dates,              # corresponding dates
        "ARIMA": arima_forecast,
        "Prophet": prophet_forecast,
        "SIR": sir_forecast,
        "LSTM": lstm_forecast,
        "Weights": dict(zip(model_names, np.round(weights, 3))),
        "RMSE": dict(zip(model_names, np.round(rmses, 2))),
        "PeakDate": peak_date,                # Timestamp
        "PeakValue": float(peak_value)
    }
