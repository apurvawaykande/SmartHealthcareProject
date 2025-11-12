# app.py
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
from symptom_data import get_advice_for_symptom  # custom symptom logic
from chatbot import get_chat_response
from deep_translator import GoogleTranslator




import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ensemble import get_ensemble_forecast


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Healthcare Assistant", layout="wide")
st.title("🧠 Smart Healthcare Assistant")
st.markdown("Your AI-powered health advisor combining AI chat and hybrid epidemic forecasting.")

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.subheader("💬 Chat Settings")
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []

# Language selection
language_option = st.sidebar.selectbox(
    "Select Language",
    ["English", "Hindi", "Marathi", "Gujarati", "Tamil", "Telugu",
        "Kannada", "Malayalam", "Bengali", "Punjabi", "Odia",
        "Spanish", "French", "German", "Chinese"]
)

LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Odia": "or",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN"
}
target_lang = LANG_CODES.get(language_option, "en")

# -----------------------------
# CHAT INTERFACE
# -----------------------------
chat_container = st.container()

def display_chat():
    """Display conversation with styling"""
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f'<div style="background-color:#0B93F6; color:white; padding:10px; border-radius:10px; margin:5px 0 5px 30%; text-align:right;">'
                f'<b>{sender}:</b> {message}</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#E5E5EA; color:black; padding:10px; border-radius:10px; margin:5px 30% 5px 0; text-align:left;">'
                f'<b>{sender}:</b> {message}</div>', unsafe_allow_html=True
            )
    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# Display chat history first
with chat_container:
    display_chat()

# -----------------------------
# INPUT BOX WITH AUTO-CLEAR
# -----------------------------
def handle_user_input():
    user_text = st.session_state.user_input.strip()
    if user_text:
        # Add user message
        st.session_state.chat_history.append(("You", user_text))

        # Get bot reply
        bot_reply = get_chat_response(
            user_text,
            chat_history=st.session_state.chat_history,
            target_lang=target_lang
        )
        st.session_state.chat_history.append(("Bot", bot_reply))

        # Clear the input box safely
        st.session_state.user_input = ""

# Text input box that calls handle_user_input() automatically on change
st.text_input(
    "Type your question here:",
    key="user_input",
    on_change=handle_user_input,
)


# -----------------------------
# EPIDEMIC FORECASTING (Hybrid SIR + ARIMA)
# -----------------------------
st.subheader("📈 Epidemic Forecasting Simulation (Real Data)")

@st.cache_data
def load_covid_data():
    """
    Try to fetch remote CSV. If network error occurs, attempt to load a local cached copy
    (csv_cache/countries-aggregated.csv). If neither is available, raise an informative error.
    """
    url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    local_cache_dir = os.path.join(os.path.dirname(__file__), "csv_cache")
    os.makedirs(local_cache_dir, exist_ok=True)
    local_path = os.path.join(local_cache_dir, "countries-aggregated.csv")

    try:
        df = pd.read_csv(url, parse_dates=["Date"], dayfirst=False)
        # save a local cached copy for offline use
        try:
            df.to_csv(local_path, index=False)
        except Exception:
            pass
        return df
    except Exception as e:
        # try to load local cached copy
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path, parse_dates=["Date"])
                st.warning("⚠️ Could not fetch remote COVID data — using local cached copy.")
                return df
            except Exception as e_local:
                raise RuntimeError(f"Failed loading both remote and local COVID data: {e_local}") from e_local
        else:
            raise RuntimeError(f"Failed to fetch COVID data from the internet: {e}. Consider running once with network access or placing a cached CSV at: {local_path}") from e

@st.cache_data
def load_population_data():
    """
    Load population dataset (attempt remote, then local cache). Returns dataframe with columns:
    Country, Population (numeric)
    """
    url = "https://raw.githubusercontent.com/datasets/population/master/data/population.csv"
    local_cache_dir = os.path.join(os.path.dirname(__file__), "csv_cache")
    os.makedirs(local_cache_dir, exist_ok=True)
    local_path = os.path.join(local_cache_dir, "population.csv")

    try:
        df = pd.read_csv(url)
        try:
            df.to_csv(local_path, index=False)
        except Exception:
            pass
    except Exception:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            st.warning("⚠️ Could not fetch remote population data — using local cached copy.")
        else:
            raise RuntimeError("Failed to load population data from the internet and no local cache found.")

    # use the most recent year available per the dataset
    latest_year = int(df['Year'].max())
    df = df[df['Year'] == latest_year]
    df = df[['Country Name', 'Value']].copy()
    df.columns = ['Country', 'Population']
    # ensure numeric
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    return df

# Load and merge datasets (errors will show in Streamlit)
try:
    covid_data = load_covid_data()
    pop_data = load_population_data()
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# Many datasets use slightly different country names; merge best-effort
covid_data = covid_data.rename(columns={"Country": "Country", "Confirmed": "Confirmed", "Recovered": "Recovered", "Deaths": "Deaths", "Date": "Date"})
merged = covid_data.merge(pop_data, on="Country", how="left")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.subheader("Simulation Parameters (Real Data)")

country = st.sidebar.selectbox("Select Country", merged['Country'].unique())

# Streamlit date_input returns datetime.date objects.
# Convert them immediately to pandas Timestamp (pd.to_datetime) so comparisons work with pandas Timestamps.
start_date_input = st.sidebar.date_input("Start Date", merged['Date'].min().date())
end_date_input = st.sidebar.date_input("End Date", merged['Date'].max().date())

# Convert to pandas Timestamp (this avoids Timestamp vs datetime.date comparison errors)
start_date = pd.to_datetime(start_date_input)
end_date = pd.to_datetime(end_date_input)

# ⚠️ Validate date range (realistic pandemic window)
pandemic_start = pd.to_datetime("2020-01-01")
pandemic_end = pd.to_datetime("2022-12-31")

if start_date < pandemic_start or end_date > pandemic_end:
    st.warning("⚠️ Please select a realistic COVID-19 period (Jan 1, 2020 – Dec 31, 2022).")

country_data = merged[
    (merged['Country'] == country) &
    (merged['Date'] >= start_date) &
    (merged['Date'] <= end_date)
].copy()

if country_data.empty:
    st.error("❌ No data found for the selected range. Please adjust your date range.")
    st.stop()

# Ensure numeric columns and fill missing sensibly
for col in ['Confirmed', 'Recovered', 'Deaths']:
    if col not in country_data.columns:
        country_data[col] = 0
country_data['Confirmed'] = pd.to_numeric(country_data['Confirmed'], errors='coerce').fillna(0)
country_data['Recovered'] = pd.to_numeric(country_data['Recovered'], errors='coerce').fillna(0)
country_data['Population'] = pd.to_numeric(country_data['Population'], errors='coerce')

# Auto population value (with fallback)
pop_val = country_data['Population'].iloc[0] if not pd.isna(country_data['Population'].iloc[0]) else np.nan
if pd.isna(pop_val) or pop_val <= 0:
    # try to get from pop_data directly
    try:
        pop_val = int(pop_data.loc[pop_data['Country'] == country, 'Population'].iloc[0])
    except Exception:
        pop_val = 1_000_000  # fallback default
population = int(pop_val)
st.sidebar.info(f"🧍 Population (N): {population:,}")

I0 = max(1, int(country_data['Confirmed'].iloc[0]))  # avoid 0 infections
R0 = int(country_data['Recovered'].iloc[0]) if not pd.isna(country_data['Recovered'].iloc[0]) else 0
S0 = max(0, population - I0 - R0)
days = len(country_data)

beta = st.sidebar.slider("Transmission Rate (β)", 0.0, 1.0, 0.3, step=0.01)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.0, 1.0, 0.1, step=0.01)

# Validation checks
if beta == 0 or gamma == 0:
    st.sidebar.warning("⚠️ Beta and Gamma should be > 0 to simulate properly.")
if population < I0 + R0:
    st.sidebar.error("⚠️ Population must be greater than I₀ + R₀.")

# -----------------------------
# SIR MODEL FUNCTION
# -----------------------------
def sir_model(S0, I0, R0, beta, gamma, days):
    S, I, R = [float(S0)], [float(I0)], [float(R0)]
    N = S0 + I0 + R0
    for _ in range(days):
        new_infected = beta * S[-1] * I[-1] / N if N > 0 else 0
        new_recovered = gamma * I[-1]
        S_next = S[-1] - new_infected
        I_next = I[-1] + new_infected - new_recovered
        R_next = R[-1] + new_recovered

        # prevent invalid values
        if np.isnan(S_next) or np.isnan(I_next) or np.isnan(R_next):
            S_next, I_next, R_next = S[-1], I[-1], R[-1]

        S_next = min(max(S_next, 0), N)
        I_next = min(max(I_next, 0), N)
        R_next = min(max(R_next, 0), N)

        S.append(S_next)
        I.append(I_next)
        R.append(R_next)
    return S, I, R

S, I, R = sir_model(S0, I0, R0, beta, gamma, days)

# -----------------------------
# ARIMA FORECAST (AI Trend)
# -----------------------------
st.sidebar.subheader("AI Trend Adjustment (Optional)")
use_ai_trend = st.sidebar.checkbox("Apply ARIMA adjustment to infected cases?", value=False)
forecast_days = int(st.sidebar.number_input("Future days to forecast", min_value=1, max_value=60, value=30))

if use_ai_trend:
    try:
        # use Confirmed series, forward-fill missing values
        series = country_data['Confirmed'].fillna(method='ffill').values
        if len(series) < 5:
            raise ValueError("Not enough points for ARIMA; need at least 5 data points.")
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        future_forecast = model_fit.forecast(steps=forecast_days)
        I_forecast = np.concatenate([series, future_forecast])
        full_days = range(len(I_forecast))
        forecast_start_date = country_data['Date'].iloc[-1]
        st.info(f"📆 Forecast starts from: {forecast_start_date.date()}")
        st.sidebar.success(f"✅ ARIMA forecast extended for {forecast_days} future days.")
    except Exception as e:
        st.sidebar.error(f"ARIMA failed: {e}")
        I_forecast = country_data['Confirmed'].values
        full_days = range(len(I_forecast))
else:
    I_forecast = country_data['Confirmed'].values
    full_days = range(len(I_forecast))

# -----------------------------
# PLOTTING
# -----------------------------
fig, ax = plt.subplots()
ax.plot(range(days), S[:-1], label="Susceptible (SIR)")
ax.plot(full_days, I_forecast, label="Infected (AI Forecast)" if use_ai_trend else "Infected")
ax.plot(range(days), R[:-1], label="Recovered (SIR)")
ax.axvline(x=days, color='gray', linestyle='--', label="Forecast start")
ax.set_xlabel("Days")
ax.set_ylabel("Number of People")
ax.set_title(f"Hybrid Epidemic Simulation + {forecast_days}-Day Forecast for {country}")
ax.legend()
st.pyplot(fig)

# -----------------------------
# SUMMARY SECTION
# -----------------------------
total_infected = int(max(I_forecast)) if len(I_forecast) > 0 else 0
peak_day = int(np.argmax(I_forecast)) if len(I_forecast) > 0 else 0
peak_date = country_data['Date'].iloc[0] + pd.to_timedelta(peak_day, unit='D')
total_recovered = int(R[-1]) if not np.isnan(R[-1]) else 0

if use_ai_trend:
    initial_cases = int(country_data['Confirmed'].values[-1])
    final_cases = int(I_forecast[-1])
    change = final_cases - initial_cases

    if change > 0:
        trend = f"an **increase** of approximately **{abs(change):,} cases**."
    elif change < 0:
        trend = f"a **decrease** of approximately **{abs(change):,} cases**."
    else:
        trend = "no major change in infection numbers."

    st.markdown(f"""
    ### 📊 Summary of Forecast (Hybrid AI + Epidemiological Model)
    - 🧠 **AI Forecast Applied:** ARIMA model extended the prediction by **{forecast_days} days**.  
    - 📆 **Forecast starts from:** {forecast_start_date.date()}.  
    - 🦠 **Peak infection:** Around **{peak_date.date()}** with **{total_infected:,} active cases**.  
    - 💪 **Estimated total recoveries:** {total_recovered:,}.  
    - 📈 During the AI-predicted period, the model shows {trend}  
    - 🧩 The hybrid forecast visualizes near-future epidemic evolution.
    """)
else:
    st.markdown(f"""
    ### 📊 Summary of Forecast (SIR Model Only)
    - 🦠 **Peak infection:** Around **{peak_date.date()}** with **{total_infected:,} active cases**.  
    - 💪 **Estimated total recoveries:** {total_recovered:,}.  
    - 📉 The forecast shows how infections may rise and then decline as recovery increases over time.
    """)

# -----------------------------
# EXPLANATION
# -----------------------------
st.markdown("""
### 📘 Understanding the Forecast
This chart simulates how an infectious disease spreads and recovers:
- **🟦 Susceptible (S):** Healthy but at risk.
- **🟩 Infected (I):** Currently infected (AI-forecasted if applied).
- **🟧 Recovered (R):** People who have recovered.

The **dotted line** marks where **AI-based prediction** begins.
⚙️ *This hybrid model combines SIR and ARIMA for educational and research purposes — not live medical data.*
""")

# ---------------------------------------
# SECTION: Ensemble Forecasting Module
# ---------------------------------------
st.subheader("🤝 Ensemble Forecast (Hybrid LSTM + ARIMA)")

if st.button("Run Ensemble Forecast for Selected Country"):
    with st.spinner("Running ensemble forecast..."):
        try:
            # Run ensemble forecast
            forecast = get_ensemble_forecast(country_data, forecast_days=30)
            st.success("✅ Forecast complete!")

            # Peak info
            st.write("📅 Peak Infection Date:", forecast["PeakDate"].date())
            st.write("📈 Peak Value:", round(forecast["PeakValue"], 2))

            # -----------------------------
            # PLOTTING ENSEMBLE FORECAST WITH ALL MODELS
            # -----------------------------
            num_pred_days = len(forecast["Predicted"])
            forecast_dates = pd.date_range(
                start=pd.to_datetime(country_data['Date'].iloc[-1]) + pd.Timedelta(days=1),
                periods=num_pred_days,
                freq='D'
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Actual data
            ax.plot(pd.to_datetime(country_data['Date']), country_data['Confirmed'],
                    label="Confirmed (Actual)", color="black", linewidth=2)

            # Individual model predictions
            ax.plot(forecast_dates, forecast["ARIMA"], label="ARIMA Forecast", linestyle="--")
            ax.plot(forecast_dates, forecast["Prophet"], label="Prophet Forecast", linestyle="--")
            ax.plot(forecast_dates, forecast["SIR"], label="SIR Forecast", linestyle="--")
            ax.plot(forecast_dates, forecast["LSTM"], label="LSTM Forecast", linestyle="--")

            # Ensemble (highlighted)
            ax.plot(forecast_dates, forecast["Predicted"],
                    label="Ensemble Forecast (LSTM + ARIMA Hybrid)",
                    color="blue", linewidth=2.5)

            # Peak marker
            ax.axvline(x=forecast["PeakDate"], color='red', linestyle='--', label="Peak Infection")

            # Labels and styling
            ax.set_xlabel("Date")
            ax.set_ylabel("Confirmed Cases")
            ax.set_title(f"Comparison of Model Predictions and Ensemble Forecast for {country}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

            st.pyplot(fig)

            # -----------------------------
            # SUMMARY (CLEANED)
            # -----------------------------
            initial_cases = int(country_data['Confirmed'].values[-1])
            final_cases = int(forecast["Predicted"][-1])
            change = final_cases - initial_cases

            if change > 0:
                trend = f"an *increase* of approximately *{abs(change):,} cases*."
            elif change < 0:
                trend = f"a *decrease* of approximately *{abs(change):,} cases*."
            else:
                trend = "no major change in infection numbers."

            st.markdown(f"""
            ### 📊 Summary of Ensemble Forecast
            - 🧠 *Forecast applied:* Hybrid LSTM + ARIMA for *{num_pred_days} days*.  
            - 📆 *Forecast starts from:* {forecast_dates[0].date() if num_pred_days>0 else 'N/A'}.  
            - 🦠 *Peak infection:* Around *{forecast['PeakDate'].date()}* with *{round(forecast['PeakValue']):,} active cases*.  
            - 📈 During the AI-predicted period, the model shows {trend}  
            - 🧩 The ensemble forecast visualizes near-future epidemic evolution.
            """)

        except Exception as e:
            st.error(f"❌ Forecast failed: {e}")
