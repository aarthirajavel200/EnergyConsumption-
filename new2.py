import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
try:
    from pmdarima import auto_arima
except ImportError:
    st.error("pmdarima module not found. Install it using: pip install pmdarima")

# Custom Styling
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f4; }
        .main-title { color: #4A90E2; text-align: center; }
        .stMarkdown { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>TNEB Bill Prediction</h1>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📂 Upload your energy consumption dataset (CSV)", type=["csv"], help="Ensure your file contains 'date' and 'energyconsumption' columns.")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    st.write("### 🔍 Data Preview")
    st.write(df.head())

    if 'date' in df.columns and 'energyconsumption' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        if df['energyconsumption'].isna().sum() > 0 or not np.isfinite(df['energyconsumption']).all():
            st.error("Dataset contains NaN or infinite values. Please clean the data before proceeding.")
        else:
            df['energyconsumption'].fillna(method='ffill', inplace=True)
            df['month'] = df.index.month
            df['year'] = df.index.year
            energy_series = df['energyconsumption'].resample('M').sum()

            selected_month = st.selectbox("📅 Select a Month for Comparison", list(range(1, 13)), format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
            future_year = st.slider("🔮 Select Year for Prediction", 2025, 2030, 2025)
            
            train_data = energy_series[-min(len(energy_series), 36):]  # Adjust for available months
            
            if len(train_data) >= 6:  # Minimum required for ARIMA
                try:
                    best_model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
                    st.write(f"✅ Optimal ARIMA Order: {best_model.order}")
                    model = ARIMA(train_data, order=best_model.order)
                    model_fit = model.fit()
                    steps = (future_year - 2024) * 12 + selected_month
                    forecast_results = model_fit.get_forecast(steps=steps)
                    forecast_values = forecast_results.predicted_mean.clip(lower=0)
                except Exception as e:
                    st.warning(f"⚠️ ARIMA model failed: {e}. Using Moving Average as fallback.")
                    forecast_values = pd.Series(train_data.rolling(window=min(3, len(train_data))).mean().iloc[-1], index=[future_year])
            else:
                st.warning("⚠️ Not enough data for ARIMA. Using Moving Average as fallback.")
                forecast_values = pd.Series(train_data.rolling(window=min(3, len(train_data))).mean().iloc[-1], index=[future_year])

            predicted_month = f"{pd.to_datetime(str(selected_month), format='%m').strftime('%B')} {future_year}"
            forecast_df = pd.DataFrame({
                'Year': [2023, 2024, future_year],
                'Energy Consumption': [
                    max(df[(df['month'] == selected_month) & (df['year'] == 2023)]['energyconsumption'].sum(), 0),
                    max(df[(df['month'] == selected_month) & (df['year'] == 2024)]['energyconsumption'].sum(), 0),
                    forecast_values.iloc[-1] if not forecast_values.empty else 0
                ]
            })

            st.markdown(f"## 📊 Predicted Energy Consumption for {predicted_month}")
            st.dataframe(forecast_df, use_container_width=True)

            # TNEB Bill Calculation
            rate_per_unit = st.number_input("💰 Enter TNEB Rate per Unit (₹)", value=6.0)
            forecast_df['Estimated Bill'] = forecast_df['Energy Consumption'] * rate_per_unit
            st.write("### 💡 Estimated TNEB Bills (₹)")
            st.dataframe(forecast_df[['Year', 'Estimated Bill']], use_container_width=True)

            # Enhanced Pie Chart
            st.write("### 📊 Energy Consumption Distribution")
            if (forecast_df['Energy Consumption'] > 0).any():
                plt.figure(figsize=(8, 8))
                sns.set_palette("pastel")
                plt.pie(forecast_df['Energy Consumption'], labels=forecast_df['Year'], autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99'], startangle=140)
                plt.title(f"Energy Consumption for {pd.to_datetime(str(selected_month), format='%m').strftime('%B')}")
                st.pyplot(plt)
            else:
                st.warning("⚠️ No valid energy consumption data available for visualization.")
    else:
        st.error("❌ Dataset must contain 'date' and 'energyconsumption' columns.")