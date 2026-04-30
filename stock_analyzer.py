import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from datetime import timedelta
from prophet import Prophet
import json
import requests
from bs4 import BeautifulSoup
import io

# Set the page configuration
st.set_page_config(
    page_title="Stock Data Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- Data Loading and Processing ---
@st.cache_data(ttl=600)
def get_stock_data_live(tickers, start_date, end_date):
    """
    Fetches live stock data from Yahoo Finance and ensures a consistent DataFrame.
    This version downloads tickers one by one to avoid multi-level column indexes.
    """
    if not tickers:
        return pd.DataFrame()

    all_data = []

    for ticker in tickers:
        try:
            # Download data for each ticker individually for robust processing
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                st.warning(f"No data available for {ticker}. Skipping.")
                continue

            # Explicitly check for and flatten multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Standardize column names
            data.rename(columns={
                'Open': 'Open Price',
                'High': 'High Price',
                'Low': 'Low Price',
                'Close': 'Close Price',
                'Volume': 'Volume Traded'
            }, inplace=True)
            
            # Reset index and add Ticker column
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            all_data.append(data)
            
        except Exception as e:
            st.error(f"Failed to download data for {ticker}: {e}. Skipping this ticker.")

    if not all_data:
        st.warning("No data found for the selected tickers. Please try again.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

# New function to handle file uploads
@st.cache_data
def load_uploaded_csv(uploaded_file):
    """Loads and preprocesses data from an uploaded CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # --- Ensure the 'Date' column is parsed correctly ---
            if 'Date' not in df.columns:
                st.error("CSV must contain a 'Date' column.")
                return None
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)

            # --- Standardize column names for compatibility ---
            column_map = {
                'Close': 'Close Price',
                'Adj Close': 'Close Price',
                'Open': 'Open Price',
                'High': 'High Price',
                'Low': 'Low Price',
                'Volume': 'Volume Traded'
            }
            df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

            # --- Check for Ticker column ---
            if 'Ticker' not in df.columns:
                st.warning("No 'Ticker' column found — assigning default ticker 'STOCK'.")
                df['Ticker'] = 'STOCK'

            # --- Final validation ---
            required_cols = {'Date', 'Ticker', 'Close Price'}
            if not required_cols.issubset(df.columns):
                st.error("CSV must contain at least 'Date', 'Ticker', and 'Close Price' columns.")
                return None

            return df

        except Exception as e:
            st.error(f"Error loading file: {e}. Please ensure it is a valid CSV with 'Date' and 'Ticker' columns.")
            return None

    return None


def calculate_metrics(df, ma_period, rsi_period, macd_fast, macd_slow, macd_signal):
    """Calculates additional metrics like daily returns, moving averages, RSI, and MACD."""
    if df.empty or 'Ticker' not in df.columns or 'Date' not in df.columns:
        return df

    df_copy = df.copy()
    
    # Ensure a consistent index before calculations
    df_copy.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    # Use a loop to calculate metrics for each ticker separately to avoid multi-indexing errors
    for ticker, group in df_copy.groupby('Ticker'):
        # Calculate Daily Return
        df_copy.loc[group.index, 'Daily Return'] = group['Close Price'].pct_change()
        
        # Calculate Moving Average
        df_copy.loc[group.index, 'Moving Average'] = group['Close Price'].rolling(window=ma_period).mean()
        
        # Calculate RSI
        def calculate_rsi(series, period):
            delta = series.diff().dropna()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df_copy.loc[group.index, 'RSI'] = calculate_rsi(group['Close Price'], rsi_period)

        # Calculate MACD and MACD Signal Line
        def calculate_macd_line(series, fast, slow):
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            return exp1 - exp2

        def calculate_macd_signal_line(series, signal):
            return series.ewm(span=signal, adjust=False).mean()

        macd_series = calculate_macd_line(group['Close Price'], macd_fast, macd_slow)
        df_copy.loc[group.index, 'MACD'] = macd_series
        df_copy.loc[group.index, 'MACD Signal'] = calculate_macd_signal_line(macd_series, macd_signal)
        
    return df_copy

# --- Main App Logic ---
st.title("Interactive Stock Data Dashboard 📈")
st.markdown("""
This dashboard provides advanced analysis tools using either a local dataset or live data.
Use the sidebar on the left to select stocks, date ranges, and analysis options.
""")

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

use_live_data = st.sidebar.checkbox("Use Live Data", value=False)

if use_live_data:
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 'PG', 'KO']
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers:",
        options=all_tickers,
        default=['AAPL', 'MSFT']
    )
    today = pd.to_datetime('today').date()
    start_date = st.sidebar.date_input("Start Date:", value=today - pd.DateOffset(months=6))
    end_date = st.sidebar.date_input("End Date:", value=today)

    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        st.stop()
    
    with st.spinner("Fetching live data..."):
        df = get_stock_data_live(selected_tickers, start_date, end_date)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
    
    df = load_uploaded_csv(uploaded_file)
    if df is None:
        st.info("Please upload a CSV file or check 'Use Live Data' to get started.")
        st.stop()
        
    all_tickers = df['Ticker'].unique().tolist()
    selected_tickers = st.sidebar.multiselect("Select Tickers:", options=all_tickers, default=all_tickers[:1])

    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        st.stop()

    df = df[df['Ticker'].isin(selected_tickers)]
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = df['Date'].min().to_pydatetime().date()
    end_date = df['Date'].max().to_pydatetime().date()

ma_period = st.sidebar.slider("Moving Average Period:", min_value=5, max_value=200, value=20)
rsi_period = st.sidebar.slider("RSI Period:", min_value=5, max_value=30, value=14)
macd_fast = st.sidebar.slider("MACD Fast Period:", min_value=5, max_value=20, value=12)
macd_slow = st.sidebar.slider("MACD Slow Period:", min_value=20, max_value=50, value=26)
macd_signal = st.sidebar.slider("MACD Signal Period:", min_value=5, max_value=20, value=9)
prediction_days = st.sidebar.slider("Prediction Period (Days):", min_value=1, max_value=30, value=7)

df = calculate_metrics(df, ma_period, rsi_period, macd_fast, macd_slow, macd_signal)

if df.empty:
    st.warning("No data found for the selected tickers or date range. Please adjust your selections.")
    st.stop()

# --- Visualization Layout ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Price & Volume Trends",
    "Price Charts and Forecast",
    "Technical Analysis",
    "Stock Comparison",
    "Portfolio Tracker",
    "Key Statistics",
    "Time Series & Model Accuracy",
    "Risk & Volatility",
    "News Sentiment Analysis",
    "Backtesting Engine",
    "Financial Report"
])

# --- Prophet Model Function ---
def run_prophet_model(df, prediction_days, selected_ticker):
    """
    Runs the Prophet model on the selected stock data and returns the forecast and evaluation metrics.
    """
    prophet_df = df[df['Ticker'] == selected_ticker][['Date', 'Close Price']].copy()
    
    if prophet_df.empty or len(prophet_df) < 10: 
        return None, None
        
    prophet_df.rename(columns={'Date': 'ds', 'Close Price': 'y'}, inplace=True)
    
    # Initialize and fit the Prophet model
    m = Prophet(interval_width=0.95)
    m.fit(prophet_df)
    
    # Create a future dataframe for predictions
    future = m.make_future_dataframe(periods=prediction_days)
    forecast = m.predict(future)
    
    # Evaluate model performance using historical data
    train_df = prophet_df.iloc[:int(0.8 * len(prophet_df))]
    test_df = prophet_df.iloc[int(0.8 * len(prophet_df)):]
    
    m_test = Prophet()
    m_test.fit(train_df)
    future_test = m_test.make_future_dataframe(periods=len(test_df))
    forecast_test = m_test.predict(future_test)
    
    test_pred = forecast_test[['ds', 'yhat']].tail(len(test_df))
    test_pred.rename(columns={'ds': 'Date', 'yhat': 'Predicted'}, inplace=True)
    test_df.rename(columns={'ds': 'Date', 'y': 'Actual'}, inplace=True)
    
    comparison_df = pd.merge(test_df, test_pred, on='Date')
    rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Predicted']))

    return forecast, rmse

# --- Random Forest Model Function ---
def run_random_forest_model(df, prediction_days, selected_ticker):
    """
    Runs a Random Forest Regressor model and returns the forecast and evaluation metrics.
    """
    rf_df = df[df['Ticker'] == selected_ticker][['Date', 'Close Price']].copy()

    if rf_df.empty or len(rf_df) < 10:
        st.warning("Not enough data points to generate a reliable forecast.")
        return None, None

    # Use a date as a numeric feature for the model
    rf_df['Date'] = pd.to_datetime(rf_df['Date'], errors='coerce')
    rf_df = rf_df.dropna(subset=['Date'])
    rf_df['Date_Ordinal'] = rf_df['Date'].apply(lambda date: date.toordinal())


    # Split data
    X = rf_df[['Date_Ordinal']]
    y = rf_df['Close Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the model
    st.write("Training rows:", len(X_train), " | Test rows:", len(X_test))
    st.write("Date range:", df['Date'].min(), "to", df['Date'].max())

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Generate a future forecast
    last_date = X['Date_Ordinal'].iloc[-1]
    future_dates_ordinal = np.arange(last_date + 1, last_date + 1 + prediction_days)
    future_predictions = model.predict(pd.DataFrame({'Date_Ordinal': future_dates_ordinal}))
    future_dates = [pd.Timestamp.fromordinal(int(d)) for d in future_dates_ordinal]

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })

    return forecast_df, rmse

# --- Linear Regression Model Function ---
def run_linear_regression_model(df, prediction_days, selected_ticker):
    """
    Runs a Linear Regression model and returns the forecast and evaluation metrics.
    """
    lr_df = df[df['Ticker'] == selected_ticker][['Date', 'Close Price']].copy()

    if lr_df.empty or len(lr_df) < 10:
        st.warning("Not enough data points to generate a reliable forecast.")
        return None, None

    # Use a date as a numeric feature for the model
    lr_df['Date'] = pd.to_datetime(lr_df['Date'], errors='coerce')
    lr_df = lr_df.dropna(subset=['Date'])
    lr_df['Date_Ordinal'] = lr_df['Date'].apply(lambda date: date.toordinal())


    # Split data
    X = lr_df[['Date_Ordinal']]
    y = lr_df['Close Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the model
    st.write("Training rows:", len(X_train), " | Test rows:", len(X_test))
    st.write("Date range:", df['Date'].min(), "to", df['Date'].max())

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Generate a future forecast
    last_date = X['Date_Ordinal'].iloc[-1]
    future_dates_ordinal = np.arange(last_date + 1, last_date + 1 + prediction_days)
    future_predictions = model.predict(pd.DataFrame({'Date_Ordinal': future_dates_ordinal}))
    future_dates = [pd.Timestamp.fromordinal(int(d)) for d in future_dates_ordinal]

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })

    return forecast_df, rmse

# --- NEW: Voting Regressor Model Function ---
def run_voting_regressor_model(df, prediction_days, selected_ticker):
    """
    Runs a Voting Regressor model (ensemble of Linear Regression and Random Forest)
    and returns the forecast and evaluation metrics.
    """
    vr_df = df[df['Ticker'] == selected_ticker][['Date', 'Close Price']].copy()

    if vr_df.empty or len(vr_df) < 10:
        st.warning("Not enough data points to generate a reliable forecast.")
        return None, None

    vr_df['Date'] = pd.to_datetime(vr_df['Date'], errors='coerce')
    vr_df = vr_df.dropna(subset=['Date'])
    vr_df['Date_Ordinal'] = vr_df['Date'].apply(lambda date: date.toordinal())


    X = vr_df[['Date_Ordinal']]
    y = vr_df['Close Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define base estimators
    st.write("Training rows:", len(X_train), " | Test rows:", len(X_test))
    st.write("Date range:", df['Date'].min(), "to", df['Date'].max())

    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create the Voting Regressor
    model = VotingRegressor(
        estimators=[('lr', lr_model), ('rf', rf_model)]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    last_date = X['Date_Ordinal'].iloc[-1]
    future_dates_ordinal = np.arange(last_date + 1, last_date + 1 + prediction_days)
    future_predictions = model.predict(pd.DataFrame({'Date_Ordinal': future_dates_ordinal}))
    future_dates = [pd.Timestamp.fromordinal(int(d)) for d in future_dates_ordinal]

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions
    })

    return forecast_df, rmse

# NEW: Function to generate synthetic OHLC data for predictions
def generate_synthetic_candlestick(forecast_df, last_historical_close):
    """
    Generates synthetic Open, High, and Low prices for a forecast dataframe
    based on the predicted Close price (yhat).
    """
    synthetic_df = forecast_df.copy()
    
    if synthetic_df.empty:
        return synthetic_df
    
    # Calculate a volatility-based spread for realistic looking OHLC
    # Use the spread from the last historical day or a small fixed percentage
    spread = abs(synthetic_df['yhat'].diff()).mean() * 0.5 if len(synthetic_df) > 1 else last_historical_close * 0.01

    # Open price is the previous day's close
    synthetic_df['Open Price'] = synthetic_df['yhat'].shift(1).fillna(last_historical_close)
    
    # High and Low are based on a spread around the close price
    synthetic_df['High Price'] = synthetic_df[['Open Price', 'yhat']].max(axis=1) + spread
    synthetic_df['Low Price'] = synthetic_df[['Open Price', 'yhat']].min(axis=1) - spread
    
    synthetic_df['Close Price'] = synthetic_df['yhat']

    return synthetic_df[['ds', 'Open Price', 'High Price', 'Low Price', 'Close Price']]

def calculate_directional_accuracy(actual_df, predicted_df):
    """
    Calculates the accuracy of a trading strategy by comparing predicted
    price movements with actual price movements.

    Args:
        actual_df (pd.DataFrame): The DataFrame with actual market data.
        predicted_df (pd.DataFrame): The DataFrame with predicted market data.

    Returns:
        float: The accuracy of the predictions in percentage.
               Returns None if data is not valid.
    """
    # Ensure both dataframes have the necessary columns
    required_columns = ['Open Price', 'Close Price']
    if not all(col in actual_df.columns for col in required_columns) or \
       not all(col in predicted_df.columns for col in required_columns):
        st.error("Error: Both CSV files must contain 'Open Price' and 'Close Price' columns.")
        return None

    # Determine the predicted movement: True for 'up' (Close > Open), False for 'down'
    predicted_direction = predicted_df['Close Price'] > predicted_df['Open Price']

    # Determine the actual movement: True for 'up' (Close > Open), False for 'down'
    actual_direction = actual_df['Close Price'] > actual_df['Open Price']

    # Compare the predicted and actual movements. We'll only compare the number of predictions
    # that are common to both data sets.
    min_rows = min(len(predicted_direction), len(actual_direction))
    
    # Calculate the number of correct predictions by element-wise comparison
    correct_predictions = (predicted_direction[:min_rows] == actual_direction[:min_rows]).sum()
    
    # Calculate the total number of predictions considered
    total_predictions = min_rows

    if total_predictions == 0:
        st.warning("No common data points to compare.")
        return 0.0

    # Calculate accuracy and convert to percentage
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

with tab1:
    st.header("Price & Volume Trends")
    st.markdown("This section shows the historical trends of closing prices, trading volume, and the moving average.")

    available_tickers = sorted(list(df['Ticker'].unique()))
    view_options = ["All Selected Tickers"] + available_tickers
    selected_view = st.selectbox("Select a Ticker for Detailed View:", options=view_options)
    
    display_df = df
    if selected_view != "All Selected Tickers":
        display_df = df[df['Ticker'] == selected_view]

    st.subheader("Closing Price with Moving Average")
    fig_price_ma = px.line(
        display_df,
        x='Date',
        y='Close Price',
        color='Ticker' if selected_view == "All Selected Tickers" else None,
        title="Closing Price with Moving Average",
        labels={'Close Price': 'Price ($)', 'Date': 'Date'},
        hover_data={'Open Price': ':.2f', 'High Price': ':.2f', 'Low Price': ':.2f', 'Close Price': ':.2f'}
    )
    
    if selected_view == "All Selected Tickers":
        for ticker in available_tickers:
            ticker_data = df[df['Ticker'] == ticker].set_index('Date')['Moving Average']
            fig_price_ma.add_trace(
                go.Scatter(
                    x=ticker_data.index,
                    y=ticker_data.values,
                    mode='lines',
                    name=f'{ticker} {ma_period}-Day MA',
                    line=dict(dash='dash')
                )
            )
    else:
        ma_data = display_df.set_index('Date')['Moving Average']
        fig_price_ma.add_trace(
            go.Scatter(
                x=ma_data.index,
                y=ma_data.values,
                mode='lines',
                name=f'{selected_view} {ma_period}-Day MA',
                line=dict(dash='dash')
            )
        )
        
    st.plotly_chart(fig_price_ma, use_container_width=True)
    

    st.subheader("Daily Volume Traded")
    fig_volume = px.bar(
        display_df,
        x='Date',
        y='Volume Traded',
        color='Ticker' if selected_view == "All Selected Tickers" else None,
        title="Daily Volume Traded",
        labels={'Volume Traded': 'Volume', 'Date': 'Date'}
    )
    st.plotly_chart(fig_volume, use_container_width=True)

with tab2:
    st.header("Price Charts and Forecasts")
    st.markdown("This chart visualizes historical and forecasted price data. You can choose to display the forecast as a simple line or a synthetic candlestick chart.")
    
    selected_ticker_chart = st.selectbox(
        "Select a Ticker:",
        options=sorted(list(df['Ticker'].unique())),
        key="chart_select"
    )

    if selected_ticker_chart:
        historical_df = df[df['Ticker'] == selected_ticker_chart].copy()
        forecast, rmse = run_prophet_model(df, prediction_days, selected_ticker_chart)

        if historical_df.empty or forecast is None:
            st.warning("Not enough data to display the chart.")
        else:
            show_predicted_candlestick = st.checkbox("Show Predicted Candlestick Chart", value=False)
            
            fig_combined = go.Figure()
            
            # Plot historical candlestick chart
            fig_combined.add_trace(go.Candlestick(
                x=historical_df['Date'],
                open=historical_df['Open Price'],
                high=historical_df['High Price'],
                low=historical_df['Low Price'],
                close=historical_df['Close Price'],
                name='Historical Price'
            ))

            if show_predicted_candlestick:
                synthetic_df = generate_synthetic_candlestick(forecast, historical_df['Close Price'].iloc[-1])
                fig_combined.add_trace(go.Candlestick(
                    x=synthetic_df['ds'],
                    open=synthetic_df['Open Price'],
                    high=synthetic_df['High Price'],
                    low=synthetic_df['Low Price'],
                    close=synthetic_df['Close Price'],
                    name='Predicted Price (Synthetic)'
                ))
                st.info("The predicted candlestick chart uses synthetic Open, High, and Low values based on the forecasted closing price. It is for visual representation only.")
            else:
                # Plot predicted line chart
                fig_combined.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(dash='dash', color='blue')
                ))

            fig_combined.update_layout(
                title=f'Historical Price and Prophet Forecast for {selected_ticker_chart}',
                yaxis_title='Stock Price',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            # Display forecast table below chart
            if forecast is not None:
                st.subheader(f"📊 Forecasted Prices for {selected_ticker_chart}")
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days)
                forecast_table.rename(
                columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'},
                inplace=True
    )
                forecast_table['Date'] = pd.to_datetime(forecast_table['Date']).dt.date
                st.dataframe(forecast_table, use_container_width=True)
            else:
                st.info("Forecast data is not available to display as a table.")


with tab3:
    st.header("Technical Analysis")
    st.markdown("This section shows common technical indicators.")
    
    selected_ticker_tech = st.selectbox(
        "Select a Ticker for Technical Indicators:",
        options=sorted(list(df['Ticker'].unique())),
        key="tech_select"
    )

    if selected_ticker_tech:
        tech_df = df[df['Ticker'] == selected_ticker_tech]
        
        st.subheader("MACD (Moving Average Convergence Divergence)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=tech_df['Date'], y=tech_df['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=tech_df['Date'], y=tech_df['MACD Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
        fig_macd.update_layout(title=f'MACD for {selected_ticker_tech}', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_macd, use_container_width=True)
        
        st.subheader("RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=tech_df['Date'], y=tech_df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.add_trace(go.Scatter(x=tech_df['Date'], y=[70] * len(tech_df), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
        fig_rsi.add_trace(go.Scatter(x=tech_df['Date'], y=[30] * len(tech_df), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
        fig_rsi.update_layout(title=f'RSI for {selected_ticker_tech}', yaxis_range=[0, 100], xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.subheader("Alerts")
        if st.button("Check for RSI Alerts"):
            if not tech_df.empty:
                current_rsi = tech_df['RSI'].iloc[-1]
                if current_rsi > 70:
                    st.warning(f"🚨 **Alert:** The RSI for {selected_ticker_tech} is {current_rsi:.2f}, indicating it may be **overbought**.")
                elif current_rsi < 30:
                    st.warning(f"🚨 **Alert:** The RSI for {selected_ticker_tech} is {current_rsi:.2f}, indicating it may be **oversold**.")
                else:
                    st.info("The RSI is currently within the normal range.")
            

with tab4:
    st.header("Stock Comparison")
    st.markdown("Compare the performance of two different stocks side-by-side.")
    col1, col2 = st.columns(2)
    
    available_tickers_comp = sorted(list(df['Ticker'].unique()))
    with col1:
        ticker1 = st.selectbox("Select First Ticker:", options=available_tickers_comp, key="comp1")
    with col2:
        ticker2 = st.selectbox("Select Second Ticker:", options=available_tickers_comp, key="comp2")
    
    if ticker1 and ticker2 and ticker1 != ticker2:
        comp_df = df[df['Ticker'].isin([ticker1, ticker2])].copy()
        
        st.subheader("Price Comparison")
        fig_comp_price = go.Figure()
        fig_comp_price.add_trace(go.Scatter(x=comp_df[comp_df['Ticker'] == ticker1]['Date'], y=comp_df[comp_df['Ticker'] == ticker1]['Close Price'], mode='lines', name=ticker1))
        fig_comp_price.add_trace(go.Scatter(x=comp_df[comp_df['Ticker'] == ticker2]['Date'], y=comp_df[comp_df['Ticker'] == ticker2]['Close Price'], mode='lines', name=ticker2))
        fig_comp_price.update_layout(title=f'Price Comparison: {ticker1} vs {ticker2}', yaxis_title='Price ($)')
        st.plotly_chart(fig_comp_price, use_container_width=True)
        
        st.subheader("Daily Return Comparison")
        fig_comp_returns = go.Figure()
        fig_comp_returns.add_trace(go.Scatter(x=comp_df[comp_df['Ticker'] == ticker1]['Date'], y=comp_df[comp_df['Ticker'] == ticker1]['Daily Return'], mode='lines', name=ticker1))
        fig_comp_returns.add_trace(go.Scatter(x=comp_df[comp_df['Ticker'] == ticker2]['Date'], y=comp_df[comp_df['Ticker'] == ticker2]['Daily Return'], mode='lines', name=ticker2))
        fig_comp_returns.update_layout(title=f'Daily Return Comparison: {ticker1} vs {ticker2}', yaxis_title='Daily Return')
        st.plotly_chart(fig_comp_returns, use_container_width=True)

        # New: Correlation Heatmap
        st.subheader("Correlation Heatmap")
        returns_df = df[df['Ticker'].isin(selected_tickers)].pivot_table(index='Date', columns='Ticker', values='Daily Return')
        correlation_matrix = returns_df.corr()
        
        fig_corr = px.imshow(correlation_matrix, 
                             text_auto=True, 
                             aspect="auto",
                             color_continuous_scale='RdBu_r',
                             title='Correlation Heatmap of Daily Returns')
        st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.info("Please select two different tickers to compare.")

with tab5:
    st.header("Portfolio Tracker")
    st.markdown("Build a mock portfolio and track its performance with live data.")
    st.warning("Your portfolio is temporary and will reset when you refresh the page.")

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}
    
    col_add, col_remove = st.columns(2)
    
    with col_add:
        st.subheader("Add/Update Stock")
        with st.form("add_form"):
            ticker_add = st.text_input("Ticker Symbol (e.g., AAPL):").upper()
            shares_add = st.number_input("Number of Shares:", min_value=1)
            add_button = st.form_submit_button("Add to Portfolio")
            if add_button and ticker_add and shares_add:
                st.session_state.portfolio[ticker_add] = shares_add
    
    with col_remove:
        st.subheader("Remove Stock")
        tickers_to_remove = st.multiselect("Select Tickers to Remove:", options=list(st.session_state.portfolio.keys()))
        if st.button("Remove Selected"):
            for t in tickers_to_remove:
                del st.session_state.portfolio[t]

    if st.session_state.portfolio:
        st.subheader("Your Portfolio Summary")
        
        portfolio_data_list = []
        total_value = 0.0

        for ticker, shares in st.session_state.portfolio.items():
            current_price = 0.0
            try:
                # Fetching data using yfinance
                latest_price_data = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
                
                if not latest_price_data.empty:
                    # Safely get the close price as a float
                    current_price = latest_price_data['Close'].iloc[-1]
                else:
                    st.warning(f"Could not fetch latest price for {ticker}. The ticker might be invalid.")
            except Exception as e:
                st.warning(f"Failed to fetch price for {ticker}. Error: {e}. Using $0.00.")

            total_stock_value = shares * current_price
            total_value += total_stock_value
            
            portfolio_data_list.append({
                'Ticker': ticker,
                'Shares': float(shares), # Ensure shares is a float
                'Current Price': float(current_price), # Ensure current_price is a float
                'Total Value': float(total_stock_value) # Ensure total_stock_value is a float
            })
            
        portfolio_data = pd.DataFrame(portfolio_data_list)
        st.dataframe(portfolio_data, width='stretch')
        
        # Now that we have a single total_value (which is a float), we can format it
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    else:
        st.info("Your portfolio is empty. Add some stocks to begin tracking.")
        
with tab6:
    st.header("Key Statistics")
    st.markdown("A quick overview of descriptive statistics for the filtered data.")
    
    if len(selected_tickers) == 1:
        desc_df = df[df['Ticker'] == selected_tickers[0]].set_index('Date').describe().T
        st.dataframe(desc_df, use_container_width=True)
    else:
        st.info("Please select a single ticker in the sidebar to view its key statistics.")

with tab7:
    st.header("Time Series & Model Accuracy")
    
    # NEW: Section for comparing directional accuracy
    st.markdown("### Directional Accuracy Analysis")
    st.markdown("""
    This tool allows you to directly compare the predicted market direction (up or down) from your own model against the actual historical data. 
    It calculates the percentage of days where your model correctly predicted the market's movement.
    """)
    
    col_actual, col_predicted = st.columns(2)
    with col_actual:
        actual_file = st.file_uploader("Upload Actual Data CSV:", type=['csv'], key="actual_file")
    with col_predicted:
        predicted_file = st.file_uploader("Upload Predicted Data CSV:", type=['csv'], key="predicted_file")
        
    if st.button("Calculate Directional Accuracy"):
        if actual_file is not None and predicted_file is not None:
            actual_df = pd.read_csv(io.StringIO(actual_file.getvalue().decode("utf-8")))
            predicted_df = pd.read_csv(io.StringIO(predicted_file.getvalue().decode("utf-8")))
            
            accuracy = calculate_directional_accuracy(actual_df, predicted_df)
            
            if accuracy is not None:
                st.markdown("---")
                st.subheader("Accuracy Results")
                st.info(f"The directional accuracy of your prediction is: **{accuracy:.2f}%**")
        else:
            st.warning("Please upload both 'Actual Data' and 'Predicted Data' CSV files.")
            
    st.markdown("---")
    
    st.markdown("### Model Accuracy Explained")
    st.markdown(
        """
        To find the best model, we compare their performance on historical data they've never seen. The primary metric for this comparison is the **Root Mean Squared Error (RMSE)**.

        **What is RMSE?**
        RMSE measures the average difference between the model's predicted values and the actual values. A lower RMSE means the model's predictions are closer to the real data, indicating higher accuracy. Our goal is to find the model with the lowest RMSE.
        """
    )
    
    if len(selected_tickers) == 1:
        selected_ticker_pred = selected_tickers[0]
        
        # Run all three models and collect RMSE scores
        # prophet_forecast, prophet_rmse = run_prophet_model(df, prediction_days, selected_ticker_pred)
        # lr_forecast, lr_rmse = run_linear_regression_model(df, prediction_days, selected_ticker_pred)
        # rf_forecast, rf_rmse = run_random_forest_model(df, prediction_days, selected_ticker_pred)
        # vr_forecast, vr_rmse = run_voting_regressor_model(df, prediction_days, selected_ticker_pred)

        # --- Directional Accuracy based Evaluation ---
        st.subheader("Model Directional Accuracy Comparison")

        def get_directional_accuracy(y_true, y_pred):
            """Calculate % of correctly predicted directions (up/down)."""
            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true, y_pred = y_true[:min_len], y_pred[:min_len]
            return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100

        direction_scores = {}

# Prophet Model
        prophet_forecast, _ = run_prophet_model(df, prediction_days, selected_ticker_pred)
        if prophet_forecast is not None:
            hist = df[df['Ticker'] == selected_ticker_pred]['Close Price'].values
            preds = prophet_forecast['yhat'].values[-len(hist):]
            direction_scores['Prophet'] = get_directional_accuracy(hist, preds)

# Linear Regression Model
        lr_forecast, _ = run_linear_regression_model(df, prediction_days, selected_ticker_pred)
        if lr_forecast is not None:
            hist = df[df['Ticker'] == selected_ticker_pred]['Close Price'].values
            preds = lr_forecast['yhat'].values[-len(hist):]
            direction_scores['Linear Regression'] = get_directional_accuracy(hist, preds)

# Random Forest Model
        rf_forecast, _ = run_random_forest_model(df, prediction_days, selected_ticker_pred)
        if rf_forecast is not None:
            hist = df[df['Ticker'] == selected_ticker_pred]['Close Price'].values
            preds = rf_forecast['yhat'].values[-len(hist):]
            direction_scores['Random Forest'] = get_directional_accuracy(hist, preds)

# Voting Regressor Model
        vr_forecast, _ = run_voting_regressor_model(df, prediction_days, selected_ticker_pred)
        if vr_forecast is not None:
            hist = df[df['Ticker'] == selected_ticker_pred]['Close Price'].values
            preds = vr_forecast['yhat'].values[-len(hist):]
            direction_scores['Voting Regressor'] = get_directional_accuracy(hist, preds)

        if not direction_scores:
            st.error("Not enough data to evaluate directional accuracy for this ticker.")
            st.stop()

# Display the results
        acc_df = pd.DataFrame(list(direction_scores.items()), columns=['Model', 'Directional Accuracy (%)'])
        fig_acc = px.bar(acc_df,
                        x='Model',
                        y='Directional Accuracy (%)',
                        title='Directional Accuracy of Forecasting Models',
                        color='Model')
        st.plotly_chart(fig_acc, use_container_width=True)

        best_model = max(direction_scores, key=direction_scores.get)
        st.success(f"✅ Based on Directional Accuracy, the **{best_model}** model performed best.")

        # Show best model forecast
        if best_model == 'Prophet':
            forecast_to_show = prophet_forecast
        elif best_model == 'Linear Regression':
            forecast_to_show = lr_forecast
        elif best_model == 'Random Forest':
            forecast_to_show = rf_forecast
        else:
            forecast_to_show = vr_forecast


        # rmse_scores = {}
        # if prophet_rmse is not None:
        #     rmse_scores['Prophet'] = prophet_rmse
        # if lr_rmse is not None:
        #     rmse_scores['Linear Regression'] = lr_rmse
        # if rf_rmse is not None:
        #     rmse_scores['Random Forest'] = rf_rmse
        # if vr_rmse is not None:
        #     rmse_scores['Voting Regressor'] = vr_rmse

        # if not rmse_scores:
        #     st.error("Not enough historical data for this ticker to train any prediction models.")
        #     st.stop()
            
        # st.subheader("Model Accuracy Comparison (RMSE)")
        # st.markdown("A lower RMSE indicates a more accurate model.")
        
        # rmse_df = pd.DataFrame(list(rmse_scores.items()), columns=['Model', 'RMSE'])
        # fig_rmse = px.bar(rmse_df, 
        #                  x='Model', 
        #                  y='RMSE', 
        #                  title='RMSE Comparison of Forecasting Models',
        #                  color='Model')
        # st.plotly_chart(fig_rmse, use_container_width=True)
        
        # # Display forecast of the best performing model
        # best_model = min(rmse_scores, key=rmse_scores.get)
        # st.info(f"Based on the RMSE, the **{best_model}** model is the most accurate for this dataset.")
        
        st.subheader(f"Forecast from the {best_model} Model")
        
        if best_model == 'Prophet':
            forecast_to_show = prophet_forecast
        elif best_model == 'Linear Regression':
            forecast_to_show = lr_forecast
        elif best_model == 'Random Forest':
            forecast_to_show = rf_forecast
        else: # Voting Regressor
            forecast_to_show = vr_forecast

        historical_df = df[df['Ticker'] == selected_ticker_pred].copy()
        last_historical_close = historical_df['Close Price'].iloc[-1]
        synthetic_forecast_df = generate_synthetic_candlestick(forecast_to_show.iloc[-prediction_days:], last_historical_close)
        
        fig_predicted = go.Figure(data=[go.Candlestick(
            x=synthetic_forecast_df['ds'],
            open=synthetic_forecast_df['Open Price'],
            high=synthetic_forecast_df['High Price'],
            low=synthetic_forecast_df['Low Price'],
            close=synthetic_forecast_df['Close Price'],
            name='Predicted Price'
        )])
        fig_predicted.update_layout(
            title=f'Predicted Price Forecast for {selected_ticker_pred}',
            yaxis_title='Stock Price',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_predicted, use_container_width=True)
        
        st.subheader("Predicted Prices")
        st.dataframe(forecast_to_show[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'}), use_container_width=True)

    else:
        st.info("Please select a single ticker from the sidebar to use the forecasting feature.")


with tab8:
    st.header("Risk & Volatility Analysis")
    st.markdown("""
    This section analyzes the risk of a stock by measuring its volatility, or how much its price fluctuates. 
    The **standard deviation of daily returns** is a common metric for this. A higher standard deviation indicates a higher-risk, higher-reward stock.
    """)
    
    selected_tickers_volatility = st.multiselect(
        "Select Tickers for Volatility Comparison:",
        options=sorted(list(df['Ticker'].unique())),
        default=sorted(list(df['Ticker'].unique()))[:2]
    )

    if selected_tickers_volatility:
        volatility_df = df[df['Ticker'].isin(selected_tickers_volatility)].copy()
        
        # Calculate Rolling Volatility (e.g., 20-day)
        volatility_df['Rolling Volatility'] = volatility_df.groupby('Ticker')['Daily Return'].transform(lambda x: x.rolling(window=20).std() * np.sqrt(252)) # Annualized
        
        st.subheader("Annualized Rolling Volatility")
        fig_vol = px.line(
            volatility_df,
            x='Date',
            y='Rolling Volatility',
            color='Ticker',
            title='20-Day Annualized Rolling Volatility',
            labels={'Rolling Volatility': 'Volatility'},
            hover_name='Ticker'
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.subheader("Average Volatility Comparison")
        avg_vol = volatility_df.groupby('Ticker')['Rolling Volatility'].mean().reset_index()
        fig_avg_vol = px.bar(
            avg_vol,
            x='Ticker',
            y='Rolling Volatility',
            color='Ticker',
            title='Average Volatility of Selected Tickers'
        )
        st.plotly_chart(fig_avg_vol, use_container_width=True)
        
        # New: Risk Conclusion
        st.subheader("Risk Conclusion")
        if not avg_vol.empty:
            sorted_vol = avg_vol.sort_values(by='Rolling Volatility', ascending=False)
            highest_vol = sorted_vol.iloc[0]
            lowest_vol = sorted_vol.iloc[-1]
            
            st.info(f"""
            Based on the data, **{highest_vol['Ticker']}** shows the highest average volatility at **{highest_vol['Rolling Volatility']:.2f}**. 
            This suggests it has experienced larger price swings and is considered the **highest-risk** stock among those selected.

            In contrast, **{lowest_vol['Ticker']}** has the lowest average volatility at **{lowest_vol['Rolling Volatility']:.2f}**, 
            making it the **lowest-risk** option in this group.

            Remember, higher volatility can lead to higher potential returns, but it also carries a greater risk of loss.
            """)

    else:
        st.info("Please select at least two tickers to compare their volatility.")

with tab9:
    st.header("News Sentiment Analysis")
    st.markdown("""
    This is a conceptual demonstration of how a news sentiment analysis feature would work. 
    **Note:** This is a simulated analysis as real-time news data retrieval requires an external API not available in this environment.
    A real implementation would fetch live headlines and use natural language processing to determine the overall sentiment.
    """)
    
    selected_ticker_news = st.selectbox(
        "Select a Ticker for News Sentiment:",
        options=sorted(list(df['Ticker'].unique())),
        key="news_select"
    )
    
    st.info(f"Simulating news sentiment for **{selected_ticker_news}**...")
    
    # Static, simulated sentiment data
    simulated_sentiment = {
        'AAPL': {'sentiment': 'Positive', 'headlines': ['iPhone sales soar in Q3', 'Apple stock hits new all-time high']},
        'MSFT': {'sentiment': 'Neutral', 'headlines': ['Microsoft announces new partnership', 'Stock holds steady after earnings report']},
        'GOOGL': {'sentiment': 'Negative', 'headlines': ['Regulatory concerns plague Google', 'Google fined for anti-competitive practices']}
    }
    
    if selected_ticker_news in simulated_sentiment:
        sentiment_data = simulated_sentiment[selected_ticker_news]
        
        col_sentiment, col_headlines = st.columns([1, 2])
        
        with col_sentiment:
            sentiment = sentiment_data['sentiment']
            if sentiment == 'Positive':
                st.success(f"Overall Sentiment: **{sentiment}** ✅")
            elif sentiment == 'Neutral':
                st.info(f"Overall Sentiment: **{sentiment}** 😐")
            else:
                st.error(f"Overall Sentiment: **{sentiment}** 🔻")

        with col_headlines:
            st.subheader("Key Headlines (Simulated)")
            for headline in sentiment_data['headlines']:
                st.write(f"- {headline}")
    else:
        st.warning("Simulated sentiment data not available for this ticker.")

with tab10:
    st.header("Backtesting Engine")
    st.markdown("""
    Test a simple Moving Average Crossover strategy on historical data. This engine simulates trades and
    calculates the strategy's hypothetical performance.
    """)
    
    if len(selected_tickers) == 1:
        ticker_for_backtest = selected_tickers[0]
        st.info(f"Backtesting the strategy for **{ticker_for_backtest}**.")
        
        fast_ma = st.number_input("Fast MA Period:", min_value=5, max_value=50, value=10)
        slow_ma = st.number_input("Slow MA Period:", min_value=20, max_value=200, value=50)

        if st.button("Run Backtest"):
            backtest_df = df[df['Ticker'] == ticker_for_backtest].copy()
            backtest_df['Fast_MA'] = backtest_df['Close Price'].rolling(window=fast_ma).mean()
            backtest_df['Slow_MA'] = backtest_df['Close Price'].rolling(window=slow_ma).mean()

            # Generate trading signals
            backtest_df['Signal'] = 0
            backtest_df['Signal'][fast_ma:] = np.where(backtest_df['Fast_MA'][fast_ma:] > backtest_df['Slow_MA'][fast_ma:], 1, 0)
            backtest_df['Position'] = backtest_df['Signal'].diff()

            initial_cash = 10000
            backtest_df['Portfolio_Value'] = initial_cash
            cash = initial_cash
            shares = 0
            
            for i, row in backtest_df.iterrows():
                if row['Position'] == 1: # Buy signal
                    shares_to_buy = cash // row['Close Price']
                    if shares_to_buy > 0:
                        shares += shares_to_buy
                        cash -= shares_to_buy * row['Close Price']
                elif row['Position'] == -1: # Sell signal
                    if shares > 0:
                        cash += shares * row['Close Price']
                        shares = 0
                
                # Calculate portfolio value at the end of the day
                backtest_df.loc[i, 'Portfolio_Value'] = cash + shares * row['Close Price']

            total_return = (backtest_df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash * 100
            
            st.subheader("Backtest Results")
            st.write(f"Initial Portfolio Value: **${initial_cash:,.2f}**")
            st.write(f"Final Portfolio Value: **${backtest_df['Portfolio_Value'].iloc[-1]:,.2f}**")
            st.metric("Total Return", f"{total_return:.2f}%")

            fig_backtest = px.line(backtest_df, x='Date', y='Portfolio_Value', title='Portfolio Value Over Time')
            st.plotly_chart(fig_backtest, use_container_width=True)
            
    else:
        st.info("Please select a single ticker from the sidebar to use the backtesting engine.")

with tab11:
    st.header("Financial Report")
    st.markdown("Here you can find a simulated financial report for the selected stock. The data provided is for demonstration purposes only and should not be used for investment decisions.")
    
    selected_ticker_report = st.selectbox(
        "Select a Ticker for the Report:",
        options=sorted(list(df['Ticker'].unique())),
        key="report_select"
    )
    
    if selected_ticker_report:
        
        @st.cache_data(ttl=600)
        def get_financial_data(ticker):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                return info
            except Exception as e:
                st.error(f"Could not retrieve financial data for {ticker}. Error: {e}")
                return None
        
        data = get_financial_data(selected_ticker_report)
        
        if data:
            st.subheader(f"Financial Summary for {selected_ticker_report}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Cap", f"${data.get('marketCap', 'N/A'):,.2f}")
                st.metric("P/E Ratio", f"{data.get('trailingPE', 'N/A'):.2f}")
                st.metric("Dividend Yield", f"{data.get('dividendYield', 'N/A'):.2%}" if data.get('dividendYield') else 'N/A')
                st.metric("Beta", f"{data.get('beta', 'N/A'):.2f}")

            with col2:
                st.metric("Forward P/E", f"{data.get('forwardPE', 'N/A'):.2f}")
                st.metric("EPS (TTM)", f"${data.get('trailingEps', 'N/A'):.2f}")
                st.metric("52 Week High", f"${data.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
                st.metric("52 Week Low", f"${data.get('fiftyTwoWeekLow', 'N/A'):.2f}")
                
            st.markdown("---")
            st.subheader("Business Summary")
            st.write(data.get('longBusinessSummary', 'Summary not available.'))
            
            st.markdown("---")
            st.subheader("Analyst Ratings (Simulated)")
            
            # This is a simulated section as live analyst data is complex to retrieve
            simulated_ratings = {
                'AAPL': 'Hold', 'MSFT': 'Buy', 'GOOGL': 'Strong Buy',
                'AMZN': 'Buy', 'TSLA': 'Hold', 'NVDA': 'Strong Buy'
            }
            rating = simulated_ratings.get(selected_ticker_report, "N/A (Simulated)")
            st.write(f"The consensus analyst rating is: **{rating}**.")
            st.info("Simulated data for demonstration purposes only.")
    else:
        st.info("Please select a ticker to view its financial report.")
