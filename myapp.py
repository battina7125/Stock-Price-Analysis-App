import yfinance as yf
import streamlit as st
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Calculating RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD (Moving Average Convergence Divergence)
def calculate_macd(data, slow=26, fast=12, signal=9):
    fast_ema = data.ewm(span=fast, adjust=False).mean()
    slow_ema = data.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Predict future stock prices
def predict_stock_prices(df, forecast_days=30):
    df['Prediction'] = df['Close'].shift(-forecast_days)
    X = np.array(df.drop(['Prediction'], axis=1))[:-forecast_days]
    y = np.array(df['Prediction'])[:-forecast_days]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_days:]
    predictions = model.predict(X_forecast)
    return predictions

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #333;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1e3c72;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .stMultiSelect label {
        color: #1e3c72;
        font-family: 'Arial', sans-serif;
    }
    .stDateInput label {
        color: #1e3c72;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #1e3c72;
        color: white;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #1e3c72;
    }
    .stExpanderHeader {
        background-color: #1e3c72;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .footer {
        text-align: center;
        padding: 10px 0;
        color: #666;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.write("""
# 📈 Stock Price App

Easily explore the **closing price** and **volume** of popular international stocks. No financial expertise needed!
""")

# Extended list of ticker symbols
available_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'BRK-B', 'JPM', 'V', 
    'DIS', 'BABA', 'ORCL', 'CSCO', 'INTC', 'PFE', 'KO', 'PEP', 'BA', 'MCD', 'NKE', 
    'ADBE', 'PYPL', 'CRM', 'IBM', 'XOM', 'WMT', 'CVX', 'T', 'UNH', 'MA', 'PG', 
    'JNJ', 'VZ', 'HD', 'BAC', 'LLY', 'MRK', 'ABBV', 'DHR', 'COST', 'TM', 'PM', 
    'WFC', 'BMY', 'QCOM', 'AVGO', 'AMD', 'TXN', 'NEE', 'RTX', 'LIN', 'SPGI', 'AXP', 
    'MS', 'GS', 'BLK', 'INTU', 'ADP', 'ISRG', 'NOW', 'UNP', 'CAT', 'TMO', 'SCHW', 
    'AMT', 'DE', 'HON', 'CI', 'MDT', 'MMM', 'GILD', 'PLD', 'AMGN', 'SYK', 'C', 
    'GE', 'CB', 'ZTS', 'MMC', 'PNC', 'ICE', 'ELV', 'CL', 'USB', 'TGT', 'VRTX', 
    'SO', 'DUK', 'AON', 'BDX', 'EQIX', 'TFC', 'ADSK', 'PSA', 'COF', 'CCI', 'SBUX', 
    'COP', 'BK', 'ADI', 'REGN', 'MO', 'LMT', 'LOW', 'SPG', 'GM', 'FIS', 'ECL', 
    'F', 'FDX', 'GIS', 'NSC', 'KMB', 'BAX', 'AIG', 'APH', 'MAR', 'ETN', 'PGR', 
    'SHW', 'DG', 'HCA', 'CTAS', 'CMG', 'LRCX', 'EW', 'MCK', 'AFL', 'LUV', 'HPQ', 
    'EMR', 'CSX', 'CLX', 'BBY', 'DOW', 'WBA', 'ROST', 'XEL', 'EBAY', 'SYY', 
    'KR', 'ADM', 'D', 'MPC', 'VLO', 'SPGI', 'BKNG'
]

# Multi-select dropdown for ticker symbols
tickerSymbols = st.multiselect(
    'Select Ticker Symbols (Tip: Choose the stocks you are interested in)',
    available_tickers,
    default=['AAPL', 'MSFT', 'GOOGL']
)

# Get today's date
today = datetime.date.today()

# Set the default date range (e.g., last 5 years)
default_start_date = today - datetime.timedelta(days=5*365)
default_end_date = today

# Date range selection using date input for precise selection
start_date = st.date_input(
    "Select start date (Tip: Choose a past date to see historical performance)", 
    value=default_start_date, 
    min_value=datetime.date(2010, 1, 1), 
    max_value=today
)
end_date = st.date_input(
    "Select end date (Tip: Choose today's date to see the latest data)", 
    value=default_end_date, 
    min_value=datetime.date(2010, 1, 1), 
    max_value=today
)

# Ensure start date is before end date
if start_date > end_date:
    st.error("⚠️ Error: End date must be after the start date.")
else:
    # Initializing a dictionary to store analysis data
    analysis_data = {}

    # Displaying data for selected stocks
    for symbol in tickerSymbols:
        tickerData = yf.Ticker(symbol)
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
        
        if not tickerDf.empty:
            # Calculating additional indicators
            rsi = calculate_rsi(tickerDf['Close'])
            macd, signal_line = calculate_macd(tickerDf['Close'])
            predicted_prices = predict_stock_prices(tickerDf)
            
            # Storing data for analysis
            closing_prices = tickerDf['Close']
            analysis_data[symbol] = {
                "start_price": closing_prices.iloc[0],
                "end_price": closing_prices.iloc[-1],
                "max_price": closing_prices.max(),
                "min_price": closing_prices.min(),
                "price_change": closing_prices.iloc[-1] - closing_prices.iloc[0],
                "volatility": np.std(closing_prices),
                "moving_avg_short": closing_prices.rolling(window=50).mean().iloc[-1],
                "moving_avg_long": closing_prices.rolling(window=200).mean().iloc[-1],
                "support_level": closing_prices.min(),
                "resistance_level": closing_prices.max(),
                "volume_avg": tickerDf['Volume'].mean(),
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "predicted_prices": predicted_prices
            }

            # Displaying  charts side by side with titles
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"## 📉 Closing Price of {symbol}")
                st.line_chart(closing_prices)
            with col2:
                st.write(f"## 📊 Volume Price of {symbol}")
                st.line_chart(tickerDf.Volume)
            
            with st.expander(f"More details for {symbol} (Click to expand)"):
                st.write(f"### 🟢 Open Price of {symbol}")
                st.line_chart(tickerDf.Open)
                
                st.write(f"### 🔺 High Price of {symbol}")
                st.line_chart(tickerDf.High)
                
                st.write(f"### 🔻 Low Price of {symbol}")
                st.line_chart(tickerDf.Low)
                st.write(f"### 📈 RSI of {symbol}")
                st.line_chart(rsi)
                st.write(f"### 📊 MACD and Signal Line of {symbol}")
                macd_df = pd.DataFrame({"MACD": macd, "Signal Line": signal_line})
                st.line_chart(macd_df)
                st.write(f"### 🔮 Predicted Prices for {symbol} (Next 30 Days)")
                st.line_chart(predicted_prices)
        else:
            st.write(f"No data available for {symbol} within the selected date range.")

    # Analysis section with detailed insights
    st.write("## 📊 Analysis Summary and Detailed Insights")
    if analysis_data:
        for symbol, data in analysis_data.items():
            st.write(f"### {symbol}")
            st.write(f"- **Starting Price:** ${data['start_price']:.2f}")
            st.write(f"- **Ending Price:** ${data['end_price']:.2f}")
            st.write(f"- **Max Price:** ${data['max_price']:.2f}")
            st.write(f"- **Min Price:** ${data['min_price']:.2f}")
            st.write(f"- **Price Change:** ${data['price_change']:.2f}")
            st.write(f"- **Volatility (Standard Deviation):** ${data['volatility']:.2f}")
            st.write(f"- **50-Day Moving Average:** ${data['moving_avg_short']:.2f}")
            st.write(f"- **200-Day Moving Average:** ${data['moving_avg_long']:.2f}")
            st.write(f"- **Support Level:** ${data['support_level']:.2f}")
            st.write(f"- **Resistance Level:** ${data['resistance_level']:.2f}")
            st.write(f"- **Average Volume:** {data['volume_avg']:.0f}")
            st.write(f"- **RSI (14-day):** {data['rsi']:.2f}")
            st.write(f"- **MACD:** {data['macd']:.2f}")
            st.write(f"- **Signal Line:** {data['signal_line']:.2f}")

            # Analysis based on the data
            trend = "uptrend" if data['end_price'] > data['moving_avg_long'] else "downtrend"
            volatility_comment = "high volatility" if data['volatility'] > data['price_change'] * 0.05 else "low volatility"
            moving_avg_comparison = (
                "is trading above both the short-term and long-term averages, indicating a strong bullish trend."
                if data['moving_avg_short'] > data['moving_avg_long'] else
                "is trading below the long-term average, indicating potential bearish sentiment."
            )
            sentiment = "Positive" if trend == "uptrend" and data['price_change'] > 0 else "Negative"
            recommendation = "Buy" if sentiment == "Positive" and data['volatility'] < data['price_change'] * 0.05 else "Hold" if sentiment == "Positive" else "Sell"

            st.write(f"**Trend Analysis:** {symbol} appears to be in an {trend} based on current price trends.")
            st.write(f"**Volatility Analysis:** The stock has {volatility_comment}, which could indicate a {volatility_comment.replace('volatility', '')} market environment.")
            st.write(f"**Moving Average Analysis:** The stock {moving_avg_comparison}")
            st.write(f"**Support and Resistance Levels:** The nearest support is around ${data['support_level']:.2f}, and resistance is at ${data['resistance_level']:.2f}.")
            st.write(f"**RSI Analysis:** The RSI is currently at {data['rsi']:.2f}, indicating that the stock is {'overbought' if data['rsi'] > 70 else 'oversold' if data['rsi'] < 30 else 'in a neutral range'}.")
            st.write(f"**MACD Analysis:** The MACD is {'above' if data['macd'] > data['signal_line'] else 'below'} the signal line, indicating a {'bullish' if data['macd'] > data['signal_line'] else 'bearish'} signal.")
            st.write(f"**Investment Recommendation:** Based on the analysis, a **{recommendation}** recommendation is suggested.")
    else:
        st.write("No analysis available for the selected stocks and date range.")

# Footer with guidance
st.markdown("""
    <hr>
    <div class="footer">
        © 2024 Stock Price App | Built with Streamlit | <a href="#">Help & Support</a>
    </div>
    """, unsafe_allow_html=True)
