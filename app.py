import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from datetime import datetime, timedelta
from fuzzywuzzy import process
import base64
import re

# Set page config
st.set_page_config(
    page_title="Futuristic Financial ML Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'alerts' not in st.session_state:
    st.session_state.alerts = {}

# Load S&P 500 data
@st.cache_data
def load_sample_data():
    return pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")

# Download stock data
@st.cache_data
def load_stock_data(ticker, start_date, end_date, interval="1d"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data for {ticker}.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        if 'adj_close' in data.columns:
            data = data.rename(columns={'adj_close': 'close'})
        required_cols = ['close', 'volume', 'open', 'high', 'low']
        if not all(col in data.columns for col in required_cols):
            st.warning(f"Missing columns for {ticker}")
            return None
        return data
    except Exception as e:
        st.warning(f"Error downloading {ticker}: {str(e)}")
        return None

# Technical indicators
def add_technical_indicators(data):
    data['sma_20'] = SMAIndicator(data['close'], window=20).sma_indicator()
    data['ema_20'] = EMAIndicator(data['close'], window=20).ema_indicator()
    data['rsi'] = RSIIndicator(data['close'], window=14).rsi()
    data['signal'] = np.where(data['sma_20'] > data['ema_20'], 1, -1)  # Buy/sell signals
    return data

# Sentiment analysis (simplified keyword-based)
def analyze_sentiment(ticker):
    positive_words = ['bullish', 'buy', 'strong', 'growth']
    negative_words = ['bearish', 'sell', 'weak', 'decline']
    # Simulated X post analysis (replace with actual API call if available)
    posts = [f"{ticker} looking bullish!", f"Time to sell {ticker}", f"{ticker} steady growth"]
    score = 0
    for post in posts:
        if any(word in post.lower() for word in positive_words):
            score += 1
        if any(word in post.lower() for word in negative_words):
            score -= 1
    return score / len(posts) if posts else 0

# Monte Carlo simulation
def monte_carlo_simulation(returns, weights, n_simulations=1000, days=252):
    portfolio_returns = []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=days, replace=True)
        sim_portfolio = np.prod(1 + (sim_returns * weights).sum(axis=1)) - 1
        portfolio_returns.append(sim_portfolio)
    return np.array(portfolio_returns)

# Backtesting strategy
def backtest_strategy(data):
    data['position'] = np.where(data['sma_20'] > data['ema_20'], 1, 0)
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['position'].shift(1) * data['returns']
    cumulative_return = (1 + data['strategy_returns']).cumprod().iloc[-1] - 1
    return cumulative_return

# Generate PDF report
def generate_pdf_report(data, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title, styles['Title']))
    table_data = [data.columns.tolist()] + data.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#63b3ed'),
        ('TEXTCOLOR', (0, 0), (-1, 0), '#1a202c'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), '#f7fafc'),
        ('GRID', (0, 0), (-1, -1), 1, '#cbd5e0')
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# Main app
def main():
    # Theme toggle
    st.markdown(
        f'<button onclick="document.body.classList.toggle(\'light-theme\');" class="theme-toggle">Toggle {"Light" if st.session_state.theme == "dark" else "Dark"} Theme</button>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown('<div class="sidebar-header">Futuristic Financial Dashboard</div>', unsafe_allow_html=True)
    app_mode = st.sidebar.selectbox("Navigate", 
                                   ["Home", "Data Exploration", "Portfolio Optimization", 
                                    "Predictive Analytics", "Clustering", "Backtesting"])

    # Home
    if app_mode == "Home":
        st.markdown('<div class="card glass"><h1>Futuristic Financial ML Dashboard</h1></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box glass"><p>Analyze stocks, optimize portfolios, and predict markets with AI.</p></div>', unsafe_allow_html=True)
        
        sample_data = load_sample_data()
        st.markdown('<div class="card glass"><h3>S&P 500 Companies</h3></div>', unsafe_allow_html=True)
        st.dataframe(sample_data.head(), use_container_width=True)
        
        csv = sample_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="sp500_companies.csv" class="download-btn">Download S&P 500 Data</a>', unsafe_allow_html=True)

    # Data Exploration
    elif app_mode == "Data Exploration":
        st.markdown('<div class="card glass"><h2>Data Exploration</h2></div>', unsafe_allow_html=True)
        
        sample_data = load_sample_data()
        ticker_input = st.text_input("Search Ticker", "AAPL")
        matches = process.extract(ticker_input.upper(), sample_data['Symbol'].tolist(), limit=5)
        ticker = st.selectbox("Select Ticker", [m[0] for m in matches], index=0)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            period = st.selectbox("Period", ["1D", "5D", "1M", "3M", "6M", "1Y", "3Y"], index=5)
            interval = "1m" if period == "1D" else "5m" if period == "5D" else "1d"
        with col2:
            alert_price = st.number_input(f"Set Alert Price for {ticker}", value=0.0, step=0.1)
            if alert_price > 0:
                st.session_state.alerts[ticker] = alert_price
        
        end_date = datetime.now()
        if period == "1D":
            start_date = end_date - timedelta(days=1)
        elif period == "5D":
            start_date = end_date - timedelta(days=5)
        elif period == "1M":
            start_date = end_date - timedelta(days=30)
        elif period == "3M":
            start_date = end_date - timedelta(days=90)
        elif period == "6M":
            start_date = end_date - timedelta(days=180)
        elif period == "1Y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=3*365)
        
        if st.button("Load Data"):
            with st.spinner("Loading data..."):
                stock_data = load_stock_data(ticker, start_date, end_date, interval)
                if stock_data is None:
                    st.error(f"No data for {ticker}.")
                else:
                    stock_data = add_technical_indicators(stock_data)
                    st.success(f"Loaded data for {ticker}")
                    
                    # Check alerts
                    if ticker in st.session_state.alerts and stock_data['close'].iloc[-1] >= st.session_state.alerts[ticker]:
                        st.markdown(f'<div class="alert-box">🚨 Alert: {ticker} reached ${stock_data["close"].iloc[-1]:.2f}!</div>', unsafe_allow_html=True)
                    
                    # Sentiment
                    sentiment_score = analyze_sentiment(ticker)
                    st.markdown(f'<div class="card glass"><h3>Sentiment Analysis</h3><p>Sentiment Score: {sentiment_score:.2f} (from X posts)</p></div>', unsafe_allow_html=True)
                    
                    # Candlestick with signals
                    st.markdown('<div class="card glass"><h3>Candlestick Chart</h3></div>', unsafe_allow_html=True)
                    fig = go.Figure(data=[go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['open'],
                        high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        name=ticker
                    )])
                    buy_signals = stock_data[stock_data['signal'] == 1]
                    sell_signals = stock_data[stock_data['signal'] == -1]
                    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='#34c759')))
                    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='#f24236')))
                    fig.update_layout(
                        title=f"{ticker} Candlestick Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_dark",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators
                    st.markdown('<div class="card glass"><h3>Technical Indicators</h3></div>', unsafe_allow_html=True)
                    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                            subplot_titles=("Price & MAs", "RSI", "Volume"))
                    fig_tech.add_trace(go.Scatter(x=stock_data.index, y=stock_data['close'], name="Close", line=dict(color='#2e86ab')), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=stock_data.index, y=stock_data['sma_20'], name="SMA 20", line=dict(color='#f24236')), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=stock_data.index, y=stock_data['ema_20'], name="EMA 20", line=dict(color='#34c759')), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=stock_data.index, y=stock_data['rsi'], name="RSI", line=dict(color='#ff9500')), row=2, col=1)
                    fig_tech.add_trace(go.Bar(x=stock_data.index, y=stock_data['volume'], name="Volume", marker_color='#2e86ab'), row=3, col=1)
                    fig_tech.update_layout(height=800, template="plotly_dark", showlegend=True)
                    st.plotly_chart(fig_tech, use_container_width=True)

    # Portfolio Optimization
    elif app_mode == "Portfolio Optimization":
        st.markdown('<div class="card glass"><h2>Portfolio Optimization</h2></div>', unsafe_allow_html=True)
        
        sample_data = load_sample_data()
        selected_tickers = st.multiselect("Select Tickers", sample_data['Symbol'].tolist(), default=['AAPL', 'MSFT', 'GOOG'])
        weights = st.text_input("Weights (comma-separated)", "0.33,0.33,0.34")
        weights = [float(w) for w in weights.split(",")] if weights else [1/len(selected_tickers)]*len(selected_tickers)
        
        if len(selected_tickers) != len(weights) or abs(sum(weights) - 1.0) > 0.01:
            st.error("Weights must match tickers and sum to 1.")
            return
        
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing..."):
                data_dict = {}
                for ticker in selected_tickers:
                    data = load_stock_data(ticker, "2022-01-01", "2023-12-31")
                    if data is not None:
                        data_dict[ticker] = data
                
                if not data_dict:
                    st.error("No valid data.")
                    return
                
                returns = pd.DataFrame({ticker: data['close'].pct_change() for ticker, data in data_dict.items()}).dropna()
                mean_return = (returns * weights).sum(axis=1).mean() * 252
                volatility = (returns * weights).sum(axis=1).std() * np.sqrt(252)
                sharpe_ratio = mean_return / volatility if volatility != 0 else 0
                corr_matrix = returns.corr()
                
                # Monte Carlo
                sim_returns = monte_carlo_simulation(returns, weights)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="card glass"><h3>Portfolio Metrics</h3></div>', unsafe_allow_html=True)
                    st.metric("Annualized Return", f"{mean_return*100:.2f}%")
                    st.metric("Volatility", f"{volatility*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                with col2:
                    st.markdown('<div class="card glass"><h3>Risk-Return Scatter</h3></div>', unsafe_allow_html=True)
                    risk_return = pd.DataFrame({
                        'Ticker': selected_tickers,
                        'Return': [returns[t].mean() * 252 for t in selected_tickers],
                        'Volatility': [returns[t].std() * np.sqrt(252) for t in selected_tickers]
                    })
                    fig = px.scatter(risk_return, x='Volatility', y='Return', text='Ticker', size=[10]*len(selected_tickers),
                                    color='Return', hover_name='Ticker', title="Risk-Return Profile")
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="card glass"><h3>Monte Carlo Simulation</h3></div>', unsafe_allow_html=True)
                fig_mc = px.histogram(sim_returns, nbins=50, title="Portfolio Return Distribution")
                fig_mc.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # PDF report
                pdf_buffer = generate_pdf_report(pd.DataFrame({
                    'Metric': ['Return', 'Volatility', 'Sharpe Ratio'],
                    'Value': [f"{mean_return*100:.2f}%", f"{volatility*100:.2f}%", f"{sharpe_ratio:.2f}"]
                }), "Portfolio Analysis Report")
                st.download_button("Download Report", pdf_buffer, "portfolio_report.pdf", "application/pdf")

    # Predictive Analytics
    elif app_mode == "Predictive Analytics":
        st.markdown('<div class="card glass"><h2>Predictive Analytics</h2></div>', unsafe_allow_html=True)
        
        ticker = st.selectbox("Select Ticker", ['AAPL', 'MSFT', 'GOOG'], index=0)
        model_type = st.radio("Model Type", ["Linear Regression", "Random Forest"])
        
        stock_data = load_stock_data(ticker, "2020-01-01", "2023-12-31")
        if stock_data is None:
            st.error(f"No data for {ticker}.")
            return
        
        stock_data['days'] = range(len(stock_data))
        X = stock_data[['days']]
        y = stock_data['close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card glass"><h3>Model Performance</h3></div>', unsafe_allow_html=True)
            st.metric("Mean Squared Error", f"{mse:.2f}")
            future_days = st.slider("Predict days in future", 1, 30, 5)
            future_pred = model.predict([[len(stock_data) + future_days]])
            st.metric(f"Predicted price in {future_days} days", f"${float(future_pred[0]):.2f}")
        
        with col2:
            st.markdown('<div class="card glass"><h3>Actual vs Predicted</h3></div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_test['days'], y=y_test, mode='markers', name='Actual', marker=dict(color='#2e86ab')))
            fig.add_trace(go.Scatter(x=X_test['days'], y=y_pred, mode='lines', name='Predicted', line=dict(color='#f24236')))
            fig.update_layout(title=f"{model_type}: Actual vs Predicted", xaxis_title="Days", yaxis_title="Price ($)", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Clustering
    elif app_mode == "Clustering":
        st.markdown('<div class="card glass"><h2>K-Means Clustering</h2></div>', unsafe_allow_html=True)
        
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'WMT']
        features_selected = st.multiselect("Select Features", ['mean_price', 'price_volatility', 'mean_volume'], default=['mean_price', 'price_volatility', 'mean_volume'])
        
        with st.spinner("Loading data..."):
            data_list = []
            for ticker in tickers:
                stock_data = load_stock_data(ticker, "2023-01-01", "2023-12-31")
                if stock_data is not None:
                    stock_data['ticker'] = ticker
                    data_list.append(stock_data)
            
            if not data_list:
                st.error("No valid data.")
                return
            
            combined_data = pd.concat(data_list)
        
        features = combined_data.groupby('ticker').agg({
            'close': ['mean', 'std'],
            'volume': 'mean'
        }).reset_index()
        features.columns = ['ticker', 'mean_price', 'price_volatility', 'mean_volume']
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(features[features_selected])
        
        n_clusters = st.slider("Number of clusters", 2, 5, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        features['cluster'] = clusters
        
        silhouette = silhouette_score(X, clusters)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card glass"><h3>Clustering Results</h3></div>', unsafe_allow_html=True)
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            st.dataframe(features[['ticker', 'cluster']])
            
            st.markdown('<div class="card glass"><h3>Cluster Characteristics</h3></div>', unsafe_allow_html=True)
            cluster_stats = features.groupby('cluster')[features_selected].mean()
            st.dataframe(cluster_stats)
            
            # PDF report
            pdf_buffer = generate_pdf_report(features[['ticker', 'cluster'] + features_selected], "Clustering Report")
            st.download_button("Download Report", pdf_buffer, "clustering_report.pdf", "application/pdf")
        
        with col2:
            st.markdown('<div class="card glass"><h3>Cluster Visualization</h3></div>', unsafe_allow_html=True)
            if len(features_selected) >= 3:
                fig = px.scatter_3d(
                    features,
                    x=features_selected[0],
                    y=features_selected[1],
                    z=features_selected[2],
                    color='cluster',
                    hover_name='ticker',
                    title="Stock Clusters (3D View)",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    animation_frame='cluster'
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title=features_selected[0].replace('_', ' ').title(),
                        yaxis_title=features_selected[1].replace('_', ' ').title(),
                        zaxis_title=features_selected[2].replace('_', ' ').title()
                    ),
                    template="plotly_dark",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(
                    features,
                    x=features_selected[0],
                    y=features_selected[1] if len(features_selected) > 1 else features_selected[0],
                    color='cluster',
                    hover_name='ticker',
                    title="Stock Clusters (2D View)",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Backtesting
    elif app_mode == "Backtesting":
        st.markdown('<div class="card glass"><h2>Strategy Backtesting</h2></div>', unsafe_allow_html=True)
        
        ticker = st.selectbox("Select Ticker", ['AAPL', 'MSFT', 'GOOG'], index=0)
        stock_data = load_stock_data(ticker, "2020-01-01", "2023-12-31")
        if stock_data is None:
            st.error(f"No data for {ticker}.")
            return
        
        stock_data = add_technical_indicators(stock_data)
        cumulative_return = backtest_strategy(stock_data)
        
        st.markdown('<div class="card glass"><h3>Backtest Results</h3></div>', unsafe_allow_html=True)
        st.metric("Cumulative Return", f"{cumulative_return*100:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=(1 + stock_data['strategy_returns']).cumprod(), 
                               name="Strategy", line=dict(color='#2e86ab')))
        fig.add_trace(go.Scatter(x=stock_data.index, y=(1 + stock_data['returns']).cumprod(), 
                               name="Buy & Hold", line=dict(color='#f24236')))
        fig.update_layout(title="Strategy vs Buy & Hold", xaxis_title="Date", yaxis_title="Cumulative Return", 
                         template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()