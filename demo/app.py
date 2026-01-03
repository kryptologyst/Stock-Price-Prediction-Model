"""Streamlit demo application for stock price prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import YahooDataLoader, DataPreprocessor
from features import TechnicalFeatureEngineer
from models import LSTMTrainer
from backtest import VectorBTBacktest
from utils import set_seed


# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration."""
    return OmegaConf.load("configs/config.yaml")

def load_model(symbol: str, config):
    """Load trained model for symbol."""
    try:
        model_path = f"models/lstm_{symbol}.pt"
        trainer = LSTMTrainer(config.models.lstm)
        trainer.load_model(model_path)
        return trainer
    except:
        return None

def get_stock_data(symbol: str, config):
    """Get stock data for symbol."""
    data_loader = YahooDataLoader(config.data)
    data = data_loader.download_data(symbol)
    return data

def create_features(data: pd.DataFrame, config):
    """Create features for the data."""
    feature_engineer = TechnicalFeatureEngineer(config.features)
    features_df, feature_columns = feature_engineer.prepare_features(data)
    data_with_features = pd.concat([data, features_df], axis=1)
    return data_with_features.dropna(), feature_columns

def make_prediction(trainer, data: pd.DataFrame, config):
    """Make prediction using the trained model."""
    preprocessor = DataPreprocessor(config.features)
    X, y = preprocessor.create_sequences(data, "close")
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    predictions = trainer.predict(X)
    return predictions.flatten()

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This is a research and educational demonstration only.</strong></p>
        <ul>
            <li>This model is NOT financial advice</li>
            <li>Past performance does not guarantee future results</li>
            <li>All predictions may be inaccurate</li>
            <li>Do not use this for actual trading decisions</li>
            <li>Always consult with qualified financial advisors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=config.data.symbols,
        index=0
    )
    
    # Date range
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime(config.data.start_date).date()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime(config.data.end_date).date()
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    lookback_window = st.sidebar.slider(
        "Lookback Window",
        min_value=10,
        max_value=120,
        value=config.features.lookback_window
    )
    
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        options=[1, 5, 10, 20],
        index=0
    )
    
    # Load data
    with st.spinner(f"Loading data for {symbol}..."):
        try:
            data = get_stock_data(symbol, config)
            if data.empty:
                st.error(f"No data available for {symbol}")
                return
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Create features
    with st.spinner("Creating features..."):
        data_with_features, feature_columns = create_features(data, config)
    
    # Load model
    with st.spinner("Loading model..."):
        trainer = load_model(symbol, config)
        if trainer is None:
            st.error(f"No trained model found for {symbol}. Please run training first.")
            return
    
    # Make predictions
    with st.spinner("Making predictions..."):
        predictions = make_prediction(trainer, data_with_features, config)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{symbol} Stock Analysis")
        
        # Price chart with predictions
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price History & Predictions", "Volume"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price data
        price_data = data_with_features['close'].iloc[config.features.lookback_window:]
        pred_data = pd.Series(predictions, index=price_data.index)
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                name='Actual Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=pred_data.index,
                y=pred_data.values,
                name='Predicted Price',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add volume
        volume_data = data_with_features['volume'].iloc[config.features.lookback_window:]
        fig.add_trace(
            go.Bar(
                x=volume_data.index,
                y=volume_data.values,
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title=f"{symbol} Stock Price Prediction"
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        # Calculate metrics
        actual_prices = data_with_features['close'].iloc[config.features.lookback_window:]
        actual_prices = actual_prices.reset_index(drop=True)
        
        # Calculate prediction metrics
        mse = np.mean((actual_prices - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_prices - predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        
        # Display metrics
        st.metric("RMSE", f"${rmse:.2f}")
        st.metric("MAE", f"${mae:.2f}")
        st.metric("MAPE", f"{mape:.2f}%")
        
        # Feature importance (placeholder)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_columns[:10],  # Top 10 features
            'Importance': np.random.random(10)  # Placeholder
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=True)
        
        fig_importance = go.Figure(go.Bar(
            x=feature_importance['Importance'],
            y=feature_importance['Feature'],
            orientation='h'
        ))
        fig_importance.update_layout(
            title="Top 10 Features",
            height=400
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Trading signals section
    st.subheader("Trading Signals")
    
    # Create signals
    signal_threshold = st.slider("Signal Threshold (%)", 0.5, 5.0, 2.0, 0.1)
    
    # Calculate signals
    returns = (predictions - actual_prices) / actual_prices
    signals = pd.Series(0, index=range(len(returns)))
    signals[returns > signal_threshold / 100] = 1  # Buy
    signals[returns < -signal_threshold / 100] = -1  # Sell
    
    # Display signal statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Buy Signals", f"{sum(signals == 1)}")
    with col2:
        st.metric("Sell Signals", f"{sum(signals == -1)}")
    with col3:
        st.metric("Hold Signals", f"{sum(signals == 0)}")
    
    # Signal chart
    fig_signals = go.Figure()
    
    # Add price
    fig_signals.add_trace(go.Scatter(
        x=list(range(len(actual_prices))),
        y=actual_prices,
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add buy signals
    buy_signals = actual_prices[signals == 1]
    fig_signals.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals.values,
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Buy Signal'
    ))
    
    # Add sell signals
    sell_signals = actual_prices[signals == -1]
    fig_signals.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals.values,
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell Signal'
    ))
    
    fig_signals.update_layout(
        title="Trading Signals",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig_signals, use_container_width=True)
    
    # Model information
    with st.expander("Model Information"):
        st.write(f"**Model Type:** LSTM Neural Network")
        st.write(f"**Lookback Window:** {lookback_window} days")
        st.write(f"**Features Used:** {len(feature_columns)}")
        st.write(f"**Training Period:** {config.data.start_date} to {config.data.train_end_date}")
        st.write(f"**Test Period:** {config.data.test_start_date} to {config.data.end_date}")
        
        st.write("**Feature List:**")
        for i, feature in enumerate(feature_columns):
            st.write(f"{i+1}. {feature}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>This is a research demonstration. Not financial advice.</p>
        <p>Built with Streamlit, PyTorch, and VectorBT</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
