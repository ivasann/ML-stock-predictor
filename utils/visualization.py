"""
Visualization utilities for ML Analytics Suite
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def setup_dark_style():
    """Configure matplotlib for dark theme."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1A1A2E',
        'axes.facecolor': '#16213E',
        'axes.edgecolor': '#EAEAEA',
        'axes.labelcolor': '#EAEAEA',
        'text.color': '#EAEAEA',
        'xtick.color': '#EAEAEA',
        'ytick.color': '#EAEAEA',
        'grid.color': '#2A2A4E',
        'legend.facecolor': '#16213E',
        'legend.edgecolor': '#EAEAEA',
    })


def create_stock_chart(
    df: pd.DataFrame,
    predicted_dates: Optional[List] = None,
    predicted_prices: Optional[np.ndarray] = None,
    indicators: Optional[List[str]] = None,
    title: str = "Stock Price Prediction",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create stock price chart with actual, predicted values and indicators.
    
    Args:
        df: DataFrame with at least Date and Close columns
        predicted_dates: List of future dates (optional)
        predicted_prices: array of predicted prices (optional)
        indicators: List of indicators to overlay ('MA_7', 'BB', etc.)
        title: Chart title
        figsize: Figure dimensions
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual price
    ax.plot(df.index, df['Close'], label='Actual Price', color='#1E88E5', linewidth=2)
    
    # Plot overlays
    if indicators:
        if 'MA' in indicators or 'Moving Averages' in indicators:
            if 'MA_7' in df.columns: ax.plot(df.index, df['MA_7'], label='MA 7', alpha=0.7, linestyle='--')
            if 'MA_21' in df.columns: ax.plot(df.index, df['MA_21'], label='MA 21', alpha=0.7, linestyle='--')
            if 'MA_50' in df.columns: ax.plot(df.index, df['MA_50'], label='MA 50', alpha=0.7, linestyle='--')
        
        if 'BB' in indicators or 'Bollinger Bands' in indicators:
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1, label='Bollinger Bands')
                ax.plot(df.index, df['BB_Upper'], color='gray', alpha=0.3, linewidth=1)
                ax.plot(df.index, df['BB_Lower'], color='gray', alpha=0.3, linewidth=1)

    # Plot predictions
    if predicted_dates is not None and predicted_prices is not None:
        ax.plot(predicted_dates, predicted_prices, 
                label='Predicted Price', color='#43A047', linewidth=2, linestyle='--')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (₹)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def create_indicators_chart(
    df: pd.DataFrame,
    indicator: str = 'RSI',
    title: str = "Technical Indicator",
    figsize: Tuple[int, int] = (12, 4)
) -> Figure:
    """Create a chart for a specific technical indicator."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if indicator == 'RSI' and 'RSI' in df.columns:
        ax.plot(df.index, df['RSI'], color='#FB8C00', linewidth=1.5)
        ax.axhline(70, color='#E53935', linestyle='--', alpha=0.5)
        ax.axhline(30, color='#43A047', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI')
    
    elif indicator == 'MACD' and 'MACD' in df.columns:
        ax.plot(df.index, df['MACD'], label='MACD', color='#1E88E5')
        ax.plot(df.index, df['MACD_Signal'], label='Signal', color='#FB8C00')
        # Histogram
        hist = df['MACD'] - df['MACD_Signal']
        ax.bar(df.index, hist, label='Hist', color=['#43A047' if x > 0 else '#E53935' for x in hist], alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylabel('MACD')

    elif indicator == 'Stochastic' and 'Stochastic_K' in df.columns:
        ax.plot(df.index, df['Stochastic_K'], label='%K', color='#1E88E5')
        ax.plot(df.index, df['Stochastic_D'], label='%D', color='#FB8C00')
        ax.axhline(80, color='#E53935', linestyle='--', alpha=0.5)
        ax.axhline(20, color='#43A047', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylabel('Stochastic')
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def create_forecast_chart(
    categories: List[str],
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Sales Forecast",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create bar chart comparing actual vs forecasted sales.
    
    Args:
        categories: Category labels
        actual: Actual sales values
        predicted: Predicted sales values
        title: Chart title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, actual, width, label='Actual', color='#1E88E5', alpha=0.8)
    bars2 = ax.bar(x + width/2, predicted, width, label='Predicted', color='#43A047', alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Sales (₹)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'₹{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_churn_gauge(
    probability: float,
    title: str = "Churn Risk",
    figsize: Tuple[int, int] = (6, 4)
) -> Figure:
    """
    Create a gauge chart showing churn probability.
    
    Args:
        probability: Churn probability (0-1)
        title: Chart title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Determine color based on risk level
    if probability >= 0.7:
        color = '#E53935'  # High risk - Red
        risk_label = 'HIGH RISK'
    elif probability >= 0.4:
        color = '#FB8C00'  # Medium risk - Orange
        risk_label = 'MEDIUM RISK'
    else:
        color = '#43A047'  # Low risk - Green
        risk_label = 'LOW RISK'
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, [1] * 100, color='#2A2A4E', linewidth=20, alpha=0.3)
    
    filled_theta = np.linspace(0, np.pi * probability, 100)
    ax.plot(filled_theta, [1] * 100, color=color, linewidth=20)
    
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Add text
    ax.text(np.pi/2, 0.5, f'{probability*100:.1f}%', 
            ha='center', va='center', fontsize=24, fontweight='bold', color=color)
    ax.text(np.pi/2, 0.1, risk_label, 
            ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return fig


def create_feature_importance_chart(
    features: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create horizontal bar chart showing feature importances.
    
    Args:
        features: Feature names
        importances: Importance scores
        title: Chart title
        top_n: Number of top features to show
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort and get top N
    indices = np.argsort(importances)[-top_n:]
    top_features = [features[i] for i in indices]
    top_importances = importances[indices]
    
    # Create gradient colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_importances, color=colors)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, top_importances):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_confusion_matrix_chart(
    cm: np.ndarray,
    labels: List[str] = ['No Churn', 'Churn'],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Create confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        title: Chart title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    
    plt.tight_layout()
    return fig


def create_time_series_decomposition(
    data: pd.Series,
    title: str = "Time Series Analysis",
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Create time series decomposition chart.
    
    Args:
        data: Time series data
        title: Chart title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Original
    axes[0].plot(data.index, data.values, color='#1E88E5', linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=10)
    axes[0].set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Trend (rolling mean)
    trend = data.rolling(window=30).mean()
    axes[1].plot(data.index, trend.values, color='#43A047', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=10)
    
    # Seasonality (detrended)
    seasonal = data - trend
    axes[2].plot(data.index, seasonal.values, color='#FB8C00', linewidth=1)
    axes[2].set_ylabel('Seasonal', fontsize=10)
    
    # Residual
    residual = data - trend - seasonal.rolling(window=7).mean()
    axes[3].plot(data.index, residual.values, color='#E53935', linewidth=1, alpha=0.7)
    axes[3].set_ylabel('Residual', fontsize=10)
    axes[3].set_xlabel('Date', fontsize=11)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def embed_figure_in_tk(fig: Figure, parent_widget) -> FigureCanvasTkAgg:
    """
    Embed a matplotlib figure into a Tkinter widget.
    
    Args:
        fig: Matplotlib Figure
        parent_widget: Tkinter parent widget
        
    Returns:
        FigureCanvasTkAgg canvas
    """
    canvas = FigureCanvasTkAgg(fig, master=parent_widget)
    canvas.draw()
    return canvas
