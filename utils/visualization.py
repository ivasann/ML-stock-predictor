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
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: Optional[np.ndarray] = None,
    title: str = "Stock Price Prediction",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create stock price chart with actual vs predicted values.
    
    Args:
        dates: Date index
        actual: Actual prices
        predicted: Predicted prices (optional)
        title: Chart title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dates, actual, label='Actual', color='#1E88E5', linewidth=2)
    
    if predicted is not None:
        ax.plot(dates[-len(predicted):], predicted, 
                label='Predicted', color='#43A047', linewidth=2, linestyle='--')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (₹)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
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
