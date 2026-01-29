"""
Data preprocessing utilities for ML Analytics Suite
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training from time series data.
    
    Args:
        data: Scaled time series data
        seq_length: Number of time steps to look back
        
    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def scale_data(data: pd.Series, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale data to range [0, 1] using MinMaxScaler.
    
    Args:
        data: Pandas series to scale
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (scaled data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    else:
        scaled_data = scaler.transform(data.values.reshape(-1, 1))
    return scaled_data, scaler


def encode_categorical(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns using LabelEncoder.
    
    Args:
        df: DataFrame to encode
        columns: List of column names to encode
        
    Returns:
        Tuple of (encoded DataFrame, encoders dict)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            
    return df_encoded, encoders


def prepare_features(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        
    Returns:
        Tuple of (X features, y target)
    """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame with potential missing values
        strategy: 'mean', 'median', 'mode', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif df_clean[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                
    return df_clean


def add_technical_indicators(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """
    Add technical indicators for stock analysis.
    
    Args:
        df: DataFrame with stock prices
        close_col: Name of closing price column
        
    Returns:
        DataFrame with added indicators
    """
    df_ind = df.copy()
    close = df_ind[close_col]
    
    # Moving Averages
    df_ind['MA_7'] = close.rolling(window=7).mean()
    df_ind['MA_21'] = close.rolling(window=21).mean()
    df_ind['MA_50'] = close.rolling(window=50).mean()
    
    # EMA
    df_ind['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df_ind['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    
    # RSI (14-day)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ind['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df_ind['MACD'] = df_ind['EMA_12'] - df_ind['EMA_26']
    df_ind['MACD_Signal'] = df_ind['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df_ind['BB_Middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df_ind['BB_Upper'] = df_ind['BB_Middle'] + (bb_std * 2)
    df_ind['BB_Lower'] = df_ind['BB_Middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high = df_ind['High']
    low = df_ind['Low']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df_ind['ATR'] = tr.rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df_ind['Stochastic_K'] = (close - low_14) * 100 / (high_14 - low_14)
    df_ind['Stochastic_D'] = df_ind['Stochastic_K'].rolling(window=3).mean()
    
    # Volume Moving Average (if Volume exists)
    if 'Volume' in df_ind.columns:
        df_ind['Volume_MA'] = df_ind['Volume'].rolling(window=20).mean()
    
    return df_ind


def calculate_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features for churn prediction.
    
    Args:
        df: Customer DataFrame
        
    Returns:
        DataFrame with additional churn-related features
    """
    df_churn = df.copy()
    
    # Engagement score (if components exist)
    if all(col in df.columns for col in ['monthly_transactions', 'app_sessions']):
        df_churn['engagement_score'] = (
            df_churn['monthly_transactions'] * 0.4 + 
            df_churn['app_sessions'] * 0.6
        )
    
    # Balance trend (if balance columns exist)
    if 'current_balance' in df.columns and 'avg_balance' in df.columns:
        df_churn['balance_trend'] = df_churn['current_balance'] / (df_churn['avg_balance'] + 1)
    
    # Customer tenure buckets
    if 'tenure_months' in df.columns:
        df_churn['tenure_bucket'] = pd.cut(
            df_churn['tenure_months'],
            bins=[0, 6, 12, 24, 48, 999],
            labels=['0-6m', '6-12m', '1-2y', '2-4y', '4y+']
        )
    
    return df_churn
