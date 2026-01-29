# Configuration settings for ML Analytics Suite

# Stock Predictor Settings
STOCK_SYMBOLS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'SBI': 'SBIN.NS',
    'BHARTI': 'BHARTIARTL.NS',
    'ADANI': 'ADANIPORTS.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
}

LSTM_CONFIG = {
    'sequence_length': 60,
    'epochs': 50,
    'batch_size': 32,
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate': 0.2,
}

# Sales Forecaster Settings
OUTLET_TYPES = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
ITEM_CATEGORIES = ['Food', 'Drinks', 'Non-Consumable']
OUTLET_SIZES = ['Small', 'Medium', 'High']

GB_CONFIG = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
}

# Churn Analyzer Settings
RISK_THRESHOLDS = {
    'high': 0.7,
    'medium': 0.4,
    'low': 0.0,
}

LR_CONFIG = {
    'max_iter': 1000,
    'C': 1.0,
    'solver': 'lbfgs',
}

# GUI Settings
THEME = 'dark'
WINDOW_SIZE = '1400x900'
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#43A047',
    'danger': '#E53935',
    'warning': '#FB8C00',
    'background': '#1A1A2E',
    'card': '#16213E',
    'text': '#EAEAEA',
}
