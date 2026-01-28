# ML Analytics Suite

A comprehensive machine learning analytics platform for Indian markets featuring:

## 🚀 Features

### 📈 Stock Price Predictor (LSTM)
- Real-time data from NSE/BSE via Yahoo Finance
- LSTM neural network for time-series prediction
- Technical indicators: RSI, MACD, Bollinger Bands
- Support for stocks like SUBEX, TCS, INFOSYS, WIPRO, RELIANCE
- 30/60/90 day price forecasting

### 🛒 Sales Forecasting (Gradient Boosting)  
- BigMart-style retail data generation
- Gradient Boosting Regressor for demand prediction
- Category-wise sales forecasting
- Inventory optimization recommendations
- Feature importance analysis

### 👥 Customer Churn Analyzer (Logistic Regression)
- Fintech customer data simulation
- Logistic Regression for churn classification
- Risk scoring: High/Medium/Low
- Feature impact analysis
- Retention recommendations

## 📦 Installation

```bash
# Navigate to project directory
cd ml_analytics_suite

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage

### Run the GUI Dashboard
```bash
python main.py
```

### Run Individual Models (Demo)
```bash
# Stock Predictor
python models/stock_predictor.py

# Sales Forecaster
python models/sales_forecaster.py

# Churn Analyzer
python models/churn_analyzer.py
```

## 📁 Project Structure

```
ml_analytics_suite/
├── main.py                    # Unified GUI application
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── models/
│   ├── stock_predictor.py     # LSTM stock price model
│   ├── sales_forecaster.py    # Gradient Boosting sales model
│   └── churn_analyzer.py      # Logistic Regression churn model
├── data/
│   └── (sample datasets)
├── utils/
│   ├── preprocessing.py       # Data preprocessing utilities
│   └── visualization.py       # Chart generation helpers
└── config/
    └── settings.py            # Configuration constants
```

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+ (for LSTM)
- scikit-learn 1.3+
- CustomTkinter 5.2+
- yfinance 0.2.28+
- XGBoost 1.7+

## 📊 Screenshots

The application features a modern dark-themed interface with:
- Tabbed navigation for each model
- Interactive charts and visualizations
- Real-time prediction capabilities
- Statistical analysis panels

## 🇮🇳 Indian Market Focus

- NSE/BSE stock symbols
- ₹ (INR) currency formatting
- Indian retail patterns (BigMart-style)
- Fintech use cases for Indian banks

## 📝 License

MIT License - Free for educational and commercial use.
