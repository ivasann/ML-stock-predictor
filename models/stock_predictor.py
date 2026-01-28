"""
LSTM Stock Price Predictor for NSE/BSE stocks
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    # Fallback for systems without TensorFlow
    Sequential = None
    print("Warning: TensorFlow not available. Using simplified model.")

from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCK_SYMBOLS, LSTM_CONFIG
from utils.preprocessing import create_sequences, scale_data, add_technical_indicators


class StockPredictor:
    """LSTM-based stock price predictor for Indian markets (NSE/BSE)."""
    
    def __init__(self, symbol: str = 'SUBEX'):
        """
        Initialize the stock predictor.
        
        Args:
            symbol: Stock symbol (short name like 'SUBEX', 'TCS')
        """
        self.symbol = symbol
        self.ticker = STOCK_SYMBOLS.get(symbol.upper(), f"{symbol}.NS")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.config = LSTM_CONFIG
        self.history = None
        self.data = None
        
    def fetch_data(self, period: str = '2y') -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            period: Data period ('1y', '2y', '5y', 'max')
            
        Returns:
            DataFrame with stock data
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Add technical indicators
            self.data = add_technical_indicators(self.data)
            self.data = self.data.dropna()
            
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return sample data for demo
            return self._generate_sample_data()
    
    def _generate_sample_data(self, days: int = 500) -> pd.DataFrame:
        """Generate sample stock data for demo purposes."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement
        np.random.seed(42)
        base_price = 85  # Subex-like price around ₹85
        returns = np.random.normal(0.0005, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some volatility clustering
        volatility = np.abs(np.random.normal(0, 1, days))
        prices = prices * (1 + volatility * 0.01)
        
        self.data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, days)
        }, index=dates)
        
        self.data = add_technical_indicators(self.data)
        self.data = self.data.dropna()
        
        return self.data
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            self.fetch_data()
        
        # Use closing price
        close_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = create_sequences(scaled_data, self.config['sequence_length'])
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self) -> Sequential:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        if Sequential is None:
            raise ImportError("TensorFlow is required for LSTM model")
        
        model = Sequential([
            LSTM(
                self.config['lstm_units_1'],
                return_sequences=True,
                input_shape=(self.config['sequence_length'], 1)
            ),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units_2'], return_sequences=False),
            Dropout(self.config['dropout_rate']),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted prices (inverse scaled)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def predict_future(self, days: int = 30) -> Tuple[List[datetime], np.ndarray]:
        """
        Predict future stock prices.
        
        Args:
            days: Number of days to predict ahead
            
        Returns:
            Tuple of (future dates, predicted prices)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get last sequence
        last_sequence = self.scaler.transform(
            self.data['Close'].values[-self.config['sequence_length']:].reshape(-1, 1)
        )
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next day
            next_pred = self.model.predict(
                current_sequence.reshape(1, self.config['sequence_length'], 1),
                verbose=0
            )
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Generate future dates
        last_date = self.data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return future_dates, predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.model.predict(X_test, verbose=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = np.mean((predictions - y_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_actual))
        mape = np.mean(np.abs((y_actual - predictions) / y_actual)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def save_model(self, path: str = 'models/stock_model.h5'):
        """Save trained model to disk."""
        if self.model is not None:
            self.model.save(path)
            joblib.dump(self.scaler, path.replace('.h5', '_scaler.pkl'))
    
    def load_model(self, path: str = 'models/stock_model.h5'):
        """Load trained model from disk."""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        self.scaler = joblib.load(path.replace('.h5', '_scaler.pkl'))
    
    def get_stock_info(self) -> Dict:
        """Get current stock information."""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            return {
                'name': info.get('longName', self.symbol),
                'symbol': self.ticker,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
            }
        except:
            return {'name': self.symbol, 'symbol': self.ticker}


def run_stock_prediction_demo():
    """Run a demonstration of the stock predictor."""
    print("=" * 60)
    print("LSTM Stock Price Predictor - Demo")
    print("=" * 60)
    
    # Initialize predictor for Subex
    predictor = StockPredictor('SUBEX')
    
    print(f"\n📈 Fetching data for {predictor.ticker}...")
    data = predictor.fetch_data('2y')
    print(f"   Loaded {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Current price: ₹{data['Close'].iloc[-1]:.2f}")
    
    print("\n🔧 Preparing data for training...")
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    if Sequential is not None:
        print("\n🧠 Building and training LSTM model...")
        predictor.build_model()
        predictor.train(X_train, y_train, X_test, y_test)
        
        print("\n📊 Evaluating model...")
        metrics = predictor.evaluate(X_test, y_test)
        print(f"   RMSE: ₹{metrics['rmse']:.2f}")
        print(f"   MAE: ₹{metrics['mae']:.2f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        print("\n🔮 Predicting next 30 days...")
        future_dates, predictions = predictor.predict_future(30)
        print(f"   Predicted price range: ₹{min(predictions):.2f} - ₹{max(predictions):.2f}")
    else:
        print("\n⚠️ TensorFlow not available - skipping model training")
    
    print("\n✅ Demo complete!")
    return predictor


if __name__ == '__main__':
    run_stock_prediction_demo()
