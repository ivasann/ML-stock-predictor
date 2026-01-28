"""
Gradient Boosting Sales Forecaster for Indian Retailers (BigMart-style)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, List, Optional
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import GB_CONFIG, OUTLET_TYPES, ITEM_CATEGORIES, OUTLET_SIZES


class SalesForecaster:
    """Gradient Boosting-based sales forecaster for retail inventory optimization."""
    
    def __init__(self):
        """Initialize the sales forecaster."""
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.config = GB_CONFIG
        self.feature_names = []
        self.feature_importances = None
        self.data = None
        
    def generate_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate BigMart-style sample retail data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic retail data
        """
        np.random.seed(42)
        
        # Generate item identifiers
        item_ids = [f'FD{i:04d}' for i in range(n_samples)]
        
        # Item features
        item_weight = np.random.uniform(4, 25, n_samples)
        item_fat_content = np.random.choice(['Low Fat', 'Regular'], n_samples, p=[0.4, 0.6])
        item_visibility = np.random.uniform(0.01, 0.25, n_samples)
        item_type = np.random.choice([
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
            'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
            'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
            'Breads', 'Starchy Foods', 'Others', 'Seafood'
        ], n_samples)
        
        # MRP with realistic distribution for Indian retail
        item_mrp = np.random.choice([
            np.random.uniform(30, 100),   # Budget items
            np.random.uniform(100, 200),  # Mid-range
            np.random.uniform(200, 500),  # Premium
        ], n_samples, p=[0.5, 0.35, 0.15])
        item_mrp = np.array([np.random.uniform(30, 100) if np.random.random() < 0.5 
                           else np.random.uniform(100, 300) for _ in range(n_samples)])
        
        # Outlet features
        outlet_ids = np.random.choice([f'OUT{i:03d}' for i in range(10)], n_samples)
        outlet_establishment_year = np.random.choice(range(1985, 2020), n_samples)
        outlet_size = np.random.choice(OUTLET_SIZES, n_samples, p=[0.3, 0.45, 0.25])
        outlet_location = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_samples, p=[0.3, 0.4, 0.3])
        outlet_type = np.random.choice(OUTLET_TYPES, n_samples, p=[0.15, 0.45, 0.25, 0.15])
        
        # Generate target (sales) based on features with realistic patterns
        base_sales = item_mrp * 20  # Base sales proportional to MRP
        
        # Modifiers
        type_modifier = np.where(np.isin(outlet_type, ['Supermarket Type2', 'Supermarket Type3']), 1.3, 1.0)
        size_modifier = np.where(outlet_size == 'High', 1.4, np.where(outlet_size == 'Medium', 1.2, 1.0))
        location_modifier = np.where(outlet_location == 'Tier 1', 1.5, np.where(outlet_location == 'Tier 2', 1.2, 1.0))
        visibility_modifier = 1 + (item_visibility * 2)
        
        item_outlet_sales = base_sales * type_modifier * size_modifier * location_modifier * visibility_modifier
        item_outlet_sales += np.random.normal(0, item_outlet_sales * 0.1)  # Add noise
        item_outlet_sales = np.maximum(item_outlet_sales, 50)  # Minimum sales
        
        self.data = pd.DataFrame({
            'Item_Identifier': item_ids,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_ids,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location,
            'Outlet_Type': outlet_type,
            'Item_Outlet_Sales': item_outlet_sales
        })
        
        return self.data
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load retail data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with retail data
        """
        self.data = pd.read_csv(filepath)
        return self.data
    
    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess data for model training.
        
        Args:
            df: DataFrame to preprocess (uses self.data if None)
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        # Handle missing values
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        df['Outlet_Size'].fillna('Medium', inplace=True)
        
        # Standardize fat content
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 'low fat': 'Low Fat',
            'reg': 'Regular', 'regular': 'Regular'
        })
        
        # Calculate outlet age
        current_year = 2026
        df['Outlet_Age'] = current_year - df['Outlet_Establishment_Year']
        
        # Create item category (simplified)
        food_items = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast', 
                      'Frozen Foods', 'Snack Foods', 'Baking Goods', 'Canned',
                      'Breads', 'Starchy Foods', 'Seafood']
        drink_items = ['Soft Drinks', 'Hard Drinks']
        
        df['Item_Category'] = df['Item_Type'].apply(
            lambda x: 'Food' if x in food_items else ('Drinks' if x in drink_items else 'Non-Consumable')
        )
        
        # Encode categorical variables
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Category']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col + '_Encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    df[col + '_Encoded'] = self.encoders[col].transform(df[col])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (X features, y target)
        """
        self.feature_names = [
            'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age',
            'Item_Fat_Content_Encoded', 'Item_Type_Encoded', 'Outlet_Size_Encoded',
            'Outlet_Location_Type_Encoded', 'Outlet_Type_Encoded', 'Item_Category_Encoded'
        ]
        
        # Filter to available features
        available_features = [f for f in self.feature_names if f in df.columns]
        self.feature_names = available_features
        
        X = df[self.feature_names].values
        y = df['Item_Outlet_Sales'].values
        
        return X, y
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None, 
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the Gradient Boosting model.
        
        Args:
            X: Feature matrix (generates if None)
            y: Target vector (generates if None)
            test_size: Proportion for test split
            
        Returns:
            Dictionary with training metrics
        """
        if X is None or y is None:
            if self.data is None:
                self.generate_sample_data()
            df_processed = self.preprocess_data()
            X, y = self.prepare_features(df_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.model = GradientBoostingRegressor(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Store feature importances
        self.feature_importances = self.model.feature_importances_
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make sales predictions for new data.
        
        Args:
            df: DataFrame with item/outlet features
            
        Returns:
            Predicted sales values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df_processed = self.preprocess_data(df)
        X, _ = self.prepare_features(df_processed)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_inventory_recommendations(self, predictions: np.ndarray, 
                                       df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate inventory optimization recommendations.
        
        Args:
            predictions: Predicted sales values
            df: Original DataFrame with item info
            
        Returns:
            DataFrame with recommendations
        """
        recommendations = df[['Item_Identifier', 'Item_Type', 'Outlet_Identifier']].copy()
        recommendations['Predicted_Sales'] = predictions
        
        # Calculate safety stock (20% buffer)
        recommendations['Safety_Stock'] = predictions * 0.2
        
        # Recommended order quantity
        recommendations['Recommended_Order'] = np.ceil(predictions * 1.2)  # 20% buffer
        
        # Priority based on predicted sales
        recommendations['Priority'] = pd.cut(
            predictions, 
            bins=[0, 1000, 2500, 5000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Estimated revenue
        if 'Item_MRP' in df.columns:
            recommendations['Est_Revenue'] = predictions * df['Item_MRP'] * 0.85  # 15% margin
        
        return recommendations.sort_values('Predicted_Sales', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importances is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.feature_importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def forecast_by_category(self) -> pd.DataFrame:
        """
        Get sales forecast aggregated by category.
        
        Returns:
            DataFrame with category-wise forecasts
        """
        if self.data is None:
            self.generate_sample_data()
        
        df_processed = self.preprocess_data()
        predictions = self.predict(self.data)
        
        df_processed['Predicted_Sales'] = predictions
        
        category_forecast = df_processed.groupby('Item_Category').agg({
            'Predicted_Sales': ['sum', 'mean', 'count'],
            'Item_Outlet_Sales': 'sum'
        }).round(2)
        
        category_forecast.columns = ['Total_Predicted', 'Avg_Predicted', 
                                     'Item_Count', 'Total_Actual']
        category_forecast['Accuracy'] = (
            1 - abs(category_forecast['Total_Predicted'] - category_forecast['Total_Actual']) 
            / category_forecast['Total_Actual']
        ) * 100
        
        return category_forecast.reset_index()
    
    def save_model(self, path: str = 'models/sales_model.pkl'):
        """Save trained model to disk."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.encoders,
                'feature_names': self.feature_names
            }, path)
    
    def load_model(self, path: str = 'models/sales_model.pkl'):
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoders = data['encoders']
        self.feature_names = data['feature_names']


def run_sales_forecast_demo():
    """Run a demonstration of the sales forecaster."""
    print("=" * 60)
    print("Gradient Boosting Sales Forecaster - Demo")
    print("=" * 60)
    
    forecaster = SalesForecaster()
    
    print("\n📦 Generating BigMart-style sample data...")
    data = forecaster.generate_sample_data(5000)
    print(f"   Generated {len(data)} records")
    print(f"   Outlets: {data['Outlet_Identifier'].nunique()}")
    print(f"   Item types: {data['Item_Type'].nunique()}")
    
    print("\n🔧 Training Gradient Boosting model...")
    metrics = forecaster.train()
    print(f"   Train RMSE: ₹{metrics['train_rmse']:.2f}")
    print(f"   Test RMSE: ₹{metrics['test_rmse']:.2f}")
    print(f"   Test R² Score: {metrics['test_r2']:.4f}")
    
    print("\n📊 Feature Importance:")
    importance = forecaster.get_feature_importance()
    for _, row in importance.head(5).iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n📈 Category-wise Forecast:")
    category_forecast = forecaster.forecast_by_category()
    for _, row in category_forecast.iterrows():
        print(f"   {row['Item_Category']}: ₹{row['Total_Predicted']:,.0f} "
              f"(Accuracy: {row['Accuracy']:.1f}%)")
    
    print("\n✅ Demo complete!")
    return forecaster


if __name__ == '__main__':
    run_sales_forecast_demo()
