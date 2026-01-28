"""
Logistic Regression Customer Churn Analyzer for Fintech
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Tuple, Dict, List, Optional
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LR_CONFIG, RISK_THRESHOLDS


class ChurnAnalyzer:
    """Logistic Regression-based customer churn analyzer for fintech services."""
    
    def __init__(self):
        """Initialize the churn analyzer."""
        self.model = None
        self.scaler = StandardScaler()
        self.config = LR_CONFIG
        self.feature_names = []
        self.coefficients = None
        self.data = None
        self.encoders = {}
        
    def generate_sample_data(self, n_customers: int = 3000) -> pd.DataFrame:
        """
        Generate synthetic fintech customer data.
        
        Args:
            n_customers: Number of customer records
            
        Returns:
            DataFrame with customer data
        """
        np.random.seed(42)
        
        # Customer demographics
        customer_ids = [f'CUST{i:06d}' for i in range(n_customers)]
        age = np.random.randint(18, 65, n_customers)
        gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.52, 0.48])
        
        # Location (Indian cities)
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 
                  'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow']
        location = np.random.choice(cities, n_customers)
        
        # Account features
        tenure_months = np.random.exponential(24, n_customers).astype(int) + 1
        tenure_months = np.clip(tenure_months, 1, 120)
        
        account_type = np.random.choice(['Savings', 'Current', 'Premium'], 
                                        n_customers, p=[0.6, 0.25, 0.15])
        
        # Financial activity
        current_balance = np.random.exponential(50000, n_customers)
        avg_balance = current_balance * np.random.uniform(0.8, 1.2, n_customers)
        
        monthly_transactions = np.random.poisson(15, n_customers)
        avg_transaction_amount = np.random.exponential(2000, n_customers)
        
        # Digital engagement
        app_sessions = np.random.poisson(20, n_customers)
        web_logins = np.random.poisson(5, n_customers)
        
        # Service issues
        support_tickets = np.random.poisson(1, n_customers)
        complaints_last_6m = np.random.poisson(0.5, n_customers)
        
        # Product usage
        has_credit_card = np.random.choice([0, 1], n_customers, p=[0.4, 0.6])
        has_loan = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
        has_investment = np.random.choice([0, 1], n_customers, p=[0.75, 0.25])
        num_products = has_credit_card + has_loan + has_investment + 1
        
        # Generate churn label based on realistic patterns
        churn_probability = np.zeros(n_customers)
        
        # Factors that increase churn
        churn_probability += (complaints_last_6m > 1) * 0.2
        churn_probability += (tenure_months < 6) * 0.15
        churn_probability += (monthly_transactions < 5) * 0.1
        churn_probability += (app_sessions < 5) * 0.1
        churn_probability += (support_tickets > 2) * 0.15
        churn_probability += (num_products == 1) * 0.1
        
        # Factors that decrease churn
        churn_probability -= (tenure_months > 24) * 0.1
        churn_probability -= (has_loan == 1) * 0.15
        churn_probability -= (num_products >= 3) * 0.1
        churn_probability -= (current_balance > 100000) * 0.1
        
        # Normalize and add noise
        churn_probability = np.clip(churn_probability, 0.05, 0.85)
        churn_probability += np.random.normal(0, 0.1, n_customers)
        churn_probability = np.clip(churn_probability, 0, 1)
        
        # Generate actual churn labels
        churned = (np.random.random(n_customers) < churn_probability).astype(int)
        
        self.data = pd.DataFrame({
            'customer_id': customer_ids,
            'age': age,
            'gender': gender,
            'location': location,
            'tenure_months': tenure_months,
            'account_type': account_type,
            'current_balance': current_balance.round(2),
            'avg_balance': avg_balance.round(2),
            'monthly_transactions': monthly_transactions,
            'avg_transaction_amount': avg_transaction_amount.round(2),
            'app_sessions': app_sessions,
            'web_logins': web_logins,
            'support_tickets': support_tickets,
            'complaints_last_6m': complaints_last_6m,
            'has_credit_card': has_credit_card,
            'has_loan': has_loan,
            'has_investment': has_investment,
            'num_products': num_products,
            'churned': churned
        })
        
        return self.data
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load customer data from CSV file."""
        self.data = pd.read_csv(filepath)
        return self.data
    
    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess data for model training.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        # Calculate derived features
        df['balance_ratio'] = df['current_balance'] / (df['avg_balance'] + 1)
        df['engagement_score'] = (
            df['app_sessions'] * 0.5 + 
            df['web_logins'] * 0.3 + 
            df['monthly_transactions'] * 0.2
        )
        df['issue_score'] = df['support_tickets'] + df['complaints_last_6m'] * 2
        
        # Tenure buckets
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 12, 24, 48, 999],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'location', 'account_type']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    # Handle unseen categories
                    known_classes = set(self.encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else self.encoders[col].classes_[0])
                    df[col + '_encoded'] = self.encoders[col].transform(df[col])
        
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
            'age', 'tenure_months', 'current_balance', 'avg_balance',
            'monthly_transactions', 'avg_transaction_amount',
            'app_sessions', 'web_logins', 'support_tickets',
            'complaints_last_6m', 'has_credit_card', 'has_loan',
            'has_investment', 'num_products', 'balance_ratio',
            'engagement_score', 'issue_score', 'tenure_bucket',
            'gender_encoded', 'account_type_encoded'
        ]
        
        # Filter to available features
        available_features = [f for f in self.feature_names if f in df.columns]
        self.feature_names = available_features
        
        X = df[self.feature_names].values
        y = df['churned'].values if 'churned' in df.columns else None
        
        return X, y
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None,
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the Logistic Regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test split proportion
            
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
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(
            max_iter=self.config['max_iter'],
            C=self.config['C'],
            solver=self.config['solver'],
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Store coefficients
        self.coefficients = self.model.coef_[0]
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_prob_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_prob_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get churn probability for customers.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            Array of churn probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df_processed = self.preprocess_data(df)
        X, _ = self.prepare_features(df_processed)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn labels.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            Binary churn predictions
        """
        probabilities = self.predict_proba(df)
        return (probabilities >= 0.5).astype(int)
    
    def get_risk_assessment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive risk assessment for customers.
        
        Args:
            df: Customer DataFrame
            
        Returns:
            DataFrame with risk assessments
        """
        probabilities = self.predict_proba(df)
        
        risk_df = df[['customer_id']].copy() if 'customer_id' in df.columns else pd.DataFrame()
        risk_df['churn_probability'] = probabilities
        risk_df['churn_percentage'] = (probabilities * 100).round(1)
        
        # Assign risk levels
        def assign_risk(prob):
            if prob >= RISK_THRESHOLDS['high']:
                return 'HIGH'
            elif prob >= RISK_THRESHOLDS['medium']:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        risk_df['risk_level'] = probabilities
        risk_df['risk_level'] = risk_df['risk_level'].apply(assign_risk)
        
        # Add key factors contributing to churn
        if len(df) > 0:
            risk_df['tenure_months'] = df['tenure_months'].values
            risk_df['monthly_transactions'] = df['monthly_transactions'].values
            risk_df['num_products'] = df['num_products'].values
            risk_df['complaints'] = df['complaints_last_6m'].values
        
        return risk_df.sort_values('churn_probability', ascending=False)
    
    def get_high_risk_customers(self, df: pd.DataFrame, 
                                threshold: float = None) -> pd.DataFrame:
        """
        Get customers at high risk of churning.
        
        Args:
            df: Customer DataFrame
            threshold: Custom probability threshold
            
        Returns:
            DataFrame with high-risk customers
        """
        if threshold is None:
            threshold = RISK_THRESHOLDS['high']
        
        risk_df = self.get_risk_assessment(df)
        high_risk = risk_df[risk_df['churn_probability'] >= threshold]
        
        return high_risk
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.
        
        Returns:
            DataFrame with feature importances
        """
        if self.coefficients is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.coefficients,
            'Abs_Importance': np.abs(self.coefficients)
        }).sort_values('Abs_Importance', ascending=False)
        
        # Add interpretation
        importance_df['Effect'] = importance_df['Coefficient'].apply(
            lambda x: 'Increases Churn' if x > 0 else 'Decreases Churn'
        )
        
        return importance_df
    
    def get_retention_recommendations(self, customer: pd.Series) -> List[str]:
        """
        Generate personalized retention recommendations.
        
        Args:
            customer: Single customer row
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if customer.get('num_products', 0) == 1:
            recommendations.append("Offer bundled products (credit card + investment)")
        
        if customer.get('app_sessions', 0) < 10:
            recommendations.append("Send app engagement rewards and tutorials")
        
        if customer.get('tenure_months', 0) < 6:
            recommendations.append("Assign dedicated relationship manager")
        
        if customer.get('complaints_last_6m', 0) > 0:
            recommendations.append("Proactive outreach to resolve past issues")
        
        if customer.get('current_balance', 0) > 100000:
            recommendations.append("Upgrade to premium account with benefits")
        
        if customer.get('monthly_transactions', 0) < 5:
            recommendations.append("Offer cashback on transactions")
        
        if not recommendations:
            recommendations.append("Continue regular engagement - low risk customer")
        
        return recommendations
    
    def get_churn_summary(self) -> Dict:
        """
        Get overall churn analysis summary.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            return {}
        
        probabilities = self.predict_proba(self.data)
        
        summary = {
            'total_customers': len(self.data),
            'high_risk_count': sum(probabilities >= RISK_THRESHOLDS['high']),
            'medium_risk_count': sum((probabilities >= RISK_THRESHOLDS['medium']) & 
                                     (probabilities < RISK_THRESHOLDS['high'])),
            'low_risk_count': sum(probabilities < RISK_THRESHOLDS['medium']),
            'avg_churn_probability': probabilities.mean(),
            'actual_churn_rate': self.data['churned'].mean() if 'churned' in self.data.columns else None
        }
        
        summary['high_risk_percentage'] = (summary['high_risk_count'] / summary['total_customers']) * 100
        summary['estimated_revenue_at_risk'] = summary['high_risk_count'] * self.data['current_balance'].mean()
        
        return summary
    
    def save_model(self, path: str = 'models/churn_model.pkl'):
        """Save trained model to disk."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.encoders,
                'feature_names': self.feature_names
            }, path)
    
    def load_model(self, path: str = 'models/churn_model.pkl'):
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoders = data['encoders']
        self.feature_names = data['feature_names']


def run_churn_analysis_demo():
    """Run a demonstration of the churn analyzer."""
    print("=" * 60)
    print("Logistic Regression Churn Analyzer - Demo")
    print("=" * 60)
    
    analyzer = ChurnAnalyzer()
    
    print("\n👥 Generating fintech customer data...")
    data = analyzer.generate_sample_data(3000)
    print(f"   Generated {len(data)} customer records")
    print(f"   Actual churn rate: {data['churned'].mean()*100:.1f}%")
    
    print("\n🔧 Training Logistic Regression model...")
    metrics = analyzer.train()
    print(f"   Test Accuracy: {metrics['test_accuracy']*100:.1f}%")
    print(f"   Precision: {metrics['precision']*100:.1f}%")
    print(f"   Recall: {metrics['recall']*100:.1f}%")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\n📊 Feature Importance (Top 5):")
    importance = analyzer.get_feature_importance()
    for _, row in importance.head(5).iterrows():
        effect = "↑" if row['Coefficient'] > 0 else "↓"
        print(f"   {effect} {row['Feature']}: {row['Coefficient']:.4f}")
    
    print("\n⚠️ Churn Summary:")
    summary = analyzer.get_churn_summary()
    print(f"   High Risk: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
    print(f"   Medium Risk: {summary['medium_risk_count']}")
    print(f"   Low Risk: {summary['low_risk_count']}")
    print(f"   Est. Revenue at Risk: ₹{summary['estimated_revenue_at_risk']:,.0f}")
    
    print("\n🎯 Sample High-Risk Customers:")
    high_risk = analyzer.get_high_risk_customers(data).head(3)
    for _, customer in high_risk.iterrows():
        print(f"   {customer.get('customer_id', 'N/A')}: "
              f"{customer['churn_percentage']:.1f}% risk")
    
    print("\n✅ Demo complete!")
    return analyzer


if __name__ == '__main__':
    run_churn_analysis_demo()
