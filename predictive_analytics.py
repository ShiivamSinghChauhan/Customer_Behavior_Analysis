import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalytics:
    """
    Handles predictive analytics including CLV prediction, churn analysis, and next purchase prediction.
    """
    
    def __init__(self):
        self.clv_model = None
        self.churn_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def calculate_clv(self, df):
        """Calculate Customer Lifetime Value using historical data and prediction."""
        # Calculate customer-level metrics
        customer_metrics = self.prepare_customer_features(df)
        
        # Calculate historical CLV
        customer_metrics['Historical_CLV'] = customer_metrics['Total_Revenue']
        
        # Predict future CLV
        predicted_clv = self.predict_future_clv(customer_metrics, df)
        
        # Combine historical and predicted CLV
        customer_metrics['Predicted_CLV'] = predicted_clv
        customer_metrics['CLV'] = customer_metrics['Historical_CLV'] + customer_metrics['Predicted_CLV']
        
        return customer_metrics[['Customer_ID', 'CLV', 'Historical_CLV', 'Predicted_CLV', 
                                'Avg_Order_Value', 'Purchase_Frequency', 'Days_Since_Last_Purchase']]
    
    def prepare_customer_features(self, df):
        """Prepare customer-level features for modeling."""
        # Calculate customer metrics
        customer_metrics = df.groupby('Customer_ID').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Transaction_Date': ['min', 'max'],
            'Units_Sold': ['sum', 'mean'],
            'Category': lambda x: x.mode()[0] if not x.empty else 'Unknown',
            'Region': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'Customer_ID', 'Total_Revenue', 'Avg_Order_Value', 'Purchase_Frequency',
            'First_Purchase', 'Last_Purchase', 'Total_Units', 'Avg_Units_Per_Order',
            'Preferred_Category', 'Region'
        ]
        
        # Calculate additional features
        reference_date = df['Transaction_Date'].max()
        customer_metrics['Days_Since_Last_Purchase'] = (reference_date - customer_metrics['Last_Purchase']).dt.days
        customer_metrics['Customer_Lifespan'] = (customer_metrics['Last_Purchase'] - customer_metrics['First_Purchase']).dt.days + 1
        customer_metrics['Purchase_Rate'] = customer_metrics['Purchase_Frequency'] / customer_metrics['Customer_Lifespan']
        
        # Handle missing values
        customer_metrics['Purchase_Rate'] = customer_metrics['Purchase_Rate'].fillna(0)
        customer_metrics['Customer_Lifespan'] = customer_metrics['Customer_Lifespan'].fillna(1)
        
        return customer_metrics
    
    def predict_future_clv(self, customer_metrics, df):
        """Predict future CLV using machine learning."""
        # Prepare features for modeling
        features = ['Avg_Order_Value', 'Purchase_Frequency', 'Days_Since_Last_Purchase', 
                   'Total_Units', 'Customer_Lifespan', 'Purchase_Rate']
        
        # Encode categorical variables
        categorical_features = ['Preferred_Category', 'Region']
        customer_features = customer_metrics.copy()
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            customer_features[col + '_encoded'] = self.label_encoders[col].fit_transform(customer_features[col])
            features.append(col + '_encoded')
        
        # Prepare training data
        X = customer_features[features].fillna(0)
        
        # Use a simple approach for future CLV prediction
        # Predict based on average revenue per day * expected future days
        expected_future_days = 365  # Predict for next year
        
        # Calculate daily revenue rate
        daily_revenue_rate = customer_metrics['Total_Revenue'] / customer_metrics['Customer_Lifespan']
        daily_revenue_rate = daily_revenue_rate.fillna(0)
        
        # Adjust for recency (customers who purchased recently are more likely to continue)
        recency_factor = np.exp(-customer_metrics['Days_Since_Last_Purchase'] / 100)  # Decay factor
        
        # Predict future CLV
        predicted_clv = daily_revenue_rate * expected_future_days * recency_factor
        
        return predicted_clv.fillna(0)
    
    def analyze_churn_risk(self, df):
        """Analyze customer churn risk."""
        customer_features = self.prepare_customer_features(df)
        
        # Define churn based on days since last purchase
        churn_threshold = 90  # Consider customer churned if no purchase in 90 days
        customer_features['Is_Churned'] = customer_features['Days_Since_Last_Purchase'] > churn_threshold
        
        # Calculate churn risk scores
        customer_features['Churn_Risk_Score'] = self.calculate_churn_risk_score(customer_features)
        
        # Categorize risk levels
        customer_features['Risk_Level'] = pd.cut(
            customer_features['Churn_Risk_Score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Calculate summary statistics
        risk_summary = {
            'low_risk_count': len(customer_features[customer_features['Risk_Level'] == 'Low']),
            'medium_risk_count': len(customer_features[customer_features['Risk_Level'] == 'Medium']),
            'high_risk_count': len(customer_features[customer_features['Risk_Level'] == 'High']),
            'total_customers': len(customer_features),
            'churn_rate': customer_features['Is_Churned'].mean(),
            'at_risk_revenue': customer_features[customer_features['Risk_Level'] == 'High']['Total_Revenue'].sum(),
            'customers_by_risk': customer_features.groupby('Risk_Level').size().to_dict()
        }
        
        risk_summary['customer_details'] = customer_features[['Customer_ID', 'Churn_Risk_Score', 'Risk_Level', 
                                                            'Days_Since_Last_Purchase', 'Total_Revenue']]
        
        return risk_summary
    
    def calculate_churn_risk_score(self, customer_features):
        """Calculate churn risk score based on customer behavior."""
        # Normalize features to 0-1 scale
        features = customer_features.copy()
        
        # Days since last purchase (higher = more risk)
        max_days = features['Days_Since_Last_Purchase'].max()
        recency_risk = features['Days_Since_Last_Purchase'] / max_days if max_days > 0 else 0
        
        # Purchase frequency (lower = more risk)
        max_frequency = features['Purchase_Frequency'].max()
        frequency_risk = 1 - (features['Purchase_Frequency'] / max_frequency) if max_frequency > 0 else 1
        
        # Average order value (lower = more risk for retention)
        max_aov = features['Avg_Order_Value'].max()
        value_risk = 1 - (features['Avg_Order_Value'] / max_aov) if max_aov > 0 else 1
        
        # Purchase rate (lower = more risk)
        max_rate = features['Purchase_Rate'].max()
        rate_risk = 1 - (features['Purchase_Rate'] / max_rate) if max_rate > 0 else 1
        
        # Weighted combination
        churn_risk_score = (
            0.4 * recency_risk +
            0.3 * frequency_risk +
            0.2 * value_risk +
            0.1 * rate_risk
        )
        
        return np.clip(churn_risk_score, 0, 1)
    
    def predict_next_purchase(self, df):
        """Predict next purchase timing and product for active customers."""
        customer_features = self.prepare_customer_features(df)
        
        # Filter active customers (purchased in last 90 days)
        active_customers = customer_features[customer_features['Days_Since_Last_Purchase'] <= 90]
        
        predictions = []
        
        for _, customer in active_customers.iterrows():
            # Predict days until next purchase based on historical pattern
            avg_days_between_purchases = customer['Customer_Lifespan'] / max(customer['Purchase_Frequency'], 1)
            
            # Adjust based on recent activity
            if customer['Days_Since_Last_Purchase'] < 30:
                predicted_days = avg_days_between_purchases * 0.8  # More likely to purchase soon
            else:
                predicted_days = avg_days_between_purchases * 1.2  # Less likely to purchase soon
            
            # Predict likely category based on purchase history
            customer_transactions = df[df['Customer_ID'] == customer['Customer_ID']]
            likely_category = customer_transactions['Category'].mode()[0] if not customer_transactions.empty else 'Unknown'
            
            # Calculate confidence based on purchase regularity
            purchase_pattern_std = customer_transactions.groupby(customer_transactions['Transaction_Date'].dt.date).size().std()
            confidence = 1 / (1 + purchase_pattern_std) if not pd.isna(purchase_pattern_std) else 0.5
            
            predictions.append({
                'Customer_ID': customer['Customer_ID'],
                'Predicted_Days_Until_Next_Purchase': max(1, int(predicted_days)),
                'Likely_Category': likely_category,
                'Confidence_Score': min(confidence, 1.0),
                'Expected_Order_Value': customer['Avg_Order_Value'],
                'Total_Revenue': customer['Total_Revenue']
            })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by confidence and expected value
        predictions_df = predictions_df.sort_values(['Confidence_Score', 'Expected_Order_Value'], ascending=False)
        
        return predictions_df
    
    def forecast_revenue(self, df, days_ahead=30):
        """Forecast revenue for the next period."""
        # Prepare daily revenue data
        daily_revenue = df.groupby(df['Transaction_Date'].dt.date)['Revenue'].sum().reset_index()
        daily_revenue.columns = ['Date', 'Revenue']
        daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
        daily_revenue = daily_revenue.sort_values('Date')
        
        # Simple moving average forecast
        window_size = min(7, len(daily_revenue) // 2)  # 7-day moving average or half the data
        if len(daily_revenue) >= window_size:
            recent_avg = daily_revenue['Revenue'].tail(window_size).mean()
        else:
            recent_avg = daily_revenue['Revenue'].mean()
        
        # Account for trend
        if len(daily_revenue) >= 2:
            trend = (daily_revenue['Revenue'].iloc[-1] - daily_revenue['Revenue'].iloc[0]) / len(daily_revenue)
        else:
            trend = 0
        
        # Generate forecast
        forecast_dates = pd.date_range(
            start=daily_revenue['Date'].max() + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        forecasted_revenue = []
        for i, date in enumerate(forecast_dates):
            # Apply trend and some seasonality (weekly pattern)
            day_of_week = date.dayofweek
            weekly_multiplier = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)  # Simple weekly seasonality
            
            forecast_value = (recent_avg + trend * i) * weekly_multiplier
            forecasted_revenue.append(max(0, forecast_value))  # Ensure non-negative
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Revenue': forecasted_revenue
        })
        
        return forecast_df
