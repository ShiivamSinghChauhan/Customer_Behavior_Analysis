import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

class AnomalyDetectionSystem:
    """
    Automated alert system for significant behavior changes and anomaly detection in customer data.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            'revenue_spike': 2.0,  # 2x normal revenue
            'transaction_spike': 3.0,  # 3x normal transactions
            'churn_spike': 0.2,  # 20% increase in churn indicators
            'conversion_drop': 0.3,  # 30% drop in conversion
            'new_customer_surge': 2.5  # 2.5x normal new customers
        }
        self.active_alerts = []
        
    def detect_revenue_anomalies(self, df):
        """Detect unusual revenue patterns."""
        alerts = []
        
        # Daily revenue analysis
        df['date'] = pd.to_datetime(df['Transaction_Date']).dt.date
        daily_revenue = df.groupby('date')['Revenue'].sum()
        
        if len(daily_revenue) > 7:
            # Calculate baseline (previous 7 days average)
            recent_avg = daily_revenue.tail(7).mean()
            historical_avg = daily_revenue.head(-1).mean() if len(daily_revenue) > 8 else recent_avg
            
            # Revenue spike detection
            if recent_avg > historical_avg * self.alert_thresholds['revenue_spike']:
                alerts.append({
                    'type': 'Revenue Spike',
                    'severity': 'high',
                    'message': f'Revenue spike detected: {recent_avg/historical_avg:.1f}x normal levels',
                    'current_value': recent_avg,
                    'baseline_value': historical_avg,
                    'timestamp': datetime.now()
                })
            
            # Revenue drop detection
            elif recent_avg < historical_avg * 0.5:
                alerts.append({
                    'type': 'Revenue Drop',
                    'severity': 'critical',
                    'message': f'Significant revenue drop: {(1-recent_avg/historical_avg)*100:.1f}% below normal',
                    'current_value': recent_avg,
                    'baseline_value': historical_avg,
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def detect_transaction_anomalies(self, df):
        """Detect unusual transaction volume patterns."""
        alerts = []
        
        # Hourly transaction analysis
        df['datetime'] = pd.to_datetime(df['Transaction_Date'])
        df['hour'] = df['datetime'].dt.hour
        hourly_transactions = df.groupby(['datetime', 'hour']).size()
        
        if len(hourly_transactions) > 24:
            recent_hourly_avg = hourly_transactions.tail(24).mean()
            historical_hourly_avg = hourly_transactions.head(-24).mean() if len(hourly_transactions) > 48 else recent_hourly_avg
            
            # Transaction spike
            if recent_hourly_avg > historical_hourly_avg * self.alert_thresholds['transaction_spike']:
                alerts.append({
                    'type': 'Transaction Volume Spike',
                    'severity': 'medium',
                    'message': f'Unusual transaction volume: {recent_hourly_avg/historical_hourly_avg:.1f}x normal rate',
                    'current_value': recent_hourly_avg,
                    'baseline_value': historical_hourly_avg,
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def detect_customer_behavior_anomalies(self, df):
        """Detect anomalies in customer behavior patterns."""
        alerts = []
        
        # New customer analysis
        df['date'] = pd.to_datetime(df['Transaction_Date']).dt.date
        
        # Find first transaction per customer
        first_transactions = df.groupby('Customer_ID')['Transaction_Date'].min().reset_index()
        first_transactions['date'] = pd.to_datetime(first_transactions['Transaction_Date']).dt.date
        daily_new_customers = first_transactions.groupby('date').size()
        
        if len(daily_new_customers) > 7:
            recent_new_avg = daily_new_customers.tail(7).mean()
            historical_new_avg = daily_new_customers.head(-7).mean() if len(daily_new_customers) > 14 else recent_new_avg
            
            # New customer surge
            if recent_new_avg > historical_new_avg * self.alert_thresholds['new_customer_surge']:
                alerts.append({
                    'type': 'New Customer Surge',
                    'severity': 'medium',
                    'message': f'Unusual new customer acquisition: {recent_new_avg/historical_new_avg:.1f}x normal rate',
                    'current_value': recent_new_avg,
                    'baseline_value': historical_new_avg,
                    'timestamp': datetime.now()
                })
        
        # Customer retention analysis
        customer_frequency = df.groupby('Customer_ID').size()
        one_time_buyers_ratio = (customer_frequency == 1).mean()
        
        if one_time_buyers_ratio > 0.8:
            alerts.append({
                'type': 'High Churn Risk',
                'severity': 'high',
                'message': f'Unusually high one-time buyer ratio: {one_time_buyers_ratio:.1%}',
                'current_value': one_time_buyers_ratio,
                'baseline_value': 0.6,  # Expected ratio
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def detect_product_anomalies(self, df):
        """Detect anomalies in product performance."""
        alerts = []
        
        # Product performance analysis
        product_performance = df.groupby('Product_ID').agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count'
        }).reset_index()
        
        # Find products with unusual performance
        revenue_threshold = product_performance['Revenue'].quantile(0.95)
        high_performers = product_performance[product_performance['Revenue'] > revenue_threshold]
        
        if len(high_performers) > 0:
            for _, product in high_performers.iterrows():
                alerts.append({
                    'type': 'Product Performance Spike',
                    'severity': 'low',
                    'message': f'Product {product["Product_ID"]} showing exceptional performance',
                    'current_value': product['Revenue'],
                    'baseline_value': product_performance['Revenue'].median(),
                    'timestamp': datetime.now(),
                    'product_id': product['Product_ID']
                })
        
        # Category shift analysis
        category_trends = df.groupby(['Category']).agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count'
        })
        
        # Check for significant category shifts (simplified)
        top_category = category_trends['Revenue'].idxmax()
        top_category_share = category_trends.loc[top_category, 'Revenue'] / category_trends['Revenue'].sum()
        
        if top_category_share > 0.6:
            alerts.append({
                'type': 'Category Concentration',
                'severity': 'medium',
                'message': f'{top_category} dominates sales ({top_category_share:.1%} of revenue)',
                'current_value': top_category_share,
                'baseline_value': 0.4,  # Expected max category share
                'timestamp': datetime.now(),
                'category': top_category
            })
        
        return alerts
    
    def detect_marketing_anomalies(self, df):
        """Detect anomalies in marketing performance."""
        alerts = []
        
        if all(col in df.columns for col in ['Clicks', 'Impressions', 'Ad_Spend', 'Revenue']):
            # Calculate marketing metrics
            df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, np.nan)
            df['ROAS'] = df['Revenue'] / df['Ad_Spend'].replace(0, np.nan)
            df['CPC'] = df['Ad_Spend'] / df['Clicks'].replace(0, np.nan)
            
            # CTR anomalies
            avg_ctr = df['CTR'].mean()
            if avg_ctr < 0.01:  # Less than 1% CTR
                alerts.append({
                    'type': 'Low Click-Through Rate',
                    'severity': 'medium',
                    'message': f'Unusually low CTR detected: {avg_ctr:.2%}',
                    'current_value': avg_ctr,
                    'baseline_value': 0.02,
                    'timestamp': datetime.now()
                })
            
            # ROAS anomalies
            avg_roas = df['ROAS'].mean()
            if avg_roas < 2.0:  # Less than 2x return
                alerts.append({
                    'type': 'Low Return on Ad Spend',
                    'severity': 'high',
                    'message': f'Poor ROAS performance: {avg_roas:.1f}x return',
                    'current_value': avg_roas,
                    'baseline_value': 3.0,
                    'timestamp': datetime.now()
                })
            
            # High CPC alert
            avg_cpc = df['CPC'].mean()
            if avg_cpc > 2.0:  # High cost per click
                alerts.append({
                    'type': 'High Cost Per Click',
                    'severity': 'medium',
                    'message': f'Elevated CPC detected: ${avg_cpc:.2f}',
                    'current_value': avg_cpc,
                    'baseline_value': 1.0,
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def detect_geographic_anomalies(self, df):
        """Detect anomalies in geographic performance."""
        alerts = []
        
        # Regional performance analysis
        regional_performance = df.groupby('Region').agg({
            'Revenue': ['sum', 'mean'],
            'Customer_ID': 'nunique'
        }).reset_index()
        
        regional_performance.columns = ['Region', 'Total_Revenue', 'Avg_Revenue', 'Unique_Customers']
        regional_performance['Revenue_Per_Customer'] = regional_performance['Total_Revenue'] / regional_performance['Unique_Customers']
        
        # Check for regional performance imbalances
        revenue_std = regional_performance['Revenue_Per_Customer'].std()
        revenue_mean = regional_performance['Revenue_Per_Customer'].mean()
        
        if revenue_std / revenue_mean > 0.5:  # High coefficient of variation
            best_region = regional_performance.loc[regional_performance['Revenue_Per_Customer'].idxmax(), 'Region']
            worst_region = regional_performance.loc[regional_performance['Revenue_Per_Customer'].idxmin(), 'Region']
            
            alerts.append({
                'type': 'Regional Performance Imbalance',
                'severity': 'medium',
                'message': f'Large performance gap between {best_region} and {worst_region}',
                'current_value': revenue_std / revenue_mean,
                'baseline_value': 0.3,
                'timestamp': datetime.now(),
                'best_region': best_region,
                'worst_region': worst_region
            })
        
        return alerts
    
    def run_comprehensive_anomaly_detection(self, df):
        """Run all anomaly detection algorithms on the dataset."""
        all_alerts = []
        
        try:
            # Run individual detection methods
            revenue_alerts = self.detect_revenue_anomalies(df)
            transaction_alerts = self.detect_transaction_anomalies(df)
            customer_alerts = self.detect_customer_behavior_anomalies(df)
            product_alerts = self.detect_product_anomalies(df)
            marketing_alerts = self.detect_marketing_anomalies(df)
            geographic_alerts = self.detect_geographic_anomalies(df)
            
            # Combine all alerts
            all_alerts.extend(revenue_alerts)
            all_alerts.extend(transaction_alerts)
            all_alerts.extend(customer_alerts)
            all_alerts.extend(product_alerts)
            all_alerts.extend(marketing_alerts)
            all_alerts.extend(geographic_alerts)
            
            # Sort by severity
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            all_alerts.sort(key=lambda x: severity_order.get(x['severity'], 4))
            
            # Store active alerts
            self.active_alerts = all_alerts
            
        except Exception as e:
            # Fallback alert if detection fails
            all_alerts.append({
                'type': 'System Alert',
                'severity': 'medium',
                'message': 'Anomaly detection encountered an issue during analysis',
                'current_value': None,
                'baseline_value': None,
                'timestamp': datetime.now(),
                'error': str(e)
            })
        
        return all_alerts
    
    def get_alert_summary(self):
        """Get summary of current alerts."""
        if not self.active_alerts:
            return {
                'total_alerts': 0,
                'critical_alerts': 0,
                'high_alerts': 0,
                'medium_alerts': 0,
                'low_alerts': 0,
                'latest_alert': None
            }
        
        summary = {
            'total_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts if a['severity'] == 'critical']),
            'high_alerts': len([a for a in self.active_alerts if a['severity'] == 'high']),
            'medium_alerts': len([a for a in self.active_alerts if a['severity'] == 'medium']),
            'low_alerts': len([a for a in self.active_alerts if a['severity'] == 'low']),
            'latest_alert': self.active_alerts[0] if self.active_alerts else None
        }
        
        return summary
    
    def configure_alert_thresholds(self, new_thresholds):
        """Configure custom alert thresholds."""
        self.alert_thresholds.update(new_thresholds)
        return "Alert thresholds updated successfully"
    
    def export_alerts(self):
        """Export current alerts for external systems."""
        return {
            'timestamp': datetime.now().isoformat(),
            'alert_count': len(self.active_alerts),
            'alerts': self.active_alerts,
            'thresholds': self.alert_thresholds
        }
    
    def clear_alerts(self):
        """Clear all active alerts."""
        cleared_count = len(self.active_alerts)
        self.active_alerts = []
        return f"Cleared {cleared_count} alerts"