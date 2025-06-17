import pandas as pd
import numpy as np
import json
from datetime import datetime
import streamlit as st

class Utils:
    """
    Utility functions for the customer behavior analysis application.
    """
    
    @staticmethod
    def format_currency(amount):
        """Format currency values."""
        if amount >= 1000000:
            return f"${amount/1000000:.1f}M"
        elif amount >= 1000:
            return f"${amount/1000:.1f}K"
        else:
            return f"${amount:.2f}"
    
    @staticmethod
    def format_percentage(value):
        """Format percentage values."""
        return f"{value:.1%}"
    
    @staticmethod
    def format_number(number):
        """Format large numbers."""
        if number >= 1000000:
            return f"{number/1000000:.1f}M"
        elif number >= 1000:
            return f"{number/1000:.1f}K"
        else:
            return f"{number:,.0f}"
    
    @staticmethod
    def calculate_statistical_significance(group1, group2, alpha=0.05):
        """Calculate statistical significance between two groups."""
        from scipy import stats
        
        try:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            is_significant = p_value < alpha
            
            return {
                'is_significant': is_significant,
                'p_value': p_value,
                't_statistic': t_stat,
                'alpha': alpha,
                'interpretation': 'Statistically significant' if is_significant else 'Not statistically significant'
            }
        except Exception as e:
            return {
                'is_significant': False,
                'p_value': None,
                't_statistic': None,
                'alpha': alpha,
                'interpretation': f'Error in calculation: {str(e)}'
            }
    
    @staticmethod
    def calculate_confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for a dataset."""
        from scipy import stats
        
        try:
            mean = np.mean(data)
            sem = stats.sem(data)  # Standard error of the mean
            margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
            
            return {
                'mean': mean,
                'lower_bound': mean - margin_of_error,
                'upper_bound': mean + margin_of_error,
                'margin_of_error': margin_of_error,
                'confidence_level': confidence
            }
        except Exception as e:
            return {
                'mean': np.mean(data) if len(data) > 0 else 0,
                'lower_bound': None,
                'upper_bound': None,
                'margin_of_error': None,
                'confidence_level': confidence,
                'error': str(e)
            }
    
    @staticmethod
    def detect_outliers(data, method='iqr'):
        """Detect outliers in a dataset."""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            return {
                'outliers': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'method': 'IQR'
            }
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            threshold = 3
            outliers = data[z_scores > threshold]
            
            return {
                'outliers': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100,
                'threshold': threshold,
                'method': 'Z-Score'
            }
    
    @staticmethod
    def generate_summary_report(df):
        """Generate a comprehensive summary report."""
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_records': len(df),
                'date_range': {
                    'start': df['Transaction_Date'].min().isoformat(),
                    'end': df['Transaction_Date'].max().isoformat()
                }
            },
            'business_metrics': {
                'total_revenue': float(df['Revenue'].sum()),
                'average_order_value': float(df['Revenue'].mean()),
                'median_order_value': float(df['Revenue'].median()),
                'total_transactions': len(df),
                'unique_customers': df['Customer_ID'].nunique(),
                'unique_products': df['Product_ID'].nunique(),
                'categories': df['Category'].nunique(),
                'regions': df['Region'].nunique()
            },
            'customer_analysis': {
                'customer_metrics': Utils._calculate_customer_metrics(df),
                'geographic_distribution': df['Region'].value_counts().to_dict(),
                'category_preferences': df.groupby('Customer_ID')['Category'].apply(lambda x: x.mode()[0]).value_counts().to_dict()
            },
            'product_analysis': {
                'category_performance': df.groupby('Category').agg({
                    'Revenue': 'sum',
                    'Transaction_ID': 'count',
                    'Units_Sold': 'sum'
                }).to_dict(),
                'top_products': df.groupby('Product_ID')['Revenue'].sum().nlargest(10).to_dict()
            },
            'temporal_analysis': {
                'monthly_trends': df.groupby(df['Transaction_Date'].dt.month)['Revenue'].sum().to_dict(),
                'daily_patterns': df.groupby(df['Transaction_Date'].dt.day_name())['Revenue'].sum().to_dict(),
                'seasonal_patterns': df.groupby(df['Transaction_Date'].dt.quarter)['Revenue'].sum().to_dict()
            }
        }
        
        # Add marketing analysis if data available
        if all(col in df.columns for col in ['Clicks', 'Impressions', 'Ad_Spend']):
            report['marketing_analysis'] = {
                'total_clicks': int(df['Clicks'].sum()),
                'total_impressions': int(df['Impressions'].sum()),
                'total_ad_spend': float(df['Ad_Spend'].sum()),
                'average_ctr': float(df['Ad_CTR'].mean()),
                'average_cpc': float(df['Ad_CPC'].mean()),
                'roas': float(df['Revenue'].sum() / df['Ad_Spend'].sum()) if df['Ad_Spend'].sum() > 0 else 0
            }
        
        return report
    
    @staticmethod
    def _calculate_customer_metrics(df):
        """Calculate customer-level metrics for the report."""
        customer_metrics = df.groupby('Customer_ID').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Transaction_Date': ['min', 'max'],
            'Units_Sold': 'sum'
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'Customer_ID', 'Total_Revenue', 'Avg_Order_Value', 'Purchase_Frequency',
            'First_Purchase', 'Last_Purchase', 'Total_Units'
        ]
        
        # Calculate additional metrics
        reference_date = df['Transaction_Date'].max()
        customer_metrics['Days_Since_Last_Purchase'] = (reference_date - customer_metrics['Last_Purchase']).dt.days
        customer_metrics['Customer_Lifespan'] = (customer_metrics['Last_Purchase'] - customer_metrics['First_Purchase']).dt.days + 1
        
        return {
            'total_customers': len(customer_metrics),
            'average_clv': float(customer_metrics['Total_Revenue'].mean()),
            'median_clv': float(customer_metrics['Total_Revenue'].median()),
            'average_purchase_frequency': float(customer_metrics['Purchase_Frequency'].mean()),
            'average_order_value': float(customer_metrics['Avg_Order_Value'].mean()),
            'customer_retention_stats': {
                'active_customers_30_days': len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] <= 30]),
                'active_customers_90_days': len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] <= 90]),
                'inactive_customers_90_days': len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] > 90])
            }
        }
    
    @staticmethod
    def export_to_csv(data, filename):
        """Export data to CSV format."""
        try:
            if isinstance(data, pd.DataFrame):
                csv_string = data.to_csv(index=False)
                return csv_string
            else:
                df = pd.DataFrame(data)
                csv_string = df.to_csv(index=False)
                return csv_string
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return None
    
    @staticmethod
    def export_to_json(data, filename):
        """Export data to JSON format."""
        try:
            if isinstance(data, pd.DataFrame):
                json_string = data.to_json(orient='records', indent=2)
                return json_string
            else:
                json_string = json.dumps(data, indent=2, default=str)
                return json_string
        except Exception as e:
            st.error(f"Error exporting to JSON: {str(e)}")
            return None
    
    @staticmethod
    def validate_data_quality(df):
        """Validate data quality and return quality metrics."""
        quality_report = {
            'total_records': len(df),
            'duplicate_records': df.duplicated().sum(),
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'quality_score': 0,
            'recommendations': []
        }
        
        # Calculate quality score
        score = 100
        
        # Deduct for missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= missing_percentage * 2
        
        # Deduct for duplicates
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_percentage * 3
        
        # Check for obvious data quality issues
        if 'Revenue' in df.columns:
            negative_revenue = (df['Revenue'] < 0).sum()
            if negative_revenue > 0:
                score -= 10
                quality_report['recommendations'].append(f"Found {negative_revenue} records with negative revenue")
        
        if 'Units_Sold' in df.columns:
            zero_units = (df['Units_Sold'] <= 0).sum()
            if zero_units > 0:
                score -= 5
                quality_report['recommendations'].append(f"Found {zero_units} records with zero or negative units sold")
        
        quality_report['quality_score'] = max(0, min(100, score))
        
        return quality_report
    
    @staticmethod
    def create_data_dictionary(df):
        """Create a data dictionary for the dataset."""
        data_dict = {}
        
        for column in df.columns:
            data_dict[column] = {
                'data_type': str(df[column].dtype),
                'non_null_count': df[column].count(),
                'null_count': df[column].isnull().sum(),
                'unique_values': df[column].nunique(),
                'sample_values': df[column].dropna().head(5).tolist()
            }
            
            # Add statistics for numeric columns
            if df[column].dtype in ['int64', 'float64']:
                data_dict[column].update({
                    'mean': float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max())
                })
            
            # Add frequency for categorical columns
            elif df[column].dtype == 'object':
                data_dict[column]['value_counts'] = df[column].value_counts().head(10).to_dict()
        
        return data_dict
