import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for e-commerce transaction data.
    """
    
    def __init__(self):
        self.required_columns = [
            'Transaction_ID', 'Customer_ID', 'Product_ID', 'Transaction_Date',
            'Units_Sold', 'Revenue', 'Category', 'Region'
        ]
        self.optional_columns = [
            'Discount_Applied', 'Clicks', 'Impressions', 'Conversion_Rate',
            'Ad_CTR', 'Ad_CPC', 'Ad_Spend'
        ]
    
    def load_data(self, file):
        """Load data from uploaded file."""
        try:
            df = pd.read_csv(file)
            self.validate_data(df)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self, df):
        """Validate that required columns exist in the dataset."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Check for minimum data requirements
        if len(df) < 100:
            raise Exception("Dataset must contain at least 100 transactions for meaningful analysis")
    
    def preprocess_data(self, df):
        """Clean and preprocess the data."""
        df_processed = df.copy()
        
        # Convert date column
        df_processed['Transaction_Date'] = pd.to_datetime(df_processed['Transaction_Date'])
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Remove outliers
        df_processed = self.remove_outliers(df_processed)
        
        # Add derived features
        df_processed = self.add_derived_features(df_processed)
        
        # Ensure data types
        df_processed = self.ensure_data_types(df_processed)
        
        return df_processed
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Transaction_Date' and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers using IQR method for revenue and units sold."""
        for col in ['Revenue', 'Units_Sold']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def add_derived_features(self, df):
        """Add derived features for analysis."""
        # Calculate days since last transaction (for RFM analysis)
        reference_date = df['Transaction_Date'].max()
        df['Days_Since_Transaction'] = (reference_date - df['Transaction_Date']).dt.days
        
        # Add time-based features
        df['Year'] = df['Transaction_Date'].dt.year
        df['Month'] = df['Transaction_Date'].dt.month
        df['Day_of_Week'] = df['Transaction_Date'].dt.dayofweek
        df['Quarter'] = df['Transaction_Date'].dt.quarter
        
        # Calculate average price per unit
        df['Price_Per_Unit'] = df['Revenue'] / df['Units_Sold']
        
        # Add discount indicator if discount column exists
        if 'Discount_Applied' in df.columns:
            df['Has_Discount'] = df['Discount_Applied'] > 0
        
        # Calculate marketing efficiency metrics if available
        if all(col in df.columns for col in ['Revenue', 'Ad_Spend']):
            df['Marketing_ROI'] = df['Revenue'] / df['Ad_Spend'].replace(0, np.nan)
        
        return df
    
    def ensure_data_types(self, df):
        """Ensure proper data types for all columns."""
        # Ensure numeric columns are numeric
        numeric_columns = ['Units_Sold', 'Revenue', 'Price_Per_Unit']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_columns = ['Customer_ID', 'Product_ID', 'Category', 'Region']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def get_data_summary(self, df):
        """Generate a summary of the processed data."""
        summary = {
            'total_transactions': len(df),
            'unique_customers': df['Customer_ID'].nunique(),
            'unique_products': df['Product_ID'].nunique(),
            'date_range': {
                'start': df['Transaction_Date'].min(),
                'end': df['Transaction_Date'].max()
            },
            'total_revenue': df['Revenue'].sum(),
            'avg_order_value': df['Revenue'].mean(),
            'categories': df['Category'].nunique(),
            'regions': df['Region'].nunique()
        }
        
        return summary
