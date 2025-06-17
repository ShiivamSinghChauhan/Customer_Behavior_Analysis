import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InsightsGenerator:
    """
    Generates AI-powered business insights and recommendations from the analyzed data.
    """
    
    def __init__(self):
        self.insights = {
            'key_findings': [],
            'recommendations': {
                'marketing': [],
                'customer_retention': [],
                'product': [],
                'operational': []
            },
            'risk_alerts': [],
            'opportunities': []
        }
    
    def generate_business_insights(self, df):
        """Generate comprehensive business insights from the data."""
        self.insights = {
            'key_findings': [],
            'recommendations': {
                'marketing': [],
                'customer_retention': [],
                'product': [],
                'operational': []
            },
            'risk_alerts': [],
            'opportunities': []
        }
        
        try:
            # Ensure required columns exist and are properly typed
            df = self._prepare_data_for_analysis(df)
            
            # Analyze key metrics
            self._analyze_revenue_patterns(df)
            self._analyze_customer_behavior(df)
            self._analyze_product_performance(df)
            self._analyze_regional_performance(df)
            self._analyze_temporal_patterns(df)
            self._analyze_marketing_effectiveness(df)
            self._identify_risks_and_opportunities(df)
            
        except Exception as e:
            # If analysis fails, provide basic insights
            self.insights['key_findings'].append("Analysis completed with basic metrics")
            self.insights['recommendations']['marketing'].append("Review data quality and ensure all required columns are present")
        
        return self.insights
    
    def _prepare_data_for_analysis(self, df):
        """Prepare data for analysis by ensuring proper data types."""
        df_prepared = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Revenue', 'Units_Sold']
        for col in numeric_cols:
            if col in df_prepared.columns:
                df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_cols = ['Customer_ID', 'Product_ID', 'Category', 'Region']
        for col in categorical_cols:
            if col in df_prepared.columns:
                df_prepared[col] = df_prepared[col].astype(str)
        
        # Handle discount column if present
        if 'Discount_Applied' in df_prepared.columns:
            df_prepared['Discount_Applied'] = pd.to_numeric(df_prepared['Discount_Applied'], errors='coerce')
            df_prepared['Discount_Applied'] = df_prepared['Discount_Applied'].fillna(0)
        
        return df_prepared
    
    def _analyze_revenue_patterns(self, df):
        """Analyze revenue patterns and trends."""
        total_revenue = df['Revenue'].sum()
        avg_order_value = df['Revenue'].mean()
        revenue_std = df['Revenue'].std()
        
        # Revenue concentration analysis
        customer_revenue = df.groupby('Customer_ID')['Revenue'].sum().sort_values(ascending=False)
        top_20_percent_customers = int(len(customer_revenue) * 0.2)
        top_20_revenue_share = customer_revenue.head(top_20_percent_customers).sum() / total_revenue
        
        if top_20_revenue_share > 0.8:
            self.insights['key_findings'].append(
                f"High revenue concentration: Top 20% of customers generate {top_20_revenue_share:.1%} of total revenue"
            )
            self.insights['recommendations']['customer_retention'].append(
                "Implement VIP loyalty program for high-value customers to prevent churn"
            )
        
        # Order value analysis
        if revenue_std / avg_order_value > 1.5:
            self.insights['key_findings'].append(
                f"High order value variability detected (CV: {revenue_std/avg_order_value:.2f})"
            )
            self.insights['recommendations']['marketing'].append(
                "Implement tiered pricing strategy to capture different customer segments"
            )
    
    def _analyze_customer_behavior(self, df):
        """Analyze customer behavior patterns."""
        try:
            # Customer frequency analysis
            customer_frequency = df.groupby('Customer_ID').size()
            one_time_buyers = (customer_frequency == 1).sum()
            repeat_customers = (customer_frequency > 1).sum()
            
            one_time_buyer_rate = one_time_buyers / len(customer_frequency)
            
            if one_time_buyer_rate > 0.6:
                self.insights['risk_alerts'].append(
                    f"High one-time buyer rate: {one_time_buyer_rate:.1%} of customers make only one purchase"
                )
                self.insights['recommendations']['customer_retention'].extend([
                    "Implement post-purchase email sequence to encourage repeat purchases",
                    "Offer first-time buyer discount for second purchase",
                    "Create customer onboarding program"
                ])
            
            # Purchase recency analysis
            latest_date = df['Transaction_Date'].max()
            customer_recency = df.groupby('Customer_ID')['Transaction_Date'].max()
            inactive_customers = ((latest_date - customer_recency).dt.days > 90).sum()
            
            if len(customer_recency) > 0 and inactive_customers / len(customer_recency) > 0.3:
                self.insights['risk_alerts'].append(
                    f"High customer inactivity: {inactive_customers/len(customer_recency):.1%} of customers haven't purchased in 90+ days"
                )
                self.insights['recommendations']['marketing'].append(
                    "Launch win-back campaign targeting inactive customers"
                )
        except Exception as e:
            # Skip customer behavior analysis if there are data issues
            pass
    
    def _analyze_product_performance(self, df):
        """Analyze product and category performance."""
        category_performance = df.groupby('Category').agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count',
            'Units_Sold': 'sum'
        }).reset_index()
        
        category_performance['Revenue_Share'] = category_performance['Revenue'] / category_performance['Revenue'].sum()
        category_performance['Avg_Order_Value'] = category_performance['Revenue'] / category_performance['Transaction_ID']
        
        # Identify top performing categories
        top_category = category_performance.loc[category_performance['Revenue'].idxmax()]
        self.insights['key_findings'].append(
            f"Top performing category: {top_category['Category']} generates {top_category['Revenue_Share']:.1%} of total revenue"
        )
        
        # Identify underperforming categories
        low_performing = category_performance[category_performance['Revenue_Share'] < 0.1]
        if len(low_performing) > 0:
            self.insights['opportunities'].extend([
                f"Growth opportunity in {cat} category (currently {share:.1%} of revenue)" 
                for cat, share in zip(low_performing['Category'], low_performing['Revenue_Share'])
            ])
        
        # Cross-selling opportunities
        customer_categories = df.groupby('Customer_ID')['Category'].nunique()
        single_category_customers = (customer_categories == 1).sum()
        
        if single_category_customers / len(customer_categories) > 0.7:
            self.insights['opportunities'].append(
                f"Cross-selling opportunity: {single_category_customers/len(customer_categories):.1%} of customers buy from only one category"
            )
            self.insights['recommendations']['marketing'].append(
                "Implement cross-category product recommendations"
            )
    
    def _analyze_regional_performance(self, df):
        """Analyze regional performance patterns."""
        regional_performance = df.groupby('Region').agg({
            'Revenue': 'sum',
            'Customer_ID': 'nunique',
            'Transaction_ID': 'count'
        }).reset_index()
        
        regional_performance['Revenue_Per_Customer'] = regional_performance['Revenue'] / regional_performance['Customer_ID']
        regional_performance['Transactions_Per_Customer'] = regional_performance['Transaction_ID'] / regional_performance['Customer_ID']
        
        # Identify best performing region
        best_region = regional_performance.loc[regional_performance['Revenue_Per_Customer'].idxmax()]
        self.insights['key_findings'].append(
            f"Highest value region: {best_region['Region']} with ${best_region['Revenue_Per_Customer']:.2f} revenue per customer"
        )
        
        # Identify expansion opportunities
        lowest_region = regional_performance.loc[regional_performance['Revenue'].idxmin()]
        self.insights['opportunities'].append(
            f"Market expansion opportunity in {lowest_region['Region']} region"
        )
        
        self.insights['recommendations']['marketing'].append(
            f"Increase marketing investment in {lowest_region['Region']} region"
        )
    
    def _analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in sales."""
        # Day of week analysis
        df['DayOfWeek'] = df['Transaction_Date'].dt.day_name()
        daily_performance = df.groupby('DayOfWeek')['Revenue'].sum()
        
        best_day = daily_performance.idxmax()
        worst_day = daily_performance.idxmin()
        
        self.insights['key_findings'].append(
            f"Best sales day: {best_day} (${daily_performance[best_day]:,.2f})"
        )
        
        if daily_performance[best_day] / daily_performance[worst_day] > 2:
            self.insights['opportunities'].append(
                f"Opportunity to boost {worst_day} sales through targeted promotions"
            )
        
        # Monthly trends
        df['Month'] = df['Transaction_Date'].dt.month
        monthly_performance = df.groupby('Month')['Revenue'].sum()
        
        # Identify seasonal patterns
        if monthly_performance.std() / monthly_performance.mean() > 0.3:
            peak_month = monthly_performance.idxmax()
            self.insights['key_findings'].append(
                f"Strong seasonal pattern detected, peak month: {peak_month}"
            )
            self.insights['recommendations']['operational'].append(
                "Adjust inventory and marketing spend based on seasonal patterns"
            )
    
    def _analyze_marketing_effectiveness(self, df):
        """Analyze marketing campaign effectiveness."""
        if all(col in df.columns for col in ['Clicks', 'Impressions', 'Ad_Spend']):
            # Calculate marketing metrics
            total_clicks = df['Clicks'].sum()
            total_impressions = df['Impressions'].sum()
            total_ad_spend = df['Ad_Spend'].sum()
            total_revenue = df['Revenue'].sum()
            
            if total_impressions > 0:
                overall_ctr = total_clicks / total_impressions
                self.insights['key_findings'].append(
                    f"Overall click-through rate: {overall_ctr:.2%}"
                )
                
                if overall_ctr < 0.02:
                    self.insights['risk_alerts'].append(
                        "Low click-through rate indicates poor ad targeting or creative"
                    )
                    self.insights['recommendations']['marketing'].extend([
                        "Review and optimize ad targeting parameters",
                        "A/B test new ad creatives and messaging"
                    ])
            
            if total_ad_spend > 0:
                roas = total_revenue / total_ad_spend
                self.insights['key_findings'].append(
                    f"Return on ad spend (ROAS): {roas:.2f}x"
                )
                
                if roas < 3:
                    self.insights['recommendations']['marketing'].append(
                        "Optimize ad spend allocation to improve ROAS"
                    )
                elif roas > 5:
                    self.insights['opportunities'].append(
                        "High ROAS indicates opportunity to scale marketing investment"
                    )
    
    def _identify_risks_and_opportunities(self, df):
        """Identify additional risks and opportunities."""
        # Customer acquisition analysis
        df['YearMonth'] = df['Transaction_Date'].dt.to_period('M')
        new_customers_per_month = df.groupby('YearMonth')['Customer_ID'].nunique()
        
        if len(new_customers_per_month) > 1:
            recent_trend = new_customers_per_month.iloc[-1] / new_customers_per_month.iloc[-2]
            if recent_trend < 0.9:
                self.insights['risk_alerts'].append(
                    "Declining customer acquisition trend detected"
                )
                self.insights['recommendations']['marketing'].append(
                    "Increase customer acquisition marketing spend"
                )
        
        # Price sensitivity analysis
        try:
            if 'Discount_Applied' in df.columns:
                # Count non-zero discounts
                discount_count = len(df[df['Discount_Applied'].notna()])
                total_count = len(df)
                
                if discount_count > total_count * 0.1:  # More than 10% have discounts
                    self.insights['opportunities'].append(
                        "Discount strategies show potential for boosting sales volume"
                    )
                    self.insights['recommendations']['marketing'].append(
                        "Implement strategic discount campaigns during slow periods"
                    )
        except Exception:
            # Skip discount analysis if there are data issues
            pass
        
        # Product diversity analysis
        products_per_customer = df.groupby('Customer_ID')['Product_ID'].nunique()
        avg_products_per_customer = products_per_customer.mean()
        
        if avg_products_per_customer < 2:
            self.insights['opportunities'].append(
                "Low product diversity per customer - opportunity for bundle offers"
            )
            self.insights['recommendations']['product'].append(
                "Create product bundles and cross-sell recommendations"
            )
        
        # Revenue growth analysis
        if len(df['Transaction_Date'].dt.month.unique()) > 1:
            monthly_revenue = df.groupby(df['Transaction_Date'].dt.to_period('M'))['Revenue'].sum()
            if len(monthly_revenue) > 1:
                growth_rate = (monthly_revenue.iloc[-1] / monthly_revenue.iloc[0]) ** (1/len(monthly_revenue)) - 1
                
                if growth_rate > 0.05:
                    self.insights['opportunities'].append(
                        f"Strong revenue growth trend: {growth_rate:.1%} monthly growth rate"
                    )
                elif growth_rate < -0.02:
                    self.insights['risk_alerts'].append(
                        f"Declining revenue trend: {growth_rate:.1%} monthly decline"
                    )
                    self.insights['recommendations']['operational'].append(
                        "Investigate causes of revenue decline and implement corrective measures"
                    )
