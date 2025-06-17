import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    """
    Handles all visualization tasks for the customer behavior analysis.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
    
    def plot_revenue_distribution(self, df):
        """Plot revenue distribution histogram."""
        fig = px.histogram(
            df, 
            x='Revenue', 
            nbins=50,
            title='Revenue Distribution',
            template=self.template,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="Frequency",
            showlegend=False
        )
        return fig
    
    def plot_transaction_trends(self, df):
        """Plot transaction trends over time."""
        daily_transactions = df.groupby(df['Transaction_Date'].dt.date).agg({
            'Transaction_ID': 'count',
            'Revenue': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Transaction Count', 'Daily Revenue'),
            vertical_spacing=0.1
        )
        
        # Transaction count
        fig.add_trace(
            go.Scatter(
                x=daily_transactions['Transaction_Date'],
                y=daily_transactions['Transaction_ID'],
                mode='lines+markers',
                name='Transactions',
                line=dict(color=self.color_palette[0])
            ),
            row=1, col=1
        )
        
        # Daily revenue
        fig.add_trace(
            go.Scatter(
                x=daily_transactions['Transaction_Date'],
                y=daily_transactions['Revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color=self.color_palette[1])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template=self.template,
            title="Transaction Trends Over Time",
            height=600
        )
        return fig
    
    def plot_category_distribution(self, df):
        """Plot category distribution."""
        category_stats = df.groupby('Category').agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count'
        }).reset_index()
        
        fig = px.bar(
            category_stats,
            x='Category',
            y='Revenue',
            title='Revenue by Category',
            template=self.template,
            color='Revenue',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Total Revenue ($)"
        )
        return fig
    
    def plot_region_distribution(self, df):
        """Plot region distribution."""
        region_stats = df.groupby('Region').agg({
            'Revenue': 'sum',
            'Customer_ID': 'nunique'
        }).reset_index()
        
        fig = px.pie(
            region_stats,
            values='Revenue',
            names='Region',
            title='Revenue Distribution by Region',
            template=self.template,
            color_discrete_sequence=self.color_palette
        )
        return fig
    
    def plot_rfm_segments(self, segment_counts):
        """Plot RFM segment distribution."""
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title='Customer Segment Distribution (RFM Analysis)',
            template=self.template,
            color=segment_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            xaxis_tickangle=-45
        )
        return fig
    
    def plot_clusters_3d(self, rfm_data, cluster_labels):
        """Plot 3D cluster visualization."""
        fig = px.scatter_3d(
            rfm_data,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color=cluster_labels,
            title='Customer Clusters (3D RFM Analysis)',
            template=self.template,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Recency (days)",
                yaxis_title="Frequency",
                zaxis_title="Monetary Value ($)"
            )
        )
        return fig
    
    def plot_clv_distribution(self, clv_data):
        """Plot CLV distribution."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('CLV Distribution', 'CLV vs Purchase Frequency')
        )
        
        # CLV histogram
        fig.add_trace(
            go.Histogram(
                x=clv_data['CLV'],
                nbinsx=50,
                name='CLV Distribution',
                marker_color=self.color_palette[0]
            ),
            row=1, col=1
        )
        
        # CLV vs Frequency scatter
        fig.add_trace(
            go.Scatter(
                x=clv_data['Purchase_Frequency'],
                y=clv_data['CLV'],
                mode='markers',
                name='CLV vs Frequency',
                marker=dict(color=self.color_palette[1], opacity=0.6)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template=self.template,
            title="Customer Lifetime Value Analysis",
            height=400
        )
        return fig
    
    def plot_churn_risk(self, churn_analysis):
        """Plot churn risk analysis."""
        risk_data = churn_analysis['customers_by_risk']
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "bar"}]],
            subplot_titles=('Risk Level Distribution', 'At-Risk Revenue')
        )
        
        # Pie chart for risk levels
        fig.add_trace(
            go.Pie(
                labels=list(risk_data.keys()),
                values=list(risk_data.values()),
                name="Risk Levels"
            ),
            row=1, col=1
        )
        
        # Bar chart for customer counts
        fig.add_trace(
            go.Bar(
                x=list(risk_data.keys()),
                y=list(risk_data.values()),
                name="Customer Count",
                marker_color=self.color_palette[:len(risk_data)]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template=self.template,
            title="Churn Risk Analysis",
            height=400
        )
        return fig
    
    def plot_marketing_by_category(self, df):
        """Plot marketing effectiveness by category."""
        if all(col in df.columns for col in ['Clicks', 'Impressions', 'Ad_Spend']):
            marketing_stats = df.groupby('Category').agg({
                'Clicks': 'sum',
                'Impressions': 'sum',
                'Ad_Spend': 'sum',
                'Revenue': 'sum'
            }).reset_index()
            
            marketing_stats['CTR'] = marketing_stats['Clicks'] / marketing_stats['Impressions']
            marketing_stats['ROAS'] = marketing_stats['Revenue'] / marketing_stats['Ad_Spend']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Clicks by Category', 'CTR by Category', 
                              'Ad Spend by Category', 'ROAS by Category')
            )
            
            # Clicks
            fig.add_trace(
                go.Bar(x=marketing_stats['Category'], y=marketing_stats['Clicks'], 
                      name='Clicks', marker_color=self.color_palette[0]),
                row=1, col=1
            )
            
            # CTR
            fig.add_trace(
                go.Bar(x=marketing_stats['Category'], y=marketing_stats['CTR'], 
                      name='CTR', marker_color=self.color_palette[1]),
                row=1, col=2
            )
            
            # Ad Spend
            fig.add_trace(
                go.Bar(x=marketing_stats['Category'], y=marketing_stats['Ad_Spend'], 
                      name='Ad Spend', marker_color=self.color_palette[2]),
                row=2, col=1
            )
            
            # ROAS
            fig.add_trace(
                go.Bar(x=marketing_stats['Category'], y=marketing_stats['ROAS'], 
                      name='ROAS', marker_color=self.color_palette[3]),
                row=2, col=2
            )
            
            fig.update_layout(
                template=self.template,
                title="Marketing Performance by Category",
                height=600,
                showlegend=False
            )
        else:
            # Fallback visualization
            fig = go.Figure()
            fig.add_annotation(
                text="Marketing data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    def plot_marketing_by_region(self, df):
        """Plot marketing effectiveness by region."""
        if all(col in df.columns for col in ['Clicks', 'Ad_Spend']):
            marketing_stats = df.groupby('Region').agg({
                'Clicks': 'sum',
                'Ad_Spend': 'sum',
                'Revenue': 'sum'
            }).reset_index()
            
            marketing_stats['CPC'] = marketing_stats['Ad_Spend'] / marketing_stats['Clicks']
            marketing_stats['ROAS'] = marketing_stats['Revenue'] / marketing_stats['Ad_Spend']
            
            fig = px.scatter(
                marketing_stats,
                x='CPC',
                y='ROAS',
                size='Revenue',
                color='Region',
                title='Marketing Efficiency by Region (CPC vs ROAS)',
                template=self.template,
                color_discrete_sequence=self.color_palette
            )
            fig.update_layout(
                xaxis_title="Cost Per Click ($)",
                yaxis_title="Return on Ad Spend"
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Marketing data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    def create_conversion_funnel(self, df):
        """Create conversion funnel data."""
        if all(col in df.columns for col in ['Impressions', 'Clicks']):
            total_impressions = df['Impressions'].sum()
            total_clicks = df['Clicks'].sum()
            total_transactions = len(df)
            
            funnel_data = {
                'Stage': ['Impressions', 'Clicks', 'Transactions'],
                'Count': [total_impressions, total_clicks, total_transactions],
                'Conversion_Rate': [100, (total_clicks/total_impressions)*100 if total_impressions > 0 else 0, 
                                  (total_transactions/total_clicks)*100 if total_clicks > 0 else 0]
            }
        else:
            # Fallback funnel
            total_customers = df['Customer_ID'].nunique()
            total_transactions = len(df)
            
            funnel_data = {
                'Stage': ['Unique Customers', 'Transactions'],
                'Count': [total_customers, total_transactions],
                'Conversion_Rate': [100, (total_transactions/total_customers)*100 if total_customers > 0 else 0]
            }
        
        return pd.DataFrame(funnel_data)
    
    def plot_conversion_funnel(self, funnel_data):
        """Plot conversion funnel."""
        fig = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent initial",
            marker=dict(color=self.color_palette[:len(funnel_data)])
        ))
        
        fig.update_layout(
            title="Conversion Funnel",
            template=self.template
        )
        return fig
    
    def plot_daily_patterns(self, df):
        """Plot daily purchase patterns."""
        df['Day_Name'] = df['Transaction_Date'].dt.day_name()
        daily_stats = df.groupby('Day_Name').agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count'
        }).reset_index()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats['Day_Name'] = pd.Categorical(daily_stats['Day_Name'], categories=day_order, ordered=True)
        daily_stats = daily_stats.sort_values('Day_Name')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue by Day of Week', 'Transactions by Day of Week')
        )
        
        fig.add_trace(
            go.Bar(x=daily_stats['Day_Name'], y=daily_stats['Revenue'], 
                  name='Revenue', marker_color=self.color_palette[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=daily_stats['Day_Name'], y=daily_stats['Transaction_ID'], 
                  name='Transactions', marker_color=self.color_palette[1]),
            row=1, col=2
        )
        
        fig.update_layout(
            template=self.template,
            title="Daily Purchase Patterns",
            height=400,
            showlegend=False
        )
        return fig
    
    def plot_monthly_patterns(self, df):
        """Plot monthly purchase patterns."""
        df['Month_Name'] = df['Transaction_Date'].dt.month_name()
        monthly_stats = df.groupby('Month_Name').agg({
            'Revenue': 'sum',
            'Transaction_ID': 'count'
        }).reset_index()
        
        # Create month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_stats['Month_Name'] = pd.Categorical(monthly_stats['Month_Name'], categories=month_order, ordered=True)
        monthly_stats = monthly_stats.sort_values('Month_Name')
        
        fig = px.line(
            monthly_stats,
            x='Month_Name',
            y='Revenue',
            title='Monthly Revenue Trends',
            template=self.template,
            markers=True,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue ($)"
        )
        return fig
    
    def plot_category_correlation(self, df):
        """Plot category correlation heatmap."""
        # Create category matrix
        customer_categories = df.groupby(['Customer_ID', 'Category'])['Revenue'].sum().unstack(fill_value=0)
        
        # Calculate correlation
        correlation_matrix = customer_categories.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Product Category Correlation Matrix',
            template=self.template,
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        return fig
    
    def plot_seasonal_trends(self, df):
        """Plot seasonal trends."""
        df['Quarter'] = df['Transaction_Date'].dt.quarter
        seasonal_stats = df.groupby(['Quarter', 'Category'])['Revenue'].sum().reset_index()
        
        fig = px.bar(
            seasonal_stats,
            x='Quarter',
            y='Revenue',
            color='Category',
            title='Seasonal Revenue Trends by Category',
            template=self.template,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Revenue ($)"
        )
        return fig
    
    def create_performance_dashboard(self, df):
        """Create a comprehensive performance dashboard."""
        # Calculate key metrics
        total_revenue = df['Revenue'].sum()
        total_customers = df['Customer_ID'].nunique()
        avg_order_value = df['Revenue'].mean()
        total_transactions = len(df)
        
        # Create dashboard layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Top Categories', 'Regional Performance', 'Customer Segments'),
            specs=[[{"secondary_y": False}, {"type": "domain"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Revenue trend
        daily_revenue = df.groupby(df['Transaction_Date'].dt.date)['Revenue'].sum()
        fig.add_trace(
            go.Scatter(x=daily_revenue.index, y=daily_revenue.values, 
                      mode='lines', name='Daily Revenue'),
            row=1, col=1
        )
        
        # Top categories
        category_revenue = df.groupby('Category')['Revenue'].sum().nlargest(5)
        fig.add_trace(
            go.Pie(labels=category_revenue.index, values=category_revenue.values, 
                  name="Top Categories"),
            row=1, col=2
        )
        
        # Regional performance
        region_stats = df.groupby('Region')['Revenue'].sum()
        fig.add_trace(
            go.Bar(x=region_stats.index, y=region_stats.values, 
                  name='Regional Revenue'),
            row=2, col=1
        )
        
        # Customer distribution by region
        customer_region = df.groupby('Region')['Customer_ID'].nunique()
        fig.add_trace(
            go.Bar(x=customer_region.index, y=customer_region.values, 
                  name='Customers by Region'),
            row=2, col=2
        )
        
        fig.update_layout(
            template=self.template,
            title=f"Performance Dashboard - Total Revenue: ${total_revenue:,.2f}",
            height=800,
            showlegend=False
        )
        
        return fig
