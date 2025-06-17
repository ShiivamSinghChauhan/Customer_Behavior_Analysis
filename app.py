import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from customer_segmentation import CustomerSegmentation
from predictive_analytics import PredictiveAnalytics
from visualization import Visualizer
from insights_generator import InsightsGenerator
from utils import Utils

# Page configuration
st.set_page_config(
    page_title="AI Customer Behavior Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def main():
    st.title("🛍️ AI-Powered Customer Behavior Analysis")
    st.markdown("""
    ### Comprehensive E-commerce Analytics Dashboard
    Upload your transaction data to unlock deep customer insights, predictive analytics, and actionable business recommendations.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Module",
        ["📊 Data Overview", "👥 Customer Segmentation", "🔮 Predictive Analytics", 
         "📈 Marketing Analysis", "🛒 Purchase Patterns", "📋 Insights & Recommendations", "📤 Export Results"]
    )
    
    # Data upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your e-commerce transaction data (CSV format)"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            if not st.session_state.data_loaded:
                with st.spinner("Loading and processing data..."):
                    data_processor = DataProcessor()
                    df = data_processor.load_data(uploaded_file)
                    processed_data = data_processor.preprocess_data(df)
                    
                    st.session_state.processed_data = processed_data
                    st.session_state.data_loaded = True
                    
                    st.sidebar.success(f"✅ Data loaded successfully! {len(df):,} transactions processed")
            
            # Initialize analysis modules
            if st.session_state.data_loaded:
                df = st.session_state.processed_data
                
                # Initialize analysis classes
                segmentation = CustomerSegmentation()
                predictive = PredictiveAnalytics()
                visualizer = Visualizer()
                insights = InsightsGenerator()
                
                # Route to different pages
                if page == "📊 Data Overview":
                    show_data_overview(df, visualizer)
                elif page == "👥 Customer Segmentation":
                    show_customer_segmentation(df, segmentation, visualizer)
                elif page == "🔮 Predictive Analytics":
                    show_predictive_analytics(df, predictive, visualizer)
                elif page == "📈 Marketing Analysis":
                    show_marketing_analysis(df, visualizer)
                elif page == "🛒 Purchase Patterns":
                    show_purchase_patterns(df, visualizer)
                elif page == "📋 Insights & Recommendations":
                    show_insights_recommendations(df, insights, visualizer)
                elif page == "📤 Export Results":
                    show_export_results(df, segmentation, predictive)
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please ensure your CSV file has the required columns and format.")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show sample data format
        st.subheader("Expected Data Format")
        sample_data = {
            'Transaction_ID': ['tx001', 'tx002', 'tx003'],
            'Customer_ID': ['cust001', 'cust002', 'cust001'],
            'Product_ID': ['prod001', 'prod002', 'prod003'],
            'Transaction_Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Units_Sold': [2, 1, 3],
            'Revenue': [29.99, 49.99, 89.97],
            'Category': ['Electronics', 'Clothing', 'Books'],
            'Region': ['North America', 'Europe', 'Asia']
        }
        st.table(pd.DataFrame(sample_data))

def show_data_overview(df, visualizer):
    st.header("📊 Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Unique Customers", f"{df['Customer_ID'].nunique():,}")
    with col3:
        st.metric("Total Revenue", f"${df['Revenue'].sum():,.2f}")
    with col4:
        st.metric("Avg Order Value", f"${df['Revenue'].mean():.2f}")
    
    # Data quality metrics
    st.subheader("Data Quality Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"- {col}: {missing} ({missing/len(df)*100:.1f}%)")
        else:
            st.success("✅ No missing values found")
    
    with col2:
        st.write("Data Types:")
        for col, dtype in df.dtypes.items():
            st.write(f"- {col}: {dtype}")
    
    # Visualizations
    st.subheader("Data Distribution")
    
    # Revenue distribution
    fig_revenue = visualizer.plot_revenue_distribution(df)
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Transaction trends
    fig_trends = visualizer.plot_transaction_trends(df)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Category analysis
    col1, col2 = st.columns(2)
    with col1:
        fig_category = visualizer.plot_category_distribution(df)
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        fig_region = visualizer.plot_region_distribution(df)
        st.plotly_chart(fig_region, use_container_width=True)

def show_customer_segmentation(df, segmentation, visualizer):
    st.header("👥 Customer Segmentation")
    
    # RFM Analysis
    st.subheader("RFM Analysis")
    with st.spinner("Performing RFM analysis..."):
        rfm_data = segmentation.calculate_rfm(df)
        rfm_segments = segmentation.create_rfm_segments(rfm_data)
    
    # Display RFM metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Recency (days)", f"{rfm_data['Recency'].mean():.1f}")
    with col2:
        st.metric("Avg Frequency", f"{rfm_data['Frequency'].mean():.1f}")
    with col3:
        st.metric("Avg Monetary Value", f"${rfm_data['Monetary'].mean():.2f}")
    
    # RFM Segment Distribution
    segment_counts = rfm_segments['RFM_Segment'].value_counts()
    fig_segments = visualizer.plot_rfm_segments(segment_counts)
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Detailed segment analysis
    st.subheader("Segment Profiles")
    segment_profiles = segmentation.analyze_segments(df, rfm_segments)
    
    for segment, profile in segment_profiles.items():
        with st.expander(f"{segment} - {profile['count']} customers ({profile['percentage']:.1f}%)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Average Revenue:** ${profile['avg_revenue']:.2f}")
                st.write(f"**Average Frequency:** {profile['avg_frequency']:.1f}")
                st.write(f"**Average Recency:** {profile['avg_recency']:.1f} days")
            with col2:
                st.write(f"**Total Revenue:** ${profile['total_revenue']:.2f}")
                st.write(f"**Revenue Share:** {profile['revenue_share']:.1f}%")
                st.write("**Characteristics:** " + profile['characteristics'])
    
    # K-means clustering
    st.subheader("Advanced Clustering Analysis")
    with st.spinner("Performing K-means clustering..."):
        kmeans_result = segmentation.perform_kmeans_clustering(rfm_data)
    
    # Cluster visualization
    fig_clusters = visualizer.plot_clusters_3d(rfm_data, kmeans_result['labels'])
    st.plotly_chart(fig_clusters, use_container_width=True)

def show_predictive_analytics(df, predictive, visualizer):
    st.header("🔮 Predictive Analytics")
    
    # Customer Lifetime Value Prediction
    st.subheader("Customer Lifetime Value (CLV) Prediction")
    with st.spinner("Calculating CLV predictions..."):
        clv_predictions = predictive.calculate_clv(df)
    
    # CLV metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average CLV", f"${clv_predictions['CLV'].mean():.2f}")
    with col2:
        st.metric("Median CLV", f"${clv_predictions['CLV'].median():.2f}")
    with col3:
        st.metric("Top 10% CLV", f"${clv_predictions['CLV'].quantile(0.9):.2f}")
    
    # CLV distribution
    fig_clv = visualizer.plot_clv_distribution(clv_predictions)
    st.plotly_chart(fig_clv, use_container_width=True)
    
    # Churn Risk Analysis
    st.subheader("Churn Risk Assessment")
    with st.spinner("Analyzing churn risk..."):
        churn_analysis = predictive.analyze_churn_risk(df)
    
    # Churn metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High Risk Customers", f"{churn_analysis['high_risk_count']:,}")
        st.metric("Medium Risk Customers", f"{churn_analysis['medium_risk_count']:,}")
    with col2:
        st.metric("Low Risk Customers", f"{churn_analysis['low_risk_count']:,}")
        st.metric("At-Risk Revenue", f"${churn_analysis['at_risk_revenue']:,.2f}")
    
    # Churn risk visualization
    fig_churn = visualizer.plot_churn_risk(churn_analysis)
    st.plotly_chart(fig_churn, use_container_width=True)
    
    # Next Purchase Prediction
    st.subheader("Next Purchase Prediction")
    with st.spinner("Predicting next purchases..."):
        purchase_predictions = predictive.predict_next_purchase(df)
    
    # Show top predictions
    st.write("Top 10 Next Purchase Predictions:")
    st.dataframe(purchase_predictions.head(10))

def show_marketing_analysis(df, visualizer):
    st.header("📈 Marketing Analysis")
    
    # Marketing effectiveness metrics
    st.subheader("Campaign Effectiveness Overview")
    
    # Calculate marketing metrics
    marketing_metrics = {
        'total_clicks': df['Clicks'].sum(),
        'total_impressions': df['Impressions'].sum(),
        'avg_ctr': df['Ad_CTR'].mean(),
        'avg_cpc': df['Ad_CPC'].mean(),
        'total_spend': df['Ad_Spend'].sum(),
        'roas': df['Revenue'].sum() / df['Ad_Spend'].sum() if df['Ad_Spend'].sum() > 0 else 0
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clicks", f"{marketing_metrics['total_clicks']:,}")
        st.metric("Average CTR", f"{marketing_metrics['avg_ctr']:.3f}")
    with col2:
        st.metric("Total Impressions", f"{marketing_metrics['total_impressions']:,}")
        st.metric("Average CPC", f"${marketing_metrics['avg_cpc']:.2f}")
    with col3:
        st.metric("Total Ad Spend", f"${marketing_metrics['total_spend']:,.2f}")
        st.metric("ROAS", f"{marketing_metrics['roas']:.2f}x")
    
    # Marketing performance by category
    fig_marketing_cat = visualizer.plot_marketing_by_category(df)
    st.plotly_chart(fig_marketing_cat, use_container_width=True)
    
    # Marketing performance by region
    fig_marketing_region = visualizer.plot_marketing_by_region(df)
    st.plotly_chart(fig_marketing_region, use_container_width=True)
    
    # Conversion funnel
    st.subheader("Conversion Funnel Analysis")
    funnel_data = visualizer.create_conversion_funnel(df)
    fig_funnel = visualizer.plot_conversion_funnel(funnel_data)
    st.plotly_chart(fig_funnel, use_container_width=True)

def show_purchase_patterns(df, visualizer):
    st.header("🛒 Purchase Patterns")
    
    # Temporal patterns
    st.subheader("Temporal Purchase Patterns")
    
    # Daily patterns
    fig_daily = visualizer.plot_daily_patterns(df)
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Monthly patterns
    fig_monthly = visualizer.plot_monthly_patterns(df)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Product affinity analysis
    st.subheader("Product Category Affinity")
    
    # Category correlation
    fig_correlation = visualizer.plot_category_correlation(df)
    st.plotly_chart(fig_correlation, use_container_width=True)
    
    # Seasonal trends
    st.subheader("Seasonal Trends")
    fig_seasonal = visualizer.plot_seasonal_trends(df)
    st.plotly_chart(fig_seasonal, use_container_width=True)

def show_insights_recommendations(df, insights, visualizer):
    st.header("📋 Insights & Recommendations")
    
    # Generate comprehensive insights
    with st.spinner("Generating AI-powered insights..."):
        business_insights = insights.generate_business_insights(df)
    
    # Key findings
    st.subheader("🔍 Key Findings")
    for i, finding in enumerate(business_insights['key_findings'], 1):
        st.write(f"{i}. {finding}")
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    for category, recommendations in business_insights['recommendations'].items():
        with st.expander(f"{category.title()} Recommendations"):
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Risk alerts
    if business_insights['risk_alerts']:
        st.subheader("⚠️ Risk Alerts")
        for alert in business_insights['risk_alerts']:
            st.warning(alert)
    
    # Opportunities
    st.subheader("🎯 Growth Opportunities")
    for opportunity in business_insights['opportunities']:
        st.success(opportunity)
    
    # Performance summary
    st.subheader("📊 Performance Summary")
    performance_fig = visualizer.create_performance_dashboard(df)
    st.plotly_chart(performance_fig, use_container_width=True)

def show_export_results(df, segmentation, predictive):
    st.header("📤 Export Results")
    
    st.write("Export your analysis results in various formats:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Exports")
        
        # Export processed data
        if st.button("Export Processed Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        # Export RFM analysis
        if st.button("Export RFM Analysis (CSV)"):
            rfm_data = segmentation.calculate_rfm(df)
            csv = rfm_data.to_csv(index=False)
            st.download_button(
                label="Download RFM Analysis",
                data=csv,
                file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("📋 Reports")
        
        # Generate summary report
        if st.button("Generate Summary Report (JSON)"):
            report = Utils.generate_summary_report(df)
            json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="Download Report",
                data=json_str,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
