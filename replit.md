# AI-Powered Customer Behavior Analysis Dashboard

## Overview

This is a comprehensive AI-powered customer behavior analysis application built with Streamlit. The system processes e-commerce transaction data to provide deep insights into customer behavior, segmentation, predictive analytics, and actionable business recommendations. The application includes advanced features for real-time data streaming, NLP sentiment analysis, A/B testing, and automated anomaly detection.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web-based dashboard interface
- **Layout**: Wide layout with expandable sidebar navigation
- **Visualization**: Plotly for interactive charts and graphs, with matplotlib/seaborn as secondary options
- **User Interface**: Multi-page application with modular navigation system

### Backend Architecture
- **Language**: Python 3.11+
- **Data Processing**: Pandas and NumPy for data manipulation and analysis
- **Machine Learning**: Scikit-learn for predictive modeling and clustering
- **Analytics Engine**: Custom modules for specialized analysis tasks

### Module Structure
The application follows a modular architecture with specialized components:

**Core Analysis Modules:**
- `DataProcessor`: Handles data loading, validation, and preprocessing
- `CustomerSegmentation`: Implements RFM analysis and customer clustering
- `PredictiveAnalytics`: Manages CLV prediction and churn analysis
- `Visualizer`: Creates interactive visualizations and charts
- `InsightsGenerator`: Generates AI-powered business insights and recommendations
- `Utils`: Provides utility functions for formatting and calculations

**Advanced Feature Modules:**
- `RealTimeDataStreamer`: Handles real-time data streaming from APIs, WebSockets, or demo mode
- `NLPSentimentAnalyzer`: Advanced sentiment analysis using Hugging Face models with fallback keyword analysis
- `ABTestingFramework`: Complete A/B testing framework with statistical significance testing
- `AnomalyDetectionSystem`: Automated anomaly detection and alert system for behavior changes

## Key Components

### Data Processing Pipeline
- **Input Validation**: Ensures required columns are present and data quality meets minimum standards
- **Preprocessing**: Handles missing values, outlier detection, and data type conversions
- **Feature Engineering**: Creates derived metrics for advanced analytics

### Customer Segmentation Engine
- **RFM Analysis**: Calculates Recency, Frequency, and Monetary value scores
- **Clustering**: Uses K-means clustering with silhouette score optimization
- **Segment Classification**: Maps customers to predefined business segments (Champions, At Risk, etc.)

### Predictive Analytics Module
- **Customer Lifetime Value (CLV)**: Predicts future customer value using Random Forest
- **Churn Prediction**: Identifies at-risk customers using classification models
- **Feature Engineering**: Creates customer-level metrics from transaction data

### Visualization Framework
- **Interactive Charts**: Plotly-based visualizations with customizable themes
- **Dashboard Layout**: Multi-subplot layouts for comprehensive data views
- **Export Capabilities**: Support for various output formats

### Insights Generation System
- **Pattern Recognition**: Automated detection of business-critical patterns
- **Recommendation Engine**: Generates actionable recommendations across marketing, retention, and operations
- **Risk Assessment**: Identifies potential business risks and opportunities

## Data Flow

1. **Data Ingestion**: CSV files uploaded through Streamlit file uploader
2. **Validation**: Checks for required columns and data quality standards
3. **Preprocessing**: Cleans data, handles missing values, and creates derived features
4. **Analysis**: Parallel processing of segmentation, predictive analytics, and insights generation
5. **Visualization**: Creates interactive charts and dashboards
6. **Output**: Presents results through multi-page interface with export capabilities

### Expected Data Schema
The system expects the following core columns:
- `Transaction_ID`: Unique transaction identifier
- `Customer_ID`: Unique customer identifier
- `Product_ID`: Product identifier
- `Transaction_Date`: Transaction timestamp
- `Units_Sold`: Quantity purchased
- `Revenue`: Transaction revenue
- `Category`: Product category
- `Region`: Geographic region

Optional columns for enhanced analysis:
- `Discount_Applied`: Discount percentage
- `Clicks`, `Impressions`: Marketing metrics
- `Ad_CTR`, `Ad_CPC`, `Ad_Spend`: Advertising data

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Statistical plotting

### Development Dependencies
- **Python 3.11+**: Runtime environment
- **UV**: Package management
- **Nix**: Development environment management

### System Dependencies (via Nix)
- Cairo, FFmpeg, FreeType: Graphics and media processing
- GTK3, GObject Introspection: GUI framework support
- QHULL, TCL/TK: Mathematical and visualization libraries

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale deployment
- **Port Configuration**: Streamlit runs on port 5000
- **Workflow**: Automated deployment with parallel task execution

### Environment Setup
- Nix-based package management ensures consistent environment
- All dependencies managed through `pyproject.toml` and Nix configuration
- Streamlit configuration optimized for cloud deployment

### Scalability Considerations
- Modular architecture supports horizontal scaling
- Stateless design with session state management
- Efficient memory usage through lazy loading and data streaming

## Changelog

- June 17, 2025: Initial setup with core customer behavior analysis features
- June 17, 2025: Integrated advanced features including real-time streaming, NLP sentiment analysis, A/B testing framework, and anomaly detection system

## User Preferences

Preferred communication style: Simple, everyday language.