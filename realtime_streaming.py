import pandas as pd
import numpy as np
import streamlit as st
import asyncio
import websockets
import json
import threading
import time
from datetime import datetime, timedelta
import requests
from queue import Queue

class RealTimeDataStreamer:
    """
    Handles real-time data streaming integration for live customer behavior tracking.
    """
    
    def __init__(self):
        self.data_queue = Queue()
        self.is_streaming = False
        self.stream_thread = None
        self.websocket_url = None
        self.api_endpoint = None
        
    def setup_streaming_source(self, source_type, config):
        """Setup streaming data source configuration."""
        if source_type == "websocket":
            self.websocket_url = config.get("url")
        elif source_type == "api":
            self.api_endpoint = config.get("endpoint")
            self.api_headers = config.get("headers", {})
            self.polling_interval = config.get("interval", 30)  # seconds
        
    def start_streaming(self, source_type="api"):
        """Start real-time data streaming."""
        if self.is_streaming:
            return "Streaming already active"
        
        self.is_streaming = True
        
        if source_type == "websocket" and self.websocket_url:
            self.stream_thread = threading.Thread(target=self._websocket_stream)
        elif source_type == "api" and self.api_endpoint:
            self.stream_thread = threading.Thread(target=self._api_polling_stream)
        else:
            # Demo mode with simulated data
            self.stream_thread = threading.Thread(target=self._demo_stream)
        
        self.stream_thread.start()
        return "Streaming started successfully"
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        return "Streaming stopped"
    
    def _websocket_stream(self):
        """Handle WebSocket streaming data."""
        async def websocket_handler():
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    while self.is_streaming:
                        data = await websocket.recv()
                        parsed_data = json.loads(data)
                        self.data_queue.put(parsed_data)
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
        
        # Run the async function
        try:
            asyncio.run(websocket_handler())
        except Exception as e:
            st.error(f"WebSocket streaming error: {str(e)}")
    
    def _api_polling_stream(self):
        """Handle API polling for real-time data."""
        while self.is_streaming:
            try:
                response = requests.get(self.api_endpoint, headers=self.api_headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self.data_queue.put(data)
                time.sleep(self.polling_interval)
            except Exception as e:
                st.error(f"API polling error: {str(e)}")
                time.sleep(self.polling_interval)
    
    def _demo_stream(self):
        """Generate demo streaming data for testing."""
        customer_ids = [f"Customer_{i}" for i in range(1, 1001)]
        products = [f"Product_{i}" for i in range(1, 101)]
        categories = ["Electronics", "Clothing", "Books", "Home Appliances", "Toys"]
        regions = ["North America", "Europe", "Asia"]
        
        while self.is_streaming:
            # Generate realistic streaming transaction data
            transaction = {
                "Transaction_ID": f"stream_{int(time.time())}_{np.random.randint(1000, 9999)}",
                "Customer_ID": np.random.choice(customer_ids),
                "Product_ID": np.random.choice(products),
                "Transaction_Date": datetime.now().isoformat(),
                "Units_Sold": np.random.randint(1, 5),
                "Revenue": round(np.random.uniform(10, 500), 2),
                "Category": np.random.choice(categories),
                "Region": np.random.choice(regions),
                "Discount_Applied": round(np.random.uniform(0, 0.3), 2),
                "Clicks": np.random.randint(1, 50),
                "Impressions": np.random.randint(50, 500),
                "Ad_CTR": round(np.random.uniform(0.01, 0.2), 4),
                "Ad_CPC": round(np.random.uniform(0.1, 2.0), 2),
                "Ad_Spend": round(np.random.uniform(5, 100), 2)
            }
            
            self.data_queue.put(transaction)
            time.sleep(np.random.uniform(1, 5))  # Random interval between 1-5 seconds
    
    def get_recent_data(self, max_items=100):
        """Get recent streaming data."""
        recent_data = []
        count = 0
        
        while not self.data_queue.empty() and count < max_items:
            recent_data.append(self.data_queue.get())
            count += 1
        
        if recent_data:
            return pd.DataFrame(recent_data)
        return pd.DataFrame()
    
    def get_streaming_metrics(self):
        """Get real-time streaming metrics."""
        recent_df = self.get_recent_data(50)
        
        if recent_df.empty:
            return {
                "total_transactions": 0,
                "revenue_last_hour": 0,
                "avg_order_value": 0,
                "unique_customers": 0,
                "top_category": "N/A"
            }
        
        # Convert Transaction_Date to datetime
        recent_df['Transaction_Date'] = pd.to_datetime(recent_df['Transaction_Date'])
        
        # Filter last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        last_hour_df = recent_df[recent_df['Transaction_Date'] >= one_hour_ago]
        
        metrics = {
            "total_transactions": len(recent_df),
            "revenue_last_hour": last_hour_df['Revenue'].sum() if not last_hour_df.empty else 0,
            "avg_order_value": recent_df['Revenue'].mean() if not recent_df.empty else 0,
            "unique_customers": recent_df['Customer_ID'].nunique() if not recent_df.empty else 0,
            "top_category": recent_df['Category'].mode()[0] if not recent_df.empty else "N/A"
        }
        
        return metrics
    
    def detect_real_time_anomalies(self, df):
        """Detect anomalies in real-time data."""
        if df.empty or len(df) < 10:
            return []
        
        anomalies = []
        
        # Revenue anomaly detection
        revenue_mean = df['Revenue'].mean()
        revenue_std = df['Revenue'].std()
        revenue_threshold = revenue_mean + 2 * revenue_std
        
        high_revenue_transactions = df[df['Revenue'] > revenue_threshold]
        if not high_revenue_transactions.empty:
            anomalies.append({
                "type": "High Revenue Alert",
                "message": f"Detected {len(high_revenue_transactions)} unusually high-value transactions",
                "severity": "medium"
            })
        
        # Sudden spike in transactions
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df = df.sort_values('Transaction_Date')
        
        # Check for transaction spikes in 5-minute windows
        df['time_window'] = df['Transaction_Date'].dt.floor('5T')
        transaction_counts = df.groupby('time_window').size()
        
        if len(transaction_counts) > 1:
            recent_avg = transaction_counts.mean()
            latest_count = transaction_counts.iloc[-1]
            
            if latest_count > recent_avg * 2:
                anomalies.append({
                    "type": "Transaction Spike",
                    "message": f"Unusual transaction spike detected: {latest_count} transactions in last 5 minutes",
                    "severity": "high"
                })
        
        # New customer surge
        recent_customers = df['Customer_ID'].nunique()
        if recent_customers > 20:  # Threshold for demo
            anomalies.append({
                "type": "Customer Surge",
                "message": f"High customer activity: {recent_customers} unique customers in recent activity",
                "severity": "low"
            })
        
        return anomalies