import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Handles customer segmentation using RFM analysis and clustering algorithms.
    """
    
    def __init__(self):
        self.rfm_segments = {
            'Champions': {'R': [4, 5], 'F': [4, 5], 'M': [4, 5]},
            'Loyal Customers': {'R': [3, 5], 'F': [3, 5], 'M': [3, 5]},
            'Potential Loyalists': {'R': [3, 5], 'F': [1, 3], 'M': [1, 3]},
            'Recent Customers': {'R': [4, 5], 'F': [1, 2], 'M': [1, 2]},
            'Promising': {'R': [3, 4], 'F': [1, 2], 'M': [1, 2]},
            'Customers Needing Attention': {'R': [2, 3], 'F': [2, 3], 'M': [2, 3]},
            'About to Sleep': {'R': [2, 3], 'F': [1, 2], 'M': [1, 2]},
            'At Risk': {'R': [1, 2], 'F': [2, 4], 'M': [2, 4]},
            'Cannot Lose Them': {'R': [1, 2], 'F': [4, 5], 'M': [4, 5]},
            'Hibernating': {'R': [1, 2], 'F': [1, 2], 'M': [1, 2]}
        }
    
    def calculate_rfm(self, df):
        """Calculate RFM metrics for each customer."""
        # Calculate the reference date (latest transaction date + 1 day)
        reference_date = df['Transaction_Date'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby('Customer_ID').agg({
            'Transaction_Date': lambda x: (reference_date - x.max()).days,  # Recency
            'Transaction_ID': 'count',  # Frequency
            'Revenue': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate RFM score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
    
    def create_rfm_segments(self, rfm_data):
        """Create customer segments based on RFM scores."""
        rfm_segments = rfm_data.copy()
        
        # Initialize segment column
        rfm_segments['RFM_Segment'] = 'Others'
        
        # Assign segments based on RFM scores
        for segment, criteria in self.rfm_segments.items():
            mask = (
                (rfm_segments['R_Score'].between(criteria['R'][0], criteria['R'][1])) &
                (rfm_segments['F_Score'].between(criteria['F'][0], criteria['F'][1])) &
                (rfm_segments['M_Score'].between(criteria['M'][0], criteria['M'][1]))
            )
            rfm_segments.loc[mask, 'RFM_Segment'] = segment
        
        return rfm_segments
    
    def analyze_segments(self, df, rfm_segments):
        """Analyze characteristics of each segment."""
        # Merge transaction data with segments
        df_with_segments = df.merge(rfm_segments[['Customer_ID', 'RFM_Segment']], on='Customer_ID')
        
        segment_analysis = {}
        
        for segment in rfm_segments['RFM_Segment'].unique():
            segment_data = df_with_segments[df_with_segments['RFM_Segment'] == segment]
            segment_customers = rfm_segments[rfm_segments['RFM_Segment'] == segment]
            
            analysis = {
                'count': len(segment_customers),
                'percentage': len(segment_customers) / len(rfm_segments) * 100,
                'total_revenue': segment_data['Revenue'].sum(),
                'avg_revenue': segment_data['Revenue'].mean(),
                'avg_frequency': segment_customers['Frequency'].mean(),
                'avg_recency': segment_customers['Recency'].mean(),
                'avg_monetary': segment_customers['Monetary'].mean(),
                'revenue_share': segment_data['Revenue'].sum() / df['Revenue'].sum() * 100,
                'top_categories': segment_data['Category'].value_counts().head(3).to_dict(),
                'avg_order_value': segment_data['Revenue'].mean(),
                'characteristics': self.get_segment_characteristics(segment)
            }
            
            segment_analysis[segment] = analysis
        
        return segment_analysis
    
    def get_segment_characteristics(self, segment):
        """Get characteristics description for each segment."""
        characteristics = {
            'Champions': 'High-value customers who buy frequently and recently',
            'Loyal Customers': 'Consistent customers with good purchase history',
            'Potential Loyalists': 'Recent customers with potential for growth',
            'Recent Customers': 'New customers who made recent purchases',
            'Promising': 'New customers with good potential',
            'Customers Needing Attention': 'Customers showing declining engagement',
            'About to Sleep': 'Customers at risk of becoming inactive',
            'At Risk': 'Previous good customers who haven\'t purchased recently',
            'Cannot Lose Them': 'High-value customers at risk of churning',
            'Hibernating': 'Inactive customers with low engagement'
        }
        
        return characteristics.get(segment, 'Other customer segment')
    
    def perform_kmeans_clustering(self, rfm_data, n_clusters=None):
        """Perform K-means clustering on RFM data."""
        # Prepare data for clustering
        features = ['Recency', 'Frequency', 'Monetary']
        X = rfm_data[features].values
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X_scaled)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Add cluster labels to RFM data
        rfm_clustered = rfm_data.copy()
        rfm_clustered['Cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(rfm_clustered)
        
        return {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_centers': kmeans.cluster_centers_,
            'rfm_clustered': rfm_clustered,
            'cluster_analysis': cluster_analysis,
            'scaler': scaler
        }
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method."""
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(X) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection - find the point where improvement slows down
        if len(inertias) >= 3:
            # Calculate the rate of change
            rates = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            # Find the elbow point
            for i in range(1, len(rates)):
                if rates[i-1] / rates[i] > 2:  # Significant change in rate
                    return K_range[i]
        
        # Default to 4 clusters if elbow method doesn't work well
        return 4
    
    def analyze_clusters(self, rfm_clustered):
        """Analyze characteristics of K-means clusters."""
        cluster_analysis = {}
        
        for cluster in rfm_clustered['Cluster'].unique():
            cluster_data = rfm_clustered[rfm_clustered['Cluster'] == cluster]
            
            analysis = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(rfm_clustered) * 100,
                'avg_recency': cluster_data['Recency'].mean(),
                'avg_frequency': cluster_data['Frequency'].mean(),
                'avg_monetary': cluster_data['Monetary'].mean(),
                'total_monetary': cluster_data['Monetary'].sum(),
                'characteristics': self.get_cluster_characteristics(cluster_data)
            }
            
            cluster_analysis[f'Cluster_{cluster}'] = analysis
        
        return cluster_analysis
    
    def get_cluster_characteristics(self, cluster_data):
        """Generate characteristics for a cluster based on RFM values."""
        avg_recency = cluster_data['Recency'].mean()
        avg_frequency = cluster_data['Frequency'].mean()
        avg_monetary = cluster_data['Monetary'].mean()
        
        characteristics = []
        
        # Recency characteristics
        if avg_recency <= 30:
            characteristics.append("Recent buyers")
        elif avg_recency <= 90:
            characteristics.append("Moderately recent buyers")
        else:
            characteristics.append("Inactive buyers")
        
        # Frequency characteristics
        if avg_frequency >= 10:
            characteristics.append("Frequent buyers")
        elif avg_frequency >= 5:
            characteristics.append("Occasional buyers")
        else:
            characteristics.append("Infrequent buyers")
        
        # Monetary characteristics
        if avg_monetary >= 1000:
            characteristics.append("High-value customers")
        elif avg_monetary >= 500:
            characteristics.append("Medium-value customers")
        else:
            characteristics.append("Low-value customers")
        
        return ", ".join(characteristics)
