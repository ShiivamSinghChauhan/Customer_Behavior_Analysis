import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import re
from datetime import datetime
import time

class NLPSentimentAnalyzer:
    """
    Advanced NLP analysis using Hugging Face models for customer review sentiment analysis.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.models = {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "emotion": "j-hartmann/emotion-english-distilroberta-base",
            "satisfaction": "nlptown/bert-base-multilingual-uncased-sentiment"
        }
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
    def setup_api_key(self, api_key):
        """Setup Hugging Face API key."""
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def query_huggingface_api(self, text, model_name):
        """Query Hugging Face Inference API."""
        if not self.api_key:
            return self._fallback_sentiment_analysis(text)
            
        try:
            response = requests.post(
                f"{self.api_url}{self.models[model_name]}",
                headers=self.headers,
                json={"inputs": text},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model loading, wait and retry
                time.sleep(2)
                response = requests.post(
                    f"{self.api_url}{self.models[model_name]}",
                    headers=self.headers,
                    json={"inputs": text},
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
            
            return self._fallback_sentiment_analysis(text)
            
        except Exception as e:
            st.warning(f"API request failed, using fallback analysis: {str(e)}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text):
        """Fallback sentiment analysis using keyword-based approach."""
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 
                         'wonderful', 'fantastic', 'outstanding', 'satisfied', 'happy', 'pleased']
        
        # Negative keywords
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed', 
                         'poor', 'unsatisfied', 'angry', 'frustrated', 'broken', 'defective']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return [{"label": "POSITIVE", "score": 0.7 + (positive_count * 0.1)}]
        elif negative_count > positive_count:
            return [{"label": "NEGATIVE", "score": 0.7 + (negative_count * 0.1)}]
        else:
            return [{"label": "NEUTRAL", "score": 0.6}]
    
    def analyze_customer_reviews(self, reviews_data):
        """Analyze sentiment of customer reviews."""
        if reviews_data.empty:
            return pd.DataFrame()
        
        results = []
        
        for idx, review in reviews_data.iterrows():
            review_text = str(review.get('review_text', ''))
            
            if len(review_text.strip()) < 10:  # Skip very short reviews
                continue
            
            # Analyze sentiment
            sentiment_result = self.query_huggingface_api(review_text, 'sentiment')
            
            # Extract sentiment information
            if sentiment_result and len(sentiment_result) > 0:
                sentiment_info = sentiment_result[0]
                sentiment_label = sentiment_info.get('label', 'NEUTRAL')
                sentiment_score = sentiment_info.get('score', 0.5)
            else:
                sentiment_label = 'NEUTRAL'
                sentiment_score = 0.5
            
            # Map sentiment labels
            if sentiment_label in ['LABEL_2', 'POSITIVE']:
                sentiment = 'Positive'
            elif sentiment_label in ['LABEL_0', 'NEGATIVE']:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            results.append({
                'review_id': review.get('review_id', idx),
                'customer_id': review.get('customer_id', f'customer_{idx}'),
                'product_id': review.get('product_id', f'product_{idx}'),
                'review_text': review_text,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'review_length': len(review_text),
                'timestamp': review.get('timestamp', datetime.now())
            })
        
        return pd.DataFrame(results)
    
    def analyze_product_feedback(self, df):
        """Analyze product feedback and generate insights."""
        if df.empty:
            return {}
        
        # Overall sentiment distribution
        sentiment_dist = df['sentiment'].value_counts(normalize=True)
        
        # Product-wise sentiment
        product_sentiment = df.groupby('product_id')['sentiment'].value_counts(normalize=True).unstack(fill_value=0)
        
        # Sentiment trends over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        sentiment_trends = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        # Key insights
        insights = {
            'overall_sentiment': sentiment_dist.to_dict(),
            'product_sentiment': product_sentiment.to_dict(),
            'sentiment_trends': sentiment_trends.to_dict(),
            'total_reviews': len(df),
            'avg_sentiment_score': df['sentiment_score'].mean(),
            'positive_ratio': sentiment_dist.get('Positive', 0),
            'negative_ratio': sentiment_dist.get('Negative', 0)
        }
        
        # Generate recommendations
        recommendations = self._generate_sentiment_recommendations(insights)
        insights['recommendations'] = recommendations
        
        return insights
    
    def _generate_sentiment_recommendations(self, insights):
        """Generate recommendations based on sentiment analysis."""
        recommendations = []
        
        positive_ratio = insights.get('positive_ratio', 0)
        negative_ratio = insights.get('negative_ratio', 0)
        
        if negative_ratio > 0.3:
            recommendations.append({
                'type': 'Quality Improvement',
                'priority': 'High',
                'message': f'High negative sentiment ({negative_ratio:.1%}). Review product quality and customer service.'
            })
        
        if positive_ratio > 0.7:
            recommendations.append({
                'type': 'Marketing Opportunity',
                'priority': 'Medium',
                'message': f'Strong positive sentiment ({positive_ratio:.1%}). Leverage customer testimonials in marketing.'
            })
        
        if insights.get('total_reviews', 0) < 50:
            recommendations.append({
                'type': 'Feedback Collection',
                'priority': 'Medium',
                'message': 'Low review volume. Implement strategies to encourage customer feedback.'
            })
        
        return recommendations
    
    def analyze_support_tickets(self, tickets_data):
        """Analyze customer support ticket sentiment."""
        if tickets_data.empty:
            return {}
        
        results = []
        
        for idx, ticket in tickets_data.iterrows():
            ticket_text = str(ticket.get('description', ''))
            
            if len(ticket_text.strip()) < 10:
                continue
            
            # Analyze sentiment and urgency
            sentiment_result = self.query_huggingface_api(ticket_text, 'sentiment')
            
            if sentiment_result and len(sentiment_result) > 0:
                sentiment_info = sentiment_result[0]
                sentiment_score = sentiment_info.get('score', 0.5)
                
                # Determine urgency based on sentiment and keywords
                urgency_keywords = ['urgent', 'immediate', 'asap', 'emergency', 'critical', 'broken', 'not working']
                has_urgency = any(keyword in ticket_text.lower() for keyword in urgency_keywords)
                
                urgency_level = 'High' if (sentiment_score < 0.3 or has_urgency) else 'Medium' if sentiment_score < 0.6 else 'Low'
            else:
                sentiment_score = 0.5
                urgency_level = 'Medium'
            
            results.append({
                'ticket_id': ticket.get('ticket_id', idx),
                'customer_id': ticket.get('customer_id', f'customer_{idx}'),
                'sentiment_score': sentiment_score,
                'urgency_level': urgency_level,
                'ticket_text': ticket_text,
                'category': ticket.get('category', 'General'),
                'timestamp': ticket.get('timestamp', datetime.now())
            })
        
        support_df = pd.DataFrame(results)
        
        # Generate support insights
        insights = {
            'total_tickets': len(support_df),
            'high_urgency_tickets': len(support_df[support_df['urgency_level'] == 'High']),
            'avg_sentiment_score': support_df['sentiment_score'].mean(),
            'urgency_distribution': support_df['urgency_level'].value_counts().to_dict(),
            'category_sentiment': support_df.groupby('category')['sentiment_score'].mean().to_dict()
        }
        
        return insights
    
    def generate_review_summary(self, sentiment_data):
        """Generate AI-powered review summary."""
        if sentiment_data.empty:
            return "No review data available for analysis."
        
        positive_reviews = sentiment_data[sentiment_data['sentiment'] == 'Positive']
        negative_reviews = sentiment_data[sentiment_data['sentiment'] == 'Negative']
        
        summary = []
        
        # Overall sentiment
        total_reviews = len(sentiment_data)
        positive_pct = len(positive_reviews) / total_reviews * 100
        negative_pct = len(negative_reviews) / total_reviews * 100
        
        summary.append(f"Analysis of {total_reviews} customer reviews:")
        summary.append(f"• {positive_pct:.1f}% positive sentiment")
        summary.append(f"• {negative_pct:.1f}% negative sentiment")
        
        # Key themes from positive reviews
        if not positive_reviews.empty:
            common_positive_words = self._extract_key_themes(positive_reviews['review_text'].tolist())
            summary.append(f"• Positive themes: {', '.join(common_positive_words[:5])}")
        
        # Key themes from negative reviews
        if not negative_reviews.empty:
            common_negative_words = self._extract_key_themes(negative_reviews['review_text'].tolist())
            summary.append(f"• Areas for improvement: {', '.join(common_negative_words[:5])}")
        
        return "\n".join(summary)
    
    def _extract_key_themes(self, reviews):
        """Extract key themes from reviews using simple text analysis."""
        # Combine all reviews
        text = " ".join(reviews).lower()
        
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'a', 'an'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        word_freq = {}
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top words
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)