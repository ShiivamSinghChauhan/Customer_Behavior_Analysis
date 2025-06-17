import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from datetime import datetime, timedelta
import json
import uuid

class ABTestingFramework:
    """
    A/B testing framework for marketing campaign optimization and customer behavior experiments.
    """
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        
    def create_ab_test(self, test_name, test_type, config):
        """Create a new A/B test experiment."""
        test_id = str(uuid.uuid4())
        
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'test_type': test_type,  # 'marketing_campaign', 'pricing', 'ui_element', 'product_recommendation'
            'start_date': datetime.now(),
            'end_date': config.get('end_date'),
            'target_metric': config.get('target_metric', 'conversion_rate'),
            'significance_level': config.get('significance_level', 0.05),
            'minimum_sample_size': config.get('minimum_sample_size', 100),
            'variants': config.get('variants', ['A', 'B']),
            'traffic_allocation': config.get('traffic_allocation', [0.5, 0.5]),
            'status': 'active',
            'description': config.get('description', ''),
            'hypothesis': config.get('hypothesis', ''),
            'success_criteria': config.get('success_criteria', '')
        }
        
        self.active_tests[test_id] = test_config
        return test_id, test_config
    
    def assign_user_to_variant(self, test_id, user_id):
        """Assign a user to a test variant based on traffic allocation."""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        variants = test['variants']
        allocation = test['traffic_allocation']
        
        # Use hash-based assignment for consistent user experience
        hash_value = hash(f"{test_id}_{user_id}") % 100
        cumulative_allocation = 0
        
        for i, variant in enumerate(variants):
            cumulative_allocation += allocation[i] * 100
            if hash_value < cumulative_allocation:
                return variant
        
        return variants[-1]  # Fallback to last variant
    
    def record_test_event(self, test_id, user_id, variant, event_type, event_value=1):
        """Record an event for A/B test analysis."""
        if test_id not in self.active_tests:
            return False
        
        if test_id not in self.test_results:
            self.test_results[test_id] = []
        
        event_record = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'variant': variant,
            'event_type': event_type,  # 'impression', 'click', 'conversion', 'purchase'
            'event_value': event_value
        }
        
        self.test_results[test_id].append(event_record)
        return True
    
    def analyze_test_results(self, test_id):
        """Analyze A/B test results and determine statistical significance."""
        if test_id not in self.active_tests or test_id not in self.test_results:
            return None
        
        test_config = self.active_tests[test_id]
        test_data = pd.DataFrame(self.test_results[test_id])
        
        if test_data.empty:
            return {
                'test_id': test_id,
                'status': 'insufficient_data',
                'message': 'No data collected yet'
            }
        
        # Calculate metrics by variant
        variant_metrics = {}
        
        for variant in test_config['variants']:
            variant_data = test_data[test_data['variant'] == variant]
            
            # Calculate key metrics
            total_users = variant_data['user_id'].nunique()
            total_impressions = len(variant_data[variant_data['event_type'] == 'impression'])
            total_clicks = len(variant_data[variant_data['event_type'] == 'click'])
            total_conversions = len(variant_data[variant_data['event_type'] == 'conversion'])
            total_revenue = variant_data[variant_data['event_type'] == 'purchase']['event_value'].sum()
            
            click_rate = total_clicks / total_impressions if total_impressions > 0 else 0
            conversion_rate = total_conversions / total_users if total_users > 0 else 0
            revenue_per_user = total_revenue / total_users if total_users > 0 else 0
            
            variant_metrics[variant] = {
                'total_users': total_users,
                'impressions': total_impressions,
                'clicks': total_clicks,
                'conversions': total_conversions,
                'revenue': total_revenue,
                'click_rate': click_rate,
                'conversion_rate': conversion_rate,
                'revenue_per_user': revenue_per_user
            }
        
        # Statistical significance testing
        significance_results = self._calculate_statistical_significance(variant_metrics, test_config)
        
        # Generate recommendations
        recommendations = self._generate_test_recommendations(variant_metrics, significance_results, test_config)
        
        analysis_result = {
            'test_id': test_id,
            'test_name': test_config['test_name'],
            'start_date': test_config['start_date'],
            'analysis_date': datetime.now(),
            'variant_metrics': variant_metrics,
            'significance_results': significance_results,
            'recommendations': recommendations,
            'status': 'analyzed'
        }
        
        return analysis_result
    
    def _calculate_statistical_significance(self, variant_metrics, test_config):
        """Calculate statistical significance between variants."""
        variants = list(variant_metrics.keys())
        target_metric = test_config['target_metric']
        significance_level = test_config['significance_level']
        
        if len(variants) < 2:
            return {'significant': False, 'p_value': None, 'confidence_level': None}
        
        # Compare first two variants (A vs B)
        variant_a = variants[0]
        variant_b = variants[1]
        
        metrics_a = variant_metrics[variant_a]
        metrics_b = variant_metrics[variant_b]
        
        # Choose appropriate test based on target metric
        if target_metric == 'conversion_rate':
            # Proportion test for conversion rates
            successes_a = metrics_a['conversions']
            trials_a = metrics_a['total_users']
            successes_b = metrics_b['conversions']
            trials_b = metrics_b['total_users']
            
            if trials_a < 30 or trials_b < 30:
                return {
                    'significant': False,
                    'p_value': None,
                    'confidence_level': None,
                    'message': 'Insufficient sample size for statistical testing'
                }
            
            # Two-proportion z-test
            p1 = successes_a / trials_a if trials_a > 0 else 0
            p2 = successes_b / trials_b if trials_b > 0 else 0
            
            pooled_p = (successes_a + successes_b) / (trials_a + trials_b)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/trials_a + 1/trials_b))
            
            if se == 0:
                z_score = 0
            else:
                z_score = (p1 - p2) / se
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
        elif target_metric == 'revenue_per_user':
            # T-test for continuous metrics
            revenue_a = [metrics_a['revenue_per_user']] * metrics_a['total_users']
            revenue_b = [metrics_b['revenue_per_user']] * metrics_b['total_users']
            
            if len(revenue_a) < 30 or len(revenue_b) < 30:
                return {
                    'significant': False,
                    'p_value': None,
                    'confidence_level': None,
                    'message': 'Insufficient sample size for statistical testing'
                }
            
            t_stat, p_value = stats.ttest_ind(revenue_a, revenue_b)
        
        else:
            return {
                'significant': False,
                'p_value': None,
                'confidence_level': None,
                'message': 'Unsupported target metric for statistical testing'
            }
        
        is_significant = p_value < significance_level
        confidence_level = (1 - significance_level) * 100
        
        return {
            'significant': is_significant,
            'p_value': p_value,
            'confidence_level': confidence_level,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'effect_size': self._calculate_effect_size(metrics_a, metrics_b, target_metric)
        }
    
    def _calculate_effect_size(self, metrics_a, metrics_b, target_metric):
        """Calculate effect size between variants."""
        if target_metric == 'conversion_rate':
            rate_a = metrics_a['conversion_rate']
            rate_b = metrics_b['conversion_rate']
            if rate_a == 0:
                return float('inf') if rate_b > 0 else 0
            return (rate_b - rate_a) / rate_a
        
        elif target_metric == 'revenue_per_user':
            revenue_a = metrics_a['revenue_per_user']
            revenue_b = metrics_b['revenue_per_user']
            if revenue_a == 0:
                return float('inf') if revenue_b > 0 else 0
            return (revenue_b - revenue_a) / revenue_a
        
        return 0
    
    def _generate_test_recommendations(self, variant_metrics, significance_results, test_config):
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not significance_results.get('significant', False):
            recommendations.append({
                'type': 'Continue Testing',
                'priority': 'Medium',
                'message': 'Results are not statistically significant yet. Continue collecting data or increase sample size.'
            })
        else:
            # Determine winning variant
            target_metric = test_config['target_metric']
            variants = list(variant_metrics.keys())
            
            best_variant = max(variants, key=lambda v: variant_metrics[v].get(target_metric, 0))
            effect_size = significance_results.get('effect_size', 0)
            
            if effect_size > 0.1:  # 10% improvement
                recommendations.append({
                    'type': 'Implement Winner',
                    'priority': 'High',
                    'message': f'Variant {best_variant} shows significant improvement ({effect_size:.1%}). Recommend full rollout.'
                })
            else:
                recommendations.append({
                    'type': 'Marginal Improvement',
                    'priority': 'Low',
                    'message': f'Variant {best_variant} shows statistical significance but small effect size ({effect_size:.1%}).'
                })
        
        # Sample size recommendations
        min_sample = test_config['minimum_sample_size']
        actual_samples = [variant_metrics[v]['total_users'] for v in variant_metrics]
        
        if any(sample < min_sample for sample in actual_samples):
            recommendations.append({
                'type': 'Sample Size',
                'priority': 'Medium',
                'message': f'Some variants have fewer than {min_sample} users. Consider extending test duration.'
            })
        
        return recommendations
    
    def get_test_performance_summary(self):
        """Get summary of all test performance."""
        summary = {
            'total_active_tests': len(self.active_tests),
            'total_completed_tests': 0,
            'significant_results': 0,
            'average_effect_size': 0
        }
        
        effect_sizes = []
        
        for test_id in self.active_tests:
            if test_id in self.test_results:
                analysis = self.analyze_test_results(test_id)
                if analysis and analysis.get('significance_results'):
                    sig_results = analysis['significance_results']
                    if sig_results.get('significant'):
                        summary['significant_results'] += 1
                        effect_size = sig_results.get('effect_size', 0)
                        if effect_size != float('inf'):
                            effect_sizes.append(effect_size)
        
        if effect_sizes:
            summary['average_effect_size'] = np.mean(effect_sizes)
        
        return summary
    
    def stop_test(self, test_id, reason="Manual stop"):
        """Stop an active A/B test."""
        if test_id in self.active_tests:
            self.active_tests[test_id]['status'] = 'stopped'
            self.active_tests[test_id]['end_date'] = datetime.now()
            self.active_tests[test_id]['stop_reason'] = reason
            return True
        return False
    
    def export_test_results(self, test_id):
        """Export test results for external analysis."""
        if test_id not in self.test_results:
            return None
        
        test_data = pd.DataFrame(self.test_results[test_id])
        test_config = self.active_tests.get(test_id, {})
        
        export_data = {
            'test_config': test_config,
            'raw_data': test_data.to_dict('records'),
            'analysis': self.analyze_test_results(test_id)
        }
        
        return export_data