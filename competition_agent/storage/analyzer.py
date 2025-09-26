"""
Data analysis and insights generation
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from .models import Database, CompetitorType, ImpactLevel

class CompetitorAnalyzer:
    def __init__(self, db: Database):
        self.db = db
    
    def generate_weekly_insights(self, days: int = 180) -> Dict[str, Any]:
        """Generate comprehensive insights from collected data"""
        activity = self._analyze_competitor_activity(days)
        features = self._analyze_features(days)
        sentiment = self._analyze_sentiment_trends(days)
        developments = self._identify_key_developments(days)
        comparison = self._compare_competitors(days)
        
        return {
            "competitor_activity": activity,
            "feature_analysis": features,
            "sentiment_trends": sentiment,
            "key_developments": developments,
            "competitor_comparison": comparison
        }
    
    def _analyze_competitor_activity(self, days: int) -> Dict[str, Any]:
        """Analyze competitor activity levels and trends"""
        try:
            impact_summary = self.db.get_competitor_impact_summary(days=days)
            
            if not isinstance(impact_summary, pd.DataFrame) or impact_summary.empty:
                return self._empty_activity_data()
            
            total_activities = impact_summary['total_mentions'].sum()
            monthly_average = total_activities / (days / 30)
            high_impact_total = impact_summary['high_impact'].sum()
            high_impact_monthly = high_impact_total / (days / 30)
            
            segment_activity = impact_summary.groupby('competitor_type')['total_mentions'].sum()
            most_active = segment_activity.idxmax() if not segment_activity.empty else "No data"
            
            return {
                "total_activities": total_activities,
                "monthly_average": monthly_average,
                "high_impact_total": high_impact_total,
                "high_impact_monthly": high_impact_monthly,
                "most_active_segment": most_active,
                "impact_summary": impact_summary
            }
        except Exception as e:
            print(f"Error analyzing competitor activity: {str(e)}")
            return self._empty_activity_data()
    
    def _analyze_features(self, days: int) -> Dict[str, Any]:
        """Analyze product features and capabilities"""
        try:
            features_df = self.db.get_feature_trends(days)
            
            if not isinstance(features_df, pd.DataFrame) or features_df.empty:
                return self._empty_feature_data()
            
            total_features = len(features_df['feature_name'].unique())
            new_features = len(features_df[
                features_df['first_seen_date'] >= datetime.now() - timedelta(days=30)
            ])
            
            return {
                "num_features": total_features,
                "new_features": new_features,
                "feature_trends": features_df
            }
        except Exception as e:
            print(f"Error analyzing features: {str(e)}")
            return self._empty_feature_data()
    
    def _analyze_sentiment_trends(self, days: int) -> Dict[str, Any]:
        """Analyze sentiment trends across competitors"""
        try:
            impact_summary = self.db.get_competitor_impact_summary(days=days)
            
            if not isinstance(impact_summary, pd.DataFrame) or impact_summary.empty:
                return self._empty_sentiment_data()
            
            avg_sentiment = impact_summary['avg_sentiment'].mean()
            sentiment_by_type = impact_summary.groupby('competitor_type')['avg_sentiment'].mean()
            
            return {
                "overall_sentiment": avg_sentiment,
                "sentiment_by_type": sentiment_by_type
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return self._empty_sentiment_data()
    
    def _identify_key_developments(self, days: int) -> List[Dict]:
        """Identify key competitive developments"""
        try:
            impact_summary = self.db.get_competitor_impact_summary(days=days)
            
            if not isinstance(impact_summary, pd.DataFrame) or impact_summary.empty:
                return []
            
            high_impact = impact_summary[impact_summary['high_impact'] > 0]
            developments = []
            
            for _, row in high_impact.iterrows():
                developments.append({
                    "competitor": row['competitor_name'],
                    "type": row['competitor_type'],
                    "impact": "High",
                    "mentions": row['high_impact']
                })
            
            return developments
        except Exception as e:
            print(f"Error identifying key developments: {str(e)}")
            return []
    
    def _compare_competitors(self, days: int) -> Dict[str, Any]:
        """Compare competitors within and across segments"""
        try:
            impact_summary = self.db.get_competitor_impact_summary(days=days)
            
            if not isinstance(impact_summary, pd.DataFrame) or impact_summary.empty:
                return self._empty_comparison_data()
            
            by_type = impact_summary.groupby('competitor_type').agg({
                'total_mentions': 'sum',
                'high_impact': 'sum',
                'avg_sentiment': 'mean'
            }).to_dict('index')
            
            return {
                "segment_comparison": by_type,
                "raw_data": impact_summary
            }
        except Exception as e:
            print(f"Error comparing competitors: {str(e)}")
            return self._empty_comparison_data()
    
    def _empty_activity_data(self) -> Dict[str, Any]:
        return {
            "total_activities": 0,
            "monthly_average": 0,
            "high_impact_total": 0,
            "high_impact_monthly": 0,
            "most_active_segment": "No data",
            "impact_summary": pd.DataFrame()
        }
    
    def _empty_feature_data(self) -> Dict[str, Any]:
        return {
            "num_features": 0,
            "new_features": 0,
            "feature_trends": pd.DataFrame()
        }
    
    def _empty_sentiment_data(self) -> Dict[str, Any]:
        return {
            "overall_sentiment": 0,
            "sentiment_by_type": pd.Series()
        }
    
    def _empty_comparison_data(self) -> Dict[str, Any]:
        return {
            "segment_comparison": {},
            "raw_data": pd.DataFrame()
        }
        
    def generate_visualizations(self) -> Dict[str, Any]:
        """Generate visualizations for the report"""
        figures = {}
        
        # Activity by competitor type
        try:
            impact_summary = self.db.get_competitor_impact_summary(days=180)
            if isinstance(impact_summary, pd.DataFrame) and not impact_summary.empty:
                plt.figure(figsize=(8, 4))
                by_type = impact_summary.groupby('competitor_type')['total_mentions'].sum()
                by_type.plot(kind='bar')
                plt.title('Activity by Competitor Type')
                plt.tight_layout()
                figures['activity_by_type'] = plt.gcf()
                plt.close()
        except Exception as e:
            print(f"Error generating activity visualization: {str(e)}")
        
        # Sentiment trends
        try:
            if isinstance(impact_summary, pd.DataFrame) and not impact_summary.empty:
                plt.figure(figsize=(8, 4))
                sns.boxplot(data=impact_summary, x='competitor_type', y='avg_sentiment')
                plt.title('Sentiment Distribution by Competitor Type')
                plt.tight_layout()
                figures['sentiment_trends'] = plt.gcf()
                plt.close()
        except Exception as e:
            print(f"Error generating sentiment visualization: {str(e)}")
        
        return figures