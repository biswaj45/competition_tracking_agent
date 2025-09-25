"""
Data analysis and insights generation
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from .models import Database, CompetitorType, ImpactLevel

class CompetitorAnalyzer:
    def __init__(self, db: Database):
        self.db = db
        
    def generate_weekly_insights(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive insights from collected data
        
        Returns:
            Dictionary containing various analyses and insights
        """
        return {
            "competitor_activity": self._analyze_competitor_activity(days),
            "feature_analysis": self._analyze_features(days),
            "sentiment_trends": self._analyze_sentiment_trends(days),
            "key_developments": self._identify_key_developments(days),
            "competitor_comparison": self._compare_competitors(days)
        }
    
    def _analyze_competitor_activity(self, days: int) -> Dict[str, Any]:
        """Analyze competitor activity levels and trends"""
        impact_summary = self.db.get_competitor_impact_summary(days)
        
        # Group by competitor type
        type_summary = impact_summary.groupby('competitor_type').agg({
            'total_mentions': 'sum',
            'high_impact': 'sum',
            'avg_relevance': 'mean',
            'avg_sentiment': 'mean'
        }).round(2)
        
        # Identify most active competitors
        most_active = impact_summary.nlargest(5, 'total_mentions')[
            ['competitor_name', 'total_mentions', 'high_impact']
        ].to_dict('records')
        
        return {
            "type_summary": type_summary.to_dict(),
            "most_active_competitors": most_active,
            "total_mentions": impact_summary['total_mentions'].sum(),
            "high_impact_events": impact_summary['high_impact'].sum()
        }
    
    def _analyze_features(self, days: int) -> Dict[str, Any]:
        """Analyze feature trends and gaps"""
        feature_trends = self.db.get_feature_trends(days)
        
        # Get features by adoption level
        feature_adoption = feature_trends.groupby('feature_name').agg({
            'adoption_count': 'first',
            'competitor_type': lambda x: list(set(x))
        }).sort_values('adoption_count', ascending=False)
        
        # Identify new features
        new_features = feature_trends[
            feature_trends['first_seen_date'] >= datetime.now() - timedelta(days=days)
        ]
        
        return {
            "top_features": feature_adoption.head(10).to_dict('index'),
            "new_features": new_features[['feature_name', 'competitor_name']].to_dict('records'),
            "feature_count": len(feature_trends['feature_name'].unique())
        }
    
    def _analyze_sentiment_trends(self, days: int) -> Dict[str, Any]:
        """Analyze sentiment trends across competitors"""
        impact_summary = self.db.get_competitor_impact_summary(days)
        
        sentiment_by_type = impact_summary.groupby('competitor_type').agg({
            'avg_sentiment': ['mean', 'std']
        }).round(3)
        
        return {
            "sentiment_by_type": sentiment_by_type.to_dict(),
            "overall_sentiment": impact_summary['avg_sentiment'].mean(),
            "sentiment_volatility": impact_summary['avg_sentiment'].std()
        }
    
    def _identify_key_developments(self, days: int) -> List[Dict]:
        """Identify key competitive developments"""
        high_impact_news = []
        
        for competitor_type in CompetitorType:
            query = """
                SELECT 
                    n.title,
                    n.published_date,
                    n.impact_level,
                    n.relevance_score,
                    c.name as competitor_name,
                    c.type as competitor_type
                FROM news n
                JOIN competitors c ON n.competitor_id = c.id
                WHERE n.impact_level = 'high'
                AND n.published_date >= datetime('now', ?)
                AND c.type = ?
                ORDER BY n.published_date DESC
            """
            
            # Query news from last X days
        with self.db.get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[f'-{days} days', competitor_type.value],
                    parse_dates=['published_date']
                )
                
                high_impact_news.extend(df.to_dict('records'))
        
        return high_impact_news
    
    def _compare_competitors(self, days: int) -> Dict[str, Any]:
        """Generate competitor comparison metrics"""
        impact_summary = self.db.get_competitor_impact_summary(days)
        
        # Calculate engagement scores
        impact_summary['engagement_score'] = (
            impact_summary['high_impact'] * 3 +
            impact_summary['medium_impact'] * 2 +
            impact_summary['low_impact']
        ) / impact_summary['total_mentions']
        
        # Group by competitor type
        type_comparison = impact_summary.groupby('competitor_type').agg({
            'total_mentions': ['mean', 'std'],
            'engagement_score': ['mean', 'std'],
            'avg_sentiment': ['mean', 'std']
        }).round(3)
        
        # Get top competitors by engagement
        top_engagement = impact_summary.nlargest(5, 'engagement_score')[
            ['competitor_name', 'engagement_score', 'avg_sentiment']
        ].to_dict('records')
        
        return {
            "type_comparison": type_comparison.to_dict(),
            "top_engagement": top_engagement
        }
    
    def generate_visualizations(self) -> Dict[str, plt.Figure]:
        """Generate visualization figures for the report"""
        figs = {}
        
        # Activity by competitor type
        impact_summary = self.db.get_competitor_impact_summary(30)  # Last 30 days
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=impact_summary,
            x='competitor_type',
            y='total_mentions',
            hue='competitor_type',
            ax=ax
        )
        ax.set_title('Activity by Competitor Type')
        ax.set_xlabel('Competitor Type')
        ax.set_ylabel('Total Mentions')
        figs['activity_by_type'] = fig
        
        # Feature adoption timeline
        feature_trends = self.db.get_feature_trends(30)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            data=feature_trends,
            x='first_seen_date',
            y='adoption_count',
            hue='competitor_type',
            size='adoption_count',
            ax=ax
        )
        ax.set_title('Feature Adoption Timeline')
        ax.set_xlabel('First Seen Date')
        ax.set_ylabel('Adoption Count')
        figs['feature_timeline'] = fig
        
        # Word cloud of features
        feature_text = ' '.join(feature_trends['feature_name'])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(feature_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Feature Word Cloud')
        figs['feature_wordcloud'] = fig
        
        return figs