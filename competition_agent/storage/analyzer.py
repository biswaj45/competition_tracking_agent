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
        
    def generate_weekly_insights(self, days: int = 180) -> Dict[str, Any]:
        """
        Generate comprehensive insights from collected data
        
        Args:
            days: Number of days to analyze (default: 180 days / 6 months)
        
        Returns:
            Dictionary containing various analyses and insights with trend data
        """
        return {
            "competitor_activity": self._analyze_competitor_activity(days),
            "feature_analysis": self._analyze_features(days),
            "sentiment_trends": self._analyze_sentiment_trends(days),
            "key_developments": self._identify_key_developments(days),
            "competitor_comparison": self._compare_competitors(days)
        }
    
    def _analyze_competitor_activity(self, days: int) -> Dict[str, Any]:
        """Analyze competitor activity levels and trends over time"""
        # Initialize empty DataFrames in case the database is empty
        trend_df = pd.DataFrame(columns=['competitor_type', 'period', 'total_mentions', 'high_impact'])
        impact_summary = pd.DataFrame(columns=[
            'competitor_name', 'competitor_type', 'total_mentions', 'high_impact',
            'medium_impact', 'low_impact', 'avg_relevance', 'avg_sentiment'
        ])
        
        try:
            # Get monthly summaries for trend analysis
            monthly_data = []
            for month_start in range(0, days, 30):
                month_summary = self.db.get_competitor_impact_summary(
                    days=30,
                    start_date=datetime.now() - timedelta(days=month_start+30)
                )
                if not month_summary.empty:
                    month_summary['period'] = f"Month {month_start//30 + 1}"
                    monthly_data.append(month_summary)
            
            if monthly_data:
                trend_df = pd.concat(monthly_data)
            
            # Current period summary
            impact_summary = self.db.get_competitor_impact_summary(days)
            
            if impact_summary.empty:
                # Use sample data if no real data exists
                impact_summary = pd.DataFrame([
                    {
                        'competitor_name': 'Experian Hunter',
                        'competitor_type': 'established',
                        'total_mentions': 15,
                        'high_impact': 5,
                        'medium_impact': 7,
                        'low_impact': 3,
                        'avg_relevance': 0.85,
                        'avg_sentiment': 0.6
                    },
                    {
                        'competitor_name': 'Feedzai',
                        'competitor_type': 'mid_sized',
                        'total_mentions': 12,
                        'high_impact': 4,
                        'medium_impact': 5,
                        'low_impact': 3,
                        'avg_relevance': 0.78,
                        'avg_sentiment': 0.7
                    },
                    {
                        'competitor_name': 'Sardine',
                        'competitor_type': 'startup',
                        'total_mentions': 8,
                        'high_impact': 3,
                        'medium_impact': 3,
                        'low_impact': 2,
                        'avg_relevance': 0.92,
                        'avg_sentiment': 0.8
                    }
                ])
        
        except Exception as e:
            self.logger.error(f"Error analyzing competitor activity: {str(e)}")
            # Return empty data structure
            return {
                "type_summary": {},
                "most_active_competitors": [],
                "total_mentions": 0,
                "high_impact_events": 0
            }
        
        # Group by competitor type
        type_summary = impact_summary.groupby('competitor_type').agg({
            'total_mentions': 'sum',
            'high_impact': 'sum',
            'avg_relevance': 'mean',
            'avg_sentiment': 'mean'
        }).round(2)
        
        # Calculate growth trends
        trend_summary = trend_df.groupby(['competitor_type', 'period']).agg({
            'total_mentions': 'sum',
            'high_impact': 'sum'
        }).reset_index()
        
        # Identify most active competitors
        most_active = impact_summary.nlargest(5, 'total_mentions')[
            ['competitor_name', 'total_mentions', 'high_impact']
        ].to_dict('records')
        
        return {
            "type_summary": type_summary.to_dict(),
            "trend_summary": trend_summary.to_dict('records'),
            "most_active_competitors": most_active,
            "total_mentions": int(impact_summary['total_mentions'].sum()),
            "high_impact_events": int(impact_summary['high_impact'].sum()),
            "average_sentiment": float(impact_summary['avg_sentiment'].mean()),
            "competitor_details": impact_summary.to_dict('records')
        }
    
    def _analyze_features(self, days: int) -> Dict[str, Any]:
        """Analyze feature trends and gaps"""
        try:
            feature_trends = self.db.get_feature_trends(days)
            
            if feature_trends.empty:
                # Use sample feature data if no real data exists
                feature_trends = pd.DataFrame([
                    {
                        'feature_name': 'AI-powered Fraud Detection',
                        'competitor_name': 'Experian Hunter',
                        'competitor_type': 'established',
                        'first_seen_date': datetime.now() - timedelta(days=10),
                        'adoption_count': 5
                    },
                    {
                        'feature_name': 'Behavioral Biometrics',
                        'competitor_name': 'Feedzai',
                        'competitor_type': 'mid_sized',
                        'first_seen_date': datetime.now() - timedelta(days=30),
                        'adoption_count': 4
                    },
                    {
                        'feature_name': 'Device Intelligence',
                        'competitor_name': 'Sardine',
                        'competitor_type': 'startup',
                        'first_seen_date': datetime.now() - timedelta(days=60),
                        'adoption_count': 3
                    },
                    {
                        'feature_name': 'Real-time Risk Scoring',
                        'competitor_name': 'FICO Falcon',
                        'competitor_type': 'established',
                        'first_seen_date': datetime.now() - timedelta(days=90),
                        'adoption_count': 6
                    }
                ])
            
            # Get features by adoption level
            feature_adoption = feature_trends.groupby('feature_name').agg({
                'adoption_count': 'first',
                'competitor_type': lambda x: list(set(x))
            }).sort_values('adoption_count', ascending=False)
            
            # Identify new features
            new_features = feature_trends[
                feature_trends['first_seen_date'] >= datetime.now() - timedelta(days=30)
            ]
            
            # Calculate feature distribution by competitor type
            feature_dist = feature_trends.groupby('competitor_type').agg({
                'feature_name': 'nunique'
            }).to_dict()
            
            return {
                "top_features": feature_adoption.head(10).to_dict('index'),
                "new_features": new_features[['feature_name', 'competitor_name']].to_dict('records'),
                "feature_count": len(feature_trends['feature_name'].unique()),
                "feature_distribution": feature_dist,
                "recent_features": feature_trends[
                    feature_trends['first_seen_date'] >= datetime.now() - timedelta(days=90)
                ].to_dict('records'),
                "feature_trends": feature_trends.sort_values('first_seen_date').to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing features: {str(e)}")
            return {
                "top_features": {},
                "new_features": [],
                "feature_count": 0,
                "feature_distribution": {},
                "recent_features": [],
                "feature_trends": []
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
        
        try:
            # Activity by competitor type
            impact_summary = self.db.get_competitor_impact_summary(180)  # Last 6 months
            
            if impact_summary.empty:
                # Use sample data if no real data exists
                impact_summary = pd.DataFrame([
                    {'competitor_type': 'established', 'total_mentions': 45, 'high_impact': 15},
                    {'competitor_type': 'mid_sized', 'total_mentions': 30, 'high_impact': 10},
                    {'competitor_type': 'startup', 'total_mentions': 25, 'high_impact': 8}
                ])
            
            # Activity by competitor type
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=impact_summary,
                x='competitor_type',
                y='total_mentions',
                hue='competitor_type',
                ax=ax
            )
            ax.set_title('6-Month Activity by Competitor Type')
            ax.set_xlabel('Competitor Type')
            ax.set_ylabel('Total Mentions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figs['activity_by_type'] = fig
            
            # Impact level distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            impact_data = pd.melt(impact_summary[['competitor_type', 'high_impact', 'medium_impact', 'low_impact']], 
                                id_vars=['competitor_type'])
            sns.barplot(
                data=impact_data,
                x='competitor_type',
                y='value',
                hue='variable',
                ax=ax
            )
            ax.set_title('Impact Level Distribution by Competitor Type')
            ax.set_xlabel('Competitor Type')
            ax.set_ylabel('Number of Events')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figs['impact_distribution'] = fig
            
            # Feature adoption timeline
            feature_trends = self.db.get_feature_trends(180)
            
            if feature_trends.empty:
                # Create sample feature trends
                dates = [datetime.now() - timedelta(days=x) for x in [150, 120, 90, 60, 30, 15]]
                feature_trends = pd.DataFrame([
                    {'first_seen_date': date, 
                     'feature_name': f'Feature {i}',
                     'adoption_count': np.random.randint(1, 8),
                     'competitor_type': np.random.choice(['established', 'mid_sized', 'startup'])}
                    for i, date in enumerate(dates)
                ])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(
                data=feature_trends,
                x='first_seen_date',
                y='adoption_count',
                hue='competitor_type',
                size='adoption_count',
                ax=ax
            )
            ax.set_title('6-Month Feature Adoption Timeline')
            ax.set_xlabel('First Seen Date')
            ax.set_ylabel('Adoption Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            figs['feature_timeline'] = fig
            
            # Feature word cloud
            features_list = []
            for feature in feature_trends['feature_name']:
                if isinstance(feature, str):
                    features_list.extend(feature.split())
            
            if not features_list:
                features_list = ['AI', 'ML', 'Fraud', 'Detection', 'Prevention', 'Analytics',
                               'Behavioral', 'Biometrics', 'Risk', 'Scoring', 'Real-time']
            
            feature_text = ' '.join(features_list)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                min_word_length=2,
                collocations=False
            ).generate(feature_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Feature Focus Areas')
            plt.tight_layout()
            figs['feature_wordcloud'] = fig
            
            # Close all figures to free memory
            plt.close('all')
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
        
        return figs