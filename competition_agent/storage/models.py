"""
Database models for storing competitor information and news
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json
import pandas as pd

class CompetitorType(Enum):
    ESTABLISHED = "established"
    MID_SIZED = "mid_sized"
    STARTUP = "startup"

class ImpactLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Competitor:
    name: str
    type: CompetitorType
    description: str
    website: str
    
@dataclass
class News:
    competitor_id: int
    title: str
    content: str
    source: str
    published_date: datetime
    url: str
    impact_level: ImpactLevel
    sentiment: Dict[str, float]
    relevance_score: float
    features: List[str]
    
class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create competitors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS competitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL,
                    description TEXT,
                    website TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create news table with analysis fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competitor_id INTEGER,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    source_type TEXT,
                    published_date DATETIME,
                    url TEXT UNIQUE,
                    impact_level TEXT,
                    sentiment_data TEXT,
                    relevance_score FLOAT,
                    features TEXT,
                    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (competitor_id) REFERENCES competitors (id)
                )
            """)
            
            # Create features table for tracking product features
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competitor_id INTEGER,
                    feature_name TEXT NOT NULL,
                    description TEXT,
                    first_seen_date DATETIME,
                    source_url TEXT,
                    status TEXT,
                    FOREIGN KEY (competitor_id) REFERENCES competitors (id),
                    UNIQUE(competitor_id, feature_name)
                )
            """)
            
            conn.commit()
    
    def add_competitor(self, competitor: Competitor) -> int:
        """Add a new competitor to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO competitors (name, type, description, website)
                VALUES (?, ?, ?, ?)
            """, (competitor.name, competitor.type.value, competitor.description, competitor.website))
            return cursor.lastrowid
    
    def add_news(self, news: News) -> int:
        """Add a news item to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO news (
                    competitor_id, title, content, source, published_date,
                    url, impact_level, sentiment_data, relevance_score, features
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                news.competitor_id,
                news.title,
                news.content,
                news.source,
                news.published_date.isoformat(),
                news.url,
                news.impact_level.value,
                json.dumps(news.sentiment),
                news.relevance_score,
                json.dumps(news.features)
            ))
            return cursor.lastrowid
    
    def get_competitor_news(self, competitor_id: int, days: int = 7) -> pd.DataFrame:
        """Get recent news for a competitor"""
        query = """
            SELECT 
                n.*, c.name as competitor_name, c.type as competitor_type
            FROM news n
            JOIN competitors c ON n.competitor_id = c.id
            WHERE n.competitor_id = ?
            AND n.published_date >= datetime('now', ?)
            ORDER BY n.published_date DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=[competitor_id, f'-{days} days'],
                parse_dates=['published_date', 'collected_at']
            )
            
            # Parse JSON columns
            df['sentiment_data'] = df['sentiment_data'].apply(json.loads)
            df['features'] = df['features'].apply(json.loads)
            
            return df
    
    def get_feature_trends(self, days: int = 30) -> pd.DataFrame:
        """Get trending features across competitors"""
        query = """
            SELECT 
                f.feature_name,
                c.name as competitor_name,
                c.type as competitor_type,
                f.first_seen_date,
                COUNT(DISTINCT c.id) as adoption_count
            FROM features f
            JOIN competitors c ON f.competitor_id = c.id
            WHERE f.first_seen_date >= datetime('now', ?)
            GROUP BY f.feature_name
            ORDER BY adoption_count DESC, f.first_seen_date DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                query,
                conn,
                params=[f'-{days} days'],
                parse_dates=['first_seen_date']
            )
    
    def get_competitor_impact_summary(self, days: int = 180, start_date: datetime = None) -> pd.DataFrame:
        """
        Get summary of competitor impact levels
        
        Args:
            days: Number of days to analyze
            start_date: Starting date for the analysis period (default: now)
        """
        if start_date is None:
            start_date = datetime.now()
            
        query = """
            SELECT 
                c.name as competitor_name,
                c.type as competitor_type,
                COUNT(*) as total_mentions,
                SUM(CASE WHEN n.impact_level = 'high' THEN 1 ELSE 0 END) as high_impact,
                SUM(CASE WHEN n.impact_level = 'medium' THEN 1 ELSE 0 END) as medium_impact,
                SUM(CASE WHEN n.impact_level = 'low' THEN 1 ELSE 0 END) as low_impact,
                AVG(n.relevance_score) as avg_relevance,
                AVG(json_extract(n.sentiment_data, '$.polarity')) as avg_sentiment
            FROM news n
            JOIN competitors c ON n.competitor_id = c.id
            WHERE n.published_date >= datetime('now', ?)
            GROUP BY c.id
            ORDER BY total_mentions DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[f'-{days} days'])