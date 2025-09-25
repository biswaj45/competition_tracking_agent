"""
Google News data collector for competitor tracking
"""
from typing import List, Dict
import requests
import feedparser
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import quote
from ..base_collector import BaseCollector

class GoogleNewsCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.base_url = "https://news.google.com/rss/search"
        
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles for a specific company from the last N days
        
        Args:
            company: Company name to search for
            days: Number of days to look back
            
        Returns:
            List of news articles with metadata
        """
        date_limit = datetime.now() - timedelta(days=days)
        # Create search query with company name and relevant terms
        query = quote(f'"{company}" (fraud OR "fraud detection" OR "fraud prevention" OR analytics)')
        
        try:
            # Fetch RSS feed from Google News
            feed_url = f"{self.base_url}?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed.entries:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < date_limit:
                        continue
                        
                    article = {
                        "title": entry.title,
                        "link": entry.link,
                        "published_date": pub_date.isoformat(),
                        "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                        "summary": self.clean_text(entry.description),
                        "content": self.clean_text(entry.description)  # Full content would require additional scraping
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article for {company}: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Google News for {company}: {str(e)}")
            return []