"""News collection using GNews package."""
from datetime import datetime
from typing import List, Dict, Any
import asyncio
from gnews import GNews

class GNewsCollector:
    def __init__(self):
        self.gnews = GNews(language='en', country='US', max_results=10)
    
    async def collect_news(self, query: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Collect news articles for a given query and date range."""
        self.gnews.start_date = start_date
        self.gnews.end_date = end_date
        
        # Get news
        articles = self.gnews.get_news(query)
        
        # Clean and standardize the output
        cleaned = []
        for article in articles:
            cleaned.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "published_date": article.get("published date", ""),
                "url": article.get("url", ""),
                "publisher": article.get("publisher", {}).get("title", ""),
                "company": query  # add the company we searched for
            })
        
        return cleaned