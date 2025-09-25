"""
Collect news from tech media sources
"""
from typing import List, Dict
import feedparser
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from ..base_collector import BaseCollector
from ..config import TECH_MEDIA, KEYWORDS

class TechMediaCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.sources = TECH_MEDIA
        self.keywords = KEYWORDS
        
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch relevant articles from tech media sources
        
        Args:
            company: Company name to search for
            days: Number of days to look back
            
        Returns:
            List of articles with title, link, date, and content
        """
        all_articles = []
        
        for source in self.sources:
            try:
                if source["url"].endswith("/feed/") or "rss" in source["url"]:
                    articles = self._fetch_rss(source["url"], company, days)
                else:
                    articles = self._fetch_web(source["url"], company, days)
                    
                all_articles.extend(articles)
                
            except Exception as e:
                self.logger.error(f"Error fetching from {source['name']}: {str(e)}")
                
        return all_articles
    
    def _fetch_rss(self, url: str, company: str, days: int) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        articles = []
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            if not self._is_relevant(entry.title + " " + entry.description, company):
                continue
                
            article = {
                "title": entry.title,
                "link": entry.link,
                "published_date": self._parse_date(entry.published),
                "source": feed.feed.title,
                "content": self.clean_text(entry.description)
            }
            articles.append(article)
            
        return articles
    
    def _fetch_web(self, url: str, company: str, days: int) -> List[Dict]:
        """Fetch articles from web pages"""
        articles = []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This is a basic implementation - would need to be customized per site
        for article in soup.find_all('article'):
            title = article.find('h2')
            if not title or not self._is_relevant(title.text, company):
                continue
                
            link = article.find('a')
            date = article.find('time')
            
            articles.append({
                "title": title.text.strip(),
                "link": link['href'] if link else "",
                "published_date": self._parse_date(date.text) if date else None,
                "source": url,
                "content": self.clean_text(article.text)
            })
            
        return articles
    
    def _is_relevant(self, text: str, company: str) -> bool:
        """Check if the article is relevant based on company name and keywords"""
        text = text.lower()
        if company.lower() in text:
            return any(keyword.lower() in text for keyword in self.keywords)
        return False
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format"""
        try:
            dt = self._format_date(date_str)
            return dt.isoformat() if dt else None
        except Exception:
            return None