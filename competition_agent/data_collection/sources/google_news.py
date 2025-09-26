"""
Google News data collector for competitor tracking
"""
from typing import List, Dict, Optional
import requests
import feedparser
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
import logging
from .site_config import PAYWALL_DOMAINS, CUSTOM_HANDLERS
from ..base_collector import BaseCollector

class GoogleNewsCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.base_url = "https://news.google.com/rss/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        
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

                    # Get full article content if possible
                    full_content = self._extract_article_content(entry.link)
                    
                    article = {
                        "title": entry.title,
                        "link": entry.link,
                        "published_date": pub_date.isoformat(),
                        "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                        "summary": self.clean_text(entry.description),
                        "content": full_content if full_content else self.clean_text(entry.description),
                        "is_paywall": self._is_paywall_site(entry.link),
                        "full_content_available": bool(full_content)
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article for {company}: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Google News for {company}: {str(e)}")
            return []

    def _is_paywall_site(self, url: str) -> bool:
        """Check if the URL belongs to a known paywall/subscription site"""
        domain = urlparse(url).netloc.lower()
        return any(paywall_domain in domain for paywall_domain in PAYWALL_DOMAINS)

    def _extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract the full article content from the URL
        Returns None if extraction fails or site is paywalled
        """
        try:
            if self._is_paywall_site(url):
                return None

            domain = urlparse(url).netloc.lower()
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for elem in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                elem.decompose()

            # Try custom handler first
            for site_domain, selector in CUSTOM_HANDLERS.items():
                if site_domain in domain:
                    content_elem = soup.find(class_=selector)
                    if content_elem:
                        return self.clean_text(content_elem.get_text())

            # Try common article content patterns
            content = None
            
            # Try article tag first
            article = soup.find('article')
            if article:
                content = article.get_text()
            
            # Try common content class names
            if not content:
                for class_name in ['post-content', 'article-content', 'entry-content', 'content-body']:
                    elem = soup.find(class_=class_name)
                    if elem:
                        content = elem.get_text()
                        break

            # Try main tag
            if not content:
                main = soup.find('main')
                if main:
                    content = main.get_text()

            # Final fallback: look for longest text block
            if not content:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    content = ' '.join(p.get_text() for p in paragraphs)

            return self.clean_text(content) if content else None

        except Exception as e:
            self.logger.warning(f"Error extracting content from {url}: {str(e)}")
            return None