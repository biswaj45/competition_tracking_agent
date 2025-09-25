"""
Collect news from tech media sources
"""
from typing import List, Dict, Optional
import feedparser
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
from ..base_collector import BaseCollector
from ..config import TECH_MEDIA, KEYWORDS
from ..content_analyzer import ContentAnalyzer

class TechMediaCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.sources = TECH_MEDIA
        self.keywords = KEYWORDS
        self.content_analyzer = ContentAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch relevant articles from tech media sources
        
        Args:
            company: Company name to search for
            days: Number of days to look back
            
        Returns:
            List of articles with title, link, date, and content
        """
        date_limit = datetime.now() - timedelta(days=days)
        all_articles = []
        
        for source in self.sources:
            try:
                # Add delay between requests to be polite
                time.sleep(2)
                
                source_articles = []
                if source["url"].endswith("/feed/") or "rss" in source["url"]:
                    source_articles = self._fetch_rss(source["url"], company, date_limit)
                else:
                    source_articles = self._fetch_web(source["url"], company, date_limit)
                
                # Add source information
                for article in source_articles:
                    article["source"] = source["name"]
                    article["source_url"] = source["url"]
                
                all_articles.extend(source_articles)
                
            except Exception as e:
                self.logger.error(f"Error fetching from {source['name']}: {str(e)}")
                continue
                
        return all_articles
    
    def _fetch_rss(self, url: str, company: str, date_limit: datetime) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        articles = []
        
        try:
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                try:
                    # Get publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed'):
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    if not pub_date or pub_date < date_limit:
                        continue
                    
                    # Get content
                    content = ""
                    if hasattr(entry, 'content'):
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    
                    # Clean content
                    if content:
                        soup = BeautifulSoup(content, 'html.parser')
                        content = self.clean_text(soup.get_text())
                    
                    # Analyze relevance
                    analysis = self.content_analyzer.analyze_content(
                        entry.title + " " + content,
                        company,
                        self.keywords
                    )
                    
                    if analysis["relevance_score"] < 0.3:
                        continue
                    
                    articles.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published_date": pub_date.isoformat(),
                        "content": content,
                        "type": "tech_media",
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing feed entry: {str(e)}")
                    continue
                    
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching feed from {url}: {str(e)}")
            return []
    
    def _fetch_web(self, url: str, company: str, date_limit: datetime) -> List[Dict]:
        """Fetch articles from web pages"""
        articles = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find RSS feed link if available
            rss_link = self._find_rss_link(soup, url)
            if rss_link:
                return self._fetch_rss(rss_link, company, date_limit)
            
            # Find articles in the page
            for article in self._find_articles(soup):
                try:
                    content = self._extract_content(article, url)
                    if not content:
                        continue
                    
                    pub_date = self._parse_date(content["date"])
                    if not pub_date or pub_date < date_limit:
                        continue
                    
                    # Analyze relevance
                    analysis = self.content_analyzer.analyze_content(
                        content["title"] + " " + content["text"],
                        company,
                        self.keywords
                    )
                    
                    if analysis["relevance_score"] < 0.3:
                        continue
                    
                    articles.append({
                        "title": content["title"],
                        "link": content["link"],
                        "published_date": pub_date.isoformat(),
                        "content": content["text"],
                        "type": "tech_media",
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching from {url}: {str(e)}")
            return []
    
    def _find_rss_link(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Find RSS feed link in the page"""
        for link in soup.find_all('link', type='application/rss+xml'):
            return urljoin(base_url, link.get('href'))
        for link in soup.find_all('link', type='application/atom+xml'):
            return urljoin(base_url, link.get('href'))
        return None
    
    def _find_articles(self, soup: BeautifulSoup) -> List[BeautifulSoup]:
        """Find article elements in the page"""
        articles = []
        
        # Common article containers
        selectors = [
            'article',
            '.post',
            '.article',
            '.news-item',
            '.blog-entry',
            'div[class*="article"]',
            'div[class*="post"]'
        ]
        
        for selector in selectors:
            found = soup.select(selector)
            if found:
                articles.extend(found)
        
        return articles
    
    def _extract_content(self, article_soup: BeautifulSoup, base_url: str) -> Optional[Dict]:
        """Extract content from an article element"""
        # Find title
        title_elem = (
            article_soup.find('h1') or 
            article_soup.find('h2') or
            article_soup.find(class_=lambda x: x and 'title' in x.lower())
        )
        
        if not title_elem:
            return None
            
        # Find link
        link_elem = title_elem.find('a') if title_elem else None
        link = urljoin(base_url, link_elem['href']) if link_elem and 'href' in link_elem.attrs else base_url
        
        # Find date
        date_elem = (
            article_soup.find('time') or
            article_soup.find('meta', {'property': 'article:published_time'}) or
            article_soup.find(class_=lambda x: x and any(word in x.lower() for word in ['date', 'time', 'published']))
        )
        
        date = ''
        if date_elem:
            if date_elem.name == 'time':
                date = date_elem.get('datetime', date_elem.text)
            elif date_elem.name == 'meta':
                date = date_elem.get('content', '')
            else:
                date = date_elem.text
        
        # Find content
        content_elem = (
            article_soup.find(class_=lambda x: x and 'content' in x.lower()) or
            article_soup.find(class_=lambda x: x and 'body' in x.lower()) or
            article_soup
        )
        
        # Remove unwanted elements
        for elem in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer']):
            elem.decompose()
        
        return {
            "title": self.clean_text(title_elem.text),
            "link": link,
            "date": date,
            "text": self.clean_text(content_elem.text)
        }
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
            
        # Try common date formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
            '%Y-%m-%dT%H:%M:%S.%f%z',  # ISO format with microseconds
            '%Y-%m-%dT%H:%M:%S',  # ISO format without timezone
            '%Y-%m-%d %H:%M:%S',
            '%B %d, %Y',
            '%b %d, %Y',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y'
        ]
        
        # Clean up the date string
        date_str = re.sub(r'\s+', ' ', date_str.strip())
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None