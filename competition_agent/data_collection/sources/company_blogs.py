"""
Collect data from company blogs and press releases
"""
from typing import List, Dict, Optional
import requests
from datetime import datetime, timedelta
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from ..base_collector import BaseCollector
from ..config import COMPANY_FEEDS, KEYWORDS
from ..content_analyzer import ContentAnalyzer

class CompanyBlogCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.content_analyzer = ContentAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch blog posts and press releases from company websites
        
        Args:
            company: Company name to fetch data for
            days: Number of days to look back
            
        Returns:
            List of blog posts and press releases
        """
        company_info = COMPANY_FEEDS.get(company)
        if not company_info:
            self.logger.warning(f"No feed configuration found for {company}")
            return []
            
        articles = []
        date_limit = datetime.now() - timedelta(days=days)
        
        # Fetch from news/press releases
        if "news_url" in company_info:
            news = self._fetch_content(
                company_info["news_url"],
                company,
                date_limit,
                content_type="press_release"
            )
            articles.extend(news)
            
        # Fetch from blog
        if "blog_url" in company_info:
            blogs = self._fetch_content(
                company_info["blog_url"],
                company,
                date_limit,
                content_type="blog"
            )
            articles.extend(blogs)
            
        return articles
    
    def _fetch_content(self, url: str, company: str, date_limit: datetime, content_type: str) -> List[Dict]:
        """Fetch content from a specific URL"""
        articles = []
        
        try:
            # First try RSS if it's a feed URL
            if url.endswith('/feed/') or 'rss' in url or 'atom' in url:
                articles.extend(self._fetch_from_feed(url, company, date_limit))
                if articles:
                    return articles
            
            # If no RSS or it failed, try web scraping
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for RSS link in the page
            rss_link = self._find_rss_link(soup, url)
            if rss_link:
                feed_articles = self._fetch_from_feed(rss_link, company, date_limit)
                if feed_articles:
                    return feed_articles
            
            # If no RSS or it failed, parse the HTML
            for article in self._find_articles(soup):
                try:
                    content = self._extract_content(article, url)
                    if not content:
                        continue
                    
                    pub_date = self._parse_date(content["date"])
                    if not pub_date or pub_date < date_limit:
                        continue
                        
                    # Analyze content relevance
                    analysis = self.content_analyzer.analyze_content(
                        content["text"],
                        company,
                        KEYWORDS
                    )
                    
                    # Skip if not relevant enough
                    if analysis["relevance_score"] < 0.3:
                        continue
                    
                    articles.append({
                        "title": content["title"],
                        "link": content["link"],
                        "published_date": pub_date.isoformat(),
                        "content": content["text"],
                        "type": content_type,
                        "source": urlparse(url).netloc,
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article: {str(e)}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching from {url}: {str(e)}")
            return []
    
    def _fetch_from_feed(self, url: str, company: str, date_limit: datetime) -> List[Dict]:
        """Fetch articles from an RSS/Atom feed"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
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
                        content,
                        company,
                        KEYWORDS
                    )
                    
                    if analysis["relevance_score"] < 0.3:
                        continue
                    
                    articles.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published_date": pub_date.isoformat(),
                        "content": content,
                        "source": feed.feed.title if hasattr(feed.feed, 'title') else urlparse(url).netloc,
                        "type": "feed",
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing feed entry: {str(e)}")
                    continue
                    
            return articles
            
        except Exception as e:
            self.logger.warning(f"Error fetching feed from {url}: {str(e)}")
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
        
        # Try different article selectors
        selectors = [
            'article',
            '.post',
            '.article',
            '.blog-post',
            '.news-item',
            '.press-release'
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
            article_soup.find(class_=lambda x: x and 'date' in x.lower()) or
            article_soup.find(class_=lambda x: x and 'meta' in x.lower())
        )
        
        # Find content
        content_elem = (
            article_soup.find(class_=lambda x: x and 'content' in x.lower()) or
            article_soup.find(class_=lambda x: x and 'body' in x.lower()) or
            article_soup.find(class_=lambda x: x and 'text' in x.lower())
        )
        
        if not content_elem:
            content_elem = article_soup
        
        return {
            "title": self.clean_text(title_elem.text),
            "link": link,
            "date": date_elem.get('datetime', '') if date_elem else '',
            "text": self.clean_text(content_elem.text)
        }
    
    def _find_articles(self, soup: BeautifulSoup) -> List:
        """Find article elements in the page"""
        # Try different common article containers
        articles = (
            soup.find_all('article') or 
            soup.find_all(class_=lambda x: x and 'post' in x.lower()) or
            soup.find_all(class_=lambda x: x and 'article' in x.lower())
        )
        return articles
    
    def _extract_content(self, article_soup) -> Dict:
        """Extract content from an article element"""
        # These selectors would need to be customized per site
        title_elem = (
            article_soup.find('h1') or 
            article_soup.find('h2') or
            article_soup.find(class_=lambda x: x and 'title' in x.lower())
        )
        
        link_elem = title_elem.find('a') if title_elem else None
        
        date_elem = (
            article_soup.find('time') or
            article_soup.find(class_=lambda x: x and 'date' in x.lower())
        )
        
        content_elem = (
            article_soup.find(class_=lambda x: x and 'content' in x.lower()) or
            article_soup.find(class_=lambda x: x and 'body' in x.lower())
        )
        
        if not (title_elem and content_elem):
            return None
            
        return {
            "title": title_elem.text.strip(),
            "link": urljoin(article_soup.url, link_elem['href']) if link_elem else "",
            "date": self._parse_date(date_elem.text) if date_elem else None,
            "text": content_elem.text.strip()
        }