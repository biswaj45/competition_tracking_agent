"""
Collect data from company blogs and press releases
"""
from typing import List, Dict
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ..base_collector import BaseCollector
from ..config import COMPANY_FEEDS, KEYWORDS
from ..content_analyzer import ContentAnalyzer

class CompanyBlogCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.content_analyzer = ContentAnalyzer()
        
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
        
        # Fetch from news/press releases
        if "news_url" in company_info:
            news = self._fetch_content(
                company_info["news_url"],
                company,
                days,
                content_type="press_release"
            )
            articles.extend(news)
            
        # Fetch from blog
        if "blog_url" in company_info:
            blogs = self._fetch_content(
                company_info["blog_url"],
                company,
                days,
                content_type="blog"
            )
            articles.extend(blogs)
            
        return articles
    
    def _fetch_content(self, url: str, company: str, days: int, content_type: str) -> List[Dict]:
        """Fetch content from a specific URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            # Different sites have different structures - this is a basic example
            for article in self._find_articles(soup):
                try:
                    content = self._extract_content(article)
                    if not content:
                        continue
                        
                    # Analyze content relevance and extract information
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
                        "published_date": content["date"],
                        "content": content["text"],
                        "type": content_type,
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing article: {str(e)}")
                    continue
                    
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching from {url}: {str(e)}")
            return []
    
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