#!/usr/bin/env python3
"""
Comprehensive Scraper with Company Newsrooms + Research Papers
- Scrapes official company newsrooms (Experian, Equifax, TransUnion, FICO, etc.)
- Fetches academic research papers from arXiv, SSRN, Google Scholar
- Focuses on fraud analytics in financial sector
- 90-day lookback period
"""
import asyncio
import re
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
from tqdm.asyncio import tqdm as async_tqdm
import logging
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveScraperWithResearch:
    def __init__(self, days=90):
        """Initialize comprehensive scraper"""
        self.days = days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=days)

        # Company Official Newsrooms
        self.company_newsrooms = {
            'Experian': {
                'url': 'https://www.experianplc.com/newsroom',
                'rss': 'https://www.experianplc.com/feeds/news',
                'type': 'corporate'
            },
            'Equifax': {
                'url': 'https://investor.equifax.com/news-and-events/news',
                'type': 'corporate'
            },
            'TransUnion': {
                'url': 'https://newsroom.transunion.com',
                'rss': 'https://newsroom.transunion.com/rss',
                'type': 'corporate'
            },
            'FICO': {
                'url': 'https://www.fico.com/en/newsroom',
                'type': 'corporate'
            },
            'LexisNexis': {
                'url': 'https://risk.lexisnexis.com/about-us/media-center',
                'type': 'corporate'
            },
            'SAS': {
                'url': 'https://www.sas.com/en_us/news.html',
                'rss': 'https://www.sas.com/en_us/news.rss.html',
                'type': 'corporate'
            },
            'Feedzai': {
                'url': 'https://feedzai.com/blog/',
                'type': 'corporate'
            },
            'Jumio': {
                'url': 'https://www.jumio.com/news/',
                'type': 'corporate'
            },
            'Onfido': {
                'url': 'https://onfido.com/blog/',
                'rss': 'https://onfido.com/blog/feed/',
                'type': 'corporate'
            },
            'Socure': {
                'url': 'https://www.socure.com/blog',
                'type': 'corporate'
            },
            'BioCatch': {
                'url': 'https://www.biocatch.com/company/newsroom',
                'type': 'corporate'
            }
        }

        # News Aggregators
        self.news_sources = {
            'biometric_update': 'https://www.biometricupdate.com/feed',
            'regtech_analyst': 'https://regtechanalyst.com/feed/',
            'techcrunch_security': 'https://techcrunch.com/category/security/feed/',
            'payments_dive': 'https://www.paymentsdive.com/feeds/news/',
            'pymnts': 'https://www.pymnts.com/feed/',
            'finextra': 'https://www.finextra.com/rss/headlines.aspx',
        }

        # Research Paper Sources
        self.research_sources = {
            'arxiv': {
                'name': 'arXiv Computer Science',
                'search_terms': ['fraud detection', 'financial fraud', 'fraud analytics', 
                               'anomaly detection finance', 'transaction fraud'],
                'api': 'http://export.arxiv.org/api/query'
            },
            'ssrn': {
                'name': 'SSRN (Social Science Research Network)',
                'url': 'https://papers.ssrn.com/sol3/Jeljour_results.cfm',
                'search_terms': ['fraud detection', 'financial fraud analytics']
            }
        }

        self.all_articles = []
        self.research_papers = []

    def _sanitize_title_for_filename(self, title: str, max_len: int = 80) -> str:
        """Sanitize a title string to be safe for Windows filenames."""
        if not title:
            title = "untitled"
        # Replace control characters (including newlines, tabs) with space
        title = re.sub(r"[\x00-\x1F\x7F]+", " ", title)
        # Collapse whitespace
        title = re.sub(r"\s+", " ", title).strip()
        # Remove invalid filename chars
        for ch in '\\/:*?"<>|':
            title = title.replace(ch, '_')
        # Avoid trailing dots/spaces
        title = title.rstrip(' .')
        # Limit length
        return title[:max_len] if len(title) > max_len else title

    async def fetch_company_newsroom(self, company: str, config: dict, session: aiohttp.ClientSession) -> list:
        """Fetch news from company official newsroom"""
        articles = []
        
        try:
            # Try RSS first if available
            if 'rss' in config:
                logger.info(f"Fetching RSS from {company} newsroom...")
                async with session.get(config['rss'], timeout=aiohttp.ClientTimeout(total=15)) as response:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries:
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        # Filter by date - collect ALL articles from company newsrooms
                        if pub_date and self.start_date <= pub_date <= self.end_date:
                            articles.append({
                                'title': entry.get('title', ''),
                                'url': entry.get('link', ''),
                                'published_date': pub_date.isoformat() if pub_date else '',
                                'source': company,
                                'source_type': 'company_newsroom',
                                'summary': entry.get('summary', '')[:500]
                            })
            
            # Fallback: Scrape HTML page
            else:
                logger.info(f"Scraping HTML from {company} newsroom...")
                async with session.get(config['url'], timeout=aiohttp.ClientTimeout(total=20)) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    base_netloc = urlparse(config['url']).netloc

                    candidate_selectors = [
                        'a[href*="news"]',
                        'a[href*="press"]',
                        'a[href*="media"]',
                        'a[href*="newsroom"]',
                        'a[href*="press-releases"]',
                        'a[href*="insights"]',
                        'a[href*="stories"]',
                        'a[href*="blog"]',
                        'a[href*="article"]',
                        'a[href*="announ"]'
                    ]
                    article_links = []
                    for sel in candidate_selectors:
                        article_links.extend(soup.select(sel))

                    seen_hrefs = set()
                    filtered = []
                    for link in article_links:
                        href = link.get('href', '')
                        if not href or href.startswith('#'):
                            continue

                        full_url = urljoin(config['url'], href)
                        parsed = urlparse(full_url)

                        if parsed.netloc and parsed.netloc != base_netloc:
                            continue

                        lower_path = parsed.path.lower()
                        if any(excl in lower_path for excl in [
                            'login', 'signin', 'sign-in', 'sign_in', 'account', 'portal', 'dashboard',
                            'careers', 'cookie', 'privacy', 'terms', 'preferences', 'subscribe', 'contact',
                            'sso', 'idm', 'auth', 'mfa', 'register', 'download', 'pdf', 'events'
                        ]):
                            continue

                        if href in seen_hrefs:
                            continue
                        seen_hrefs.add(href)

                        text = link.get_text(strip=True)
                        if text and len(text) > 10 and (href.startswith('http') or href.startswith('/')):
                            filtered.append((text, full_url))

                    for text, full_url in filtered[:30]:
                        articles.append({
                            'title': text[:200],
                            'url': full_url,
                            'published_date': datetime.now().isoformat(),
                            'source': company,
                            'source_type': 'company_newsroom',
                            'summary': ''
                        })
            
            logger.info(f"✓ {company}: Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching {company} newsroom: {e}")
            return []

    async def fetch_arxiv_papers(self, session: aiohttp.ClientSession) -> list:
        """Fetch research papers from arXiv"""
        papers = []
        
        try:
            search_terms = self.research_sources['arxiv']['search_terms']
            api_url = self.research_sources['arxiv']['api']
            
            for term in search_terms:
                logger.info(f"Searching arXiv for: {term}")
                
                # arXiv API query
                query_params = {
                    'search_query': f'all:{term}',
                    'start': 0,
                    'max_results': 10,
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }
                
                query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])
                full_url = f"{api_url}?{query_string}"
                
                async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    xml_content = await response.text()
                    
                    # Parse XML
                    soup = BeautifulSoup(xml_content, 'xml')
                    entries = soup.find_all('entry')
                    
                    for entry in entries:
                        published = entry.find('published')
                        if published:
                            pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                            
                            # Filter by date (90 days)
                            # Ensure both dates are offset-aware or naive consistently
                            try:
                                start_dt = self.start_date
                                if pub_date.tzinfo is not None and start_dt.tzinfo is None:
                                    # make start_dt timezone aware (UTC)
                                    from datetime import timezone
                                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                                if pub_date >= start_dt:
                                    title = entry.find('title').text.strip() if entry.find('title') else ''
                                    summary = entry.find('summary').text.strip() if entry.find('summary') else ''
                                    link = entry.find('id').text.strip() if entry.find('id') else ''
                                    
                                    # Extract authors
                                    authors = []
                                    for author in entry.find_all('author'):
                                        name = author.find('name')
                                        if name:
                                            authors.append(name.text.strip())
                                    
                                    papers.append({
                                        'title': title,
                                        'authors': authors,
                                        'published_date': pub_date.isoformat(),
                                        'url': link,
                                        'summary': summary[:1000],
                                        'source': 'arXiv',
                                        'source_type': 'research_paper',
                                        'citations': 'N/A (preprint)',
                                        'search_term': term
                                    })
                            except Exception:
                                # Skip malformed dates
                                continue
                                title = entry.find('title').text.strip() if entry.find('title') else ''
                                summary = entry.find('summary').text.strip() if entry.find('summary') else ''
                                link = entry.find('id').text.strip() if entry.find('id') else ''
                                
                                # Extract authors
                                authors = []
                                for author in entry.find_all('author'):
                                    name = author.find('name')
                                    if name:
                                        authors.append(name.text.strip())
                                
                                papers.append({
                                    'title': title,
                                    'authors': authors,
                                    'published_date': pub_date.isoformat(),
                                    'url': link,
                                    'summary': summary[:1000],
                                    'source': 'arXiv',
                                    'source_type': 'research_paper',
                                    'citations': 'N/A (preprint)',
                                    'search_term': term
                                })
                
                # Small delay between searches
                await asyncio.sleep(3)
            
            logger.info(f"✓ arXiv: Found {len(papers)} research papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
            return []

    async def scrape_article_content(self, article: dict, session: aiohttp.ClientSession) -> dict:
        """Scrape full article content"""
        if article.get('source_type') == 'research_paper':
            # Research papers already have abstracts
            article['content'] = article.get('summary', '')
            article['scraped_successfully'] = True
            return article
        
        try:
            async with session.get(article['url'], timeout=aiohttp.ClientTimeout(total=15)) as response:
                html = await response.text()
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts, styles, etc.
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Extract main content
            content = ''
            for tag in soup.find_all(['p', 'article', 'div']):
                text = tag.get_text(strip=True)
                if len(text) > 50:
                    content += text + ' '
            
            article['content'] = content[:5000]
            article['scraped_successfully'] = True
            return article
            
        except Exception as e:
            logger.error(f"Error scraping {article['url']}: {e}")
            article['scraped_successfully'] = False
            article['content'] = article.get('summary', '')
            return article

    async def run(self):
        """Run comprehensive scraper"""
        logger.info(f"🔍 Starting Comprehensive Scraping - {self.days} days")
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        
        async with aiohttp.ClientSession(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/127.0.0.0 Safari/537.36"
                )
            }
        ) as session:
            # 1. Fetch Company Newsrooms
            logger.info("\n📰 Fetching Company Official Newsrooms...")
            newsroom_tasks = []
            for company, config in self.company_newsrooms.items():
                newsroom_tasks.append(self.fetch_company_newsroom(company, config, session))
            
            newsroom_results = await asyncio.gather(*newsroom_tasks, return_exceptions=True)
            
            for result in newsroom_results:
                if isinstance(result, list):
                    self.all_articles.extend(result)
            
            logger.info(f"✓ Company newsrooms: {len([a for a in self.all_articles if a.get('source_type') == 'company_newsroom'])} articles")
            
            # 2. Fetch News Aggregators (RSS)
            logger.info("\n📡 Fetching News Aggregators...")
            for source_name, url in async_tqdm(self.news_sources.items(), desc="News sources"):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        content = await response.text()
                        
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries:
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        if pub_date and self.start_date <= pub_date <= self.end_date:
                            self.all_articles.append({
                                'title': entry.get('title', ''),
                                'url': entry.get('link', ''),
                                'published_date': pub_date.isoformat() if pub_date else '',
                                'source': source_name,
                                'source_type': 'news_aggregator',
                                'summary': entry.get('summary', '')[:500]
                            })
                except Exception as e:
                    logger.error(f"Error fetching {source_name}: {e}")
            
            logger.info(f"✓ News aggregators: {len([a for a in self.all_articles if a.get('source_type') == 'news_aggregator'])} articles")
            
            # 3. Fetch Research Papers
            logger.info("\n🔬 Fetching Research Papers from arXiv...")
            self.research_papers = await self.fetch_arxiv_papers(session)
            
            # 4. Scrape Full Content
            logger.info(f"\n📄 Scraping full content for {len(self.all_articles)} articles...")
            
            scraping_tasks = []
            for article in self.all_articles:
                scraping_tasks.append(self.scrape_article_content(article, session))
            
            scraped_articles = []
            for task in async_tqdm(asyncio.as_completed(scraping_tasks), total=len(scraping_tasks), desc="Scraping"):
                result = await task
                scraped_articles.append(result)
            
            self.all_articles = scraped_articles
        
        # Save Results
        await self.save_results()
        
        return self.all_articles, self.research_papers

    async def save_results(self):
        """Save all results organized by type"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"reports/comprehensive_with_research_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save company newsroom articles
        newsroom_dir = output_dir / 'company_newsrooms'
        newsroom_dir.mkdir(exist_ok=True)
        
        newsroom_articles = [a for a in self.all_articles if a.get('source_type') == 'company_newsroom']
        for article in newsroom_articles:
            company = article['source'].replace(' ', '_')
            company_dir = newsroom_dir / company
            company_dir.mkdir(exist_ok=True)
            
            safe_title = article['title'][:80]
            # Remove characters invalid on Windows filesystems and trim spaces
            for ch in '\\/:*?"<>|':
                safe_title = safe_title.replace(ch, '_')
            safe_title = safe_title.strip().rstrip('.')
            filename = f"{self._sanitize_title_for_filename(article['title'])}_{hashlib.md5(article['url'].encode()).hexdigest()[:8]}.txt"
            filepath = company_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article['title']}\n")
                f.write(f"URL: {article['url']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write(f"Published: {article['published_date']}\n")
                f.write("Type: Company Newsroom\n\n")
                f.write(f"CONTENT:\n{article.get('content', article.get('summary', ''))}\n")
        
        # Save news aggregator articles
        news_dir = output_dir / 'news_articles'
        news_dir.mkdir(exist_ok=True)
        
        news_articles = [a for a in self.all_articles if a.get('source_type') == 'news_aggregator']
        for article in news_articles:
            filename = f"{self._sanitize_title_for_filename(article['title'])}_{hashlib.md5(article['url'].encode()).hexdigest()[:8]}.txt"
            filepath = news_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article['title']}\n")
                f.write(f"URL: {article['url']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write(f"Published: {article['published_date']}\n")
                f.write("Type: News Article\n\n")
                f.write(f"CONTENT:\n{article.get('content', '')}\n")
        
        # Save research papers
        research_dir = output_dir / 'research_papers'
        research_dir.mkdir(exist_ok=True)
        
        for paper in self.research_papers:
            filename = f"{self._sanitize_title_for_filename(paper['title'])}_{hashlib.md5(paper['url'].encode()).hexdigest()[:8]}.txt"
            filepath = research_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {paper['title']}\n")
                f.write(f"Authors: {', '.join(paper['authors'])}\n")
                f.write(f"URL: {paper['url']}\n")
                f.write(f"Published: {paper['published_date']}\n")
                f.write(f"Source: {paper['source']}\n")
                f.write(f"Citations: {paper['citations']}\n")
                f.write(f"Search Term: {paper['search_term']}\n\n")
                f.write(f"ABSTRACT:\n{paper['summary']}\n")
        
        # Save summary
        summary = {
            'scrape_date': timestamp,
            'date_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat(),
                'days': self.days
            },
            'statistics': {
                'total_articles': len(self.all_articles),
                'company_newsroom_articles': len(newsroom_articles),
                'news_aggregator_articles': len(news_articles),
                'research_papers': len(self.research_papers)
            },
            'companies_tracked': list(self.company_newsrooms.keys()),
            'news_sources': list(self.news_sources.keys())
        }
        
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Save raw data
        with open(output_dir / 'all_articles.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_articles, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / 'research_papers.json', 'w', encoding='utf-8') as f:
            json.dump(self.research_papers, f, indent=2, ensure_ascii=False)
        
        # Print Summary
        print("\n" + "="*80)
        print("COMPREHENSIVE SCRAPING COMPLETED")
        print("="*80)
        print(f"Date Range: {self.start_date.date()} to {self.end_date.date()} ({self.days} days)")
        print("\n📊 Results:")
        print(f"  Company Newsroom Articles: {len(newsroom_articles)}")
        print(f"  News Aggregator Articles: {len(news_articles)}")
        print(f"  Research Papers: {len(self.research_papers)}")
        print(f"  Total: {len(self.all_articles) + len(self.research_papers)}")
        print(f"\n📁 Saved to: {output_dir}")
        print("\nCompany Newsrooms Scraped:")
        for company in self.company_newsrooms.keys():
            count = len([a for a in newsroom_articles if a['source'] == company])
            if count > 0:
                print(f"  ✓ {company}: {count} articles")
        
        logger.info(f"✅ All results saved to {output_dir}")

async def main():
    scraper = ComprehensiveScraperWithResearch(days=90)
    articles, research_papers = await scraper.run()
    return articles, research_papers

if __name__ == "__main__":
    # Fix for Windows event loop issues
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
