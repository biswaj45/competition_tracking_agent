#!/usr/bin/env python3
"""
Event Registry Scraper for Competition Tracking Agent
Uses Event Registry API (newsapi.ai) for fraud analytics, identity verification, and biometrics news
API Key: 25e5cfec-2d81-4827-9aad-3e7c35c440ba
"""
from eventregistry import *
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventRegistryScraper:
    def __init__(self, api_key: str):
        """Initialize Event Registry scraper with API key"""
        self.api_key = api_key
        # allowUseOfArchive=False restricts to last 30 days (free tier)
        self.er = EventRegistry(apiKey=api_key, allowUseOfArchive=False)
        self.articles_collected = []
        
        # Keywords for fraud analytics and identity verification
        self.search_keywords = [
            # Fraud detection
            ["fraud detection", "fraud prevention", "anti-fraud"],
            # Identity verification
            ["identity verification", "KYC", "know your customer", "identity proofing"],
            # Biometrics
            ["biometric authentication", "facial recognition", "fingerprint", "liveness detection"],
            # AML & Compliance
            ["AML", "anti-money laundering", "compliance", "KYB"],
            # Account security
            ["account takeover", "synthetic identity", "credential stuffing"],
            # Payment fraud
            ["payment fraud", "transaction monitoring", "card-not-present fraud"]
        ]
        
        # Competitors to track
        self.competitors = [
            'TransUnion', 'Experian', 'Equifax', 'LexisNexis', 'FICO', 'SAS',
            'Feedzai', 'DataVisor', 'Kount', 'Riskified', 'Forter', 'Sift', 'Signifyd',
            'Jumio', 'Onfido', 'Veriff', 'Trulioo', 'IDnow', 'Socure', 'Mitek',
            'BioCatch', 'Nuance', 'Shield', 'Sardine', 'Unit21', 'Alloy', 'Persona',
            'Microblink', 'Signicat', 'SITA', 'Indicio', 'iDAKTO', 'GET Group', 'Sumsub'
        ]

    def scrape_articles_by_keywords(self, keywords: List[str], max_items: int = 100) -> List[Dict]:
        """Scrape articles using keyword search"""
        try:
            keyword_query = QueryItems.OR(keywords)
            
            logger.info(f"🔍 Searching for: {' OR '.join(keywords)}")
            
            # Create query iterator for articles
            q = QueryArticlesIter(
                keywords=keyword_query,
                dataType=["news", "blog"],  # Include news and blog posts
                lang="eng"  # English only
            )
            
            articles = []
            # Execute query and get articles sorted by date (most recent first)
            for article in q.execQuery(self.er, sortBy="date", maxItems=max_items):
                # Only include articles with actual content (body text)
                if article.get('body') and len(article.get('body', '')) > 200:  # At least 200 chars
                    articles.append(article)
            
            logger.info(f"✅ Found {len(articles)} articles with content")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error searching for keywords {keywords}: {e}")
            return []

    def scrape_articles_by_concept(self, concept_name: str, max_items: int = 100) -> List[Dict]:
        """Scrape articles using concept-based search (more accurate)"""
        try:
            # Get concept URI
            concept_uri = self.er.getConceptUri(concept_name)
            if not concept_uri:
                logger.warning(f"⚠️ Concept not found: {concept_name}")
                return []
            
            logger.info(f"🔍 Searching for concept: {concept_name} ({concept_uri})")
            
            # Create query iterator
            q = QueryArticlesIter(
                conceptUri=concept_uri,
                dataType=["news", "blog"],
                lang="eng"
            )
            
            articles = []
            for article in q.execQuery(self.er, sortBy="date", maxItems=max_items):
                articles.append(article)
            
            logger.info(f"✅ Found {len(articles)} articles for concept '{concept_name}'")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error searching for concept {concept_name}: {e}")
            return []

    def scrape_articles_by_competitor(self, competitor: str, max_items: int = 50) -> List[Dict]:
        """Scrape articles mentioning specific competitor - ENSURE content exists"""
        try:
            logger.info(f"🔍 Searching for competitor: {competitor}")
            
            # Try to get company concept URI
            concept_uri = self.er.getConceptUri(competitor)
            
            if concept_uri:
                q = QueryArticlesIter(
                    conceptUri=concept_uri,
                    dataType=["news", "blog"],
                    lang="eng"
                )
            else:
                # Fallback to keyword search
                q = QueryArticlesIter(
                    keywords=competitor,
                    dataType=["news", "blog"],
                    lang="eng"
                )
            
            articles = []
            for article in q.execQuery(self.er, sortBy="date", maxItems=max_items):
                # ONLY include articles with substantial content (not just headlines)
                body = article.get('body', '')
                if body and len(body) > 200:  # At least 200 characters of content
                    article['competitor_mentioned'] = competitor
                    articles.append(article)
            
            if articles:
                logger.info(f"✅ Found {len(articles)} articles with content mentioning '{competitor}'")
            else:
                logger.info(f"⚠️ No articles with substantial content for '{competitor}'")
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error searching for competitor {competitor}: {e}")
            return []

    def filter_relevant_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles relevant to fraud analytics - EXCLUDE political news"""
        relevant_articles = []
        
        # Fraud-specific keywords (NOT political)
        fraud_keywords = [
            'fraud', 'identity', 'biometric', 'authentication', 'kyc', 'aml',
            'verification', 'security', 'anti-money laundering', 'synthetic',
            'account takeover', 'payment', 'transaction', 'liveness', 'facial',
            'identity theft', 'financial crime', 'scam', 'phishing'
        ]
        
        # Political keywords to EXCLUDE
        political_keywords = [
            'election', 'vote', 'campaign', 'president', 'senate', 'congress',
            'democrat', 'republican', 'government policy', 'legislation', 'trump',
            'biden', 'political party', 'governor', 'mayor', 'parliament'
        ]
        
        for article in articles:
            title = article.get('title', '').lower()
            body = article.get('body', '').lower()
            combined = f"{title} {body}"
            
            # SKIP if article is political
            is_political = any(keyword in combined for keyword in political_keywords)
            if is_political:
                continue
            
            # SKIP if no substantial content
            if not body or len(body) < 200:
                continue
            
            # Check if article is fraud-relevant
            is_relevant = False
            
            # Check for fraud/identity keywords
            for keyword in fraud_keywords:
                if keyword in combined:
                    is_relevant = True
                    break
            
            # Check for competitor mentions (ALWAYS include if competitor mentioned)
            for competitor in self.competitors:
                if competitor.lower() in combined:
                    if 'competitor_mentioned' not in article:
                        article['competitor_mentioned'] = competitor
                    is_relevant = True
                    break
            
            if is_relevant:
                relevant_articles.append(article)
        
        return relevant_articles

    def save_articles(self, articles: List[Dict], output_dir: Path) -> None:
        """Save articles to text files and JSON"""
        if not articles:
            logger.warning("⚠️ No articles to save")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = output_dir / 'eventregistry_articles.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=True)
        logger.info(f"💾 Saved JSON: {json_file}")
        
        # Save individual text files organized by date
        for i, article in enumerate(articles, 1):
            # Get article date
            date_str = article.get('date', article.get('dateTime', 'unknown_date'))
            if date_str and date_str != 'unknown_date':
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                date_folder = date_obj.strftime('%Y-%m-%d')
            else:
                date_folder = 'unknown_date'
            
            # Create date folder
            date_dir = output_dir / date_folder
            date_dir.mkdir(exist_ok=True)
            
            # Clean filename
            title = article.get('title', f'Article_{i}')
            filename = "".join(c for c in title[:80] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = filename.replace(' ', '_') + '.txt'
            
            txt_file = date_dir / filename
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article.get('title', 'N/A')}\n")
                f.write(f"URL: {article.get('url', 'N/A')}\n")
                f.write(f"Source: {article.get('source', {}).get('title', 'Unknown')}\n")
                f.write(f"Publication Date: {article.get('dateTime', article.get('date', 'N/A'))}\n")
                f.write(f"Scraped At: {datetime.now().isoformat()}\n")
                
                if 'sentiment' in article:
                    f.write(f"Sentiment: {article['sentiment']}\n")
                
                if 'competitor_mentioned' in article:
                    f.write(f"Competitor Mentioned: {article['competitor_mentioned']}\n")
                
                f.write(f"\nCONTENT:\n{article.get('body', 'N/A')}\n")
        
        logger.info(f"💾 Saved {len(articles)} text files to: {output_dir}")

    def scrape_all_sources(self, max_per_query: int = 100) -> Dict:
        """Scrape articles from all predefined queries"""
        all_articles = []
        
        logger.info(f"🚀 Starting Event Registry scraping (Last 30 days)")
        logger.info(f"🔍 Total keyword groups: {len(self.search_keywords)}")
        logger.info(f"🏢 Total competitors to track: {len(self.competitors)}")
        
        # Search by keyword groups
        logger.info(f"\n{'='*60}")
        logger.info(f"PART 1: Keyword-based Search")
        logger.info(f"{'='*60}")
        
        for i, keywords in enumerate(self.search_keywords, 1):
            logger.info(f"\n[{i}/{len(self.search_keywords)}] Keyword group: {keywords}")
            articles = self.scrape_articles_by_keywords(keywords, max_per_query)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        # Search by top competitors
        logger.info(f"\n{'='*60}")
        logger.info(f"PART 2: Competitor-based Search (Top 10)")
        logger.info(f"{'='*60}")
        
        top_competitors = self.competitors[:10]  # Focus on top 10 to save API calls
        for i, competitor in enumerate(top_competitors, 1):
            logger.info(f"\n[{i}/{len(top_competitors)}] Competitor: {competitor}")
            articles = self.scrape_articles_by_competitor(competitor, max_items=30)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            url = article.get('url', article.get('uri', ''))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # Filter for relevance
        logger.info(f"\n🔍 Filtering for relevance...")
        relevant_articles = self.filter_relevant_articles(unique_articles)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 SCRAPING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total articles fetched: {len(all_articles)}")
        logger.info(f"Unique articles: {len(unique_articles)}")
        logger.info(f"Relevant articles: {len(relevant_articles)}")
        logger.info(f"Queries executed: {len(self.search_keywords) + len(top_competitors)}")
        
        return {
            'articles': relevant_articles,
            'metadata': {
                'total_articles': len(relevant_articles),
                'unique_articles': len(unique_articles),
                'raw_articles': len(all_articles),
                'queries_executed': len(self.search_keywords) + len(top_competitors),
                'date_range': 'Last 30 days',
                'scraped_at': datetime.now().isoformat(),
                'api_service': 'Event Registry (newsapi.ai)'
            }
        }


def main():
    """Main function to run Event Registry scraping"""
    try:
        # API Key
        API_KEY = "25e5cfec-2d81-4827-9aad-3e7c35c440ba"
        
        logger.info("="*60)
        logger.info("Event Registry News Scraper")
        logger.info("="*60)
        logger.info(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
        
        # Initialize scraper
        scraper = EventRegistryScraper(API_KEY)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"reports/eventregistry_scrape_{timestamp}")
        
        # Scrape news (last 30 days - free tier limitation)
        results = scraper.scrape_all_sources(max_per_query=100)
        
        # Save results
        if results['articles']:
            scraper.save_articles(results['articles'], output_dir)
            
            # Save metadata
            metadata_file = output_dir / 'scraping_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(results['metadata'], f, indent=2)
            
            logger.info(f"\n✅ SCRAPING COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Results saved to: {output_dir}")
            logger.info(f"📊 Total relevant articles: {results['metadata']['total_articles']}")
            logger.info(f"📅 Date range: {results['metadata']['date_range']}")
        else:
            logger.warning("⚠️ No articles were retrieved")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
