#!/usr/bin/env python3
"""
Improved Competitive Intelligence Analyzer - 30 Days
Key improvements:
1. Clean summaries - remove news agency headers and dates from content
2. Filter out junk/stock data articles
3. Filter out ad content (subscription prompts, form fills)
4. Keep only fraud-specific technology keywords (remove API, cloud, generic terms)
5. Better content extraction
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import hashlib
from difflib import SequenceMatcher
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedCompetitiveIntelligenceAnalyzer:
    def __init__(self):
        """Initialize the improved CI analyzer"""
        self.similarity_threshold = 0.8
        self.fraud_keywords = {
            'identity_verification': ['identity verification', 'KYC', 'know your customer', 'identity proofing'],
            'biometrics': ['biometric', 'fingerprint', 'facial recognition', 'voice recognition', 'behavioral biometrics', 'liveness detection'],
            'fraud_detection': ['fraud detection', 'fraud prevention', 'anti-fraud', 'fraudulent activity', 'fraud analytics'],
            'account_takeover': ['account takeover', 'ATO', 'credential stuffing'],
            'synthetic_identity': ['synthetic identity', 'synthetic fraud', 'fake identity'],
            'authentication': ['multi-factor authentication', 'MFA', '2FA', 'passwordless', 'authentication'],
            'aml_compliance': ['AML', 'anti-money laundering', 'compliance', 'regulatory', 'KYB', 'know your business'],
            'risk_analytics': ['risk analytics', 'risk assessment', 'risk scoring', 'risk management'],
            'device_intelligence': ['device fingerprinting', 'device intelligence', 'device profiling'],
            'transaction_monitoring': ['transaction monitoring', 'transaction analysis', 'payment fraud']
        }
        
        # Fraud-specific technology keywords ONLY
        self.fraud_tech_keywords = [
            'AI fraud', 'machine learning fraud', 'artificial intelligence fraud',
            'blockchain identity', 'biometric', 'facial recognition', 'fingerprint',
            'liveness detection', 'behavioral analytics', 'anomaly detection',
            'neural network fraud', 'deep learning fraud', 'NLP fraud'
        ]
        
        # All competitors INCLUDING TransUnion for histogram
        self.all_competitors_with_transunion = [
            'TransUnion',
            'Experian', 'Equifax', 'LexisNexis', 'FICO', 'SAS',
            'Feedzai', 'DataVisor', 'Kount', 'Riskified', 'Forter', 'Sift', 'Signifyd',
            'Jumio', 'Onfido', 'Veriff', 'Trulioo', 'IDnow', 'Socure', 'Mitek',
            'BioCatch', 'Nuance', 'Shield', 'Sardine', 'Unit21', 'Alloy', 'Persona',
            'Microblink', 'Signicat', 'SITA', 'Indicio', 'iDAKTO', 'GET Group', 'Sumsub'
        ]
        
        # Competitors EXCLUDING TransUnion for analysis
        self.competitors_for_analysis = [c for c in self.all_competitors_with_transunion if c != 'TransUnion']
        
        # Patterns to identify junk articles
        self.junk_patterns = [
            r'^biometrics?\s+stocks?$',
            r'^biometrics?\s*companies$',
            r'stock\s+market',
            r'eod\s+stock\s+quote',
            r'historical\s+prices',
            r'company\s+symbol',
            r'view\s+all\s+companies',
            r'behavioral\s+biometrics$',  # Just category page
            r'inside\s+the\s+hackers',
            r'hacker.*toolkit'
        ]
        
        # Ad content patterns to remove
        self.ad_patterns = [
            r'Get the Full Story.*?Terms and Conditions\.?\s*[ΔÎ"]?',
            r'Complete the form to unlock.*?(?=\n\n|$)',
            r'Subscribe to our.*?(?=\n\n|$)',
            r'By completing this form.*?(?=\n\n|$)',
            r'yes Subscribe to.*?(?=\n\n|$)',
            r'no additional logins required.*?(?=\n\n|$)',
            r'receive marketing communications.*?(?=\n\n|$)',
        ]

    def is_junk_article(self, article: Dict) -> bool:
        """Identify and filter out junk articles (stock data, listings, political, hacker, etc.)"""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        
        # Check title against junk patterns
        for pattern in self.junk_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                logger.debug(f"Filtered junk article: {title}")
                return True
        
        # Filter out political/law enforcement articles not related to fraud solutions
        political_keywords = [
            'nypd', 'trump administration', 'federal lawsuit', 'police department',
            'government policy', 'oversight policy', 'unconstitutional',
            'government shutdown', 'pleads guilty', 'defense contractor'
        ]
        
        for keyword in political_keywords:
            if keyword in title or keyword in content[:200]:
                logger.debug(f"Filtered political/law enforcement article: {title}")
                return True
        
        # Filter out hacker/breach/spyware news not related to fraud prevention
        hacker_keywords = [
            'hackers threaten', 'data breach', 'breached', 'hacked', 
            'cyberattack', 'ransomware', 'leak data', 'spyware', 'zero-day',
            'malware', 'security flaws', 'exposed', 'vulnerability'
        ]
        
        # Only filter if it's primarily about breaches/hacks, not fraud prevention
        if any(keyword in title for keyword in hacker_keywords):
            # Check if it's about fraud prevention solutions
            fraud_solution_keywords = ['fraud detection', 'fraud prevention', 'identity verification', 'biometric']
            if not any(sol in content[:300] for sol in fraud_solution_keywords):
                logger.debug(f"Filtered hacker/breach article: {title}")
                return True
        
        # Check if article is mostly stock data/listings
        if 'symbol' in content and 'view' in content and len(content.split('\n')) > 20:
            if content.count('View') > 10:  # Likely a stock listing
                return True
        
        # Check if content is too short (< 300 chars for meaningful fraud analytics content)
        if len(content.strip()) < 300:
            logger.debug(f"Filtered short article: {title}")
            return True
        
        return False

    def clean_content(self, content: str) -> str:
        """Clean content by removing ads, headers, author names, related articles, and unwanted text"""
        if not content:
            return ""
        
        # Remove "Related Posts" section and everything after
        content = re.split(r'Related Posts|Latest Biometrics News|Article Topics|Comments|Leave a Reply', content, flags=re.IGNORECASE)[0]
        
        # Remove ad patterns
        for pattern in self.ad_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove author names at the beginning (Chris Burt, McConvey, etc.)
        content = re.sub(r'^(?:By\s+)?(?:Chris Burt|Joel R\. McConvey|[A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:\||$)', '', content, flags=re.MULTILINE)
        
        # Remove common header patterns (news agency, date in content)
        # Pattern like: "Title Name Nov 1, 2025, 2:45 pm EDT | Chris Burt Categories..."
        content = re.sub(r'^.*?\d{4},\s*\d{1,2}:\d{2}\s*[ap]m\s+[A-Z]{2,4}\s*\|', '', content, flags=re.MULTILINE)
        
        # Remove "Categories Biometrics News |" type headers
        content = re.sub(r'Categories?\s+[^|]+\|', '', content, flags=re.IGNORECASE)
        
        # Remove category labels like "Industry Analysis", "Trade Notes", etc.
        content = re.sub(r'^(?:Industry Analysis|Trade Notes|Civil / National ID|Biometrics News|Features and Interviews|Biometric Update Podcast)\s*', '', content, flags=re.MULTILINE)
        
        # Remove "Advertisement:" markers
        content = re.sub(r'Advertisement:\s*(?:Scroll to Continue)?', '', content, flags=re.IGNORECASE)
        
        # Remove special characters and clean up encoding issues
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII
        content = re.sub(r'[\u2018\u2019\u201C\u201D]', "'", content)  # Replace smart quotes
        
        # Remove repeated whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content

    def calculate_heading_similarity(self, heading1: str, heading2: str) -> float:
        """Calculate similarity between two article headings"""
        h1 = self.normalize_heading(heading1)
        h2 = self.normalize_heading(heading2)
        
        similarity = SequenceMatcher(None, h1, h2).ratio()
        
        words1 = set(h1.split())
        words2 = set(h2.split())
        
        if len(words1) > 0 and len(words2) > 0:
            jaccard_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            similarity = (similarity * 0.7) + (jaccard_similarity * 0.3)
        
        return similarity

    def normalize_heading(self, heading: str) -> str:
        """Normalize heading for comparison"""
        heading = heading.lower().strip()
        prefixes_to_remove = [
            'breaking:', 'exclusive:', 'update:', 'news:', 'press release:',
            'announced:', 'launches:', 'introduces:'
        ]
        
        for prefix in prefixes_to_remove:
            if heading.startswith(prefix):
                heading = heading[len(prefix):].strip()
        
        heading = re.sub(r'[^\w\s]', ' ', heading)
        heading = ' '.join(heading.split())
        
        return heading

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Deduplicate articles based on heading similarity"""
        logger.info(f"Starting deduplication of {len(articles)} articles...")
        
        similarity_groups = []
        processed_indices = set()
        
        for i, article1 in enumerate(articles):
            if i in processed_indices:
                continue
                
            current_group = [i]
            processed_indices.add(i)
            
            for j, article2 in enumerate(articles[i+1:], start=i+1):
                if j in processed_indices:
                    continue
                
                similarity = self.calculate_heading_similarity(
                    article1.get('title', ''), 
                    article2.get('title', '')
                )
                
                if similarity >= self.similarity_threshold:
                    current_group.append(j)
                    processed_indices.add(j)
            
            similarity_groups.append(current_group)
        
        deduplicated_articles = []
        duplicates_removed = 0
        
        for group in similarity_groups:
            if len(group) == 1:
                deduplicated_articles.append(articles[group[0]])
            else:
                group_articles = [articles[i] for i in group]
                best_article = self.select_best_article(group_articles)
                deduplicated_articles.append(best_article)
                duplicates_removed += len(group) - 1
        
        logger.info(f"Deduplication completed: {duplicates_removed} duplicates removed")
        logger.info(f"Final count: {len(deduplicated_articles)} unique articles")
        
        return deduplicated_articles

    def select_best_article(self, similar_articles: List[Dict]) -> Dict:
        """Select the best article from similar ones (latest date, longer content)"""
        def article_score(article):
            try:
                date_str = article.get('published_date', '')
                if date_str:
                    article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    article_date = datetime.min
            except:
                article_date = datetime.min
            
            content_length = len(article.get('content', ''))
            
            return article_date.timestamp() * 1000 + (content_length / 1000)
        
        best_article = max(similar_articles, key=article_score)
        return best_article

    def is_transunion_article(self, article: Dict) -> bool:
        """Check if article is primarily about TransUnion"""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        
        transunion_variants = ['transunion', 'trans union']
        
        for variant in transunion_variants:
            if variant in title:
                return True
        
        return False

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """Analyze deduplicated articles for competitive intelligence"""
        logger.info(f"Analyzing {len(articles)} deduplicated articles...")
        
        # Separate TransUnion articles for histogram only
        transunion_count = sum(1 for a in articles if self.is_transunion_article(a))
        
        # Exclude TransUnion articles from detailed analysis
        articles_for_analysis = [a for a in articles if not self.is_transunion_article(a)]
        
        logger.info(f"TransUnion articles: {transunion_count} (tracked for histogram only)")
        logger.info(f"Competitor articles for analysis: {len(articles_for_analysis)}")
        
        analysis = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_articles_analyzed': len(articles),
                'competitor_articles': len(articles_for_analysis),
                'transunion_articles': transunion_count,
                'analysis_period': self.get_analysis_period(articles),
                'deduplication_applied': True,
                'days_covered': 30
            },
            'competitor_analysis': {
                'activity_summary': {},
                'transunion_comparison': {
                    'transunion_count': transunion_count,
                    'included_in_histogram': True
                }
            },
            'technology_trends': {
                'trending_keywords': {}
            },
            'top_news_summaries': [],
            'fraud_domain_analysis': {},
            'strategic_insights': {}
        }
        
        # Track competitor mentions (excluding TransUnion from analysis)
        competitor_mentions = defaultdict(int)
        technology_mentions = defaultdict(int)
        fraud_domain_mentions = defaultdict(int)
        
        scored_articles = []
        
        for article in articles_for_analysis:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {content}"
            
            importance_score = self.calculate_article_importance(article)
            scored_articles.append((importance_score, article))
            
            # Count competitor mentions (excluding TransUnion)
            for competitor in self.competitors_for_analysis:
                if competitor.lower() in full_text:
                    competitor_mentions[competitor] += 1
            
            # Technology trends - ONLY fraud-specific technologies
            for tech in self.fraud_tech_keywords:
                if tech.lower() in full_text:
                    # Use short name for display
                    tech_name = tech.split()[0].upper() if ' ' in tech else tech
                    technology_mentions[tech_name] += 1
            
            # Fraud domains
            for domain, keywords in self.fraud_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        fraud_domain_mentions[domain] += 1
                        break
        
        # Add TransUnion to competitor summary for histogram
        competitor_mentions_with_tu = dict(competitor_mentions)
        competitor_mentions_with_tu['TransUnion'] = transunion_count
        
        # Sort articles by importance
        scored_articles.sort(reverse=True, key=lambda x: x[0])
        
        # Generate top 5 news summaries (cleaned, no ads) - MOST IMPORTANT ONLY
        analysis['top_news_summaries'] = self.generate_clean_news_summaries(scored_articles[:5])
        
        # Fill in analysis data
        analysis['competitor_analysis']['activity_summary'] = competitor_mentions_with_tu
        analysis['technology_trends']['trending_keywords'] = dict(technology_mentions)
        analysis['fraud_domain_analysis'] = dict(fraud_domain_mentions)
        
        # Strategic insights (excluding TransUnion)
        analysis['strategic_insights'] = self.generate_strategic_insights(
            competitor_mentions, technology_mentions, fraud_domain_mentions
        )
        
        return analysis

    def calculate_article_importance(self, article: Dict) -> float:
        """Calculate importance score for article ranking"""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        source = article.get('source', '')
        
        score = 0.0
        
        # Source credibility
        source_weights = {
            'Reuters': 3.0, 'Bloomberg': 2.8, 'Wall Street Journal': 2.5,
            'TechCrunch': 2.0, 'Biometric Update': 1.8, 'RegTech Analyst': 1.8,
            'PYMNTS': 1.5, 'InfoSecurity Magazine': 1.5
        }
        score += source_weights.get(source, 1.0)
        
        # Competitor mentions (excluding TransUnion)
        for competitor in self.competitors_for_analysis:
            if competitor.lower() in title:
                score += 2.0
            elif competitor.lower() in content:
                score += 1.0
        
        # Key fraud domains
        high_value_keywords = [
            'identity verification', 'fraud detection', 'biometric', 'AML', 'KYC',
            'account takeover', 'synthetic identity', 'liveness detection'
        ]
        
        for keyword in high_value_keywords:
            if keyword in title:
                score += 1.0
            elif keyword in content:
                score += 0.5
        
        # Innovation keywords
        innovation_keywords = [
            'launches', 'introduces', 'announces', 'partnership', 'acquisition',
            'new', 'first', 'innovative'
        ]
        
        for keyword in innovation_keywords:
            if keyword in title:
                score += 0.8
        
        # Recency bonus
        try:
            pub_date = datetime.fromisoformat(article.get('published_date', '').replace('Z', '+00:00'))
            days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
            recency_bonus = max(0, 1.0 - (days_old / 30))
            score += recency_bonus
        except:
            pass
        
        return score

    def generate_clean_news_summaries(self, top_articles: List[Tuple[float, Dict]]) -> List[Dict]:
        """Generate clean, concise, intelligent summaries (80 words max)"""
        summaries = []
        
        for i, (score, article) in enumerate(top_articles, 1):
            title = article.get('title', f'Article {i}')
            raw_content = article.get('content', '')
            
            # Clean the content first
            cleaned_content = self.clean_content(raw_content)
            
            # Skip if content is too short after cleaning
            if len(cleaned_content) < 100:
                logger.debug(f"Skipping article with insufficient content: {title}")
                continue
            
            # Generate short, intelligent summary (80 words)
            summary = self.create_intelligent_summary(cleaned_content, max_words=80)
            
            summaries.append({
                'rank': i,
                'title': title,
                'summary': summary,
                'importance_score': round(score, 2),
                'source': article.get('source', 'Unknown'),
                'published_date': article.get('published_date', ''),
                'url': article.get('url', '')
            })
        
        return summaries[:10]  # Return top 10

    def create_clean_summary(self, content: str, max_words: int = 150) -> str:
        """Create a clean summary without headers, ads, or metadata"""
        if not content:
            return "No content available."
        
        # Additional cleaning
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Skip sentences that look like headers or metadata
            skip_patterns = [
                r'^by\s+\w+\s*\|',  # "By Author |"
                r'^\w+\s+\d{1,2},\s+\d{4}',  # "Nov 1, 2025"
                r'^categories',  # "Categories"
                r'^tags',  # "Tags"
                r'^\w+\s+update',  # "Biometric Update"
            ]
            
            should_skip = False
            for pattern in skip_patterns:
                if re.match(pattern, sentence.lower()):
                    should_skip = True
                    break
            
            if should_skip:
                continue
                
            sentence_words = len(sentence.split())
            
            if word_count + sentence_words <= max_words:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                remaining_words = max_words - word_count
                if remaining_words > 10:
                    words = sentence.split()[:remaining_words]
                    summary_sentences.append(' '.join(words) + '...')
                break
        
        if not summary_sentences:
            return content[:max_words * 6]  # Fallback to character limit
        
        return '. '.join(summary_sentences) + '.'

    def create_intelligent_summary(self, content: str, max_words: int = 80) -> str:
        """Create an intelligent, flowing summary with key insights (80 words max)"""
        if not content:
            return "No content available."
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove author/editor bylines more aggressively
        content = re.sub(r'(?:by\s+)?(?:Chris Burt|McConvey|[A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:\||$)', '', content, flags=re.IGNORECASE)
        
        # Remove publication metadata at the start
        content = re.sub(r'^.*?\d{4},\s*\d{1,2}:\d{2}\s*[ap]m\s+[A-Z]{3,4}\s*(?:\|)?', '', content)
        
        # Remove category labels
        content = re.sub(r'Categories\s+[^|]+\|?\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'(?:Industry Analysis|Trade Notes|Civil / National ID|Biometrics News|Features and Interviews)\s*', '', content, flags=re.IGNORECASE)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short or metadata sentences
            if len(sentence) < 30 or len(sentence.split()) < 6:
                continue
            
            # Skip sentences with author names
            if re.search(r'\b(?:Chris Burt|McConvey|By [A-Z][a-z]+)\b', sentence, re.IGNORECASE):
                continue
            
            # Skip category/tag lines
            if re.match(r'^(?:Categories?|Tags?|Posted)', sentence, re.IGNORECASE):
                continue
            
            # Skip sentences that are just company/product names
            if len(sentence.split()) < 8 and re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$', sentence):
                continue
            
            meaningful_sentences.append(sentence)
        
        if not meaningful_sentences:
            # Emergency fallback: extract first meaningful chunk
            words = [w for w in content.split() if len(w) > 2][:max_words]
            return ' '.join(words) + '.'
        
        # Build flowing summary starting with most informative sentences
        summary_parts = []
        word_count = 0
        
        # Prioritize sentences with key business terms
        priority_terms = ['announces', 'launches', 'introduces', 'partners', 'acquired', 'partnership', 
                         'solution', 'platform', 'technology', 'system', 'fraud', 'identity', 'biometric']
        
        # Sort by priority (sentences with key terms first)
        def sentence_priority(s):
            lower_s = s.lower()
            return sum(1 for term in priority_terms if term in lower_s)
        
        prioritized = sorted(meaningful_sentences[:10], key=sentence_priority, reverse=True)
        
        # Take first few sentences up to word limit
        for sentence in prioritized:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if word_count + sentence_word_count <= max_words:
                summary_parts.append(sentence)
                word_count += sentence_word_count
                
                # Stop if we have a good amount
                if word_count >= max_words * 0.8:
                    break
            elif word_count < max_words * 0.5:
                # Add partial sentence if we don't have enough yet
                remaining = max_words - word_count
                if remaining > 10:
                    summary_parts.append(' '.join(sentence_words[:remaining]))
                    break
        
        if not summary_parts:
            # Last resort: take beginning
            words = meaningful_sentences[0].split()[:max_words] if meaningful_sentences else content.split()[:max_words]
            return ' '.join(words) + '.'
        
        # Join with proper flow
        summary = ' '.join(summary_parts)
        
        # Clean up punctuation
        summary = re.sub(r'\s+([.,!?])', r'\1', summary)  # Remove space before punctuation
        summary = re.sub(r'([.,!?])([A-Z])', r'\1 \2', summary)  # Add space after punctuation
        
        # Ensure proper ending
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Remove special characters and extra whitespace
        summary = re.sub(r'[^\x00-\x7F]+', '', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary        
        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.') and not summary.endswith('...'):
            summary += '.'
        
        return summary if summary else "Content not available."

    def generate_strategic_insights(self, competitor_mentions: Dict, 
                                  technology_mentions: Dict, 
                                  fraud_mentions: Dict) -> Dict:
        """Generate strategic insights (excluding TransUnion)"""
        insights = {
            'competitive_threats': [],
            'technology_gaps': [],
            'key_trends': []
        }
        
        # Top competitors (excluding TransUnion)
        top_competitors = sorted(competitor_mentions.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        for competitor, mentions in top_competitors:
            if mentions >= 1:  # Include all with at least 1 mention
                insights['competitive_threats'].append({
                    'competitor': competitor,
                    'activity_level': mentions,
                    'threat_level': 'High' if mentions >= 10 else ('Medium' if mentions >= 3 else 'Low')
                })
        
        # Technology trends - only fraud-specific
        top_technologies = sorted(technology_mentions.items(),
                                key=lambda x: x[1], reverse=True)[:8]
        
        for tech, mentions in top_technologies:
            if mentions >= 2:
                insights['key_trends'].append({
                    'technology': tech,
                    'mention_count': mentions,
                    'adoption_level': 'High' if mentions >= 8 else 'Emerging'
                })
        
        return insights

    def get_analysis_period(self, articles: List[Dict]) -> Dict:
        """Determine the analysis period from articles"""
        dates = []
        for article in articles:
            date_str = article.get('published_date', '')
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    dates.append(date)
                except:
                    continue
        
        if dates:
            return {
                'start_date': min(dates).isoformat(),
                'end_date': max(dates).isoformat(),
                'total_days': (max(dates) - min(dates)).days + 1
            }
        else:
            return {
                'start_date': None,
                'end_date': None,
                'total_days': 30
            }

async def main():
    """Main function to run improved CI analysis"""
    try:
        logger.info("🔍 Starting Improved 30-Day CI Analysis...")
        
        # Find the latest scraping results
        reports_dir = Path("reports")
        scrape_dirs = [d for d in reports_dir.glob("final_30days_*") if d.is_dir()]
        
        if not scrape_dirs:
            logger.error("No scraping results found!")
            return
        
        latest_dir = max(scrape_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using data from: {latest_dir}")
        
        # Load all articles
        all_articles = []
        
        for date_folder in latest_dir.iterdir():
            if date_folder.is_dir() and date_folder.name.startswith('2025-'):
                for article_file in date_folder.glob('*.txt'):
                    try:
                        with open(article_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        lines = content.split('\n')
                        title = ''
                        url = ''
                        pub_date = ''
                        
                        for line in lines[:10]:
                            if line.startswith('Title: '):
                                title = line[7:].strip()
                            elif line.startswith('URL: '):
                                url = line[5:].strip()
                            elif line.startswith('Publication Date: '):
                                pub_date = line[18:].strip()
                        
                        content_lines = []
                        in_content = False
                        for line in lines:
                            if line.startswith('CONTENT:') or in_content:
                                in_content = True
                                if line.startswith('CONTENT:'):
                                    content_lines.append(line[8:].strip())
                                else:
                                    content_lines.append(line)
                        
                        article_content = '\n'.join(content_lines).strip()
                        if not article_content:
                            article_content = content
                        
                        source = 'Unknown'
                        if url:
                            parsed_url = urlparse(url)
                            domain = parsed_url.netloc.replace('www.', '')
                            if 'biometricupdate' in domain:
                                source = 'Biometric Update'
                            elif 'techcrunch' in domain:
                                source = 'TechCrunch'
                            elif 'pymnts' in domain:
                                source = 'PYMNTS'
                            elif 'infosecurity-magazine' in domain:
                                source = 'InfoSecurity Magazine'
                            elif 'regtechanalyst' in domain:
                                source = 'RegTech Analyst'
                            else:
                                source = domain.title()
                        
                        article_data = {
                            'title': title if title else article_file.stem.replace('_', ' '),
                            'content': article_content,
                            'url': url,
                            'source': source,
                            'published_date': pub_date if pub_date else date_folder.name + 'T00:00:00',
                            'scraped_at': date_folder.name
                        }
                        
                        all_articles.append(article_data)
                        
                    except Exception as e:
                        logger.debug(f"Error loading {article_file}: {e}")
        
        if not all_articles:
            logger.error("No articles found!")
            return
        
        logger.info(f"Loaded {len(all_articles)} articles for analysis")
        
        # Initialize analyzer
        analyzer = ImprovedCompetitiveIntelligenceAnalyzer()
        
        # Filter out junk articles
        filtered_articles = [a for a in all_articles if not analyzer.is_junk_article(a)]
        logger.info(f"Filtered out {len(all_articles) - len(filtered_articles)} junk articles")
        
        # Deduplicate
        deduplicated_articles = analyzer.deduplicate_articles(filtered_articles)
        
        # Analyze and get scored articles
        analysis_results = analyzer.analyze_articles(deduplicated_articles)
        
        # Keep ONLY top 5 most important articles
        scored_articles = []
        for article in deduplicated_articles:
            if not analyzer.is_transunion_article(article):
                importance_score = analyzer.calculate_article_importance(article)
                scored_articles.append((importance_score, article))
        
        scored_articles.sort(reverse=True, key=lambda x: x[0])
        top_5_articles = [article for score, article in scored_articles[:5]]
        
        logger.info(f"📌 Keeping ONLY top 5 most important articles (from {len(deduplicated_articles)} total)")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"reports/improved_ci_analysis_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = output_dir / 'final_ci_analysis.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=True)
        
        dedup_file = output_dir / 'deduplicated_articles.json'
        with open(dedup_file, 'w', encoding='utf-8') as f:
            json.dump(top_5_articles, f, indent=2, ensure_ascii=True)
        
        logger.info(f"✅ Improved CI analysis completed!")
        logger.info(f"📊 Total articles: {len(all_articles)}")
        logger.info(f"📊 After filtering junk: {len(filtered_articles)}")
        logger.info(f"📊 After deduplication: {len(deduplicated_articles)}")
        logger.info(f"📊 TOP 5 MOST IMPORTANT: {len(top_5_articles)}")
        logger.info(f"📊 TransUnion articles: {analysis_results['metadata']['transunion_articles']}")
        logger.info(f"📊 Competitor articles: {analysis_results['metadata']['competitor_articles']}")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Print top news summaries
        print("\n" + "="*80)
        print("TOP 5 MOST IMPORTANT NEWS (Clean, No Ads)")
        print("="*80)
        
        for summary in analysis_results['top_news_summaries']:
            print(f"\n{summary['rank']}. {summary['title']}")
            print(f"   Score: {summary['importance_score']}")
            print(f"   {summary['summary'][:200]}...")
            print("-" * 60)
        
        return analysis_results, output_dir
        
    except Exception as e:
        logger.error(f"Improved CI analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
