#!/usr/bin/env python3
"""
CI Analyzer with Research Papers Support
- Analyzes company newsrooms, news articles, and research papers
- Creates Research & Innovation section
- 90-day period analysis
- Uses LLM for intelligent summarization
"""
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from difflib import SequenceMatcher

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from competition_agent.llm.hf_analyzer import HFAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveCIAnalyzer:
    def __init__(self):
        """Initialize comprehensive CI analyzer with LLM support"""
        self.similarity_threshold = 0.8
        
        # Initialize HuggingFace analyzer for LLM-based summarization
        try:
            logger.info("Initializing LLM models for summarization...")
            self.llm_analyzer = HFAnalyzer()
            self.use_llm = True
            logger.info("✓ LLM models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LLM models: {e}. Falling back to rule-based summarization.")
            self.llm_analyzer = None
            self.use_llm = False
        
        # All competitors
        self.all_competitors = [
            'TransUnion', 'Experian', 'Equifax', 'LexisNexis', 'FICO', 'SAS',
            'Feedzai', 'DataVisor', 'Kount', 'Riskified', 'Forter', 'Sift', 'Signifyd',
            'Jumio', 'Onfido', 'Veriff', 'Trulioo', 'IDnow', 'Socure', 'Mitek',
            'BioCatch', 'Nuance', 'Shield', 'Sardine', 'Unit21', 'Alloy', 'Persona'
        ]
        
        self.fraud_keywords = {
            'identity_verification': ['identity verification', 'KYC', 'know your customer'],
            'biometrics': ['biometric', 'fingerprint', 'facial recognition'],
            'fraud_detection': ['fraud detection', 'fraud prevention', 'anti-fraud'],
            'account_takeover': ['account takeover', 'ATO', 'credential stuffing'],
            'authentication': ['multi-factor authentication', 'MFA', '2FA'],
            'aml_compliance': ['AML', 'anti-money laundering', 'compliance'],
            'risk_analytics': ['risk analytics', 'risk assessment', 'risk scoring'],
            'transaction_monitoring': ['transaction monitoring', 'payment fraud']
        }

    def _is_transunion_article(self, article: Dict) -> bool:
        """Return True if the article is clearly about TransUnion.
        We still count TransUnion in activity histograms, but we exclude it from
        narrative sections (top summaries/recommendations)."""
        source = (article.get('source') or '').lower()
        title = (article.get('title') or '').lower()
        content = (article.get('content') or '').lower()
        # Consider it TransUnion if source is TransUnion or text mentions it
        return 'transunion' in source or 'transunion' in title or 'transunion' in content

    def calculate_heading_similarity(self, heading1: str, heading2: str) -> float:
        """Calculate similarity between two headings"""
        h1 = heading1.lower().strip()
        h2 = heading2.lower().strip()
        return SequenceMatcher(None, h1, h2).ratio()

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Deduplicate articles based on heading similarity"""
        logger.info(f"Deduplicating {len(articles)} articles...")
        
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
                # Keep the one with most content or latest date
                group_articles = [articles[i] for i in group]
                best_article = max(group_articles, key=lambda a: len(a.get('content', '')))
                deduplicated_articles.append(best_article)
                duplicates_removed += len(group) - 1
        
        logger.info(f"Removed {duplicates_removed} duplicates, {len(deduplicated_articles)} unique articles remaining")
        return deduplicated_articles

    def analyze_research_papers(self, papers: List[Dict]) -> Dict:
        """Analyze research papers for Research & Innovation section"""
        logger.info(f"Analyzing {len(papers)} research papers...")
        
        research_analysis = {
            'total_papers': len(papers),
            'papers_by_source': defaultdict(int),
            'key_topics': defaultdict(int),
            'top_papers': [],
            'innovation_trends': []
        }
        
        # Analyze each paper
        for paper in papers:
            source = paper.get('source', 'Unknown')
            research_analysis['papers_by_source'][source] += 1
            
            # Extract key topics from title and abstract
            title = paper.get('title', '').lower()
            summary = paper.get('summary', '').lower()
            full_text = f"{title} {summary}"
            
            # Count fraud domain mentions
            for domain, keywords in self.fraud_keywords.items():
                for keyword in keywords:
                    if keyword in full_text:
                        research_analysis['key_topics'][domain] += 1
                        break
            
            # Add to top papers (will sort by date later)
            research_analysis['top_papers'].append({
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'published_date': paper.get('published_date', ''),
                'url': paper.get('url', ''),
                'summary': paper.get('summary', '')[:500],  # First 500 chars
                'source': source,
                'citations': paper.get('citations', 'N/A')
            })
        
        # Sort top papers by date (most recent first)
        research_analysis['top_papers'].sort(
            key=lambda x: x['published_date'], 
            reverse=True
        )
        
        # Keep top 10 papers
        research_analysis['top_papers'] = research_analysis['top_papers'][:10]
        
        # Identify innovation trends
        top_topics = sorted(research_analysis['key_topics'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
        
        for topic, count in top_topics:
            research_analysis['innovation_trends'].append({
                'topic': topic.replace('_', ' ').title(),
                'paper_count': count,
                'relevance': 'High' if count >= 3 else 'Medium'
            })
        
        # Convert defaultdicts to regular dicts for JSON serialization
        research_analysis['papers_by_source'] = dict(research_analysis['papers_by_source'])
        research_analysis['key_topics'] = dict(research_analysis['key_topics'])
        
        return research_analysis

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """Analyze articles for competitive intelligence"""
        logger.info(f"Analyzing {len(articles)} articles...")
        
        analysis = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_articles': len(articles),
                'analysis_period_days': 90
            },
            'competitor_analysis': {
                'activity_by_company': defaultdict(int),
                'company_newsroom_count': 0,
                'news_aggregator_count': 0
            },
            'technology_trends': {
                'trending_keywords': defaultdict(int)
            },
            'fraud_domain_analysis': defaultdict(int),
            'top_news_summaries': [],
            'strategic_insights': {}
        }
        
        # Count by source type
        for article in articles:
            source_type = article.get('source_type', 'unknown')
            if source_type == 'company_newsroom':
                analysis['competitor_analysis']['company_newsroom_count'] += 1
            elif source_type == 'news_aggregator':
                analysis['competitor_analysis']['news_aggregator_count'] += 1
        
        # Track competitor mentions
        scored_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            full_text = f"{title} {content}"
            
            # Competitor tracking
            for competitor in self.all_competitors:
                if competitor.lower() in full_text:
                    analysis['competitor_analysis']['activity_by_company'][competitor] += 1
            
            # Technology trends
            tech_keywords = ['AI', 'ML', 'machine learning', 'artificial intelligence', 
                           'blockchain', 'API', 'cloud', 'biometric']
            for tech in tech_keywords:
                if tech.lower() in full_text:
                    analysis['technology_trends']['trending_keywords'][tech] += 1
            
            # Fraud domains
            for domain, keywords in self.fraud_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        analysis['fraud_domain_analysis'][domain] += 1
                        break
            
            # Score article
            score = self.calculate_article_importance(article)
            scored_articles.append((score, article))
        
        # Sort by importance
        scored_articles.sort(reverse=True, key=lambda x: x[0])
        
        # Generate top 5 news summaries, skipping TransUnion articles
        rank = 1
        for score, article in scored_articles:
            if self._is_transunion_article(article):
                continue  # exclude from narrative
            analysis['top_news_summaries'].append({
                'rank': rank,
                'title': article.get('title', ''),
                'summary': self.create_compact_summary(article.get('content', ''), max_words=150),
                'importance_score': round(score, 2),
                'source': article.get('source', 'Unknown'),
                'published_date': article.get('published_date', ''),
                'url': article.get('url', '')
            })
            rank += 1
            if rank > 5:
                break
        
        # Convert defaultdicts to dicts
        analysis['competitor_analysis']['activity_by_company'] = dict(analysis['competitor_analysis']['activity_by_company'])
        analysis['technology_trends']['trending_keywords'] = dict(analysis['technology_trends']['trending_keywords'])
        analysis['fraud_domain_analysis'] = dict(analysis['fraud_domain_analysis'])
        
        return analysis

    def calculate_article_importance(self, article: Dict) -> float:
        """Calculate importance score for article"""
        score = 0.0
        
        # Boost company newsroom articles
        if article.get('source_type') == 'company_newsroom':
            score += 2.0
        
        # Check for competitor mentions in title
        title = article.get('title', '').lower()
        for competitor in self.all_competitors:
            if competitor.lower() in title:
                score += 1.5
        
        # Check for high-value fraud keywords
        content = article.get('content', '').lower()
        high_value = ['fraud detection', 'identity verification', 'biometric', 'AML']
        for keyword in high_value:
            if keyword in content:
                score += 0.5
        
        # Recency bonus
        try:
            pub_date = datetime.fromisoformat(article.get('published_date', '').replace('Z', '+00:00'))
            days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
            recency_bonus = max(0, 1.0 - (days_old / 90))
            score += recency_bonus
        except:
            pass
        
        return score

    def create_compact_summary(self, content: str, max_words: int = 150) -> str:
        """Create clean, professional summary using LLM or fallback to rule-based"""
        if not content or len(content.strip()) < 20:
            return "No content available."
        
        # Try LLM-based summarization first
        if self.use_llm and self.llm_analyzer:
            try:
                # Clean the content MORE aggressively before T5
                cleaned = self._clean_text(content)
                
                # Apply additional pre-processing for LLM
                # Remove sentences that are just dates or navigation
                sentences = re.split(r'(?<=[.!?])\s+', cleaned)
                good_sentences = []
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 10:
                        continue
                    # Skip low-quality sentences
                    if self._is_low_quality_sentence(sent):
                        continue
                    # Skip sentences with too many capital letters (likely headings)
                    caps_ratio = sum(1 for c in sent if c.isupper()) / max(len(sent), 1)
                    if caps_ratio > 0.5:
                        continue
                    good_sentences.append(sent)
                
                # Rebuild cleaned content from good sentences
                cleaned = ' '.join(good_sentences)
                
                # Use T5 model directly for summarization
                if len(cleaned.split()) > 50:
                    # Use the T5 model and tokenizer directly
                    if hasattr(self.llm_analyzer, 'summary_tokenizer') and hasattr(self.llm_analyzer, 'summary_model'):
                        logger.info("Using T5 LLM for summarization...")
                        # Prepare the input - limit to most relevant content
                        input_text = "summarize: " + cleaned[:1500]  # Increased limit for better context
                        inputs = self.llm_analyzer.summary_tokenizer.encode(
                            input_text, 
                            return_tensors="pt", 
                            max_length=512, 
                            truncation=True
                        )
                        
                        # Move to correct device
                        if hasattr(self.llm_analyzer, 'torch_device'):
                            inputs = inputs.to(self.llm_analyzer.torch_device)
                        
                        # Generate summary
                        summary_ids = self.llm_analyzer.summary_model.generate(
                            inputs,
                            max_length=max_words,
                            min_length=40,
                            length_penalty=2.0,
                            num_beams=4,
                            early_stopping=True
                        )
                        
                        summary = self.llm_analyzer.summary_tokenizer.decode(
                            summary_ids[0], 
                            skip_special_tokens=True
                        )
                        
                        logger.info(f"LLM summary generated: {summary[:100]}...")
                        # Post-process the LLM summary
                        summary = self._polish_sentence(summary)
                        if summary and len(summary) > 20:
                            return summary
                    else:
                        logger.warning("T5 model or tokenizer not available")
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}. Using fallback.")
        
        # Fallback: Use rule-based summarization
        logger.info("Using rule-based summarization fallback")
        return self._rule_based_summary(content, max_words)
    
    def _rule_based_summary(self, content: str, max_words: int = 150) -> str:
        """Fallback rule-based summary with cleaning"""
        # Clean the content first
        cleaned = self._clean_text(content)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Skip sentences that are mostly dates, URLs, or navigation text
            if self._is_low_quality_sentence(sentence):
                continue
            
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_words:
                # Clean up the sentence
                cleaned_sentence = self._polish_sentence(sentence)
                if cleaned_sentence:
                    summary_sentences.append(cleaned_sentence)
                    word_count += sentence_words
            else:
                break
        
        if not summary_sentences:
            # Fallback: take first coherent chunk
            first_chunk = cleaned[:500].strip()
            if first_chunk:
                return first_chunk + "..."
            return "Content analysis in progress."
        
        summary = ' '.join(summary_sentences)
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def _clean_text(self, text: str) -> str:
        """Clean text of special characters, extra whitespace, and junk"""
        # Remove HTML entities and tags
        text = re.sub(r'&[a-zA-Z]+;|&#\d+;', ' ', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove trademark symbols and special characters
        text = re.sub(r'[™®©℠]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!?]{2,}', '.', text)
        
        # Remove control characters and non-printable chars
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated dashes, underscores
        text = re.sub(r'[-_]{3,}', ' ', text)
        
        return text.strip()
    
    def _is_low_quality_sentence(self, sentence: str) -> bool:
        """Check if sentence is low quality (dates, navigation, etc.)"""
        sentence_lower = sentence.lower()
        
        # Skip if mostly numbers/dates
        numbers = len(re.findall(r'\d+', sentence))
        words = len(sentence.split())
        if words > 0 and numbers / words > 0.4:
            return True
        
        # Skip navigation/boilerplate text
        skip_phrases = [
            'click here', 'read more', 'learn more', 'sign up', 'subscribe',
            'share this', 'follow us', 'contact us', 'privacy policy',
            'terms of service', 'all rights reserved', 'cookie policy',
            'categories:', 'tags:', 'posted on', 'published on', 'updated on',
            'author:', 'by admin', 'leave a comment', 'related posts'
        ]
        if any(phrase in sentence_lower for phrase in skip_phrases):
            return True
        
        # Skip if too short after cleaning
        if len(sentence.split()) < 5:
            return True
        
        # Skip if starts with date pattern
        if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[/-])', sentence_lower):
            return True
        
        return False
    
    def _polish_sentence(self, sentence: str) -> str:
        """Final polish on a sentence"""
        # Ensure proper capitalization
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        # Remove leading/trailing quotes if unmatched
        sentence = sentence.strip('"\'')
        
        # Fix spacing around punctuation
        sentence = re.sub(r'\s+([.,;:!?])', r'\1', sentence)
        sentence = re.sub(r'([.,;:!?])(\w)', r'\1 \2', sentence)
        
        # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        
        return sentence.strip()

async def main():
    """Main analysis function"""
    try:
        logger.info("🔍 Starting Comprehensive CI Analysis...")
        
        # Find latest scraping results
        reports_dir = Path("reports")
        scrape_dirs = list(reports_dir.glob("comprehensive_with_research_*"))
        
        if not scrape_dirs:
            logger.error("No scraping results found! Run comprehensive_scraper_with_research.py first.")
            return
        
        latest_dir = max(scrape_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using data from: {latest_dir}")
        
        # Load articles
        with open(latest_dir / 'all_articles.json', 'r', encoding='utf-8') as f:
            all_articles = json.load(f)
        
        # Load research papers
        with open(latest_dir / 'research_papers.json', 'r', encoding='utf-8') as f:
            research_papers = json.load(f)
        
        logger.info(f"Loaded {len(all_articles)} articles and {len(research_papers)} research papers")
        
        # Initialize analyzer
        analyzer = ComprehensiveCIAnalyzer()
        
        # Deduplicate articles
        deduplicated_articles = analyzer.deduplicate_articles(all_articles)
        
        # Analyze articles
        article_analysis = analyzer.analyze_articles(deduplicated_articles)
        
        # Analyze research papers
        research_analysis = analyzer.analyze_research_papers(research_papers)
        
        # Combine analyses
        comprehensive_analysis = {
            **article_analysis,
            'research_and_innovation': research_analysis
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"reports/comprehensive_ci_analysis_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'comprehensive_ci_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / 'deduplicated_articles.json', 'w', encoding='utf-8') as f:
            json.dump(deduplicated_articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Analysis completed!")
        logger.info(f"📊 Total articles: {len(all_articles)}")
        logger.info(f"📊 After deduplication: {len(deduplicated_articles)}")
        logger.info(f"📊 Research papers: {len(research_papers)}")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE CI ANALYSIS COMPLETED")
        print("="*80)
        print(f"Total Articles Analyzed: {len(all_articles)}")
        print(f"Deduplicated Articles: {len(deduplicated_articles)}")
        print(f"Research Papers: {len(research_papers)}")
        print(f"\nCompany Newsroom Articles: {article_analysis['competitor_analysis']['company_newsroom_count']}")
        print(f"News Aggregator Articles: {article_analysis['competitor_analysis']['news_aggregator_count']}")
        print("\nTop 5 Competitors by Activity:")
        
        sorted_competitors = sorted(
            article_analysis['competitor_analysis']['activity_by_company'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for competitor, count in sorted_competitors:
            print(f"  {competitor}: {count} mentions")
        
        print("\nTop Research Topics:")
        for trend in research_analysis['innovation_trends'][:5]:
            print(f"  {trend['topic']}: {trend['paper_count']} papers ({trend['relevance']} relevance)")
        
        return comprehensive_analysis, output_dir
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
