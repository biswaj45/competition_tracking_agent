#!/usr/bin/env python3
"""
News Consolidator for Event Registry Articles
Groups news by company and filters for strategic importance:
- New products/services
- Innovations/technology advances
- Investment/funding rounds
- Major customer wins
Uses LLM to judge final document quality
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re
from collections import defaultdict
from transformers import pipeline
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsConsolidator:
    def __init__(self, input_dir: Path):
        """Initialize news consolidator"""
        self.input_dir = Path(input_dir)
        self.articles = []
        self.grouped_articles = defaultdict(list)
        
        # Key competitors to track
        self.competitors = [
            'TransUnion', 'Experian', 'Equifax', 'LexisNexis', 'FICO', 'SAS',
            'Feedzai', 'DataVisor', 'Kount', 'Riskified', 'Forter', 'Sift', 'Signifyd',
            'Jumio', 'Onfido', 'Veriff', 'Trulioo', 'IDnow', 'Socure', 'Mitek',
            'BioCatch', 'Nuance', 'Shield', 'Sardine', 'Unit21', 'Alloy', 'Persona',
            'Microblink', 'Signicat', 'SITA', 'Indicio', 'iDAKTO', 'GET Group', 'Sumsub'
        ]
        
        # Strategic importance keywords
        self.strategic_keywords = {
            'product_launch': [
                'launch', 'launches', 'launched', 'introduce', 'introduces', 'introduced',
                'unveil', 'unveils', 'unveiled', 'release', 'releases', 'released',
                'new product', 'new service', 'new solution', 'new platform', 'new offering'
            ],
            'innovation': [
                'innovation', 'innovative', 'breakthrough', 'patent', 'AI-powered',
                'machine learning', 'deep learning', 'advanced technology', 'cutting-edge',
                'next-generation', 'state-of-the-art', 'proprietary', 'first-of-its-kind'
            ],
            'investment': [
                'funding', 'investment', 'raise', 'raised', 'raises', 'series A', 'series B',
                'series C', 'venture capital', 'VC', 'round', 'million', 'billion',
                'investors', 'valuation', 'IPO', 'acquisition', 'acquired', 'merger'
            ],
            'customers': [
                'partnership', 'partner', 'partners', 'partnered', 'client', 'customer',
                'contract', 'deal', 'agreement', 'collaboration', 'selects', 'selected',
                'chooses', 'chosen', 'wins', 'won', 'signs', 'signed', 'enterprise customer'
            ]
        }
        
        # Initialize zero-shot classifier for quality judgement
        try:
            logger.info("📦 Loading LLM classifier for quality assessment...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
            logger.info("✅ LLM classifier loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load LLM classifier: {e}")
            self.classifier = None

    def load_articles(self) -> None:
        """Load all articles from JSON file"""
        try:
            json_file = self.input_dir / 'eventregistry_articles.json'
            
            if not json_file.exists():
                logger.error(f"❌ JSON file not found: {json_file}")
                return
            
            with open(json_file, 'r', encoding='utf-8') as f:
                self.articles = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.articles)} articles from {json_file}")
            
        except Exception as e:
            logger.error(f"❌ Error loading articles: {e}")

    def identify_company(self, article: Dict) -> Optional[str]:
        """Identify which competitor is mentioned in the article"""
        # Check if already tagged
        if 'competitor_mentioned' in article:
            return article['competitor_mentioned']
        
        # Search in title and body
        title = article.get('title', '').lower()
        body = article.get('body', '').lower()
        combined = f"{title} {body}"
        
        for competitor in self.competitors:
            if competitor.lower() in combined:
                return competitor
        
        return None

    def is_fraud_industry_relevant(self, article: Dict) -> bool:
        """Strict check: ONLY fraud, identity, biometric, regulatory fraud content"""
        title = article.get('title', '').lower()
        body = article.get('body', '').lower()
        combined = f"{title} {body}"
        
        # MUST have at least TWO fraud/identity/biometric keywords (stricter)
        fraud_industry_keywords = [
            'fraud', 'fraudulent', 'anti-fraud', 'fraud detection', 'fraud prevention',
            'identity theft', 'identity verification', 'identity fraud', 'synthetic identity',
            'biometric', 'biometrics', 'facial recognition', 'fingerprint', 'liveness',
            'kyc', 'know your customer', 'aml', 'anti-money laundering',
            'authentication', 'account takeover', 'credential stuffing',
            'payment fraud', 'credit card fraud', 'transaction fraud',
            'phishing', 'data breach', 'pii breach', 'personal data breach',
            'regulatory action fraud', 'fraud investigation', 'fraud scheme',
            'scam', 'financial crime', 'cybersecurity breach', 'stolen identity',
            'identity proofing', 'digital identity', 'identity management'
        ]
        
        keyword_count = sum(1 for keyword in fraud_industry_keywords if keyword in combined)
        
        # Must have at least 2 fraud keywords for relevance
        if keyword_count < 2:
            return False
        
        # EXCLUDE non-fraud topics (expanded list)
        excluded_topics = [
            'xbox', 'gaming console', 'video game', 'playstation', 'nintendo',
            'medicine', 'pharmaceutical', 'drug development', 'clinical trial',
            'cannabis', 'marijuana', 'recreational market', 'dispensary',
            'eps report', 'earnings per share', 'quarterly earnings',
            'stock price', 'share price', 'dividend',
            'casino', 'gambling', 'betting', 'lottery',
            'cryptocurrency exchange', 'crypto trading', 'bitcoin price',
            'real estate', 'property market', 'housing market',
            'restaurant', 'food service', 'hospitality',
            'automotive', 'car manufacturer', 'vehicle sales',
            'entertainment', 'movie', 'television', 'streaming service',
            'mobile money wallet', 'payment processing without fraud',
            'general ai course', 'ml course online', 'training course',
            'neo-banking without fraud', 'credit scoring without fraud',
            'generic fintech', 'insurance without fraud',
            'upgrad', 'edtech', 'online learning', 'certification program',
            'degree program', 'educational platform', 'learning management'
        ]
        
        # Check source for edtech sites
        source = article.get('source', {}).get('title', '').lower()
        url = article.get('url', '').lower()
        edtech_sources = ['upgrad', 'coursera', 'udemy', 'edx', 'simplilearn', 'great learning', 'educba']
        
        # Check both source title and URL for edtech
        if any(edtech in source or edtech in url for edtech in edtech_sources):
            return False
        
        # If contains excluded topics, reject
        for excluded in excluded_topics:
            if excluded in combined:
                return False
        
        # EXCLUDE educational/tutorial articles about fraud (not industry news)
        educational_patterns = [
            'what is fraud', 'types of fraud', 'how to prevent fraud', 'fraud awareness tips',
            'fraud prevention guide', 'learn about fraud', 'understanding fraud',
            'fraud education', 'fraud training', 'fraud awareness week',
            'what type of fraud', 'different types of fraud', 'common fraud schemes',
            'fraud 101', 'fraud basics', 'introduction to fraud', 'fraud overview',
            'identity theft prevention vs', 'credit freeze vs', 'how to protect',
            'guide to', 'tips for preventing', 'steps to prevent', 'ways to avoid',
            'everything you need to know', 'complete guide to', 'beginner guide'
        ]
        
        # Check title for strong educational indicators (vs, guide, tips, how to)
        if any(pattern in title for pattern in ['vs.', 'vs ', ' guide', 'how to', 'tips for', 'ways to']):
            # If title sounds educational, be more strict
            if any(pattern in title for pattern in educational_patterns):
                return False
        
        # Check if article is educational/tutorial (not business news)
        educational_indicator_count = sum(1 for pattern in educational_patterns if pattern in combined)
        
        # If 2+ educational patterns found, likely a tutorial/guide article
        if educational_indicator_count >= 2:
            return False
        
        # Additional check: Must mention fraud/security/identity in title OR body prominently
        title_keywords = ['fraud', 'identity', 'biometric', 'breach', 'scam', 'aml', 'kyc', 'authentication', 'theft']
        has_relevant_title = any(keyword in title for keyword in title_keywords)
        
        # If no relevant keywords in title, check if body has strong fraud context (3+ keywords)
        if not has_relevant_title and keyword_count < 3:
            return False
        
        return True

    def check_strategic_importance(self, article: Dict) -> Dict[str, bool]:
        """Check if article mentions strategic events"""
        title = article.get('title', '').lower()
        body = article.get('body', '').lower()
        combined = f"{title} {body}"
        
        importance = {
            'product_launch': False,
            'innovation': False,
            'investment': False,
            'customers': False
        }
        
        for category, keywords in self.strategic_keywords.items():
            for keyword in keywords:
                if keyword in combined:
                    importance[category] = True
                    break
        
        return importance

    def is_strategically_important(self, importance: Dict[str, bool]) -> bool:
        """Check if article has any strategic importance"""
        return any(importance.values())

    def filter_strategic_articles(self) -> None:
        """Filter articles for strategic importance and group by company"""
        logger.info("🔍 Filtering for strategically important fraud industry news...")
        
        strategic_count = 0
        category_counts = defaultdict(int)
        excluded_count = 0
        
        for article in self.articles:
            # FIRST: Check if fraud industry relevant (strict filter)
            if not self.is_fraud_industry_relevant(article):
                excluded_count += 1
                continue
            
            # Identify company
            company = self.identify_company(article)
            if not company:
                continue
            
            # Check strategic importance
            importance = self.check_strategic_importance(article)
            
            if self.is_strategically_important(importance):
                article['strategic_importance'] = importance
                article['company'] = company
                self.grouped_articles[company].append(article)
                strategic_count += 1
                
                # Count categories
                for category, is_important in importance.items():
                    if is_important:
                        category_counts[category] += 1
        
        logger.info(f"✅ Found {strategic_count} strategically important FRAUD INDUSTRY articles")
        logger.info(f"❌ Excluded {excluded_count} non-fraud articles (gaming, pharma, etc.)")
        logger.info(f"📊 By category:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"   • {category}: {count}")
        logger.info(f"🏢 Companies mentioned: {len(self.grouped_articles)}")

    def generate_consolidated_document(self, output_file: Path) -> str:
        """Generate consolidated document grouped by company"""
        logger.info("📝 Generating consolidated document...")
        
        doc_lines = []
        
        # Header
        doc_lines.append("="*80)
        doc_lines.append("COMPETITIVE INTELLIGENCE REPORT")
        doc_lines.append("Strategic News - FRAUD, IDENTITY, BIOMETRIC & REGULATORY SECTORS")
        doc_lines.append("="*80)
        doc_lines.append(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
        doc_lines.append(f"Date Range: Last 30 days")
        doc_lines.append(f"Total Companies: {len(self.grouped_articles)}")
        doc_lines.append(f"Total Strategic Articles: {sum(len(articles) for articles in self.grouped_articles.values())}")
        doc_lines.append("")
        doc_lines.append("SCOPE: FRAUD INDUSTRY ONLY")
        doc_lines.append("  ✓ Fraud Detection & Prevention")
        doc_lines.append("  ✓ Identity Theft & Verification")
        doc_lines.append("  ✓ Biometric Authentication")
        doc_lines.append("  ✓ Regulatory Actions (Fraud-Related)")
        doc_lines.append("  ✓ PII Data Breaches")
        doc_lines.append("  ✓ AML & Financial Crime")
        doc_lines.append("")
        doc_lines.append("STRATEGIC FOCUS:")
        doc_lines.append("  • New Products & Services")
        doc_lines.append("  • Innovations & Technology")
        doc_lines.append("  • Investments & Funding")
        doc_lines.append("  • Major Customer Wins")
        doc_lines.append("")
        doc_lines.append("EXCLUDED: Gaming, Pharma, Cannabis, General EPS, Casinos, Crypto Trading")
        doc_lines.append("="*80)
        doc_lines.append("")
        
        # Sort companies by number of articles (most active first)
        sorted_companies = sorted(
            self.grouped_articles.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Generate content for each company
        for company, articles in sorted_companies:
            doc_lines.append("")
            doc_lines.append("="*80)
            doc_lines.append(f"COMPANY: {company}")
            doc_lines.append("="*80)
            doc_lines.append(f"Total Strategic News: {len(articles)}")
            doc_lines.append("")
            
            # Summarize strategic categories
            category_summary = defaultdict(int)
            for article in articles:
                for category, is_important in article['strategic_importance'].items():
                    if is_important:
                        category_summary[category] += 1
            
            doc_lines.append("STRATEGIC ACTIVITY BREAKDOWN:")
            for category, count in sorted(category_summary.items()):
                icon = {
                    'product_launch': '🚀',
                    'innovation': '💡',
                    'investment': '💰',
                    'customers': '🤝'
                }.get(category, '•')
                category_name = category.replace('_', ' ').title()
                doc_lines.append(f"  {icon} {category_name}: {count} articles")
            doc_lines.append("")
            doc_lines.append("-"*80)
            
            # List articles with categorization
            for i, article in enumerate(articles, 1):
                doc_lines.append(f"\n[{i}] {article.get('title', 'Untitled')}")
                doc_lines.append(f"    Source: {article.get('source', {}).get('title', 'Unknown')}")
                doc_lines.append(f"    Date: {article.get('dateTime', article.get('date', 'Unknown'))[:10]}")
                doc_lines.append(f"    URL: {article.get('url', 'N/A')}")
                
                # Show strategic categories
                categories = []
                for category, is_important in article['strategic_importance'].items():
                    if is_important:
                        categories.append(category.replace('_', ' ').title())
                doc_lines.append(f"    Categories: {', '.join(categories)}")
                doc_lines.append("")
                
                # Show FULL CONTENT (not just excerpt)
                body = article.get('body', 'No content available')
                doc_lines.append("    " + "-"*76)
                doc_lines.append("    FULL ARTICLE CONTENT:")
                doc_lines.append("    " + "-"*76)
                
                # Format content with proper indentation
                body_lines = body.split('\n')
                for line in body_lines:
                    if line.strip():  # Skip empty lines
                        doc_lines.append(f"    {line}")
                
                doc_lines.append("    " + "-"*76)
                doc_lines.append("")
        
        # Footer
        doc_lines.append("")
        doc_lines.append("="*80)
        doc_lines.append("END OF REPORT")
        doc_lines.append("="*80)
        
        # Join and save
        document = "\n".join(doc_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(document)
        
        logger.info(f"💾 Consolidated document saved: {output_file}")
        
        return document

    def assess_document_quality(self, document: str) -> Dict:
        """Use LLM to assess document quality"""
        logger.info("🤖 Assessing document quality with LLM...")
        
        if not self.classifier:
            logger.warning("⚠️ LLM classifier not available, skipping quality assessment")
            return {
                'status': 'skipped',
                'reason': 'LLM classifier not loaded'
            }
        
        try:
            # Take a sample of the document (first 2000 chars for analysis)
            sample = document[:2000]
            
            # Quality assessment labels
            quality_labels = [
                "high-quality strategic intelligence",
                "well-organized competitive analysis",
                "actionable business insights",
                "comprehensive industry overview",
                "low-quality or irrelevant content"
            ]
            
            result = self.classifier(
                sample,
                quality_labels,
                multi_label=True
            )
            
            # Parse results
            scores = dict(zip(result['labels'], result['scores']))
            
            # Determine overall quality
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            quality_assessment = {
                'status': 'completed',
                'overall_quality': top_label,
                'confidence': round(top_score, 3),
                'detailed_scores': {label: round(score, 3) for label, score in scores.items()},
                'verdict': 'PASS' if 'low-quality' not in top_label.lower() and top_score > 0.5 else 'NEEDS_REVIEW'
            }
            
            logger.info(f"✅ Quality Assessment: {quality_assessment['verdict']}")
            logger.info(f"   Overall: {top_label} (confidence: {top_score:.3f})")
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"❌ Error in quality assessment: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def generate_statistics(self) -> Dict:
        """Generate statistics about consolidated news"""
        stats = {
            'total_companies': len(self.grouped_articles),
            'total_strategic_articles': sum(len(articles) for articles in self.grouped_articles.values()),
            'companies_breakdown': {},
            'category_totals': defaultdict(int)
        }
        
        for company, articles in self.grouped_articles.items():
            stats['companies_breakdown'][company] = {
                'article_count': len(articles),
                'categories': defaultdict(int)
            }
            
            for article in articles:
                for category, is_important in article['strategic_importance'].items():
                    if is_important:
                        stats['companies_breakdown'][company]['categories'][category] += 1
                        stats['category_totals'][category] += 1
        
        return stats


def main():
    """Main function to consolidate news"""
    try:
        logger.info("="*60)
        logger.info("News Consolidator - Strategic Intelligence")
        logger.info("="*60)
        
        # Find latest Event Registry scrape folder
        reports_dir = Path("reports")
        scrape_folders = sorted(reports_dir.glob("eventregistry_scrape_*"))
        
        if not scrape_folders:
            logger.error("❌ No Event Registry scrape folders found")
            return
        
        # Use latest folder
        input_dir = scrape_folders[-1]
        logger.info(f"📁 Input directory: {input_dir}")
        
        # Initialize consolidator
        consolidator = NewsConsolidator(input_dir)
        
        # Load articles
        consolidator.load_articles()
        
        if not consolidator.articles:
            logger.error("❌ No articles to process")
            return
        
        # Filter for strategic importance
        consolidator.filter_strategic_articles()
        
        if not consolidator.grouped_articles:
            logger.warning("⚠️ No strategically important articles found")
            return
        
        # Generate consolidated document
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"reports/consolidated_strategic_news_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'strategic_intelligence_report.txt'
        document = consolidator.generate_consolidated_document(output_file)
        
        # Generate statistics
        stats = consolidator.generate_statistics()
        stats_file = output_dir / 'consolidation_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"💾 Statistics saved: {stats_file}")
        
        # Assess quality with LLM
        quality_assessment = consolidator.assess_document_quality(document)
        quality_file = output_dir / 'quality_assessment.json'
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_assessment, f, indent=2)
        logger.info(f"💾 Quality assessment saved: {quality_file}")
        
        # Final summary
        logger.info("")
        logger.info("="*60)
        logger.info("✅ CONSOLIDATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📊 Companies analyzed: {stats['total_companies']}")
        logger.info(f"📰 Strategic articles: {stats['total_strategic_articles']}")
        logger.info("")
        logger.info("📈 Strategic Categories:")
        for category, count in sorted(stats['category_totals'].items()):
            logger.info(f"   • {category.replace('_', ' ').title()}: {count}")
        logger.info("")
        logger.info("📋 Generated Files:")
        logger.info(f"   • strategic_intelligence_report.txt")
        logger.info(f"   • consolidation_statistics.json")
        logger.info(f"   • quality_assessment.json")
        logger.info("")
        
        if quality_assessment.get('status') == 'completed':
            logger.info(f"🤖 LLM Quality Verdict: {quality_assessment['verdict']}")
            logger.info(f"   Assessment: {quality_assessment['overall_quality']}")
            logger.info(f"   Confidence: {quality_assessment['confidence']:.1%}")
        
        return {
            'output_dir': str(output_dir),
            'stats': stats,
            'quality': quality_assessment
        }
        
    except Exception as e:
        logger.error(f"❌ Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
