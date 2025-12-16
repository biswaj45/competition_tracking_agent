#!/usr/bin/env python3
"""
arXiv Research Paper Scraper for Fraud Analytics Innovation
Extracts innovative research abstracts focusing on fraud detection and analytics
Filters for novel methods, not survey/compilation papers
Selects top 10 based on relevance with LLM quality judgment
"""
import requests
import xml.etree.ElementTree as ET
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time
import re
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivFraudResearchScraper:
    def __init__(self):
        """Initialize arXiv scraper for fraud analytics research"""
        self.base_url = "http://export.arxiv.org/api/query"
        self.papers_collected = []
        
        # Search queries focused on fraud analytics innovation
        self.search_queries = [
            # Transaction and payment fraud (HIGH PRIORITY)
            'transaction fraud AND (detection OR prevention)',
            'payment fraud AND (detection OR machine learning)',
            'credit card fraud AND (detection OR deep learning)',
            'financial fraud AND (banking OR payment)',
            
            # Biometric fraud (HIGH PRIORITY)
            'biometric fraud AND (detection OR liveness OR spoofing)',
            'facial recognition AND (fraud OR spoofing)',
            'fingerprint AND (fraud OR liveness detection)',
            
            # Identity fraud (HIGH PRIORITY)
            'identity fraud AND (detection OR verification)',
            'synthetic identity AND (fraud OR detection)',
            'identity theft AND (detection OR prevention)',
            
            # Banking/Financial applications
            'fraud detection AND (banking OR financial institution)',
            'AML AND (detection OR money laundering)',
            
            # Advanced techniques
            'graph neural network AND fraud',
            'federated learning AND fraud detection',
            'explainable AI AND fraud detection'
        ]
        
        # Relevance scoring weights
        self.topic_weights = {
            'transaction': 3.0,
            'payment': 3.0,
            'banking': 2.5,
            'financial': 2.5,
            'biometric': 2.5,
            'identity': 2.5,
            'credit card': 2.0,
            'fraud detection': 1.5,
            'money laundering': 1.5,
            'aml': 1.5,
            'synthetic identity': 2.0,
            'liveness': 2.0,
            'spoofing': 2.0
        }
        
        # Initialize LLM classifier for quality judgment
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
        
        # Keywords indicating innovation (not surveys)
        self.innovation_keywords = [
            'propose', 'novel', 'new method', 'introduce', 'present',
            'framework', 'architecture', 'algorithm', 'approach', 'model',
            'technique', 'system', 'develop', 'design', 'implement'
        ]
        
        # Keywords to filter out (surveys, reviews, compilations)
        self.filter_out_keywords = [
            'survey', 'review', 'overview', 'comparative study',
            'systematic review', 'literature review', 'state of the art',
            'taxonomy', 'comprehensive study', 'meta-analysis'
        ]

    def score_paper_relevance(self, paper: Dict) -> Dict:
        """
        Score paper relevance based on transaction fraud, biometric, identity focus
        with higher weight for payment/banking/financial industry context
        
        Args:
            paper: Paper dictionary with title and abstract
            
        Returns:
            Dict with relevance_score and breakdown
        """
        text = f"{paper['title']} {paper['abstract']}".lower()
        score = 0.0
        breakdown = {}
        
        # Score based on topic keywords with weights
        for keyword, weight in self.topic_weights.items():
            if keyword in text:
                score += weight
                breakdown[keyword] = weight
        
        # Bonus for banking/payment app context (high priority)
        banking_keywords = [
            'banking', 'bank', 'financial institution', 'payment app',
            'fintech', 'mobile payment', 'digital payment', 'payment system',
            'payment processing', 'payment gateway'
        ]
        banking_bonus = sum(1.0 for kw in banking_keywords if kw in text)
        if banking_bonus > 0:
            score += banking_bonus
            breakdown['banking_context_bonus'] = banking_bonus
        
        # Bonus for real-world application mentions
        application_keywords = [
            'real-world', 'production', 'deployment', 'industry', 'commercial',
            'enterprise', 'large-scale', 'real-time'
        ]
        app_bonus = sum(0.5 for kw in application_keywords if kw in text)
        if app_bonus > 0:
            score += app_bonus
            breakdown['application_bonus'] = app_bonus
        
        return {
            'relevance_score': round(score, 2),
            'breakdown': breakdown
        }
    
    def assess_paper_quality_with_llm(self, paper: Dict) -> Dict:
        """
        Use LLM to assess research paper quality
        
        Args:
            paper: Paper dictionary with title and abstract
            
        Returns:
            Dict with quality_score, verdict, and confidence
        """
        if not self.classifier:
            return {
                'quality_score': 0.5,
                'verdict': 'not_assessed',
                'confidence': 0.0,
                'reason': 'LLM classifier not available'
            }
        
        try:
            # Create assessment text
            text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            
            # Quality dimensions
            quality_labels = [
                "high research quality with novel contributions",
                "practical applicability to fraud detection systems",
                "strong experimental validation and results",
                "addresses real-world fraud challenges",
                "incremental improvement or limited scope"
            ]
            
            result = self.classifier(
                text[:1000],  # Limit to avoid token overflow
                quality_labels,
                multi_label=False
            )
            
            # Map verdict to quality score
            verdict = result['labels'][0]
            confidence = result['scores'][0]
            
            # Higher score for practical, high-quality, validated research
            quality_mapping = {
                "high research quality with novel contributions": 1.0,
                "practical applicability to fraud detection systems": 0.95,
                "strong experimental validation and results": 0.9,
                "addresses real-world fraud challenges": 0.85,
                "incremental improvement or limited scope": 0.5
            }
            
            quality_score = quality_mapping.get(verdict, 0.5)
            
            return {
                'quality_score': quality_score,
                'verdict': verdict,
                'confidence': round(confidence, 4),
                'all_scores': {label: score for label, score in zip(result['labels'], result['scores'])}
            }
            
        except Exception as e:
            logger.error(f"❌ LLM quality assessment error: {e}")
            return {
                'quality_score': 0.5,
                'verdict': 'assessment_failed',
                'confidence': 0.0,
                'reason': str(e)
            }
    
    def select_top_papers(self, papers: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Score all papers, assess quality with LLM, and select top N
        
        Args:
            papers: List of paper dictionaries
            top_n: Number of top papers to select (default: 10)
            
        Returns:
            List of top N papers with scores and rankings
        """
        logger.info(f"\n🎯 Scoring and ranking {len(papers)} papers...")
        
        scored_papers = []
        for idx, paper in enumerate(papers):
            # Calculate relevance score
            relevance_result = self.score_paper_relevance(paper)
            
            # Assess quality with LLM
            quality_result = self.assess_paper_quality_with_llm(paper)
            
            # Combined score: relevance * quality
            combined_score = relevance_result['relevance_score'] * quality_result['quality_score']
            
            scored_paper = {
                **paper,
                'relevance_score': relevance_result['relevance_score'],
                'relevance_breakdown': relevance_result['breakdown'],
                'quality_score': quality_result['quality_score'],
                'quality_verdict': quality_result['verdict'],
                'quality_confidence': quality_result['confidence'],
                'combined_score': round(combined_score, 2)
            }
            
            scored_papers.append(scored_paper)
            
            # Progress log every 10 papers
            if (idx + 1) % 10 == 0:
                logger.info(f"   Processed {idx + 1}/{len(papers)} papers...")
        
        # Sort by combined score (descending)
        scored_papers.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add ranking
        for rank, paper in enumerate(scored_papers[:top_n], 1):
            paper['ranking'] = rank
        
        # Log top papers
        logger.info(f"\n🏆 TOP {top_n} PAPERS SELECTED:")
        for paper in scored_papers[:top_n]:
            logger.info(
                f"   #{paper['ranking']}: {paper['title'][:80]}... "
                f"(Score: {paper['combined_score']}, "
                f"Relevance: {paper['relevance_score']}, "
                f"Quality: {paper['quality_score']})"
            )
        
        return scored_papers[:top_n]

    def search_arxiv(self, query: str, max_results: int = 50, days_back: int = 365) -> List[Dict]:
        """Search arXiv for papers matching the query"""
        try:
            # Calculate date range (arXiv uses submittedDate) - use timezone-aware datetime
            from datetime import timezone
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            
            # Build query parameters
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            logger.info(f"🔍 Searching arXiv for: '{query}'")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                papers = self.parse_arxiv_response(response.text)
                
                # Filter by date
                filtered_papers = []
                for paper in papers:
                    pub_date = datetime.fromisoformat(paper.get('published', '').replace('Z', '+00:00'))
                    if pub_date >= start_date:
                        filtered_papers.append(paper)
                
                logger.info(f"✅ Found {len(filtered_papers)} papers from last {days_back} days")
                return filtered_papers
            else:
                logger.error(f"❌ arXiv API Error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error searching arXiv: {e}")
            return []

    def parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """Parse arXiv API XML response"""
        papers = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_text)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            # Extract entries
            for entry in root.findall('atom:entry', ns):
                paper = {}
                
                # Basic info
                paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                paper['abstract'] = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                paper['published'] = entry.find('atom:published', ns).text.strip()
                paper['updated'] = entry.find('atom:updated', ns).text.strip()
                
                # arXiv ID and link
                paper['id'] = entry.find('atom:id', ns).text.strip()
                paper['arxiv_id'] = paper['id'].split('/abs/')[-1]
                paper['pdf_link'] = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
                paper['abs_link'] = paper['id']
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    author_name = author.find('atom:name', ns).text.strip()
                    authors.append(author_name)
                paper['authors'] = authors
                
                # Categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    categories.append(category.get('term'))
                paper['categories'] = categories
                
                # Primary category
                primary_cat = entry.find('arxiv:primary_category', ns)
                if primary_cat is not None:
                    paper['primary_category'] = primary_cat.get('term')
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"❌ Error parsing XML: {e}")
            return []

    def is_innovation_paper(self, paper: Dict) -> bool:
        """Check if paper presents innovation (not survey/review)"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        combined = f"{title} {abstract}"
        
        # Filter out surveys/reviews
        for keyword in self.filter_out_keywords:
            if keyword in combined:
                logger.debug(f"Filtered out (survey/review): {paper.get('title', '')[:60]}...")
                return False
        
        # Check for innovation keywords
        innovation_score = 0
        for keyword in self.innovation_keywords:
            if keyword in combined:
                innovation_score += 1
        
        # Must have at least 2 innovation keywords
        if innovation_score >= 2:
            return True
        
        return False

    def is_fraud_relevant(self, paper: Dict) -> bool:
        """Check if paper is relevant to fraud analytics"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        combined = f"{title} {abstract}"
        
        fraud_keywords = [
            'fraud', 'fraudulent', 'anomaly detection', 'outlier detection',
            'money laundering', 'anti-money laundering', 'aml',
            'identity theft', 'synthetic identity', 'deepfake',
            'payment fraud', 'credit card fraud', 'transaction fraud',
            'financial crime', 'cybercrime', 'phishing'
        ]
        
        for keyword in fraud_keywords:
            if keyword in combined:
                return True
        
        return False

    def save_papers(self, papers: List[Dict], output_dir: Path) -> None:
        """Save papers to JSON and individual text files"""
        if not papers:
            logger.warning("⚠️ No papers to save")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all papers to JSON with metadata
        json_file = output_dir / 'arxiv_fraud_research_top10.json'
        papers_with_metadata = {
            'metadata': {
                'total_papers': len(papers),
                'scraped_at': datetime.now().isoformat(),
                'source': 'arXiv.org',
                'focus': 'TOP 10 Innovative fraud analytics research papers',
                'selection_criteria': {
                    'scoring': 'Weighted by transaction fraud, biometric, identity topics',
                    'priority': 'Payment app, banking, financial industry context',
                    'quality_assessment': 'LLM-based quality judgment',
                    'filters': [
                        'Innovation-type papers only (not surveys/reviews)',
                        'Fraud analytics relevance',
                        'Last 30 days publications',
                        'Weighted relevance scoring',
                        'LLM quality assessment'
                    ]
                }
            },
            'papers': papers
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(papers_with_metadata, f, indent=2, ensure_ascii=True)
        logger.info(f"💾 Saved TOP 10 papers JSON: {json_file}")
        
        # Save individual abstract files with ranking
        abstracts_dir = output_dir / 'abstracts'
        abstracts_dir.mkdir(exist_ok=True)
        
        for paper in papers:
            # Get ranking if available
            ranking = paper.get('ranking', 'N/A')
            
            # Clean filename with ranking prefix
            title = paper.get('title', f'Paper_{ranking}')
            filename = "".join(c for c in title[:80] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"RANK{ranking:02d}_{filename.replace(' ', '_')}.txt"
            
            txt_file = abstracts_dir / filename
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"🏆 RANKING: #{ranking}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Title: {paper.get('title', 'N/A')}\n")
                f.write(f"{'='*80}\n\n")
                
                # Show scores if available
                if 'combined_score' in paper:
                    f.write(f"📊 SCORES:\n")
                    f.write(f"Combined Score: {paper.get('combined_score', 'N/A')}\n")
                    f.write(f"Relevance Score: {paper.get('relevance_score', 'N/A')}\n")
                    f.write(f"Quality Score: {paper.get('quality_score', 'N/A')}\n")
                    f.write(f"Quality Verdict: {paper.get('quality_verdict', 'N/A')}\n")
                    f.write(f"Quality Confidence: {paper.get('quality_confidence', 'N/A')}\n")
                    
                    if 'relevance_breakdown' in paper and paper['relevance_breakdown']:
                        f.write(f"\nRelevance Breakdown:\n")
                        for key, value in paper['relevance_breakdown'].items():
                            f.write(f"  • {key}: {value}\n")
                    f.write(f"\n{'='*80}\n\n")
                
                f.write(f"arXiv ID: {paper.get('arxiv_id', 'N/A')}\n")
                f.write(f"Published: {paper.get('published', 'N/A')[:10]}\n")
                f.write(f"Updated: {paper.get('updated', 'N/A')[:10]}\n")
                f.write(f"\nAuthors: {', '.join(paper.get('authors', ['N/A']))}\n")
                f.write(f"\nCategories: {', '.join(paper.get('categories', ['N/A']))}\n")
                f.write(f"Primary Category: {paper.get('primary_category', 'N/A')}\n")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"LINKS:\n")
                f.write(f"{'='*80}\n")
                f.write(f"Abstract: {paper.get('abs_link', 'N/A')}\n")
                f.write(f"PDF: {paper.get('pdf_link', 'N/A')}\n")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"ABSTRACT:\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"{paper.get('abstract', 'N/A')}\n")
        
        logger.info(f"💾 Saved {len(papers)} ranked abstract files to: {abstracts_dir}")
        
        # Create consolidated file with all abstracts
        consolidated_file = output_dir / 'all_abstracts_consolidated.txt'
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TOP 10 FRAUD ANALYTICS RESEARCH PAPERS - CONSOLIDATED REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}\n")
            f.write(f"Date Range: Last 30 days\n")
            f.write(f"Total Papers: {len(papers)}\n")
            f.write("="*80 + "\n\n")
            
            for ranking, paper in enumerate(papers, 1):
                f.write("\n" + "="*80 + "\n")
                f.write(f"🏆 PAPER #{ranking}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Title: {paper.get('title', 'N/A')}\n")
                f.write(f"{'='*80}\n\n")
                
                # Show scores if available
                if 'combined_score' in paper:
                    f.write(f"📊 SCORES:\n")
                    f.write(f"Combined Score: {paper.get('combined_score', 'N/A')}\n")
                    f.write(f"Relevance Score: {paper.get('relevance_score', 'N/A')}\n")
                    f.write(f"Quality Score: {paper.get('quality_score', 'N/A')}\n")
                    f.write(f"Quality Verdict: {paper.get('quality_verdict', 'N/A')}\n")
                    f.write(f"Quality Confidence: {paper.get('quality_confidence', 'N/A')}\n")
                    
                    if 'relevance_breakdown' in paper and paper['relevance_breakdown']:
                        f.write(f"\nRelevance Breakdown:\n")
                        for key, value in paper['relevance_breakdown'].items():
                            f.write(f"  • {key}: {value}\n")
                    f.write(f"\n{'='*80}\n\n")
                
                f.write(f"arXiv ID: {paper.get('arxiv_id', 'N/A')}\n")
                f.write(f"Published: {paper.get('published', 'N/A')[:10]}\n")
                f.write(f"Updated: {paper.get('updated', 'N/A')[:10]}\n")
                f.write(f"\nAuthors: {', '.join(paper.get('authors', ['N/A']))}\n")
                f.write(f"\nCategories: {', '.join(paper.get('categories', ['N/A']))}\n")
                f.write(f"Primary Category: {paper.get('primary_category', 'N/A')}\n")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"LINKS:\n")
                f.write(f"{'='*80}\n")
                f.write(f"Abstract: {paper.get('abs_link', 'N/A')}\n")
                f.write(f"PDF: {paper.get('pdf_link', 'N/A')}\n")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"ABSTRACT:\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"{paper.get('abstract', 'N/A')}\n")
                f.write("\n\n")
        
        logger.info(f"💾 Saved consolidated file: {consolidated_file}")

    def scrape_all_queries(self, max_per_query: int = 50, days_back: int = 365) -> Dict:
        """Scrape papers for all queries"""
        all_papers = []
        
        logger.info(f"🚀 Starting arXiv Research Scraping")
        logger.info(f"📅 Date range: Last {days_back} days")
        logger.info(f"🔍 Total queries: {len(self.search_queries)}")
        logger.info(f"🎯 Focus: Innovative fraud analytics research (not surveys)")
        
        for i, query in enumerate(self.search_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Query {i}/{len(self.search_queries)}: {query}")
            logger.info(f"{'='*60}")
            
            papers = self.search_arxiv(query, max_per_query, days_back)
            
            if papers:
                all_papers.extend(papers)
                # Rate limiting - arXiv asks for 3 second delay
                time.sleep(3)
        
        # Remove duplicates based on arXiv ID
        unique_papers = []
        seen_ids = set()
        
        for paper in all_papers:
            paper_id = paper.get('arxiv_id', paper.get('id'))
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
        
        # Filter for innovation papers
        logger.info(f"\n🔍 Filtering for innovative research papers...")
        innovation_papers = [p for p in unique_papers if self.is_innovation_paper(p)]
        
        # Filter for fraud relevance
        logger.info(f"🔍 Filtering for fraud analytics relevance...")
        fraud_innovation_papers = [p for p in innovation_papers if self.is_fraud_relevant(p)]
        
        # Select TOP 10 papers using weighted scoring and LLM quality assessment
        logger.info(f"\n🔍 Selecting TOP 10 papers based on weighted scoring...")
        top_papers = self.select_top_papers(fraud_innovation_papers, top_n=10)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 SCRAPING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total papers fetched: {len(all_papers)}")
        logger.info(f"Unique papers: {len(unique_papers)}")
        logger.info(f"Innovation papers (not surveys): {len(innovation_papers)}")
        logger.info(f"Fraud-relevant innovation papers: {len(fraud_innovation_papers)}")
        logger.info(f"🏆 TOP 10 selected papers: {len(top_papers)}")
        logger.info(f"Queries executed: {len(self.search_queries)}")
        
        return {
            'papers': top_papers,
            'metadata': {
                'total_papers': len(top_papers),
                'unique_papers': len(unique_papers),
                'raw_papers': len(all_papers),
                'fraud_relevant_papers': len(fraud_innovation_papers),
                'queries_executed': len(self.search_queries),
                'date_range_days': days_back,
                'scraped_at': datetime.now().isoformat(),
                'source': 'arXiv.org',
                'focus': 'TOP 10 Innovative fraud analytics research papers',
                'selection_criteria': {
                    'scoring': 'Weighted by transaction fraud, biometric, identity topics',
                    'priority': 'Payment app, banking, financial industry context',
                    'quality_assessment': 'LLM-based quality judgment',
                    'filters': [
                        'Innovation-type papers only (not surveys/reviews)',
                        'Fraud analytics relevance',
                        'Last 30 days publications',
                        'Weighted relevance scoring',
                        'LLM quality assessment'
                    ]
                }
            }
        }


def main():
    """Main function to run arXiv research scraping"""
    try:
        logger.info("="*60)
        logger.info("arXiv Fraud Analytics Research Scraper")
        logger.info("="*60)
        
        # Initialize scraper
        scraper = ArxivFraudResearchScraper()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"research/arxiv_fraud_analytics_{timestamp}")
        
        # Scrape research papers (last 30 days, max 50 per query)
        results = scraper.scrape_all_queries(max_per_query=50, days_back=30)
        
        # Save results
        if results['papers']:
            scraper.save_papers(results['papers'], output_dir)
            
            logger.info(f"\n✅ SCRAPING COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Results saved to: {output_dir}")
            logger.info(f"📊 Total innovative fraud analytics papers: {results['metadata']['total_papers']}")
            logger.info(f"📅 Date range: Last {results['metadata']['date_range_days']} days")
            logger.info(f"\n📖 Files generated:")
            logger.info(f"   • arxiv_fraud_research.json (all papers with metadata)")
            logger.info(f"   • abstracts/ folder ({len(results['papers'])} individual files)")
        else:
            logger.warning("⚠️ No papers were retrieved")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
