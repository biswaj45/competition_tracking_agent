"""
Enhanced content filtering and relevance detection with Hugging Face models
"""
from typing import List, Set, Dict, Optional
import re
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import nltk
from ..llm.hf_analyzer import HFAnalyzer

class ContentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Initialize Hugging Face analyzer
        try:
            self.hf_analyzer = HFAnalyzer()
            self.use_transformer = True
            print("Initialized Hugging Face models successfully")
        except Exception as e:
            print(f"Warning: Hugging Face initialization failed: {str(e)}")
            print("Falling back to basic analysis")
            self.use_transformer = False
            
    def analyze_content(self, text: str, company: str, keywords: List[str]) -> Dict:
        """
        Analyze content using Hugging Face models for enhanced understanding
        
        Args:
            text: Content text to analyze
            company: Company name to check relevance for
            keywords: List of relevant keywords
            
        Returns:
            Dictionary containing analysis results
        """
        if not text:
            return self._empty_analysis()
            
        # Clean and normalize text
        clean_text = self._clean_text(text)
        
        if self.use_llm:
            # Use LLM for enhanced analysis
            llm_analysis = self.llm_analyzer.analyze_content(clean_text, company, keywords)
            
            # Calculate relevance score (combine LLM and traditional methods)
            basic_relevance = self._calculate_relevance(clean_text, company, keywords)
            
            # Combine both analyses
            return {
                "relevance_score": (basic_relevance + 
                    (1 if llm_analysis["impact_level"] == "high" else 
                     0.5 if llm_analysis["impact_level"] == "medium" else 0.2)
                ) / 2,  # Average of both scores
                "sentiment": llm_analysis["sentiment"],
                "key_sentences": llm_analysis["key_quotes"],
                "features": llm_analysis["features"],
                "market_positioning": llm_analysis.get("market_positioning", ""),
                "competitive_advantages": llm_analysis.get("competitive_advantages", []),
                "impact_level": llm_analysis["impact_level"]
            }
        else:
            # Fallback to basic analysis
            relevance = self._calculate_relevance(clean_text, company, keywords)
            sentiment = self._analyze_sentiment(clean_text)
            key_sentences = self._extract_key_sentences(clean_text, company, keywords)
            features = self._identify_features(clean_text)
            
            return {
                "relevance_score": relevance,
                "sentiment": sentiment,
                "key_sentences": key_sentences,
                "features": features,
                "market_positioning": "",
                "competitive_advantages": [],
                "impact_level": "medium"  # Default to medium without LLM
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.?!]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_relevance(self, text: str, company: str, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keyword presence and context
        
        Returns:
            Float between 0 and 1 indicating relevance
        """
        text = text.lower()
        company = company.lower()
        
        # Check company name presence (direct and variations)
        company_parts = company.split()
        company_presence = sum(1 for part in company_parts if part in text)
        company_score = company_presence / len(company_parts)
        
        # Check keywords presence
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
        keyword_score = keyword_matches / len(keywords)
        
        # Calculate proximity of company name to keywords
        proximity_score = self._calculate_proximity(text, company, keywords)
        
        # Weighted average of scores
        relevance = (0.4 * company_score + 
                    0.3 * keyword_score + 
                    0.3 * proximity_score)
        
        return round(min(1.0, relevance), 2)
    
    def _calculate_proximity(self, text: str, company: str, keywords: List[str]) -> float:
        """Calculate how close keywords appear to company mentions"""
        sentences = sent_tokenize(text)
        max_proximity = 0
        
        for sentence in sentences:
            if company in sentence.lower():
                # Check if any keyword is in the same sentence
                keyword_presence = any(kw.lower() in sentence.lower() for kw in keywords)
                if keyword_presence:
                    max_proximity = 1.0
                    break
        
        return max_proximity
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the content"""
        blob = TextBlob(text)
        
        return {
            "polarity": round(blob.sentiment.polarity, 2),  # -1 to 1
            "subjectivity": round(blob.sentiment.subjectivity, 2)  # 0 to 1
        }
    
    def _extract_key_sentences(self, text: str, company: str, keywords: List[str]) -> List[str]:
        """Extract most relevant sentences from the content"""
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            # Check if sentence contains company name or keywords
            if (company.lower() in sentence.lower() or 
                any(kw.lower() in sentence.lower() for kw in keywords)):
                relevant_sentences.append(sentence.strip())
        
        return relevant_sentences[:3]  # Return top 3 most relevant sentences
    
    def _identify_features(self, text: str) -> List[str]:
        """Identify mentioned product features or capabilities"""
        feature_patterns = [
            r'launch(?:ed|es|ing)?\s+(?:new\s+)?([^.!?]+)',
            r'introduc(?:ed|es|ing)\s+(?:new\s+)?([^.!?]+)',
            r'new\s+(?:feature|capability|solution|product):\s*([^.!?]+)',
            r'announc(?:ed|es|ing)\s+([^.!?]+)'
        ]
        
        features = []
        for pattern in feature_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            features.extend(match.group(1).strip() for match in matches)
        
        return list(set(features))  # Remove duplicates
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "relevance_score": 0.0,
            "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
            "key_sentences": [],
            "features": []
        }