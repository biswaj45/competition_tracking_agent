"""
Hugging Face transformers integration for enhanced content analysis and summarization.
"""
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)

class HFAnalyzer:
    def __init__(self):
        """Initialize the Hugging Face models"""
        # Initialize sentiment analysis
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
        # Initialize summarization
        self.summary_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        # Initialize zero-shot classification for impact and features
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_model.to(self.device)
        self.summary_model.to(self.device)

    def analyze_content(self, text: str, company: str, keywords: List[str]) -> Dict:
        """
        Analyze content using Hugging Face models for enhanced understanding
        
        Args:
            text: Content text to analyze
            company: Company name for context
            keywords: List of relevant keywords
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Analyze sentiment
            sentiment = self._analyze_sentiment(text)
            
            # Classify impact level
            impact_level = self._classify_impact(text)
            
            # Extract features using zero-shot classification
            features = self._extract_features(text)
            
            # Extract key quotes through summarization
            key_quotes = self._extract_key_quotes(text)
            
            # Analyze market positioning
            market_positioning = self._analyze_positioning(text, company)
            
            # Identify competitive advantages
            competitive_advantages = self._identify_advantages(text, company)
            
            return {
                "features": features,
                "market_positioning": market_positioning,
                "competitive_advantages": competitive_advantages,
                "impact_level": impact_level,
                "sentiment": sentiment,
                "key_quotes": key_quotes
            }
            
        except Exception as e:
            print(f"Error in HF analysis: {str(e)}")
            return self._empty_analysis()
    
    def generate_summary(self, articles: List[Dict], competitor_type: str) -> str:
        """
        Generate a natural language summary of competitor activities
        
        Args:
            articles: List of article dictionaries
            competitor_type: Type of competitors (established/startups)
            
        Returns:
            Generated summary text
        """
        try:
            # Combine article texts with proper formatting
            combined_text = self._prepare_text_for_summary(articles, competitor_type)
            
            # Generate summary using T5
            inputs = self.summary_tokenizer.encode(
                "summarize: " + combined_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            summary_ids = self.summary_model.generate(
                inputs,
                max_length=300,
                min_length=100,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Summary generation failed. Please check the logs for details."

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using DistilBERT"""
        inputs = self.sentiment_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Convert to polarity score (-1 to 1)
        polarity = (scores[0][1] - scores[0][0]).item()
        
        return {
            "polarity": polarity,
            "subjectivity": abs(polarity)  # Use magnitude as subjectivity
        }

    def _classify_impact(self, text: str) -> str:
        """Classify impact level using zero-shot classification"""
        candidate_labels = ["major development", "minor update", "routine news"]
        result = self.classifier(text, candidate_labels)
        
        # Map classification to impact levels
        label_map = {
            "major development": "high",
            "minor update": "medium",
            "routine news": "low"
        }
        
        return label_map[result["labels"][0]]

    def _extract_features(self, text: str) -> List[str]:
        """Extract features using zero-shot classification"""
        feature_categories = [
            "new product launch",
            "feature update",
            "technology improvement",
            "service expansion",
            "partnership announcement"
        ]
        
        result = self.classifier(text, feature_categories)
        
        # Return categories with confidence > 0.3
        features = [
            label for label, score in zip(result["labels"], result["scores"])
            if score > 0.3
        ]
        
        return features

    def _extract_key_quotes(self, text: str) -> List[str]:
        """Extract key quotes using T5 summarization"""
        inputs = self.summary_tokenizer.encode(
            "extract key points: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        output_ids = self.summary_model.generate(
            inputs,
            max_length=150,
            num_beams=2,
            early_stopping=True
        )
        
        quotes = self.summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return quotes.split(". ")[:3]  # Return top 3 sentences

    def _analyze_positioning(self, text: str, company: str) -> str:
        """Analyze market positioning using zero-shot classification"""
        positioning_categories = [
            "market leader",
            "innovator",
            "challenger",
            "niche player"
        ]
        
        result = self.classifier(text, positioning_categories)
        return result["labels"][0]

    def _identify_advantages(self, text: str, company: str) -> List[str]:
        """Identify competitive advantages using zero-shot classification"""
        advantage_categories = [
            "technological superiority",
            "market presence",
            "customer satisfaction",
            "innovation",
            "cost efficiency"
        ]
        
        result = self.classifier(text, advantage_categories)
        
        # Return advantages with confidence > 0.3
        advantages = [
            label for label, score in zip(result["labels"], result["scores"])
            if score > 0.3
        ]
        
        return advantages

    def _prepare_text_for_summary(self, articles: List[Dict], competitor_type: str) -> str:
        """Prepare article texts for summarization"""
        texts = []
        for article in articles:
            texts.append(
                f"Title: {article['title']}\n"
                f"Content: {article['content'][:500]}..."  # Truncate for length
            )
        
        return f"Summarize {competitor_type} competitor activities:\n\n" + "\n\n".join(texts)

    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "features": [],
            "market_positioning": "No analysis available",
            "competitive_advantages": [],
            "impact_level": "low",
            "sentiment": {"polarity": 0, "subjectivity": 0},
            "key_quotes": []
        }