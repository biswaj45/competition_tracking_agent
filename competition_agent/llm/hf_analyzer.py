"""
Hugging Face transformers integration for enhanced content analysis and summarization.
"""
from typing import List, Dict, Optional
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration, 
    T5Tokenizer
)
import numpy as np
import logging

# Disable TensorFlow warnings and force PyTorch backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

class HFAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Set device for all models
        use_cuda = torch.cuda.is_available()
        self.device = 0 if use_cuda else None
        self.torch_device = torch.device("cuda" if use_cuda else "cpu")
        print("Device set to use", "cuda" if use_cuda else "cpu")

        # Initialize T5 models first (PyTorch only, more stable)
        try:
            self.summary_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.summary_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            if use_cuda:
                self.summary_model = self.summary_model.to(self.torch_device)
            self.logger.info("✓ T5 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load T5 model: {e}")
            self.summary_tokenizer = None
            self.summary_model = None

        # Initialize the other models
        try:
            # Initialize summarization pipeline (may fail due to TF dependencies)
            try:
                if use_cuda:
                    self.summarizer = pipeline(
                        "summarization",
                        model="t5-small",
                        device=self.device
                    )
                else:
                    self.summarizer = pipeline(
                        "summarization",
                        model="t5-small"
                    )
            except:
                self.summarizer = None  # Pipeline might fail, but direct T5 model still works

            # Initialize sentiment analysis
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            if use_cuda:
                self.sentiment_model = self.sentiment_model.to(self.torch_device)
            
            if use_cuda:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.sentiment_model,
                    tokenizer=self.sentiment_tokenizer,
                    device=self.device
                )
            else:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.sentiment_model,
                    tokenizer=self.sentiment_tokenizer
                )

            # Initialize zero-shot classification pipeline
            if use_cuda:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=self.device
                )
            else:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )

        except Exception as e:
            self.logger.error(f"Error loading pipeline models: {str(e)}")
            # Set pipeline models to None on failure, but T5 might still work
            if not hasattr(self, 'summarizer'):
                self.summarizer = None
            if not hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer = None
                self.sentiment_tokenizer = None
                self.sentiment_model = None
            if not hasattr(self, 'classifier'):
                self.classifier = None

    def _extract_competitor_mentions(self, text: str, company: str) -> List[Dict]:
        """Extract and analyze competitor mentions in text"""
        # Simple competitor extraction - can be enhanced with NER
        competitors = []
        company_lower = company.lower()
        
        # Split into sentences for better context
        sentences = text.split('.')
        for sentence in sentences:
            if company_lower in sentence.lower():
                competitors.append({
                    'competitor_name': company,
                    'mention_context': sentence.strip(),
                    'high_impact': False  # Will be set in analyze_content
                })
        
        return competitors

    def analyze_content(self, text: str, company: str, keywords: List[str]) -> Dict:
        """
        Analyze content using transformer models
        
        Args:
            text: Content text to analyze
            company: Company name
            keywords: Relevant keywords
            
        Returns:
            Dictionary containing analysis results
        """
        # Initialize results
        results = {
            "summary": "",
            "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
            "key_quotes": [],
            "features": [],
            "impact_level": "low",
            "market_positioning": "",
            "competitive_advantages": []
        }
        
        if not text:
            return results

        # Generate summary if text is long enough
        if len(text.split()) > 50 and self.summarizer:
            try:
                summary = self.summarizer(
                    text, 
                    max_length=130, 
                    min_length=30, 
                    do_sample=False
                )
                results["summary"] = summary[0]["summary_text"]
            except Exception as e:
                self.logger.warning(f"Error generating summary: {str(e)}")

        # Analyze sentiment
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text[:512])[0]  # Use first 512 chars for sentiment
                results["sentiment"] = {
                    "polarity": sentiment["score"] if sentiment["label"] == "POSITIVE" else -sentiment["score"],
                    "subjectivity": abs(sentiment["score"] - 0.5) * 2  # Convert to 0-1 range
                }
            except Exception as e:
                self.logger.warning(f"Error analyzing sentiment: {str(e)}")

        # Extract key quotes (sentences containing company name or keywords)
        sentences = text.split('.')
        key_quotes = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (company.lower() in sentence.lower() or 
                any(kw.lower() in sentence.lower() for kw in keywords)):
                key_quotes.append(sentence)
        results["key_quotes"] = key_quotes[:3]  # Top 3 relevant quotes

        # Identify impact level based on content analysis
        impact_score = 0
        impact_score += len(key_quotes) * 0.2  # More relevant quotes = higher impact
        impact_score += abs(results["sentiment"]["polarity"]) * 0.3  # Stronger sentiment = higher impact
        impact_score += sum(1 for kw in keywords if kw.lower() in text.lower()) * 0.1  # Keyword matches

        if impact_score > 0.7:
            results["impact_level"] = "high"
        elif impact_score > 0.3:
            results["impact_level"] = "medium"

        return results

    def generate_article_summary(self, article: Dict) -> str:
        """
        Generate a comprehensive summary of a news article
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            String containing the formatted summary
        """
        if not article.get("content"):
            return ""

        try:
            # Get base summary
            if self.summarizer:
                summary = self.summarizer(
                    article["content"],
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]["summary_text"]
            else:
                # Fallback to first few sentences
                sentences = article["content"].split('.')[:3]
                summary = '. '.join(sentences)

            # Format the summary with metadata
            formatted_summary = f"""
Title: {article['title']}
Date: {article['published_date']}
Source: {article['source']}

Summary:
{summary}

Key Points:
"""
            # Add key points if available
            if "key_quotes" in article:
                for quote in article["key_quotes"]:
                    formatted_summary += f"- {quote}\n"

            # Add sentiment if available
            if "sentiment" in article:
                sentiment = article["sentiment"]
                sentiment_str = "Positive" if sentiment["polarity"] > 0 else "Negative"
                formatted_summary += f"\nSentiment: {sentiment_str} ({abs(sentiment['polarity']):.2f})"

            return formatted_summary

        except Exception as e:
            self.logger.error(f"Error generating article summary: {str(e)}")
            return article.get("summary", "Summary generation failed")

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
            # Truncate text to a reasonable length
            text = text[:1000]  # Limit to first 1000 characters for faster processing
            print(f"\nAnalyzing content for {company}...")
            
            print("1/6 Analyzing sentiment...")
            sentiment = self._analyze_sentiment(text)
            
            print("2/6 Classifying impact...")
            impact_level = self._classify_impact(text)
            
            print("3/6 Extracting features...")
            features = self._extract_features(text)
            
            print("4/6 Extracting key quotes...")
            key_quotes = self._extract_key_quotes(text)
            
            print("5/6 Analyzing market positioning...")
            market_positioning = self._analyze_positioning(text, company)
            
            print("6/6 Identifying competitive advantages...")
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
        try:
            inputs = self.sentiment_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            if self.torch_device.type == "cuda":
                inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            
            outputs = self.sentiment_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Convert to polarity score (-1 to 1)
            polarity = (scores[0][1] - scores[0][0]).item()
            
            return {
                "polarity": polarity,
                "subjectivity": abs(polarity)  # Use magnitude as subjectivity
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"polarity": 0.0, "subjectivity": 0.0}

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
        try:
            inputs = self.summary_tokenizer(
                "extract key points: " + text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            if self.torch_device.type == "cuda":
                inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            
            output_ids = self.summary_model.generate(
                inputs["input_ids"],
                max_length=150,
                num_beams=2,
                early_stopping=True
            )
            
            quotes = self.summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return quotes.split(". ")[:3]  # Return top 3 sentences
        except Exception as e:
            self.logger.error(f"Error extracting key quotes: {str(e)}")
            return []

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