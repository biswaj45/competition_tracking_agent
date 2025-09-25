"""
OpenAI GPT integration for enhanced content analysis and summarization.
"""
import os
from typing import List, Dict, Optional
import openai
from ..config import LLM_CONFIG

class LLMAnalyzer:
    def __init__(self):
        """Initialize the LLM analyzer with OpenAI credentials"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        openai.api_key = self.api_key
        self.model = LLM_CONFIG.get('model', 'gpt-3.5-turbo')
        self.max_tokens = LLM_CONFIG.get('max_tokens', 500)
        self.temperature = LLM_CONFIG.get('temperature', 0.3)

    def analyze_content(self, text: str, company: str, keywords: List[str]) -> Dict:
        """
        Analyze content using GPT for enhanced understanding and feature extraction
        
        Args:
            text: Content text to analyze
            company: Company name for context
            keywords: List of relevant keywords
            
        Returns:
            Dictionary containing analysis results
        """
        prompt = self._build_analysis_prompt(text, company, keywords)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert competitive intelligence analyst. Analyze the given content and extract key insights."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            analysis = self._parse_llm_response(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
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
        prompt = self._build_summary_prompt(articles, competitor_type)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert at summarizing competitive intelligence. Generate a concise, insightful summary of competitor activities."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Summary generation failed: {str(e)}"

    def _build_analysis_prompt(self, text: str, company: str, keywords: List[str]) -> str:
        """Build prompt for content analysis"""
        return f"""
        Analyze the following content about {company}, focusing on these aspects:
        - Key features or capabilities mentioned
        - Market positioning and strategy
        - Competitive advantages
        - Business impact level (high/medium/low)
        - Overall sentiment
        
        Relevant keywords: {', '.join(keywords)}
        
        Content:
        {text}
        
        Please provide analysis in JSON format with these fields:
        - features: list of identified features/capabilities
        - market_positioning: key points about positioning
        - competitive_advantages: list of advantages
        - impact_level: "high", "medium", or "low"
        - sentiment: object with polarity (-1 to 1) and subjectivity (0 to 1)
        - key_quotes: list of important quotes (max 3)
        """

    def _build_summary_prompt(self, articles: List[Dict], competitor_type: str) -> str:
        """Build prompt for summary generation"""
        articles_text = "\n\n".join([
            f"Article {i+1}:\n"
            f"Title: {article['title']}\n"
            f"Content: {article['content'][:500]}..."  # Truncate for length
            for i, article in enumerate(articles)
        ])
        
        return f"""
        Generate a concise summary of {competitor_type} competitor activities based on these articles.
        Focus on:
        - Key trends and patterns
        - Notable product launches or updates
        - Strategic moves and market positioning
        - Potential threats or opportunities
        
        Articles:
        {articles_text}
        
        Please provide a coherent 2-3 paragraph summary that highlights the most significant developments
        and their potential impact on the competitive landscape.
        """

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured format"""
        try:
            import json
            return json.loads(response)
        except:
            # Fallback if response is not valid JSON
            return {
                "features": [],
                "market_positioning": "Analysis failed",
                "competitive_advantages": [],
                "impact_level": "low",
                "sentiment": {"polarity": 0, "subjectivity": 0},
                "key_quotes": []
            }
    
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