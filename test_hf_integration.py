"""
Test script for Hugging Face model integration
"""
from competition_agent.llm.hf_analyzer import HFAnalyzer
from pprint import pprint

def test_sentiment_analysis():
    print("\n=== Testing Sentiment Analysis ===")
    text = """LexisNexis Risk Solutions announced a major breakthrough in fraud detection, 
    with their new AI-powered system showing 95% accuracy in early testing. This innovative 
    solution represents a significant step forward in fighting financial crime."""
    
    analyzer = HFAnalyzer()
    result = analyzer._analyze_sentiment(text)
    print(f"Sentiment Analysis Result:")
    pprint(result)
    assert isinstance(result["polarity"], float), "Polarity should be a float"
    print("✓ Sentiment analysis test passed")

def test_impact_classification():
    print("\n=== Testing Impact Classification ===")
    text = """Breaking News: SEON raises $94M Series B for fraud prevention platform. 
    This marks one of the largest funding rounds in the fraud prevention space this year."""
    
    analyzer = HFAnalyzer()
    result = analyzer._classify_impact(text)
    print(f"Impact Classification Result: {result}")
    assert result in ["high", "medium", "low"], "Impact should be high/medium/low"
    print("✓ Impact classification test passed")

def test_feature_extraction():
    print("\n=== Testing Feature Extraction ===")
    text = """The new platform includes real-time transaction monitoring, 
    advanced behavioral analytics, and automated risk scoring. The machine learning models 
    can adapt to new fraud patterns automatically."""
    
    analyzer = HFAnalyzer()
    result = analyzer._extract_features(text)
    print(f"Extracted Features:")
    pprint(result)
    assert isinstance(result, list), "Features should be a list"
    print("✓ Feature extraction test passed")

def test_full_analysis():
    print("\n=== Testing Full Content Analysis ===")
    text = """SEON announced today the launch of their next-generation fraud prevention platform.
    The new solution combines advanced AI with real-time data analysis to detect and prevent 
    fraud attempts with unprecedented accuracy. Early adopters report a 75% reduction in false 
    positives and 90% improvement in fraud detection rates. The platform introduces innovative 
    features like behavioral biometrics and network analysis."""
    
    analyzer = HFAnalyzer()
    result = analyzer.analyze_content(
        text=text,
        company="SEON",
        keywords=["fraud prevention", "AI", "machine learning"]
    )
    
    print("\nFull Analysis Results:")
    pprint(result)
    
    # Verify all expected fields are present
    required_fields = [
        "features", "market_positioning", "competitive_advantages",
        "impact_level", "sentiment", "key_quotes"
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    print("✓ Full analysis test passed")

def test_summary_generation():
    print("\n=== Testing Summary Generation ===")
    articles = [{
        "title": "SEON Launches New Fraud Prevention Platform",
        "content": """SEON announced today the launch of their next-generation fraud prevention platform.
        The new solution combines advanced AI with real-time data analysis."""
    }, {
        "title": "LexisNexis Expands Risk Analytics Suite",
        "content": """LexisNexis Risk Solutions has expanded its analytics suite with new machine
        learning capabilities for enhanced fraud detection."""
    }]
    
    analyzer = HFAnalyzer()
    result = analyzer.generate_summary(articles, "established")
    print("\nGenerated Summary:")
    print(result)
    assert isinstance(result, str) and len(result) > 0, "Summary should be non-empty string"
    print("✓ Summary generation test passed")

def main():
    print("Starting Hugging Face Integration Tests...")
    print("Note: First run will download models - this may take a few minutes")
    
    try:
        test_sentiment_analysis()
        test_impact_classification()
        test_feature_extraction()
        test_full_analysis()
        test_summary_generation()
        
        print("\n✓✓✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()