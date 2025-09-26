"""
Test script for generating PDF report
"""
import os
from datetime import datetime, timedelta
from competition_agent.storage.models import Database, Competitor, News, CompetitorType, ImpactLevel
from competition_agent.storage.analyzer import CompetitorAnalyzer
from competition_agent.reporting.pdf_generator import CompetitorReportGenerator

def create_sample_data(db: Database):
    """Create sample data for testing"""
    # Add competitors
    competitors = [
        Competitor("Experian Hunter", CompetitorType.ESTABLISHED, 
                  "Leading fraud detection solution", "https://experian.com"),
        Competitor("Feedzai", CompetitorType.MID_SIZED,
                  "AI-powered fraud prevention", "https://feedzai.com"),
        Competitor("Sardine", CompetitorType.STARTUP,
                  "Real-time fraud prevention platform", "https://sardine.ai")
    ]
    
    competitor_ids = {}
    for comp in competitors:
        competitor_ids[comp.name] = db.add_competitor(comp)
    
    # Add news items spanning 6 months
    base_date = datetime.now()  # Current date
    
    # Most recent (2 days ago)
    db.add_news(News(
        competitor_ids["Experian Hunter"],
        "Experian Launches New Fraud Detection API",
        "Experian announced a new API for real-time fraud detection with advanced machine learning capabilities...",
        "Press Release",
        base_date - timedelta(days=2),
        "https://example.com/news1",
        ImpactLevel.HIGH,
        {"polarity": 0.8, "subjectivity": 0.4},
        0.9,
        ["API", "real-time detection", "fraud detection", "machine learning", "artificial intelligence"]
    ))
    
    # 1 month ago
    db.add_news(News(
        competitor_ids["Feedzai"],
        "Feedzai Releases Advanced User Behavior Analytics",
        "Feedzai introduces new behavioral biometrics module for enhanced fraud detection...",
        "Tech News",
        base_date - timedelta(days=30),
        "https://example.com/news4",
        ImpactLevel.HIGH,
        {"polarity": 0.7, "subjectivity": 0.4},
        0.85,
        ["behavioral biometrics", "user behavior", "fraud detection", "machine learning"]
    ))
    
    # 2 months ago
    db.add_news(News(
        competitor_ids["Experian Hunter"],
        "Experian Partners with Major Payment Provider",
        "Strategic partnership announced for enhanced payment fraud detection...",
        "Press Release",
        base_date - timedelta(days=60),
        "https://example.com/news5",
        ImpactLevel.MEDIUM,
        {"polarity": 0.6, "subjectivity": 0.3},
        0.75,
        ["partnership", "payment fraud", "integration"]
    ))
    
    # 3 months ago
    db.add_news(News(
        competitor_ids["Sardine"],
        "Sardine's Machine Learning Model Shows 95% Accuracy",
        "Revolutionary results in crypto fraud prevention...",
        "Industry Analysis",
        base_date - timedelta(days=90),
        "https://example.com/news6",
        ImpactLevel.HIGH,
        {"polarity": 0.9, "subjectivity": 0.4},
        0.95,
        ["machine learning", "crypto fraud", "model accuracy", "AI"]
    ))
    
    # 4 months ago
    db.add_news(News(
        competitor_ids["Feedzai"],
        "Feedzai's Q2 Fraud Report Released",
        "Comprehensive analysis of fraud trends and emerging threats...",
        "Research Report",
        base_date - timedelta(days=120),
        "https://example.com/news7",
        ImpactLevel.MEDIUM,
        {"polarity": 0.5, "subjectivity": 0.2},
        0.8,
        ["fraud trends", "threat analysis", "industry research"]
    ))
    
    # 5 months ago
    db.add_news(News(
        competitor_ids["Experian Hunter"],
        "Experian Introduces Device Intelligence",
        "New device fingerprinting technology for fraud prevention...",
        "Product Launch",
        base_date - timedelta(days=150),
        "https://example.com/news8",
        ImpactLevel.HIGH,
        {"polarity": 0.8, "subjectivity": 0.4},
        0.9,
        ["device fingerprinting", "fraud prevention", "technology"]
    ))
    
    # 6 months ago
    db.add_news(News(
        competitor_ids["Sardine"],
        "Sardine Expands to European Market",
        "Strategic expansion marks growing presence in fraud prevention...",
        "Business News",
        base_date - timedelta(days=180),
        "https://example.com/news9",
        ImpactLevel.HIGH,
        {"polarity": 0.7, "subjectivity": 0.3},
        0.85,
        ["expansion", "market growth", "international", "fraud prevention"]
    ))

def main():
    # Initialize database
    db_path = "competition_data.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = Database(db_path)
    create_sample_data(db)
    
    # Create analyzer
    analyzer = CompetitorAnalyzer(db)
    
    # Create report generator
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    report_gen = CompetitorReportGenerator(analyzer, output_dir)
    
    # Generate report for the last 180 days (6 months)
    print("Generating 6-month competitive intelligence report...")
    report_path = report_gen.generate_weekly_report(days=180)
    
    print(f"\nReport generated successfully!")
    print(f"Location: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()