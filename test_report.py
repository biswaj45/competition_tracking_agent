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
    
    # Add news items
    base_date = datetime(2025, 9, 25)  # Current date
    news_items = [
        News(
            competitor_ids["Experian Hunter"],
            "Experian Launches New Fraud Detection API",
            "Experian announced a new API for real-time fraud detection with advanced machine learning capabilities for instant transaction validation and risk scoring...",
            "Press Release",
            base_date - timedelta(days=2),
            "https://example.com/news1",
            ImpactLevel.HIGH,
            {"polarity": 0.8, "subjectivity": 0.4},
            0.9,
            ["API", "real-time detection", "fraud detection", "machine learning", "artificial intelligence"]
        ),
        News(
            competitor_ids["Feedzai"],
            "Feedzai Partners with Major Bank",
            "Feedzai announces partnership with leading bank...",
            "News",
            base_date - timedelta(days=4),
            "https://example.com/news2",
            ImpactLevel.MEDIUM,
            {"polarity": 0.6, "subjectivity": 0.3},
            0.7,
            ["partnership", "banking integration", "machine learning", "risk assessment", "fraud prevention"]
        ),
        News(
            competitor_ids["Sardine"],
            "Sardine Raises Series B Funding",
            "Fraud prevention startup Sardine raises $50M...",
            "TechCrunch",
            base_date - timedelta(days=1),
            "https://example.com/news3",
            ImpactLevel.HIGH,
            {"polarity": 0.9, "subjectivity": 0.5},
            0.8,
            ["funding", "expansion", "machine learning", "fraud prevention", "behavioral analytics", "risk management", "cryptocurrency"]
        )
    ]
    
    for news in news_items:
        db.add_news(news)

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
    
    # Generate report
    print("Generating competitive intelligence report...")
    report_path = report_gen.generate_weekly_report()
    
    print(f"\nReport generated successfully!")
    print(f"Location: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()