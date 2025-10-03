import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project root to path for local imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from competition_agent.news.gnews_collector import GNewsCollector

async def main():
    # Set up collector with a 1-day window
    end_date = datetime(2025, 10, 3)  # Today
    start_date = end_date - timedelta(days=1)
    
    collector = GNewsCollector()
    
    # List of companies to track
    companies = [
        "Microsoft",
        "OpenAI",
        "Anthropic",
        "Google DeepMind"
    ]
    
    # Collect news for each company
    results = {}
    for company in companies:
        print(f"Collecting news for {company}...")
        articles = await collector.collect_news(
            query=company,
            start_date=start_date,
            end_date=end_date
        )
        results[company] = articles
        print(f"Found {len(articles)} articles for {company}")
    
    # Save to JSON
    output_file = Path("collected_today.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to {output_file.resolve()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())