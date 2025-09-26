"""
Test script for data collection
"""
import json
from datetime import datetime
from competition_agent.data_collection.collector import DataCollector

def main():
    collector = DataCollector()
    
    # Collect data for the last 7 days
    print("Starting data collection...")
    results = collector.collect_all(days=180)
    
    # Save results to a JSON file for inspection
    output_file = f"collected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nData collection completed. Results saved to {output_file}")
    
    # Print some statistics
    for category, data in results.items():
        print(f"\n{category.title()} competitors:")
        print(f"- Total articles collected: {len(data)}")
        
        # Group by company
        company_stats = {}
        for article in data:
            company = article['company']
            company_stats[company] = company_stats.get(company, 0) + 1
        
        for company, count in company_stats.items():
            print(f"  - {company}: {count} articles")

if __name__ == "__main__":
    main()