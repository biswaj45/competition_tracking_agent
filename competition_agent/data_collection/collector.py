"""
Main data collection orchestrator
"""
from typing import List, Dict
import logging
from datetime import datetime
from .sources.google_news import GoogleNewsCollector
from .sources.tech_media import TechMediaCollector
from .config import COMPETITORS

class DataCollector:
    def __init__(self):
        self.collectors = {
            "google_news": GoogleNewsCollector(),
            "tech_media": TechMediaCollector()
        }
        self.logger = logging.getLogger(__name__)
        
    def collect_all(self, days: int = 7) -> Dict[str, List[Dict]]:
        """
        Collect data from all sources for all competitors
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with competitor categories and their collected data
        """
        results = {
            "established": [],
            "mid_sized": [],
            "startups": []
        }
        
        for category, companies in COMPETITORS.items():
            for company in companies:
                company_data = self.collect_for_company(company, days)
                results[category].extend(company_data)
                
        return results
    
    def collect_for_company(self, company: str, days: int = 7) -> List[Dict]:
        """
        Collect data for a specific company from all sources
        
        Args:
            company: Company name to collect data for
            days: Number of days to look back
            
        Returns:
            List of collected data points
        """
        all_data = []
        
        for source_name, collector in self.collectors.items():
            try:
                data = collector.fetch_data(company, days)
                
                # Add source and company information
                for item in data:
                    item.update({
                        "data_source": source_name,
                        "company": company,
                        "collected_at": datetime.now().isoformat()
                    })
                
                all_data.extend(data)
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {company} from {source_name}: {str(e)}")
        
        return all_data