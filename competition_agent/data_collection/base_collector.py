"""
Base collector class for implementing different data sources
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

class BaseCollector(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch data for a specific company from the last N days
        
        Args:
            company: Company name to fetch data for
            days: Number of days to look back
            
        Returns:
            List of dictionaries containing the collected data
        """
        pass
    
    def _format_date(self, date_str: str) -> Optional[datetime]:
        """Convert various date formats to datetime object"""
        try:
            # Add various date format parsing here
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            self.logger.error(f"Could not parse date: {date_str}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        return " ".join(text.split())