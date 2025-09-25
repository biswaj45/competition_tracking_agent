"""
LinkedIn data collector using their API
"""
from typing import List, Dict
import os
import requests
from datetime import datetime, timedelta
from ..base_collector import BaseCollector
from ..content_analyzer import ContentAnalyzer

class LinkedInCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        self.content_analyzer = ContentAnalyzer()
        
        if not self.access_token:
            self.logger.warning("LinkedIn access token not found in environment variables")
    
    def fetch_data(self, company: str, days: int = 7) -> List[Dict]:
        """
        Fetch company updates from LinkedIn
        
        Args:
            company: Company name or organization ID
            days: Number of days to look back
            
        Returns:
            List of company updates and posts
        """
        if not self.access_token:
            return []
            
        try:
            # First get the organization ID if we only have the company name
            org_id = self._get_organization_id(company)
            if not org_id:
                return []
                
            # Fetch company updates
            updates = self._fetch_company_updates(org_id, days)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error fetching LinkedIn data for {company}: {str(e)}")
            return []
    
    def _get_organization_id(self, company: str) -> str:
        """Get LinkedIn organization ID from company name"""
        url = "https://api.linkedin.com/v2/organizations"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        params = {
            "q": "vanityName",
            "vanityName": company.lower().replace(" ", "-")
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "elements" in data and data["elements"]:
                return data["elements"][0]["id"]
                
        except Exception as e:
            self.logger.error(f"Error getting organization ID for {company}: {str(e)}")
            
        return None
    
    def _fetch_company_updates(self, org_id: str, days: int) -> List[Dict]:
        """Fetch recent company updates"""
        url = f"https://api.linkedin.com/v2/organizations/{org_id}/updates"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        params = {
            "q": "chronological",
            "start": 0,
            "count": 50,
            "timeInterval.start": start_time
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            updates = []
            for element in data.get("elements", []):
                content = element.get("specificContent", {}).get("text", "")
                
                # Analyze content relevance
                analysis = self.content_analyzer.analyze_content(
                    content,
                    org_id,  # We'll use org_id since we know it's correct
                    []  # Add relevant keywords
                )
                
                if analysis["relevance_score"] >= 0.3:
                    updates.append({
                        "title": content[:100] + "..." if len(content) > 100 else content,
                        "link": f"https://www.linkedin.com/feed/update/{element['id']}",
                        "published_date": datetime.fromtimestamp(
                            element["created"]["time"] / 1000
                        ).isoformat(),
                        "content": content,
                        "engagement": element.get("engagement", {}),
                        "analysis": analysis
                    })
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error fetching updates for organization {org_id}: {str(e)}")
            return []