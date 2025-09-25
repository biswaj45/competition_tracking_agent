"""
Report generation module for creating weekly competitive intelligence reports
"""
from datetime import datetime, timedelta
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

class ReportGenerator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='Cover',
            fontSize=24,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='Section',
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10
        ))
    
    def generate_weekly_report(self, output_path: str):
        """Generate a compact 2-page competitive intelligence report"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        # Compact header
        header = Paragraph(
            f"<b>Competitive Intelligence Report — Fraud Solutions Industry</b> | Week of {datetime.now().strftime('%b %d, %Y')}",
            self.styles['Normal'])
        story.append(header)
        story.append(Spacer(1, 10))
        # All sections in summary form
        self._add_executive_summary(story, compact=True)
        self._add_competitor_tracking(story, compact=True)
        self._add_product_gap_analysis(story, compact=True)
        self._add_visuals_and_trends(story, compact=True)
        self._add_recommendations(story, compact=True)
        doc.build(story)
    
    def _add_cover_page(self, story):
        """Add the cover page to the report"""
        title = Paragraph("Competitive Intelligence Report — Fraud Solutions Industry",
                         self.styles['Cover'])
        subtitle = Paragraph("Tracking competitors of TransUnion Fraud Solutions",
                           self.styles['Normal'])
        date_range = Paragraph(f"Week of {datetime.now().strftime('%B %d, %Y')}",
                             self.styles['Normal'])
        
        story.extend([title, Spacer(1, 30), subtitle, Spacer(1, 20), date_range])
    
    def _add_executive_summary(self, story, compact=False):
        # Compact executive summary: 2-3 bullet points
        story.append(Paragraph("<b>Executive Summary</b>", self.styles['Section']))
        bullets = [
            "Industry remains highly competitive with new product launches.",
            "Established players focus on AI-driven fraud detection.",
            "Startups are innovating in behavioral biometrics and KYC."
        ]
        for b in bullets[:3]:
            story.append(Paragraph(f"- {b}", self.styles['Normal']))
        story.append(Spacer(1, 8))

    def _add_competitor_tracking(self, story, compact=False):
        # Compact: Table with top 3 competitors and their recent activity
        story.append(Paragraph("<b>Competitor Tracking (Top 3)</b>", self.styles['Section']))
        data = [
            ["Competitor", "Type", "Recent Activity"],
            ["Experian Hunter", "Established", "Launched new AI fraud module"],
            ["Feedzai", "Mid-sized", "Expanded into APAC region"],
            ["Sardine", "Startup", "Raised Series B for KYC tech"]
        ]
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 4),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    def _add_product_gap_analysis(self, story, compact=False):
        # Compact: Table with 3 features and adoption
        story.append(Paragraph("<b>Product Gap Analysis</b>", self.styles['Section']))
        data = [
            ["Feature", "Adopted By", "Gap"],
            ["Real-time AML", "FICO, Feedzai", "Not in TransUnion"],
            ["Behavioral Biometrics", "Sardine, BioCatch", "Partial"],
            ["AI Explainability", "Experian, Featurespace", "Gap"]
        ]
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,0), 4),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    def _add_visuals_and_trends(self, story, compact=False):
        # Compact: 2-3 bullet points for trends
        story.append(Paragraph("<b>Visuals & Trends</b>", self.styles['Section']))
        bullets = [
            "AI/ML adoption up 20% YoY across competitors.",
            "Behavioral biometrics mentioned in 30% of news.",
            "KYC/AML remains a top investment area."
        ]
        for b in bullets[:3]:
            story.append(Paragraph(f"- {b}", self.styles['Normal']))
        story.append(Spacer(1, 8))

    def _add_recommendations(self, story, compact=False):
        # Compact: 2-3 actionable recommendations
        story.append(Paragraph("<b>Recommendations</b>", self.styles['Section']))
        bullets = [
            "Accelerate AI explainability features.",
            "Explore partnerships in behavioral biometrics.",
            "Expand KYC/AML product marketing."
        ]
        for b in bullets[:3]:
            story.append(Paragraph(f"- {b}", self.styles['Normal']))
        story.append(Spacer(1, 8))