"""
PDF report generation with professional styling
"""
from typing import Dict, Any, List
import os
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import io

class ReportStyles:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CompReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a237e')  # Dark blue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='CompReportSection',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#0d47a1')  # Medium blue
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='CompReportSubsection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#1565c0')  # Light blue
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CompReportBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            leading=14
        ))
        
        # Table header style
        self.styles.add(ParagraphStyle(
            name='CompReportTableHeader',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.white,
            alignment=TA_CENTER
        ))

class CompetitorReportGenerator:
    def __init__(self, analyzer, output_dir: str):
        self.analyzer = analyzer
        self.output_dir = output_dir
        self.styles = ReportStyles()
        
    def generate_weekly_report(self) -> str:
        """Generate the weekly competitive intelligence report"""
        # Get report data
        insights = self.analyzer.generate_weekly_insights(days=7)
        figures = self.analyzer.generate_visualizations()
        
        # Setup output path
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir,
            f"competitive_intelligence_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build report content
        story = []
        self._add_cover_page(story)
        self._add_executive_summary(story, insights)
        self._add_competitor_tracking(story, insights)
        self._add_product_gap_analysis(story, insights)
        self._add_visuals_and_trends(story, insights, figures)
        self._add_recommendations(story, insights)
        
        # Generate PDF
        doc.build(story)
        return output_path
    
    def _add_cover_page(self, story: List) -> None:
        """Add the cover page"""
        # Title
        story.append(Paragraph(
            "Competitive Intelligence Report — Fraud Solutions Industry",
            self.styles.styles['CompReportTitle']
        ))
        
        # Subtitle
        story.append(Paragraph(
            "Tracking competitors of TransUnion Fraud Solutions",
            self.styles.styles['CompReportSubsection']
        ))
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        date_range = f"{start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}"
        
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Week of {date_range}",
            self.styles.styles['CompReportBody']
        ))
        
        story.append(PageBreak())
    
    def _add_executive_summary(self, story: List, insights: Dict) -> None:
        """Add executive summary section"""
        story.append(Paragraph(
            "Executive Summary",
            self.styles.styles['CompReportSection']
        ))
        
        # Key takeaways
        activity = insights['competitor_activity']
        features = insights['feature_analysis']
        
        takeaways = [
            f"Tracked {activity['total_mentions']} competitor activities across all segments",
            f"Identified {activity['high_impact_events']} high-impact developments",
            f"Observed {features['feature_count']} distinct product features",
            f"{len(features['new_features'])} new features introduced this week",
            f"Most active segment: {max(activity['type_summary'].items(), key=lambda x: x[1]['total_mentions'])[0]}"
        ]
        
        for takeaway in takeaways:
            story.append(Paragraph(
                f"• {takeaway}",
                self.styles.styles['CompReportBody']
            ))
        
        story.append(PageBreak())
    
    def _add_competitor_tracking(self, story: List, insights: Dict) -> None:
        """Add competitor tracking section"""
        story.append(Paragraph(
            "Competitor Tracking",
            self.styles.styles['CompReportSection']
        ))
        
        # Process each competitor type
        for comp_type in ['Established', 'Mid-sized', 'Startups']:
            story.append(Paragraph(
                f"Subsection {comp_type} Competitors",
                self.styles.styles['CompReportSubsection']
            ))
            
            # Add competitor analysis
            self._add_competitor_analysis(
                story,
                insights['competitor_activity'],
                comp_type.lower()
            )
        
        story.append(PageBreak())
    
    def _add_product_gap_analysis(self, story: List, insights: Dict) -> None:
        """Add product gap analysis section"""
        story.append(Paragraph(
            "Product Gap Analysis",
            self.styles.styles['CompReportSection']
        ))
        
        # Feature comparison table
        feature_data = insights['feature_analysis']['top_features']
        if feature_data:
            table_data = [['Feature', 'Adoption', 'Present in Segments']]
            for feature, info in feature_data.items():
                table_data.append([
                    feature,
                    str(info['adoption_count']),
                    ', '.join(info['competitor_type'])
                ])
            
            table = Table(table_data, colWidths=[200, 100, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            story.append(table)
        
        story.append(PageBreak())
    
    def _add_visuals_and_trends(self, story: List, insights: Dict, figures: Dict) -> None:
        """Add visualizations section"""
        story.append(Paragraph(
            "Visuals & Trends",
            self.styles.styles['CompReportSection']
        ))
        
        # Add matplotlib figures
        for name, fig in figures.items():
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight')
            img_data.seek(0)
            
            img = Image(img_data, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        
        story.append(PageBreak())
    
    def _add_recommendations(self, story: List, insights: Dict) -> None:
        """Add recommendations section"""
        story.append(Paragraph(
            "Recommendations",
            self.styles.styles['CompReportSection']
        ))
        
        # Generate recommendations based on insights
        feature_gaps = insights['feature_analysis'].get('new_features', [])
        competitor_activity = insights['competitor_activity']
        
        recommendations = [
            "Product Development:",
            "• Evaluate adoption of emerging features: " + 
            ", ".join(f['feature_name'] for f in feature_gaps[:3]),
            "\nMarket Positioning:",
            "• Focus on high-activity segments: " +
            ", ".join(
                k for k, v in competitor_activity['type_summary'].items()
                if v['total_mentions'] > competitor_activity['total_mentions'] / 3
            ),
            "\nCompetitive Response:",
            "• Monitor high-impact competitors: " +
            ", ".join(
                c['competitor_name'] for c in 
                competitor_activity['most_active_competitors'][:3]
            )
        ]
        
        for rec in recommendations:
            story.append(Paragraph(
                rec,
                self.styles.styles['CompReportBody']
            ))
    
    def _add_competitor_analysis(self, story: List, activity: Dict, comp_type: str) -> None:
        """Add analysis for a specific competitor type"""
        relevant_competitors = [
            c for c in activity['most_active_competitors']
            if c['competitor_name'].lower() in comp_type
        ]
        
        if relevant_competitors:
            table_data = [['Competitor', 'Mentions', 'High Impact Events']]
            for comp in relevant_competitors:
                table_data.append([
                    comp['competitor_name'],
                    str(comp['total_mentions']),
                    str(comp['high_impact'])
                ])
                
            table = Table(table_data, colWidths=[200, 100, 100])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))