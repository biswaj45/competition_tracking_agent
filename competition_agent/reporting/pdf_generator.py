"""
PDF report generation with professional styling and enhanced Hugging Face summaries
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
from ..llm.hf_analyzer import HFAnalyzer  # Import the Hugging Face analyzer

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
        
        # Initialize Hugging Face analyzer for enhanced summaries
        try:
            self.hf_analyzer = HFAnalyzer()
            self.use_transformer = True
            print("Initialized Hugging Face models for report generation")
        except Exception as e:
            print(f"Warning: Hugging Face initialization failed: {str(e)}")
            print("Falling back to basic summaries")
            self.use_transformer = False
            
    def generate_weekly_report(self, days: int = 180) -> str:
        """Generate the weekly competitor intelligence report"""
        # Get insights from analyzer
        insights = self.analyzer.generate_weekly_insights(days)
        
        # Set up the document
        output_file = os.path.join(
            self.output_dir,
            f"competitive_intelligence_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
        
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        story = []
        
        # Add title
        story.append(Paragraph(
            f"Competitive Intelligence Report<br/>Week of {datetime.now().strftime('%B %d, %Y')}",
            self.styles.styles['CompReportTitle']
        ))
        story.append(Spacer(1, 20))
        
        # Add executive summary
        story.append(Paragraph("Executive Summary (6-Month Analysis)", self.styles.styles['CompReportSection']))
        bullets = self._generate_executive_summary(insights)
        for bullet in bullets:
            story.append(Paragraph(bullet, self.styles.styles['CompReportBody']))
        story.append(Spacer(1, 20))
        
        # Build and save the document
        doc.build(story)
        return output_file
        
    def _generate_executive_summary(self, insights: Dict) -> List[str]:
        """Generate executive summary bullet points"""
        activity = insights.get("competitor_activity", {})
        feature_analysis = insights.get("feature_analysis", {})
        
        bullets = [
            f"• Tracked {int(activity.get('total_activities', 0))} competitor activities across all segments "
            f"(avg. {activity.get('monthly_average', 0):.1f}/month)",
            
            f"• Identified {int(activity.get('high_impact_total', 0))} high-impact developments "
            f"(avg. {activity.get('high_impact_monthly', 0):.1f}/month)",
            
            f"• {feature_analysis.get('num_features', 0)} distinct product features observed over 6 months",
            
            f"• {feature_analysis.get('new_features', 0)} new features introduced in this period",
            
            f"• Most active segment: {activity.get('most_active_segment', 'Unknown')}"
        ]
        
        return bullets

    def generate_weekly_report(self, days: int = 7) -> str:
        """Generate the competitive intelligence report with enhanced summaries
        
        Args:
            days: Number of days to look back (default: 7)
        """
        # Get report data
        insights = self.analyzer.generate_weekly_insights(days=days)
        figures = self.analyzer.generate_visualizations()
        
        # Generate enhanced summaries if available
        if self.use_transformer:
            for competitor_type in ["established", "startups"]:
                if competitor_type in insights:
                    enhanced_summary = self._generate_enhanced_summary(
                        insights[competitor_type].get("articles", []),
                        competitor_type
                    )
                    if enhanced_summary:
                        insights[competitor_type]["summary"] = enhanced_summary
        
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
        self._add_cover_page(story, days)
        self._add_executive_summary(story, insights, days)
        self._add_competitor_tracking(story, insights)
        self._add_product_gap_analysis(story, insights)
        self._add_visuals_and_trends(story, insights, figures)
        self._add_recommendations(story, insights)
        
        # Generate PDF
        doc.build(story)
        return output_path
    
    def _add_cover_page(self, story: List, days: int) -> None:
        """Add the cover page with correct date range"""
        story.append(Paragraph(
            "Competitive Intelligence Report — Fraud Solutions Industry",
            self.styles.styles['CompReportTitle']
        ))
        story.append(Paragraph(
            "Tracking competitors of TransUnion Fraud Solutions",
            self.styles.styles['CompReportSubsection']
        ))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = f"{start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}"
        story.append(Spacer(1, 18))
        story.append(Paragraph(
            f"Analysis Period: {date_range}",
            self.styles.styles['CompReportBody']
        ))
        story.append(Spacer(1, 18))
    # No page break here
    
    def _add_executive_summary(self, story: List, insights: Dict, days: int) -> None:
        """Add executive summary section with correct period"""
        months = max(1, round(days / 30))
        story.append(Paragraph(
            f"Executive Summary ({months}-Month Analysis)",
            self.styles.styles['CompReportSection']
        ))
        activity = insights['competitor_activity']
        features = insights['feature_analysis']
        type_summary = activity.get('type_summary', {})
        total_mentions = sum(
            summary.get(('total_mentions', 'sum'), 0) 
            for summary in type_summary.values()
        )
        high_impact_events = sum(
            summary.get(('high_impact', 'sum'), 0)
            for summary in type_summary.values()
        )
        most_active = max(
            type_summary.items(),
            key=lambda x: x[1].get(('total_mentions', 'sum'), 0),
            default=('Unknown', {})
        )[0]
        monthly_avg_mentions = total_mentions / months
        monthly_avg_impacts = high_impact_events / months
        takeaways = [
            f"Tracked {total_mentions} competitor activities across all segments (avg. {monthly_avg_mentions:.0f}/month)",
            f"Identified {high_impact_events} high-impact developments (avg. {monthly_avg_impacts:.1f}/month)",
            f"Observed {features.get('feature_count', 0)} distinct product features over {months} months",
            f"{features.get('new_feature_count', 0)} new features introduced in this period",
            f"Most active segment: {most_active}"
        ]
        for takeaway in takeaways:
            story.append(Paragraph(
                f"• {takeaway}",
                self.styles.styles['CompReportBody']
            ))
        story.append(Spacer(1, 10))
    
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
        
    # No page break here
    
    def _add_product_gap_analysis(self, story: List, insights: Dict) -> None:
        """Add product gap analysis section"""
        story.append(Paragraph(
            "Product Gap Analysis",
            self.styles.styles['CompReportSection']
        ))
        
        # Feature comparison table
        feature_analysis = insights.get('feature_analysis', {})
        feature_data = feature_analysis.get('features', {})
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
        
    # No page break here
    
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
        
    # No page break here
    
    def _add_recommendations(self, story: List, insights: Dict) -> None:
        """Add recommendations section"""
        story.append(Paragraph(
            "Recommendations",
            self.styles.styles['CompReportSection']
        ))
        
        # Generate recommendations based on insights
        feature_analysis = insights.get('feature_analysis', {})
        competitor_activity = insights.get('competitor_activity', {})
        
        # Get feature names from analysis
        feature_names = list(feature_analysis.get('features', {}).keys())
        
        recommendations = [
            "Product Development:",
            "• Evaluate adoption of emerging features: " + 
            (", ".join(feature_names[:3]) if feature_names else "No new features identified"),
            "\nMarket Positioning:",
            "• Focus on high-activity segments: " +
            ", ".join(
                k for k, v in competitor_activity.get('type_summary', {}).items()
                if v.get(('total_mentions', 'sum'), 0) > 
                sum(x.get(('total_mentions', 'sum'), 0) for x in competitor_activity.get('type_summary', {}).values()) / 3
            ) or "No high-activity segments identified",
            "\nCompetitive Response:",
            "• Monitor high-impact competitors: " +
            ", ".join(
                c['competitor_name'] for c in 
                competitor_activity.get('most_active_competitors', [])[:3]
            ) or "No high-impact competitors identified"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(
                rec,
                self.styles.styles['CompReportBody']
            ))
    
    def _add_competitor_analysis(self, story: List, activity: Dict, comp_type: str) -> None:
        """Add analysis for a specific competitor type"""
        # Get competitor mentions from type summary if available
        type_summary = activity.get('type_summary', {})
        relevant_competitors = []
        
        # Extract competitor data from type summary
        for comp_name, stats in type_summary.items():
            if comp_name.lower() in comp_type:
                relevant_competitors.append({
                    'competitor_name': comp_name,
                    'total_mentions': stats.get(('total_mentions', 'sum'), 0),
                    'high_impact': stats.get(('high_impact', 'sum'), 0)
                })
        
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