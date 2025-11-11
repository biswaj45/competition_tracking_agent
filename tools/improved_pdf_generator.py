#!/usr/bin/env python3
"""
Final Compact PDF Report Generator - 30 Days with TransUnion Tracking
Uses original preferred style from competitive_intelligence_20251010_181805
- Navy/blue color scheme with bordered headers
- Specific font sizes and spacing
- Clickable links maintained
- Compact summaries (150 words)
- TransUnion in histogram only
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image
)
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalCompactPDFReportGenerator:
    def __init__(self, analysis_data: Dict, deduplicated_articles: List[Dict], output_dir: Path):
        """Initialize PDF generator with original style"""
        self.report_data = analysis_data
        self.articles = deduplicated_articles
        self.output_dir = output_dir
        self.chart_paths = {}
        
        # Setup custom styles matching original
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Generate charts
        self._generate_charts()

    def _setup_custom_styles(self):
        """Setup custom styles matching original PDF"""
        # Custom Title - Navy, 24pt, centered
        try:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#000080'),  # Navy
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        except KeyError:
            pass
        
        # Section Header - Dark blue, 16pt, bordered
        try:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#00008B'),  # Dark blue
                spaceAfter=12,
                spaceBefore=12,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.HexColor('#000080'),  # Navy
                borderPadding=5,
                backColor=colors.HexColor('#F0F8FF')  # Alice blue background
            ))
        except KeyError:
            pass
        
        # Sub Header - Blue, 14pt
        try:
            self.styles.add(ParagraphStyle(
                name='SubHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#0000CD'),  # Medium blue
                spaceAfter=8,
                spaceBefore=8,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
        except KeyError:
            pass
        
        # Body Text - 10pt
        try:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['BodyText'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            ))
        except KeyError:
            pass
        
        # Bullet Text - 10pt with indent
        try:
            self.styles.add(ParagraphStyle(
                name='BulletText',
                parent=self.styles['BodyText'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=3,
                alignment=TA_LEFT,
                fontName='Helvetica'
            ))
        except KeyError:
            pass

    def _generate_charts(self):
        """Generate visualization charts"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Competitor Activity Chart (with TransUnion)
        self._create_competitor_chart()
        
        # 2. Competitive Threats Pie Chart
        self._create_competitive_threats_pie_chart()
        
        # 3. Technology Trends Chart
        self._create_technology_chart()
        
        # 4. Fraud Domain Distribution
        self._create_fraud_domain_chart()

    def _create_competitor_chart(self):
        """Create competitor activity bar chart (including TransUnion)"""
        try:
            competitor_data = self.report_data['competitor_analysis']['activity_summary']
            
            if not competitor_data:
                return
            
            # Sort by activity
            sorted_competitors = dict(sorted(competitor_data.items(), key=lambda x: x[1], reverse=True)[:15])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            competitors = list(sorted_competitors.keys())
            activities = list(sorted_competitors.values())
            
            # Highlight TransUnion in different color
            colors_list = ['#DC143C' if c == 'TransUnion' else '#4682B4' for c in competitors]
            
            bars = ax.bar(competitors, activities, color=colors_list, alpha=0.8)
            ax.set_title('Competitor Activity - Article Mentions (30 Days)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Competitors', fontsize=12)
            ax.set_ylabel('Number of Articles', fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, activities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_path = self.output_dir / 'competitor_activity.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['competitor_activity'] = str(chart_path)
            logger.info(f"✓ Competitor chart created (TransUnion highlighted in red)")
        except Exception as e:
            logger.error(f"Error creating competitor chart: {e}")

    def _create_competitive_threats_pie_chart(self):
        """Create pie chart for competitive threats distribution"""
        try:
            insights = self.report_data['strategic_insights']
            
            if 'competitive_threats' not in insights or not insights['competitive_threats']:
                return
            
            # Get top competitors from competitive threats
            threats = insights['competitive_threats'][:8]  # Top 8 for pie chart
            
            if not threats:
                return
            
            competitors = [t['competitor'] for t in threats]
            activities = [t['activity_level'] for t in threats]
            
            # Create color palette - varying shades of blue with red for highest
            colors = ['#DC143C', '#4682B4', '#5F9EA0', '#6495ED', '#00BFFF', '#87CEEB', '#B0C4DE', '#ADD8E6']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(activities, labels=competitors, autopct='%1.1f%%',
                                               startangle=90, colors=colors[:len(competitors)],
                                               textprops={'fontsize': 10, 'weight': 'bold'})
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            
            ax.set_title('Competitive Threats Distribution', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            chart_path = self.output_dir / 'competitive_threats_pie.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['competitive_threats_pie'] = str(chart_path)
            logger.info("✓ Competitive threats pie chart created")
        except Exception as e:
            logger.error(f"Error creating competitive threats pie chart: {e}")

    def _create_technology_chart(self):
        """Create technology trends chart"""
        try:
            tech_data = self.report_data['technology_trends']['trending_keywords']
            
            if not tech_data:
                return
            
            sorted_tech = dict(sorted(tech_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            technologies = list(sorted_tech.keys())
            counts = list(sorted_tech.values())
            
            bars = ax.barh(technologies, counts, color='#20B2AA', alpha=0.8)
            ax.set_title('Technology Trends - Keyword Mentions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Mentions', fontsize=12)
            ax.set_ylabel('Technologies', fontsize=12)
            
            # Add value labels with proper spacing to avoid going outside frame
            max_count = max(counts) if counts else 1
            for bar, value in zip(bars, counts):
                # Place label inside bar if value is large, outside if small
                if value > max_count * 0.15:
                    ax.text(bar.get_width() - (max_count * 0.02), bar.get_y() + bar.get_height()/2,
                           f'{value}', va='center', ha='right', fontweight='bold', fontsize=9, color='white')
                else:
                    ax.text(bar.get_width() + (max_count * 0.02), bar.get_y() + bar.get_height()/2,
                           f'{value}', va='center', ha='left', fontweight='bold', fontsize=9)
            
            # Add some margin to x-axis
            ax.set_xlim(0, max_count * 1.15)
            plt.tight_layout()
            
            chart_path = self.output_dir / 'technology_trends.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['technology_trends'] = str(chart_path)
            logger.info("✓ Technology trends chart created")
        except Exception as e:
            logger.error(f"Error creating technology chart: {e}")

    def _create_fraud_domain_chart(self):
        """Create fraud domain distribution chart"""
        try:
            fraud_data = self.report_data['fraud_domain_analysis']
            
            if not fraud_data:
                return
            
            sorted_fraud = dict(sorted(fraud_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            domains = list(sorted_fraud.keys())
            counts = list(sorted_fraud.values())
            
            # Format domain names
            formatted_domains = [d.replace('_', ' ').title() for d in domains]
            
            bars = ax.barh(formatted_domains, counts, color='#FF6347', alpha=0.8)
            ax.set_title('Fraud Domain Coverage', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Articles', fontsize=12)
            ax.set_ylabel('Fraud Domains', fontsize=12)
            
            # Add value labels with proper spacing to avoid going outside frame
            max_count = max(counts) if counts else 1
            for bar, value in zip(bars, counts):
                # Place label inside bar if value is large, outside if small
                if value > max_count * 0.15:
                    ax.text(bar.get_width() - (max_count * 0.02), bar.get_y() + bar.get_height()/2,
                           f'{value}', va='center', ha='right', fontweight='bold', fontsize=9, color='white')
                else:
                    ax.text(bar.get_width() + (max_count * 0.02), bar.get_y() + bar.get_height()/2,
                           f'{value}', va='center', ha='left', fontweight='bold', fontsize=9)
            
            # Add some margin to x-axis
            ax.set_xlim(0, max_count * 1.15)
            plt.tight_layout()
            
            chart_path = self.output_dir / 'fraud_domains.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['fraud_domains'] = str(chart_path)
            logger.info("✓ Fraud domain chart created")
        except Exception as e:
            logger.error(f"Error creating fraud domain chart: {e}")

    def generate_pdf_report(self) -> str:
        """Generate compact PDF report using original style"""
        pdf_path = self.output_dir / 'Final_Competitive_Intelligence_Report_30Days.pdf'
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Cover page
        self._add_cover_page(story)
        story.append(PageBreak())
        
        # Executive Summary
        self._add_executive_summary(story)
        story.append(PageBreak())
        
        # Top News Summaries (Compact)
        self._add_top_news_section(story)
        story.append(PageBreak())
        
        # Competitor Analysis (with TransUnion histogram)
        self._add_competitor_analysis(story)
        story.append(PageBreak())
        
        # Technology & Fraud Trends
        self._add_trends_analysis(story)
        story.append(PageBreak())
        
        # Strategic Insights (excluding TransUnion)
        self._add_strategic_insights(story)
        story.append(PageBreak())
        
        # Citations
        self._add_citations_section(story)
        
        # Build PDF
        doc.build(story)
        logger.info(f"✅ PDF report generated: {pdf_path}")
        
        return str(pdf_path)

    def _add_cover_page(self, story):
        """Add cover page"""
        # Title
        title = Paragraph(
            "Competitive Intelligence Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Subtitle
        subtitle = Paragraph(
            "TransUnion Fraud Solutions Portfolio<br/>30-Day Market Analysis",
            self.styles['Heading2']
        )
        story.append(subtitle)
        story.append(Spacer(1, 0.5*inch))
        
        # Metadata table
        metadata = self.report_data['metadata']
        data = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Period:', f"{metadata.get('days_covered', 30)} Days"],
            ['Total Articles:', f"{metadata['total_articles_analyzed']:,}"],
            ['Competitor Articles:', f"{metadata['competitor_articles']:,}"],
            ['TransUnion Articles:', f"{metadata['transunion_articles']:,}"],
            ['Deduplication:', 'Applied']
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F8FF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#000080')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#000080')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(table)

    def _add_executive_summary(self, story):
        """Add executive summary"""
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        metadata = self.report_data['metadata']
        
        summary_text = f"""
        This comprehensive 30-day competitive intelligence analysis covers {metadata['total_articles_analyzed']:,} 
        articles from the fraud prevention and risk analytics industry. The report includes TransUnion activity 
        tracking for competitive benchmarking purposes (shown in competitor histogram only).
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Analyzed {metadata['competitor_articles']:,} competitor-related articles (excluding TransUnion)<br/>
        • Tracked {metadata['transunion_articles']:,} TransUnion articles for comparison baseline<br/>
        • Applied context-based deduplication to ensure data quality<br/>
        • Identified top 5 high-impact news items with compact summaries<br/>
        • Analyzed technology trends and fraud domain coverage<br/>
        """
        
        story.append(Paragraph(summary_text, self.styles['BodyText']))

    def _add_top_news_section(self, story):
        """Add top 5 news summaries (compact 150 words)"""
        story.append(Paragraph("Top 5 News Highlights", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        for summary in self.report_data['top_news_summaries']:
            # Title with clickable link
            title_html = f"""
            <b>{summary['rank']}. <link href="{summary['url']}" color="blue">{summary['title']}</link></b>
            """
            story.append(Paragraph(title_html, self.styles['SubHeader']))
            
            # Metadata
            meta_text = f"Source: {summary['source']} | Impact Score: {summary['importance_score']} | Date: {summary['published_date'][:10]}"
            story.append(Paragraph(meta_text, self.styles['BulletText']))
            story.append(Spacer(1, 0.05*inch))
            
            # Compact summary (150 words)
            story.append(Paragraph(summary['summary'], self.styles['BodyText']))
            story.append(Spacer(1, 0.15*inch))

    def _add_competitor_analysis(self, story):
        """Add competitor analysis with TransUnion histogram"""
        story.append(Paragraph("Competitor Activity Analysis", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        # Note about TransUnion
        note_text = f"""
        <b>Note:</b> TransUnion articles ({self.report_data['metadata']['transunion_articles']}) 
        are included in the histogram below for competitive benchmarking purposes only. 
        All strategic analysis focuses on competitor activities.
        """
        story.append(Paragraph(note_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))
        
        # Competitor chart (includes TransUnion)
        if 'competitor_activity' in self.chart_paths:
            img = Image(self.chart_paths['competitor_activity'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        
        # Top competitors (excluding TransUnion from narrative)
        competitor_data = {k: v for k, v in self.report_data['competitor_analysis']['activity_summary'].items() 
                          if k != 'TransUnion'}
        top_competitors = sorted(competitor_data.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_competitors:
            story.append(Paragraph("Top 5 Active Competitors:", self.styles['SubHeader']))
            
            for i, (competitor, count) in enumerate(top_competitors, 1):
                bullet = f"• <b>{competitor}</b>: {count} article mentions"
                story.append(Paragraph(bullet, self.styles['BulletText']))

    def _add_trends_analysis(self, story):
        """Add technology and fraud domain trends"""
        story.append(Paragraph("Technology & Fraud Trends", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        # Technology trends chart
        if 'technology_trends' in self.chart_paths:
            story.append(Paragraph("Technology Trends", self.styles['SubHeader']))
            img = Image(self.chart_paths['technology_trends'], width=5.5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        
        # Fraud domain chart
        if 'fraud_domains' in self.chart_paths:
            story.append(Paragraph("Fraud Domain Coverage", self.styles['SubHeader']))
            img = Image(self.chart_paths['fraud_domains'], width=5.5*inch, height=3*inch)
            story.append(img)

    def _add_strategic_insights(self, story):
        """Add strategic insights (excluding TransUnion) with pie chart"""
        story.append(Paragraph("Strategic Insights", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        insights = self.report_data['strategic_insights']
        
        # Competitive threats with pie chart
        if insights.get('competitive_threats'):
            story.append(Paragraph("Competitive Threats Distribution:", self.styles['SubHeader']))
            story.append(Spacer(1, 0.05*inch))
            
            # Add pie chart if available
            if 'competitive_threats_pie' in self.chart_paths:
                img = Image(self.chart_paths['competitive_threats_pie'], width=5*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.1*inch))
            
            # Add list of top competitors
            story.append(Paragraph("Top Competitive Threats:", self.styles['SubHeader']))
            for threat in insights['competitive_threats'][:8]:
                bullet = f"• <b>{threat['competitor']}</b>: {threat['activity_level']} mentions - {threat['threat_level']} threat level"
                story.append(Paragraph(bullet, self.styles['BulletText']))
            
            story.append(Spacer(1, 0.1*inch))
        
        # Key trends
        if insights.get('key_trends'):
            story.append(Paragraph("Key Technology Trends (Fraud-Specific):", self.styles['SubHeader']))
            
            for trend in insights['key_trends'][:5]:
                bullet = f"• <b>{trend['technology']}</b>: {trend['mention_count']} mentions - {trend['adoption_level']} adoption"
                story.append(Paragraph(bullet, self.styles['BulletText']))

    def _add_citations_section(self, story):
        """Add citations with clickable links - showing top 5 most important"""
        story.append(Paragraph("Key Sources & Citations", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        # Get top 5 articles (most important only)
        top_summaries = self.report_data['top_news_summaries'][:5]
        
        for summary in top_summaries:
            citation_html = f"""
            [{summary['rank']}] <link href="{summary['url']}" color="blue">{summary['source']}</link> - 
            {summary['title']}, {summary['published_date'][:10]}
            """
            story.append(Paragraph(citation_html, self.styles['BulletText']))
            story.append(Spacer(1, 0.05*inch))


async def main():
    """Main function to generate improved PDF"""
    try:
        logger.info("📄 Starting Improved PDF Generation...")
        
        # Find latest CI analysis (look for improved first)
        reports_dir = Path("reports")
        ci_dirs = list(reports_dir.glob("improved_ci_analysis_*"))
        
        if not ci_dirs:
            # Fallback to final CI analysis
            ci_dirs = list(reports_dir.glob("final_ci_analysis_*"))
        
        if not ci_dirs:
            logger.error("No CI analysis found! Run improved_ci_analyzer.py first.")
            return
        
        latest_dir = max(ci_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using analysis from: {latest_dir}")
        
        # Load data
        analysis_file = latest_dir / 'final_ci_analysis.json'
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        dedup_file = latest_dir / 'deduplicated_articles.json'
        with open(dedup_file, 'r', encoding='utf-8') as f:
            deduplicated_articles = json.load(f)
        
        # Generate PDF
        generator = FinalCompactPDFReportGenerator(
            analysis_data, 
            deduplicated_articles, 
            latest_dir
        )
        
        pdf_path = generator.generate_pdf_report()
        
        logger.info(f"✅ Final compact PDF generated successfully!")
        logger.info(f"📁 Location: {pdf_path}")
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
