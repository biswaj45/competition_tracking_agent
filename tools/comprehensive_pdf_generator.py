#!/usr/bin/env python3
"""
Comprehensive PDF Generator with Research & Innovation Section
- Company newsroom analysis
- News aggregator coverage
- Research papers (arXiv, SSRN)
- 90-day comprehensive report
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePDFGenerator:
    def __init__(self, analysis_data: Dict, output_dir: Path):
        """Initialize comprehensive PDF generator"""
        self.report_data = analysis_data
        self.output_dir = output_dir
        self.chart_paths = {}
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self._generate_charts()

    def _setup_custom_styles(self):
        """Setup custom styles"""
        # Custom Title
        try:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#000080'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        except KeyError:
            pass
        
        # Section Header
        try:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#00008B'),
                spaceAfter=12,
                spaceBefore=12,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=colors.HexColor('#000080'),
                borderPadding=5,
                backColor=colors.HexColor('#F0F8FF')
            ))
        except KeyError:
            pass
        
        # Sub Header
        try:
            self.styles.add(ParagraphStyle(
                name='SubHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#0000CD'),
                spaceAfter=8,
                spaceBefore=8,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
        except KeyError:
            pass
        
        # Body Text
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
        
        # Bullet Text
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
        """Generate all charts"""
        plt.style.use('seaborn-v0_8')
        
        self._create_competitor_chart()
        self._create_technology_chart()
        self._create_fraud_domain_chart()
        self._create_research_topics_chart()

    def _create_competitor_chart(self):
        """Create competitor activity chart"""
        try:
            competitor_data = self.report_data['competitor_analysis']['activity_by_company']
            
            if not competitor_data:
                return
            
            sorted_competitors = dict(sorted(competitor_data.items(), key=lambda x: x[1], reverse=True)[:15])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            competitors = list(sorted_competitors.keys())
            activities = list(sorted_competitors.values())
            
            bars = ax.bar(competitors, activities, color='#4682B4', alpha=0.8)
            ax.set_title('Competitor Activity - 90 Days', fontsize=14, fontweight='bold')
            ax.set_xlabel('Competitors', fontsize=12)
            ax.set_ylabel('Article Mentions', fontsize=12)
            
            for bar, value in zip(bars, activities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_path = self.output_dir / 'competitor_activity.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['competitor_activity'] = str(chart_path)
            logger.info("✓ Competitor chart created")
        except Exception as e:
            logger.error(f"Error creating competitor chart: {e}")

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
            ax.set_title('Technology Trends', fontsize=14, fontweight='bold')
            ax.set_xlabel('Mentions', fontsize=12)
            
            for bar, value in zip(bars, counts):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                       f'{value}', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'technology_trends.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['technology_trends'] = str(chart_path)
            logger.info("✓ Technology chart created")
        except Exception as e:
            logger.error(f"Error creating technology chart: {e}")

    def _create_fraud_domain_chart(self):
        """Create fraud domain chart"""
        try:
            fraud_data = self.report_data['fraud_domain_analysis']
            
            if not fraud_data:
                return
            
            sorted_fraud = dict(sorted(fraud_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            domains = [d.replace('_', ' ').title() for d in sorted_fraud.keys()]
            counts = list(sorted_fraud.values())
            
            bars = ax.barh(domains, counts, color='#FF6347', alpha=0.8)
            ax.set_title('Fraud Domain Coverage', fontsize=14, fontweight='bold')
            ax.set_xlabel('Articles', fontsize=12)
            
            for bar, value in zip(bars, counts):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                       f'{value}', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'fraud_domains.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['fraud_domains'] = str(chart_path)
            logger.info("✓ Fraud domain chart created")
        except Exception as e:
            logger.error(f"Error creating fraud domain chart: {e}")

    def _create_research_topics_chart(self):
        """Create research topics chart"""
        try:
            research_data = self.report_data.get('research_and_innovation', {})
            topics = research_data.get('key_topics', {})
            
            if not topics:
                return
            
            sorted_topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            topic_names = [t.replace('_', ' ').title() for t in sorted_topics.keys()]
            counts = list(sorted_topics.values())
            
            bars = ax.barh(topic_names, counts, color='#9370DB', alpha=0.8)
            ax.set_title('Research Paper Topics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Papers', fontsize=12)
            
            for bar, value in zip(bars, counts):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       f'{value}', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            chart_path = self.output_dir / 'research_topics.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_paths['research_topics'] = str(chart_path)
            logger.info("✓ Research topics chart created")
        except Exception as e:
            logger.error(f"Error creating research topics chart: {e}")

    def generate_pdf_report(self) -> str:
        """Generate comprehensive PDF report"""
        pdf_path = self.output_dir / 'Comprehensive_CI_Report_with_Research_90Days.pdf'
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Build report sections
        self._add_cover_page(story)
        story.append(PageBreak())
        
        self._add_executive_summary(story)
        story.append(PageBreak())
        
        self._add_top_news_section(story)
        story.append(PageBreak())
        
        self._add_competitor_analysis(story)
        story.append(PageBreak())
        
        self._add_trends_analysis(story)
        story.append(PageBreak())
        
        # NEW: Research & Innovation Section
        self._add_research_innovation_section(story)
        story.append(PageBreak())
        
        self._add_citations_section(story)
        
        # Build PDF
        doc.build(story)
        logger.info(f"✅ Comprehensive PDF generated: {pdf_path}")
        
        return str(pdf_path)

    def _add_cover_page(self, story):
        """Add cover page"""
        title = Paragraph(
            "Comprehensive Competitive Intelligence Report",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        subtitle = Paragraph(
            "Fraud Solutions & Financial Security<br/>90-Day Analysis with Research Insights",
            self.styles['Heading2']
        )
        story.append(subtitle)
        story.append(Spacer(1, 0.5*inch))
        
        # Metadata
        metadata = self.report_data['metadata']
        research_data = self.report_data.get('research_and_innovation', {})
        
        data = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Period:', '90 Days'],
            ['Total Articles:', f"{metadata['total_articles']:,}"],
            ['Company Newsrooms:', f"{self.report_data['competitor_analysis']['company_newsroom_count']:,}"],
            ['News Sources:', f"{self.report_data['competitor_analysis']['news_aggregator_count']:,}"],
            ['Research Papers:', f"{research_data.get('total_papers', 0):,}"],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F8FF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#000080')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#000080')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(table)

    def _add_executive_summary(self, story):
        """Add executive summary"""
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        metadata = self.report_data['metadata']
        research_data = self.report_data.get('research_and_innovation', {})
        
        summary_text = f"""
        This comprehensive 90-day competitive intelligence report analyzes {metadata['total_articles']:,} 
        articles from company newsrooms and industry news sources, plus {research_data.get('total_papers', 0)} 
        academic research papers on fraud analytics in the financial sector.
        <br/><br/>
        <b>Coverage Includes:</b><br/>
        • Official company newsroom announcements from major competitors<br/>
        • Industry news aggregators and trade publications<br/>
        • Recent academic research papers (arXiv, SSRN)<br/>
        • Technology trends and innovation patterns<br/>
        • Fraud domain analysis across multiple categories<br/>
        """
        
        story.append(Paragraph(summary_text, self.styles['BodyText']))

    def _add_top_news_section(self, story):
        """Add top 5 news"""
        story.append(Paragraph("Top 5 News Highlights", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        for summary in self.report_data['top_news_summaries']:
            title_html = f"""
            <b>{summary['rank']}. <link href="{summary['url']}" color="blue">{summary['title']}</link></b>
            """
            story.append(Paragraph(title_html, self.styles['SubHeader']))
            
            meta_text = f"Source: {summary['source']} | Score: {summary['importance_score']} | {summary['published_date'][:10]}"
            story.append(Paragraph(meta_text, self.styles['BulletText']))
            story.append(Spacer(1, 0.05*inch))
            
            story.append(Paragraph(summary['summary'], self.styles['BodyText']))
            story.append(Spacer(1, 0.15*inch))

    def _add_competitor_analysis(self, story):
        """Add competitor analysis"""
        story.append(Paragraph("Competitor Activity Analysis", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        if 'competitor_activity' in self.chart_paths:
            img = Image(self.chart_paths['competitor_activity'], width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        
        # Top competitors
        competitor_data = self.report_data['competitor_analysis']['activity_by_company']
        top_competitors = sorted(competitor_data.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_competitors:
            story.append(Paragraph("Top 5 Active Competitors:", self.styles['SubHeader']))
            for competitor, count in top_competitors:
                bullet = f"• <b>{competitor}</b>: {count} mentions"
                story.append(Paragraph(bullet, self.styles['BulletText']))

    def _add_trends_analysis(self, story):
        """Add technology and fraud trends"""
        story.append(Paragraph("Technology & Fraud Trends", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        if 'technology_trends' in self.chart_paths:
            story.append(Paragraph("Technology Trends", self.styles['SubHeader']))
            img = Image(self.chart_paths['technology_trends'], width=5.5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        
        if 'fraud_domains' in self.chart_paths:
            story.append(Paragraph("Fraud Domain Coverage", self.styles['SubHeader']))
            img = Image(self.chart_paths['fraud_domains'], width=5.5*inch, height=3*inch)
            story.append(img)

    def _add_research_innovation_section(self, story):
        """Add Research & Innovation section - NEW!"""
        story.append(Paragraph("Research & Innovation", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        research_data = self.report_data.get('research_and_innovation', {})
        
        if not research_data or research_data.get('total_papers', 0) == 0:
            story.append(Paragraph("No research papers found for this period.", self.styles['BodyText']))
            return
        
        # Overview
        overview_text = f"""
        This section highlights recent academic research on fraud analytics and financial security. 
        We analyzed {research_data['total_papers']} research papers published in the last 90 days 
        from leading sources including arXiv and SSRN.
        """
        story.append(Paragraph(overview_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
        
        # Research topics chart
        if 'research_topics' in self.chart_paths:
            story.append(Paragraph("Key Research Topics", self.styles['SubHeader']))
            img = Image(self.chart_paths['research_topics'], width=5.5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.15*inch))
        
        # Innovation trends
        if research_data.get('innovation_trends'):
            story.append(Paragraph("Emerging Innovation Trends", self.styles['SubHeader']))
            
            for trend in research_data['innovation_trends']:
                bullet = f"• <b>{trend['topic']}</b>: {trend['paper_count']} papers ({trend['relevance']} relevance)"
                story.append(Paragraph(bullet, self.styles['BulletText']))
            
            story.append(Spacer(1, 0.15*inch))
        
        # Top research papers
        story.append(Paragraph("Featured Research Papers", self.styles['SubHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        top_papers = research_data.get('top_papers', [])[:5]
        
        for i, paper in enumerate(top_papers, 1):
            # Paper title with link
            title_html = f"""
            <b>{i}. <link href="{paper['url']}" color="blue">{paper['title']}</link></b>
            """
            story.append(Paragraph(title_html, self.styles['SubHeader']))
            
            # Authors and metadata
            authors = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors += ' et al.'
            
            meta_text = f"Authors: {authors} | Published: {paper['published_date'][:10]} | Source: {paper['source']}"
            story.append(Paragraph(meta_text, self.styles['BulletText']))
            story.append(Spacer(1, 0.05*inch))
            
            # Summary/Abstract
            summary = paper['summary'][:400] + "..." if len(paper['summary']) > 400 else paper['summary']
            story.append(Paragraph(summary, self.styles['BodyText']))
            story.append(Spacer(1, 0.15*inch))

    def _add_citations_section(self, story):
        """Add citations"""
        story.append(Paragraph("Key Sources & Citations", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        # News articles
        story.append(Paragraph("Top News Articles:", self.styles['SubHeader']))
        for summary in self.report_data['top_news_summaries']:
            citation_html = f"""
            [{summary['rank']}] <link href="{summary['url']}" color="blue">{summary['source']}</link> - 
            {summary['title']}, {summary['published_date'][:10]}
            """
            story.append(Paragraph(citation_html, self.styles['BulletText']))
        
        story.append(Spacer(1, 0.15*inch))
        
        # Research papers
        research_data = self.report_data.get('research_and_innovation', {})
        if research_data and research_data.get('top_papers'):
            story.append(Paragraph("Research Papers:", self.styles['SubHeader']))
            
            for i, paper in enumerate(research_data['top_papers'][:5], 1):
                authors = ', '.join(paper['authors'][:2])
                citation_html = f"""
                [{i}] {authors} et al. - <link href="{paper['url']}" color="blue">{paper['title']}</link>, 
                {paper['source']}, {paper['published_date'][:10]}
                """
                story.append(Paragraph(citation_html, self.styles['BulletText']))

async def main():
    """Generate comprehensive PDF"""
    try:
        logger.info("📄 Starting Comprehensive PDF Generation...")
        
        # Find latest analysis
        reports_dir = Path("reports")
        analysis_dirs = list(reports_dir.glob("comprehensive_ci_analysis_*"))
        
        if not analysis_dirs:
            logger.error("No analysis found! Run comprehensive_ci_analyzer.py first.")
            return
        
        latest_dir = max(analysis_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using analysis from: {latest_dir}")
        
        # Load analysis
        with open(latest_dir / 'comprehensive_ci_analysis.json', 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Generate PDF
        generator = ComprehensivePDFGenerator(analysis_data, latest_dir)
        pdf_path = generator.generate_pdf_report()
        
        logger.info(f"✅ PDF generation complete!")
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
