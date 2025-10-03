"""Generate a fraud news report (PDF) for the last N days.

This script uses the existing GNews collector and the scraper in tools/scrape_fixed.py
to collect article metadata, scrape full article content, filter for regulatory
penalties, and produce a PDF with two sections:
 - Fraud news (all found articles)
 - Regulatory penalties (subset with penalty-related keywords)

Usage:
  python tools/fraud_report.py --days 3 --pdf-out reports/fraud_report.pdf
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
import json
from typing import List, Dict, Any, Set

# Make sure the repository root is on sys.path so we can import competition_agent
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the existing collector and scraper helpers
from competition_agent.news.gnews_collector import GNewsCollector
import tools.scrape_fixed as scraper_module

try:
    # reportlab for PDF generation
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except Exception:
    # We'll install if missing at runtime if script is executed via the assistant
    raise


DEFAULT_TAGS = [
    'fraud', 'fraud investigation', 'fraud penalty', 'financial fraud', 'scam',
    'fraudster', 'regulatory penalty', 'fraud fine', 'corporate fraud'
]

PENALTY_KEYWORDS = [
    'fine', 'fined', 'penalty', 'penalties', 'penalized', 'penalised', 'settlement',
    'sanction', 'suspended', 'license revoked', 'prosecution', 'charged', 'indicted',
    'sec', 'ftc', 'fca', 'rbi', 'sebi', 'regulator'
]


def is_penalty_related(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    for kw in PENALTY_KEYWORDS:
        if kw in lower:
            return True
    return False


async def collect_and_scrape(days: int, tags: List[str], limit_per_tag: int = 20) -> Dict[str, Any]:
    """Collect articles for the last `days` days for each tag and scrape content.

    Returns a dictionary with articles keyed by canonical URL.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    collector = GNewsCollector()
    articles_by_url: Dict[str, Dict[str, Any]] = {}

    # For each tag, collect and scrape
    for tag in tags:
        print(f"Collecting for tag: {tag}")
        collected = await collector.collect_news(tag, start_date, end_date)
        print(f"  Found {len(collected)} items for {tag}")

        for item in collected[:limit_per_tag]:
            url = item.get('url')
            if not url:
                continue
            if url in articles_by_url:
                # append tag to existing
                articles_by_url[url]['tags'].add(tag)
                continue

            articles_by_url[url] = {
                'title': item.get('title', ''),
                'description': item.get('description', ''),
                'published_date': item.get('published_date', ''),
                'url': url,
                'publisher': item.get('publisher', ''),
                'tags': set([tag]),
                'content': None,
                'scrape_error': None,
            }

    # Scrape each unique article sequentially to avoid overloading remote hosts
    urls = list(articles_by_url.keys())
    print(f"Total unique articles to scrape: {len(urls)}")

    for idx, url in enumerate(urls, start=1):
        entry = articles_by_url[url]
        print(f"\nScraping {idx}/{len(urls)}: {entry['title']}\n  {url}")

        # If it's a Google News wrapper, try to resolve first
        if 'news.google' in url:
            try:
                resolved = await scraper_module.resolve_google_news_url(url)
                if resolved and resolved != url:
                    print(f"  Resolved to: {resolved}")
                    url_to_scrape = resolved
                else:
                    url_to_scrape = url
            except Exception as e:
                print(f"  Resolver error: {e}")
                url_to_scrape = url
        else:
            url_to_scrape = url

        try:
            content = await scraper_module.scrape_url(url_to_scrape)
            if not content:
                entry['scrape_error'] = 'no_content'
            else:
                entry['content'] = content
        except Exception as e:
            entry['scrape_error'] = str(e)

    # convert tag sets to lists for JSON-serializable output
    for v in articles_by_url.values():
        v['tags'] = list(v['tags'])

    return articles_by_url


def build_pdf(articles: Dict[str, Any], pdf_path: Path, tags_used: List[str]):
    """Generate a PDF report with two sections: Fraud news and Regulatory penalties."""
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = styles['Heading1']
    heading2 = styles['Heading2']

    story = []

    # Title
    story.append(Paragraph("Fraud News Report", heading))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Tags searched: {', '.join(tags_used)}", normal))
    story.append(PageBreak())

    # Section: Fraud News (all articles)
    story.append(Paragraph("Fraud News", heading2))
    story.append(Spacer(1, 12))

    sorted_articles = sorted(articles.values(), key=lambda x: x.get('published_date') or '', reverse=True)

    for a in sorted_articles:
        title = a.get('title') or 'Untitled'
        publisher = a.get('publisher') or ''
        url = a.get('url')
        pubdate = a.get('published_date') or ''
        content = a.get('content') or ''
        excerpt = (content[:800] + '...') if content and len(content) > 800 else (content or a.get('description',''))

        story.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
        story.append(Paragraph(f"<i>{publisher}</i> — {pubdate}", normal))
        story.append(Paragraph(f"Source: {url}", normal))
        story.append(Spacer(1, 6))
        story.append(Paragraph(excerpt.replace('\n','<br/>'), normal))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # Section: Regulatory penalties
    story.append(Paragraph("Regulatory penalties", heading2))
    story.append(Spacer(1, 12))

    penalty_items = []
    for a in sorted_articles:
        combined = ' '.join(filter(None, [a.get('title', ''), a.get('content') or '', a.get('description', '')]))
        if is_penalty_related(combined):
            penalty_items.append(a)

    if not penalty_items:
        story.append(Paragraph("No regulatory penalties detected in scraped articles.", normal))
    else:
        for a in penalty_items:
            title = a.get('title') or 'Untitled'
            publisher = a.get('publisher') or ''
            url = a.get('url')
            pubdate = a.get('published_date') or ''
            content = a.get('content') or a.get('description','')

            story.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            story.append(Paragraph(f"<i>{publisher}</i> — {pubdate}", normal))
            story.append(Paragraph(f"Source: {url}", normal))
            story.append(Spacer(1, 6))
            story.append(Paragraph(content.replace('\n','<br/>'), normal))
            story.append(Spacer(1, 12))

    doc.build(story)


async def main_async(args):
    tags = args.tags or DEFAULT_TAGS
    days = args.days
    limit = args.limit

    articles = await collect_and_scrape(days, tags, limit_per_tag=limit)

    # Save intermediate JSON
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Wrote scraped metadata to: {out_json}")

    # Build PDF
    pdf_path = Path(args.pdf_out)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(articles, pdf_path, tags)
    print(f"PDF report written to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate fraud news PDF for last N days')
    parser.add_argument('--days', type=int, default=3, help='Number of days back to collect (default: 3)')
    parser.add_argument('--tags', nargs='*', help='List of search tags/queries')
    parser.add_argument('--limit', type=int, default=20, help='Max articles per tag to collect')
    parser.add_argument('--pdf-out', default='reports/fraud_report.pdf', help='Output PDF path')
    parser.add_argument('--json-out', default='reports/fraud_scraped.json', help='Output JSON with scraped contents')
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    asyncio.run(main_async(args))
