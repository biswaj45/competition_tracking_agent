

# --- COMPETITIVE INTELLIGENCE ANALYST WORKFLOW ---
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
import json
import os
from typing import List, Dict, Any, Set
import hashlib

# Ensure repo root on sys.path
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from competition_agent.news.gnews_collector import GNewsCollector
except Exception:
    GNewsCollector = None
try:
    import tools.scrape_fixed as scraper_module
except Exception:
    scraper_module = None
try:
    from competition_agent.llm.hf_analyzer import HFAnalyzer
except Exception:
    HFAnalyzer = None

REPORTLAB_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    import numpy as np
    from datetime import timezone
except Exception:
    MATPLOTLIB_AVAILABLE = False

    import time
    from dateutil import parser as date_parser

# --- CONFIGURABLE PARAMETERS ---
PRIMARY_WINDOW_DAYS = 7  # One week primary window
TREND_WINDOW_DAYS = 28  # 4 weeks trend window
FOCUS_REGIONS = ["LATAM", "Africa", "US", "UK", "Canada", "APAC", "Spain", "India"]
FOCUS_APAC = ["Hong Kong", "Philippines"]
WATCHLIST = ["Experian", "Equifax", "CRIF", "IDfy", "Shield", "Jumio", "FICO", "BureauID"]
ADDITIONAL = ["Feedzai", "Featurespace", "BioCatch", "SEON", "Alloy", "Sardine", "Ravelin"]
ALL_COMPETITORS = WATCHLIST + ADDITIONAL
TU_BASELINE = ["Digital Onboarding", "TruValidate Solutions", "IDV", "Device Intelligence"]
SCOPE_KEYWORDS = [
    "fraud", "account opening", "ATO", "device risk", "id proofing", "kyc", "behavioral biometrics",
    "synthetic id", "mule detection", "payment fraud"
]
SCOPE_EXCLUDE = ["cybersecurity", "malware", "ransomware", "phishing"]
# Add corruption and legal/penal content to explicit excludes for this CI scope
SCOPE_EXCLUDE += ["corruption", "bribery", "embezzlement", "sentenced", "prison", "trial"]

# --- HELPER: Canonicalize competitor names ---
def canonical_competitor(name: str) -> str:
    name = name.lower().replace("inc.", "").replace(",", "").replace(".", "").strip()
    for c in ALL_COMPETITORS:
        if name == c.lower() or c.lower() in name:
            return c
    # Fuzzy match for common aliases
    for c in ALL_COMPETITORS:
        if c.lower()[:5] in name:
            return c
    return name.title()

# --- HELPER: Tag region ---
def tag_region(text: str) -> str:
    t = text.lower()
    for r in FOCUS_REGIONS:
        if r.lower() in t:
            return r
    for apac in FOCUS_APAC:
        if apac.lower() in t:
            return "APAC"
    return "Global Trends"


# --- HELPER: Simple impact scoring ---
def compute_impact(article: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic impact scorer. Returns ImpactScore (0-100), Bucket, Rationale, Confidence."""
    text = ' '.join([str(article.get(k, '') or '') for k in ('title', 'summary', 'content', 'description')]).lower()
    score = 10
    reasons = []

    # High-value competitors increase baseline
    comp = (article.get('competitor') or '').lower()
    if comp:
        if any(w.lower() in comp for w in WATCHLIST):
            score += 30
            reasons.append('watchlist competitor')
        elif any(w.lower() in comp for w in ADDITIONAL):
            score += 15
            reasons.append('market competitor')

    # Keyword signals
    high_kw = ['breach', 'major', 'fraud', 'regulatory', 'fined', 'lawsuit', 'downtime', 'outage', 'exploit']
    mid_kw = ['partnership', 'acquisition', 'investment', 'launch', 'announce', 'rollout', 'expand']
    low_kw = ['discussion', 'commentary', 'opinion', 'interview']

    for kw in high_kw:
        if kw in text:
            score += 20
            reasons.append(f'contains:{kw}')
            break
    for kw in mid_kw:
        if kw in text:
            score += 10
            reasons.append(f'contains:{kw}')
            break
    for kw in low_kw:
        if kw in text:
            score += 0
            reasons.append(f'contains:{kw}')
            break

    # numeric/currency signals
    if re.search(r'\$\s?\d+|\b\d+\s?(million|billion|k)\b', text):
        score += 15
        reasons.append('financial_figure')

    # Longer articles / richer summaries increase confidence
    content_len = len((article.get('content') or '').split())
    if content_len > 300:
        score += 10
        reasons.append('long_content')

    # Cap score
    score = max(0, min(100, score))

    # Bucket assignment
    if score >= 70:
        bucket = 'High'
    elif score >= 30:
        bucket = 'Medium'
    else:
        bucket = 'Low'

    # Confidence heuristic: fraction of matched signals
    conf = min(0.99, 0.1 + 0.2 * len(reasons))

    rationale = '; '.join(reasons) if reasons else 'heuristic baseline'
    return {
        'ImpactScore': int(score),
        'Bucket': bucket,
        'Rationale': rationale,
        'Confidence': round(conf, 2)
    }


# --- HELPER: Deduplicate articles ---
def dedup_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggressive dedupe by normalized title + url/content fingerprint. Prefer higher ImpactScore and richer summary."""
    seen = {}

    def norm_text(t: str) -> str:
        return re.sub(r'[^0-9a-z]', '', (t or '').lower())[:140]

    for a in articles:
        title_key = norm_text(a.get('title') or a.get('content')[:120])
        url = (a.get('url') or '').strip()
        # content fingerprint
        content = (a.get('content') or '').strip()[:400]
        fp = hashlib.sha1(content.encode('utf-8')).hexdigest()[:10] if content else ''
        key = (title_key, url or fp)

        if key in seen:
            existing = seen[key]
            # prefer higher ImpactScore
            if (a.get('ImpactScore') or 0) > (existing.get('ImpactScore') or 0):
                seen[key] = a
            else:
                # if existing missing summary and new has one, keep new
                if not existing.get('llm_digest') and a.get('llm_digest'):
                    seen[key] = a
        else:
            seen[key] = a

    return list(seen.values())


# --- HELPER: Scope filter ---
def in_scope(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # must include at least one scope keyword
    if not any(k.lower() in t for k in SCOPE_KEYWORDS):
        return False
    # must not contain explicit excludes
    if any(e.lower() in t for e in SCOPE_EXCLUDE):
        return False
    return True


def load_existing_data(raw_scrapes_dir):
    """Load existing scraped data from raw_scrapes directory (parses .txt files).

    Splits files that contain multiple 'Title:' entries, extracts Title/Source/URL/Date
    when present, generates a short human-friendly summary, infers competitor and region,
    computes impact, and returns deduplicated articles.
    """
    articles = []
    raw_scrapes_dir = Path(raw_scrapes_dir)

    def humanize_summary(raw_summary: str, content: str) -> str:
        if raw_summary:
            s = re.sub(r"STRUCTURED:\s*\{[\s\S]*?\}\s*", "", raw_summary)
            s = re.sub(r"\s+", " ", s).strip()
            if len(s.split()) > 8 and any(p in s for p in ['.', 'Key Points', 'Summary']):
                return s
        if content:
            words = content.split()
            snippet = ' '.join(words[:60])
            if not snippet.endswith('.'):
                snippet = snippet.rstrip() + '.'
            return snippet
        return "No readable summary available."

    article_files = [f for f in raw_scrapes_dir.glob('*.txt') if not f.name.endswith('_summary.txt')]
    for file in sorted(article_files):
        raw = file.read_text(encoding='utf-8')
        chunks = re.split(r"(?=\n?Title:\s)", raw)
        if len(chunks) == 1:
            chunks = [raw]
        for chunk in chunks:
            if not chunk.strip():
                continue
            article = {}
            article['content'] = chunk.strip()
            summary_file = file.parent / f"{file.stem}_summary.txt"
            raw_summary = ''
            if summary_file.exists():
                raw_summary = summary_file.read_text(encoding='utf-8')
            article['summary'] = humanize_summary(raw_summary, article['content'])
            title_match = re.search(r"Title:\s*(.+)", chunk)
            source_match = re.search(r"Source:\s*(.+)", chunk)
            url_match = re.search(r"URL:\s*(.+)", chunk)
            date_match = re.search(r"Date:\s*(.+)", chunk)
            article['title'] = title_match.group(1).strip() if title_match else file.stem
            article['publisher'] = source_match.group(1).strip() if source_match else 'Unknown'
            article['url'] = url_match.group(1).strip() if url_match else f"https://example.com/{file.stem}"
            if date_match:
                article['published_date'] = date_match.group(1).strip()
            else:
                article['published_date'] = datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            article['competitor'] = canonical_competitor(article.get('publisher') or '')
            article['region'] = tag_region(article.get('content') or '')
            article['llm_struct'] = {}
            article['llm_digest'] = ' '.join(article['summary'].split()[:40])
            impact = compute_impact(article)
            article.update(impact)
            articles.append(article)

    return dedup_articles(articles)


def clean_summary_text(s: str) -> str:
    if not s:
        return ''
    # remove repeated STRUCTURED blocks and weird control chars
    s = re.sub(r"STRUCTURED:\s*\{[\s\S]*?\}\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # If it's too short or still contains JSON-like tokens, fallback empty
    if len(s.split()) < 6 or '{' in s or '}' in s:
        return ''
    return s


def normalize_and_merge(articles: List[dict]) -> List[dict]:
    """Ensure every article has Impact fields, readable summary, and merge obvious duplicates."""
    # Ensure impact fields
    for a in articles:
        if a.get('ImpactScore') is None:
            imp = compute_impact(a)
            a.update(imp)
        # Prefer a cleaned human-readable summary for llm_digest
        cleaned = clean_summary_text(a.get('summary') or '')
        if cleaned:
            a['llm_digest'] = ' '.join(cleaned.split()[:40])
        else:
            # If existing llm_digest looks like structured output, replace it
            existing = a.get('llm_digest') or ''
            if ('{' in existing and '}' in existing) or len(existing.split()) < 6:
                a['llm_digest'] = ' '.join((a.get('content') or '').split()[:40])

    # Merge duplicates by canonical competitor + normalized title
    merged = {}
    def norm_title(t: str) -> str:
        return re.sub(r"[^0-9a-z]", "", (t or '').lower())[:120]

    for a in articles:
        key = (canonical_competitor(a.get('competitor') or ''), norm_title(a.get('title') or ''))
        if key in merged:
            # prefer the one with higher ImpactScore and richer summary
            existing = merged[key]
            if (a.get('ImpactScore',0) > existing.get('ImpactScore',0)):
                merged[key] = a
            else:
                # keep existing; but extend content if empty
                if not existing.get('summary') and a.get('summary'):
                    existing['summary'] = a['summary']
        else:
            merged[key] = a

    out = list(merged.values())
    # Final dedupe pass with dedup_articles
    out = dedup_articles(out)
    return out

# --- HELPER: Generate visualizations ---
def generate_visualizations(articles: List[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('default')  # Use default style
    
    # 1. News Volume Trend
    dates = []
    for a in articles:
        pd = a.get('published_date')
        if not pd:
            continue
        try:
            # Accept ISO or RFC-like dates
            parsed = date_parser.parse(pd)
            # make naive local datetime for plotting
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(tz=None).replace(tzinfo=None)
            dates.append(parsed)
        except Exception:
            # skip unparsable dates
            continue

    plt.figure(figsize=(12, 6))
    if dates:
        plt.hist(dates, bins=20, color='skyblue', edgecolor='black')
        plt.title('News Volume Trend')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No dated articles available', ha='center', va='center')
    plt.tight_layout()
    vol_trend_path = out_dir / 'volume_trend.png'
    plt.savefig(vol_trend_path)
    plt.close()
    
    # 2. Competitor Heat Map
    comp_region_data = defaultdict(lambda: defaultdict(int))
    for a in articles:
        comp = a.get('competitor')
        region = a.get('region')
        if comp and region:
            comp_region_data[comp][region] += 1
    
    comp_list = sorted(set(c for c in comp_region_data.keys()))
    region_list = sorted(set(r for d in comp_region_data.values() for r in d.keys()))
    
    plt.figure(figsize=(12, 8))
    if comp_list and region_list:  # Only create heatmap if we have data
        heat_data = np.zeros((len(comp_list), len(region_list)))
        for i, comp in enumerate(comp_list):
            for j, region in enumerate(region_list):
                heat_data[i, j] = comp_region_data[comp][region]
        
        sns.heatmap(heat_data, xticklabels=region_list, yticklabels=comp_list,
                    annot=True, fmt='g', cmap='YlOrRd')
    else:
        plt.text(0.5, 0.5, 'No regional data available', ha='center', va='center')
    plt.title('Competitor Coverage by Region')
    plt.xlabel('Region')
    plt.ylabel('Competitor')
    plt.tight_layout()
    heatmap_path = out_dir / 'coverage_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    
    # 3. Impact Distribution
    impact_data = defaultdict(int)
    for a in articles:
        impact_data[a.get('Bucket', 'Unknown')] += 1
    
    plt.figure(figsize=(8, 6))
    plt.pie(impact_data.values(), labels=impact_data.keys(), autopct='%1.1f%%', 
            colors=['lightcoral', 'lightgreen', 'lightskyblue'])
    plt.title('Impact Distribution')
    plt.tight_layout()
    impact_path = out_dir / 'impact_dist.png'
    plt.savefig(impact_path)
    plt.close()
    
    return {
        'volume_trend': str(vol_trend_path),
        'coverage_heatmap': str(heatmap_path),
        'impact_dist': str(impact_path)
    }

# --- MAIN COLLECTION AND SCRAPING ---
async def collect_and_scrape(primary_days: int, trend_days: int, limit_per_tag: int = 20) -> dict:
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=primary_days)
    collector = GNewsCollector()
    hf = HFAnalyzer()
    all_articles = []
    # Use competitor names as tags
    for comp in ALL_COMPETITORS:
        print(f"Collecting for competitor: {comp}")
        collected = await collector.collect_news(comp, start_date, end_date)
        print(f"  Found {len(collected)} items for {comp}")
        for item in collected[:limit_per_tag]:
            url = item.get('url')
            if not url:
                continue
            # Scrape
            if 'news.google' in url:
                try:
                    resolved = await scraper_module.resolve_google_news_url(url)
                    url_to_scrape = resolved if resolved else url
                except Exception:
                    url_to_scrape = url
            else:
                url_to_scrape = url
            try:
                content = await scraper_module.scrape_url(url_to_scrape)
            except Exception:
                content = None
            # Summarize with T5
            # Generate LLM-based summary (ensure we have both structured analysis and a readable summary)
            summary = ""
            llm_struct = {}
            if content:
                try:
                    # structured analysis (features, key_quotes, etc.)
                    llm_struct = hf.analyze_content(content, comp, SCOPE_KEYWORDS)
                except Exception:
                    llm_struct = {}
                try:
                    # readable article summary (fallback to a short extract if unavailable)
                    # HFAnalyzer provides generate_article_summary which expects keys title, content, published_date, source/publisher
                    article_for_summary = {
                        "title": item.get("title") or "",
                        "content": content,
                        "published_date": item.get("published_date"),
                        "source": item.get("publisher")
                    }
                    summary = hf.generate_article_summary(article_for_summary) or llm_struct.get("impact_level", "")
                except Exception:
                    # best-effort fallback
                    summary = llm_struct.get("impact_level", "") or (content[:500] + "...")
            # Tag competitor, region, scope
            competitor = canonical_competitor(item.get("publisher", "") or comp)
            region = tag_region((item.get("description") or "") + " " + (content or ""))
            # Scope filter
            if not in_scope((item.get("title") or "") + " " + (item.get("description") or "") + " " + (content or "")):
                continue
            # Citation check
            if not (item.get("title") and item.get("publisher") and url and item.get("published_date")):
                continue
            article = {
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "url": url,
                "published_date": item.get("published_date"),
                "description": item.get("description"),
                "content": content,
                "summary": summary,
                "llm_struct": llm_struct,
                # also provide a short LLM digest field for table display
                "llm_digest": (summary.split('\n')[:3] and '\n'.join(summary.split('\n')[:3])) or (summary[:300] if summary else ""),
                "competitor": competitor,
                "region": region,
            }
            article.update(compute_impact(article))
            all_articles.append(article)
    # Deduplicate and normalize (ensure impact fields and human summaries)
    all_articles = dedup_articles(all_articles)
    all_articles = normalize_and_merge(all_articles)
    return all_articles

# --- PDF REPORT GENERATION ---
def build_pdf(articles: List[dict], pdf_path: Path, meta: dict):
    # Normalize and merge duplicates before building PDF
    articles = normalize_and_merge(articles)
    # Remove existing file if present and retry with delay if needed
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if pdf_path.exists():
                os.remove(str(pdf_path))  # Convert to string path
                time.sleep(1)  # Wait for file handle to be released
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Warning: could not remove existing PDF {pdf_path} after {attempt+1} attempts: {e}")
            time.sleep(1)  # Wait before retry
    
    # Create parent directory if it doesn't exist
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = styles['Heading1']
    heading2 = styles['Heading2']
    story = []
    # Cover
    story.append(Paragraph("Competitive Intelligence Report — Fraud Solutions Industry", heading))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Primary Window: {meta['primary_window']}d, Trend Window: {meta['trend_window']}d", normal))
    story.append(PageBreak())
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading2))
    story.append(Spacer(1, 8))
    # Generate a short executive paragraph: counts by impact bucket and top companies mentioned
    total = len(articles)
    high = sum(1 for a in articles if a.get('Bucket') == 'High')
    medium = sum(1 for a in articles if a.get('Bucket') == 'Medium')
    low = sum(1 for a in articles if a.get('Bucket') == 'Low')
    top_comps = {}
    for a in articles:
        top_comps[a.get('competitor')] = top_comps.get(a.get('competitor'), 0) + 1
    top_list = sorted(top_comps.items(), key=lambda x: x[1], reverse=True)[:5]
    top_str = ", ".join([f"{c} ({n})" for c, n in top_list]) if top_list else "None"
    exec_para = f"This report covers {total} events in the last {meta['primary_window']} day(s). Impact distribution: High={high}, Medium={medium}, Low={low}. Top entities in coverage: {top_str}."
    story.append(Paragraph(exec_para, normal))
    story.append(Spacer(1, 8))
    # Top 5 short takeaways for quick reading
    takeaways = []
    for a in articles:
        if a.get('competitor') in WATCHLIST and a.get('region') in FOCUS_REGIONS:
            snippet = a.get('llm_digest') or a.get('summary') or a.get('title')
            takeaways.append(f"{a['competitor']} ({a['region']}): {snippet} [Impact: {a.get('Bucket')}]")
        if len(takeaways) >= 5:
            break
    if not takeaways:
        takeaways = ["No Material Updates"]
    for t in takeaways:
        story.append(Paragraph(f"- {t}", normal))
    story.append(Spacer(1, 12))
    # Competitor Tracking
    story.append(Paragraph("Competitor Tracking", heading2))
    story.append(Spacer(1, 8))
    # Table: Competitor Tracking (full title)
    table_data = [["Competitor", "Region", "Title", "Impact", "Date"]]
    for a in articles:
        # Use full title rather than truncated sentence
        table_data.append([a.get("competitor"), a.get("region"), a.get("title"), a.get("Bucket"), a.get("published_date")])
    t = Table(table_data, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    # Product Gap Analysis (placeholder)
    story.append(Paragraph("Product Gap Analysis", heading2))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Baseline incomplete. Gaps not confirmed.", normal))
    story.append(Spacer(1, 12))
    # Summaries of Important News
    story.append(Paragraph("Summaries of Important News", heading2))
    story.append(Spacer(1, 8))
    # Sort articles by impact score and get top stories
    important_news = sorted([a for a in articles if a.get('ImpactScore', 0) >= 70], 
                          key=lambda x: x.get('ImpactScore', 0), reverse=True)[:5]
    if important_news:
        for news in important_news:
            title = f"{news.get('competitor')} - {news.get('title')}"
            story.append(Paragraph(title, ParagraphStyle('NewsSummaryTitle', parent=normal, fontName='Helvetica-Bold')))
            story.append(Spacer(1, 4))
            summary = news.get('llm_digest') or news.get('summary') or news.get('description')
            if summary:
                # Ensure summary is not too long (max 150 words)
                summary_words = summary.split()[:150]
                summary = ' '.join(summary_words)
                if len(summary_words) == 150:
                    summary += '...'
            story.append(Paragraph(summary or "No summary available", normal))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No high-impact news in this period.", normal))
    story.append(Spacer(1, 12))
    
    # Visuals & Trends
    story.append(Paragraph("Visuals & Trends", heading2))
    story.append(Spacer(1, 8))
    
    # Generate and add visualizations
    viz_dir = Path('reports/visualizations')
    viz_paths = generate_visualizations(articles, viz_dir)
    
    # Add volume trend
    story.append(Paragraph("News Volume Trend", heading2))
    story.append(Spacer(1, 4))
    story.append(Image(viz_paths['volume_trend'], width=7*inch, height=3.5*inch))
    story.append(Spacer(1, 12))
    
    # Add coverage heatmap
    story.append(Paragraph("Regional Coverage Heatmap", heading2))
    story.append(Spacer(1, 4))
    story.append(Image(viz_paths['coverage_heatmap'], width=7*inch, height=4.5*inch))
    story.append(Spacer(1, 12))
    
    # Add impact distribution
    story.append(Paragraph("Impact Distribution", heading2))
    story.append(Spacer(1, 4))
    story.append(Image(viz_paths['impact_dist'], width=5*inch, height=3.5*inch))
    story.append(Spacer(1, 12))
    # Recommendations (placeholder)
    story.append(Paragraph("Recommendations", heading2))
    story.append(Spacer(1, 8))
    story.append(Paragraph("- Accelerate AI explainability features.\n- Explore partnerships in behavioral biometrics.\n- Expand KYC/AML product marketing.", normal))
    story.append(Spacer(1, 12))
    # Appendix: Citations (hyperlinked 'link' text)
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Citations", heading2))
    for a in articles:
        safe_url = a.get('url') or ''
        # ReportLab supports basic <a href=> in Paragraph with style
        ptext = f"{a.get('title')} — {a.get('publisher')} — <a href=\"{safe_url}\">link</a> — {a.get('published_date')}"
        story.append(Paragraph(ptext, normal))
    # Build PDF with retry
    for attempt in range(max_attempts):
        try:
            doc.build(story)
            break
        except PermissionError as e:
            if attempt == max_attempts - 1:
                print(f"Permission error writing PDF {pdf_path} after {attempt+1} attempts: {e}")
                raise
            time.sleep(1)  # Wait before retry
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise

# --- JSON OUTPUT ---
def build_json(articles: List[dict], meta: dict, json_path: Path):
    # Structure per prompt
    # Also write raw scrapes and LLM summaries to a folder for audit
    raw_dir = Path("reports/raw_scrapes")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(articles):
        # Raw content
        try:
            raw_path = raw_dir / f"{i:03d}_{re.sub(r'[^0-9a-zA-Z_-]', '_', a.get('title','untitled'))[:120]}.txt"
            with raw_path.open('w', encoding='utf-8') as rf:
                rf.write(f"Title: {a.get('title')}\nSource: {a.get('publisher')}\nURL: {a.get('url')}\nDate: {a.get('published_date')}\n\n")
                rf.write(a.get('content') or '')
        except Exception:
            pass
        # LLM summary
        try:
            sum_path = raw_dir / f"{i:03d}_{re.sub(r'[^0-9a-zA-Z_-]', '_', a.get('title','untitled'))[:120]}_summary.txt"
            with sum_path.open('w', encoding='utf-8') as sf:
                sf.write(a.get('summary') or '')
                sf.write('\n\nSTRUCTURED:\n')
                sf.write(json.dumps(a.get('llm_struct', {}), ensure_ascii=False, indent=2))
        except Exception:
            pass

    out = {
        "report_meta": meta,
        "events": articles,
        "gaps": {},
        "visualization_data": {},
        "recommendations": {},
        "citations": {i: {
            "title": a["title"], "publisher": a["publisher"], "url": a["url"], "date": a["published_date"],
            "link_text": "link"
        } for i, a in enumerate(articles)},
        "known_unknowns": {}
    }
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# --- MAIN ---
async def main_async(args):
    meta = {
        "primary_window": PRIMARY_WINDOW_DAYS,
        "trend_window": TREND_WINDOW_DAYS,
        "focus_regions": FOCUS_REGIONS,
        "watchlist": WATCHLIST,
        "run_time": datetime.utcnow().isoformat()
    }
    articles = await collect_and_scrape(PRIMARY_WINDOW_DAYS, TREND_WINDOW_DAYS, limit_per_tag=10)
    # Save JSON
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    build_json(articles, meta, out_json)
    print(f"Wrote structured JSON to: {out_json}")
    # Build PDF
    pdf_path = Path(args.pdf_out)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(articles, pdf_path, meta)
    print(f"PDF report written to: {pdf_path}")



def main():
    parser = argparse.ArgumentParser(description='Competitive Intelligence Report Generator')
    parser.add_argument('--pdf-out', default='reports/ci_report.pdf', help='Output PDF path')
    parser.add_argument('--json-out', default='reports/ci_report.json', help='Output JSON path')
    parser.add_argument('--no-scrape', action='store_true', help='Use existing scraped data')
    parser.add_argument('--raw-dir', default='reports/raw_scrapes', help='Directory containing raw scrape files')
    return parser.parse_args()

if __name__ == '__main__':
    args = main()
    
    if args.no_scrape:
        # Use existing data
        articles = load_existing_data(args.raw_dir)
        # Normalize/merge duplicates and ensure impact fields
        articles = normalize_and_merge(articles)
        meta = {
            "primary_window": PRIMARY_WINDOW_DAYS,
            "trend_window": TREND_WINDOW_DAYS,
            "focus_regions": FOCUS_REGIONS,
            "watchlist": WATCHLIST,
            "run_time": datetime.utcnow().isoformat()
        }
        # Save JSON
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        build_json(articles, meta, out_json)
        print(f"Wrote structured JSON to: {out_json}")
        # Build PDF (if available)
        if REPORTLAB_AVAILABLE:
            pdf_path = Path(args.pdf_out)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            build_pdf(articles, pdf_path, meta)
            print(f"PDF report written to: {pdf_path}")
        else:
            print("ReportLab not available; skipping PDF generation.")
    else:
        asyncio.run(main_async(args))
