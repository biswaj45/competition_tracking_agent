"""
Scrape article URLs from a collected JSON file, extract article text, and optionally run HFAnalyzer to summarize.
Usage:
  python tools/scrape_and_summarize.py input.json [--output out.json] [--txt-out dir] [--summarize] [--limit N]

Defaults:
  --output: input_updated.json
  --txt-out: collected_txt_updated

Notes:
- Uses requests + BeautifulSoup for scraping; it's a heuristic extractor (tries             # Resolve Google News pages to underlying article URLs when possible
            url_to_scrape = url
            if url and 'news.google' in url:
                try:
                    resolved = await resolve_google_news_url(url)
                    if resolved:
                        print(f"Resolved Google News URL to: {resolved}")
                        url_to_scrape = resolved
                    else:
                        print(f"Could not resolve Google News URL; will attempt to scrape the Google News page itself: {url}")
                except Exception as e:
                    print(f"Error resolving Google News URL: {e}")

            print(f"Scraping ({category}) {i+1}/{limit}: {url_to_scrape}")
            try:
                scraped = await scrape_url(url_to_scrape)
            except Exception as e:
                print(f"Error scraping URL: {e}")
                scraped = None
            if scraped:, main tag, then largest <div> by text length, then fallback to <p> gather).
- If --summarize is passed, the script will attempt to import and use `competition_agent.llm.hf_analyzer.HFAnalyzer` (this will load Hugging Face models and may be slow).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
import urllib.parse
import asyncio
from playwright.async_api import async_playwright


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Try common tags first
    article_tag = soup.find("article")
    if article_tag and article_tag.get_text(strip=True):
        return article_tag.get_text(separator="\n\n", strip=True)

    main_tag = soup.find("main")
    if main_tag and main_tag.get_text(strip=True):
        return main_tag.get_text(separator="\n\n", strip=True)

    # Heuristic: find the <div> with the most text
    divs = soup.find_all("div")
    best = None
    best_len = 0
    for d in divs:
        text = d.get_text(separator=" ", strip=True)
        if len(text) > best_len:
            best_len = len(text)
            best = d
    if best and best.get_text(strip=True):
        return best.get_text(separator="\n\n", strip=True)

    # Fallback: join paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


async def scrape_url(url: str, timeout: float = 30.0) -> Optional[str]:
    # First try simple HTTP GET
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ArticleScraper/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout/2, allow_redirects=True)
        resp.raise_for_status()
        text = extract_text_from_html(resp.text)
        if text and len(text.strip()) > 200:  # if we got meaningful content
            return clean_text(text)
    except Exception as e:
        print(f"Static scraping failed for {url}: {e}")

    # If simple GET fails or returns too little content, try with Playwright
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Navigate and wait for content
            await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            await page.wait_for_timeout(2000)  # wait for any dynamic content
            
            # Try to find article content using common selectors
            content = await page.evaluate('''() => {
                function getText(elem) {
                    return elem ? elem.innerText.trim() : '';
                }
                
                // Try article tag first
                const article = document.querySelector('article');
                if (article) {
                    return getText(article);
                }
                
                // Try main content area
                const main = document.querySelector('main');
                if (main) {
                    return getText(main);
                }
                
                // Try common article content selectors
                const contentSelectors = [
                    '[role="main"]',
                    '.article-content',
                    '.post-content',
                    '.entry-content',
                    '#content-body',
                ];
                
                for (const selector of contentSelectors) {
                    const elem = document.querySelector(selector);
                    if (elem) {
                        return getText(elem);
                    }
                }
                
                // Fallback: get all paragraph text
                const paragraphs = Array.from(document.querySelectorAll('p'))
                    .map(p => getText(p))
                    .filter(text => text.length > 40);  // filter out tiny fragments
                
                return paragraphs.join('\\n\\n');
            }''')
            
            await browser.close()
            if content:
                return clean_text(content)

    except Exception as e:
        print(f"Playwright scraping failed for {url}: {e}")

    return None


async def resolve_google_news_url(url: str, timeout: float = 30.0) -> Optional[str]:
    """Given a news.google.com article URL, try to find the original article URL.
    Uses Playwright to handle JavaScript-driven pages.
    """
    try:
        # Extract the actual URL from Google News format
        if "articles/" in url:
            article_id = url.split("articles/")[1].split("?")[0]
            decoded_url = article_id.replace("CBMi", "")
            import base64
            try:
                real_url = base64.b64decode(decoded_url + "=" * (-len(decoded_url) % 4)).decode('utf-8')
                if real_url.startswith('http'):
                    return real_url
            except:
                pass

        # If direct extraction fails, try with Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            # Add handler to capture navigation events
            final_url = None
            
            async def handle_response(response):
                nonlocal final_url
                if (response.ok and 
                    response.request.resource_type == "document" and
                    'news.google' not in response.url):
                    final_url = response.url
            
            page.on("response", handle_response)
            
            # Navigate and wait for network idle
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                await page.wait_for_timeout(2000)  # extra 2s for any late redirects
            except Exception as e:
                print(f"Navigation failed: {e}")
                await browser.close()
                return None

            if final_url:
                await browser.close()
                return final_url

            # Try to find and click the main article link
            try:
                # Wait for any article link to be available
                await page.wait_for_selector('a[href]:not([href*="news.google"])', timeout=5000)
                
                # Get all non-Google-News links
                links = await page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a[href]'))
                        .map(a => a.href)
                        .filter(href => !href.includes('news.google'))
                }''')
                
                if links and len(links) > 0:
                    await browser.close()
                    return links[0]  # Return the first non-Google link found
            except Exception as e:
                print(f"Failed to extract link: {e}")
            
            await browser.close()

    except Exception as e:
        print(f"Failed to resolve URL {url}: {e}")
    
    return None
                article_link = await page.evaluate('''() => {
                    // Check all links
                    for (const a of document.querySelectorAll('a[href]')) {
                        const href = a.href;
                        if (href && !href.includes('news.google')) {
                            return href;
                        }
                    }
                    // Check meta tags
                    const og = document.querySelector('meta[property="og:url"]');
                    if (og && !og.content.includes('news.google')) {
                        return og.content;
                    }
                    return null;
                }''')
                if article_link:
                    final_url = article_link
            
            await browser.close()
            if final_url:
                return final_url

    except Exception as e:
        print(f"Playwright URL resolution failed for {url}: {e}")

    return None


def export_txt_for_articles(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for category, items in data.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for i, article in enumerate(items, start=1):
            title = article.get("title") or article.get("headline") or "untitled"
            company = article.get("company") or "unknown"
            safe_title = re.sub(r'[^0-9a-zA-Z_\-]', '_', title)[:120]
            filename = f"{i:03d}_{company}_{safe_title}.txt"
            path = cat_dir / filename
            body = article.get("content") or article.get("content_scraped") or article.get("summary") or ""
            with path.open("w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n")
                f.write(f"Company: {company}\n")
                f.write(f"Source: {article.get('source', '')}\n")
                f.write(f"URL: {article.get('url', '')}\n\n")
                f.write(body)
            total += 1
    print(f"Exported {total} txt files to {out_dir.resolve()}")


async def process_articles(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_updated.json")
    txt_out = Path(args.txt_out)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Optionally initialize HFAnalyzer
    hf = None
    if args.summarize:
        try:
            # Ensure project root is on sys.path
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from competition_agent.llm.hf_analyzer import HFAnalyzer
            hf = HFAnalyzer()
        except Exception as e:
            print(f"Failed to initialize HFAnalyzer: {e}")
            hf = None

    for category, items in data.items():
        limit = args.limit if args.limit > 0 else len(items)
        for i, article in enumerate(items[:limit]):
            url = article.get("url") or article.get("link")
            
            # If we already have content, skip — but treat placeholder HTML/link snippets as not content
            def is_placeholder_content(s: str) -> bool:
                if not s:
                    return True
                s_strip = s.strip()
                # common placeholder: an <a href=...> link snippet or extremely short summaries
                if s_strip.startswith('<a ') and 'href=' in s_strip:
                    return True
                if '<a href' in s_strip and len(s_strip) < 300:
                    return True
                # if content is just a single URL
                if re.match(r'^https?://', s_strip):
                    return True
                # otherwise consider valid if longer than threshold
                return len(s_strip) < 200

            existing = article.get("content") or article.get("content_scraped")
            if existing and not is_placeholder_content(existing):
                print(f"Skipping article (already has full content): {article.get('title', '')}")
                continue

            if not url:
                print(f"No URL for article: {article.get('title', '')}")
                continue

            # Resolve Google News pages to underlying article URLs when possible
            url_to_scrape = url
            if url and 'news.google' in url:
                resolved = await resolve_google_news_url(url)
                if resolved:
                    print(f"Resolved Google News URL to: {resolved}")
                    url_to_scrape = resolved
                else:
                    print(f"Could not resolve Google News URL; will attempt to scrape the Google News page itself: {url}")

            print(f"Scraping ({category}) {i+1}/{limit}: {url_to_scrape}")
            scraped = await scrape_url(url_to_scrape)

            if scraped:
                article["content_scraped"] = scraped
                # Optionally run HF summarization
                if hf:
                    try:
                        print("  Summarizing with HFAnalyzer...")
                        llm_out = hf.analyze_content(scraped, article.get('company', ''), [])
                        # Create a concise summary: use key_quotes or features
                        summary = ''
                        if llm_out.get('key_quotes'):
                            summary = ' '.join(llm_out['key_quotes'])
                        elif llm_out.get('features'):
                            summary = '; '.join(llm_out['features'])
                        else:
                            summary = llm_out.get('impact_level', '')
                        article['llm_summary'] = summary
                    except Exception as e:
                        print(f"  HF summarization failed: {e}")
            else:
                print(f"  Failed to scrape content for: {url}")

    # Write updated JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated JSON written to: {output_path.resolve()}")

    # Export txt files
    export_txt_for_articles(data, txt_out)
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to collected JSON")
    parser.add_argument("--output", help="Path for updated JSON", default=None)
    parser.add_argument("--txt-out", help="Directory to write per-article txt files", default="collected_txt_updated")
    parser.add_argument("--summarize", help="Run HFAnalyzer to summarize scraped content", action="store_true")
    parser.add_argument("--limit", help="Limit number of articles to process (per category)", type=int, default=0)
    return parser.parse_args()


async def process_articles(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_updated.json")
    txt_out = Path(args.txt_out)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Optionally initialize HFAnalyzer
    hf = None
    if args.summarize:
        try:
            # Ensure project root is on sys.path so local package imports work
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from competition_agent.llm.hf_analyzer import HFAnalyzer
            hf = HFAnalyzer()
        except Exception as e:
            print(f"Failed to initialize HFAnalyzer: {e}")
            hf = None

    for category, items in data.items():
        limit = args.limit if args.limit > 0 else len(items)
        for i, article in enumerate(items[:limit]):
            url = article.get("url") or article.get("link")
            # If we already have content, skip — but treat placeholder HTML/link snippets as not content
            def is_placeholder_content(s: str) -> bool:
                if not s:
                    return True
                s_strip = s.strip()
                # common placeholder: an <a href=...> link snippet or extremely short summaries
                if s_strip.startswith('<a ') and 'href=' in s_strip:
                    return True
                if '<a href' in s_strip and len(s_strip) < 300:
                    return True
                # if content is just a single URL
                if re.match(r'^https?://', s_strip):
                    return True
                # otherwise consider valid if longer than threshold
                return len(s_strip) < 200

            existing = article.get("content") or article.get("content_scraped")
            if existing and not is_placeholder_content(existing):
                print(f"Skipping article (already has full content): {article.get('title', '')}")
                continue

            if not url:
                print(f"No URL for article: {article.get('title', '')}")
                continue

            # Resolve Google News pages to underlying article URLs when possible
            url_to_scrape = url
            if url and 'news.google' in url:
                resolved = await resolve_google_news_url(url)
                if resolved:
                    print(f"Resolved Google News URL to: {resolved}")
                    url_to_scrape = resolved
                else:
                    print(f"Could not resolve Google News URL; will attempt to scrape the Google News page itself: {url}")

            print(f"Scraping ({category}) {i+1}/{limit}: {url_to_scrape}")
            scraped = await scrape_url(url_to_scrape)
            if scraped:
                article["content_scraped"] = scraped
                # Optionally run HF summarization
                if hf:
                    try:
                        print("  Summarizing with HFAnalyzer...")
                        llm_out = hf.analyze_content(scraped, article.get('company', ''), [])
                        # Create a concise summary: use key_quotes or features
                        summary = ''
                        if llm_out.get('key_quotes'):
                            summary = ' '.join(llm_out['key_quotes'])
                        elif llm_out.get('features'):
                            summary = '; '.join(llm_out['features'])
                        else:
                            summary = llm_out.get('impact_level', '')
                        article['llm_summary'] = summary
                    except Exception as e:
                        print(f"  HF summarization failed: {e}")
            else:
                print(f"  Failed to scrape content for: {url}")

    # Write updated JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated JSON written to: {output_path.resolve()}")

    # Export txt files
    export_txt_for_articles(data, txt_out)


if __name__ == "__main__":
    args = main()
    sys.exit(asyncio.run(process_articles(args)))
