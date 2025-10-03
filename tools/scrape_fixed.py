"""
Script to extract and save content from Google News articles.
Takes a JSON file with article URLs and saves their content to text files.
"""
import argparse
import json
import re
import sys
from pathlib import Path
import asyncio
import base64
from typing import Optional, Dict, Any

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from newspaper import Article
import concurrent.futures


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def newspaper_extract(url: str, timeout: float = 15.0) -> Optional[str]:
    try:
        a = Article(url, language='en')
        a.download()
        a.parse()
        text = a.text
        if text and len(text.strip()) > 200:
            return clean_text(text)
    except Exception as e:
        # Newspaper failed - return None and let Playwright try
        print(f"[Newspaper] Extraction failed for {url}: {e}")
    return None


async def scrape_url(url: str, timeout: float = 30.0) -> Optional[str]:
    if not url:
        print(f"[Scraper] No URL provided.")
        return None

    # Try newspaper3k first in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    try:
        print(f"[Scraper] Trying newspaper3k for {url}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, newspaper_extract, url, 15.0)
        if result:
            print(f"[Scraper] Newspaper extraction succeeded: {len(result)} chars")
            return result
        else:
            print(f"[Scraper] Newspaper returned no content, falling back to Playwright")
    except Exception as e:
        print(f"[Scraper] Newspaper extraction threw: {e}")

    print(f"[Scraper] Attempting to scrape: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                await page.wait_for_timeout(2000)
                print(f"[Scraper] Navigation successful: {url}")
            except Exception as e:
                print(f"[Scraper] Navigation failed: {e}")
                await browser.close()
                return None
            # Extract content using various selectors
            try:
                content = await page.evaluate('''() => {
                    function getText(elem) {
                        return elem ? elem.innerText.trim() : '';
                    }
                    const article = document.querySelector('article');
                    if (article) {
                        const text = getText(article);
                        if (text.length > 200) return text;
                    }
                    const main = document.querySelector('main');
                    if (main) {
                        const text = getText(main);
                        if (text.length > 200) return text;
                    }
                    const selectors = [
                        '[role="main"]',
                        '.article-content',
                        '.post-content',
                        '.entry-content',
                        '#content-body',
                        '.article-body',
                        '.story-body'
                    ];
                    for (const selector of selectors) {
                        const elem = document.querySelector(selector);
                        if (elem) {
                            const text = getText(elem);
                            if (text.length > 200) return text;
                        }
                    }
                    return Array.from(document.querySelectorAll('p'))
                        .map(p => getText(p))
                        .filter(text => text.length > 40)
                        .join('\\n\\n');
                }''')
                print(f"[Scraper] Content length: {len(content) if content else 0}")
            except Exception as e:
                print(f"[Scraper] Selector evaluation failed: {e}")
                content = None
            await browser.close()
            if content and len(content.strip()) > 200:
                print(f"[Scraper] Successfully extracted content.")
                return clean_text(content)
            else:
                print(f"[Scraper] Content too short or empty.")
    except Exception as e:
        print(f"[Scraper] Scraping failed: {e}")
    return None


async def resolve_google_news_url(url: str, timeout: float = 30.0) -> Optional[str]:
    """Resolve a Google News URL to its original article URL."""
    print(f"[Resolver] Attempting to resolve Google News URL: {url}")
    if not url or 'news.google' not in url:
        print(f"[Resolver] Not a Google News URL, returning as is.")
        return url

    # Helper: filter out known tracking / static / ad hosts and file extensions
    def likely_article_link(href: str) -> bool:
        if not href or not href.startswith(('http:', 'https:')):
            return False
        lower = href.lower()
        # exclude known ad/tracking/static hosts or resources
        blocked = ['googleadservices', 'pagead2.googlesyndication', 'gstatic', 'stripe',
                   'captcha-delivery', 'fundingchoicesmessages', 'doubleclick', 'amazon-adsystem',
                   'google.com/pagead', 'fonts.googleapis', '/admin-ajax.php', '/wp-admin', '.woff', '.woff2', '.css', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.pdf']
        for b in blocked:
            if b in lower:
                return False
        # avoid obvious short/tracking urls
        if len(href) < 20:
            return False
        return True

    # Try browser-based resolution with targeted link extraction
    try:
        print(f"[Resolver] Falling back to browser-based resolution (fast-mode)...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
            page = await context.new_page()

            try:
                # Use domcontentloaded to avoid waiting on ad networks
                await page.goto(url, wait_until='domcontentloaded', timeout=min(timeout, 25) * 1000)
            except Exception as e:
                print(f"[Resolver] Initial navigation failed (domcontentloaded): {e}")

            # Wait briefly for links to render, but keep it short
            try:
                await page.wait_for_selector('a[href]', timeout=5000)
            except Exception:
                # no links found fast - continue to evaluation
                pass

            # Evaluate and collect candidate hrefs
            try:
                candidates = await page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a[href]')).map(a => a.href);
                }''')
            except Exception as e:
                print(f"[Resolver] Link evaluation failed: {e}")
                candidates = []

            # Prefer og:url if present and looks like an article
            try:
                og = await page.evaluate('''() => {
                    const m = document.querySelector('meta[property="og:url"]');
                    return m ? m.content : null;
                }''')
                if og and likely_article_link(og):
                    await browser.close()
                    print(f"[Resolver] Using og:url: {og}")
                    return og
            except Exception:
                pass

            # Filter candidates and return the first plausible one
            for href in candidates:
                if likely_article_link(href):
                    await browser.close()
                    print(f"[Resolver] Found candidate link: {href}")
                    return href

            await browser.close()
    except Exception as e:
        print(f"[Resolver] Browser-based URL resolution failed: {e}")

    print(f"[Resolver] Could not resolve to a publisher URL; returning original URL.")
    return url


async def process_articles(args):
    """Process articles from input JSON, scrape content, and save results."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_updated.json")
    txt_out = Path(args.txt_out)
    txt_out.mkdir(parents=True, exist_ok=True)

    # Read input JSON
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each article
    for category, items in data.items():
        print(f"\nProcessing category: {category}")
        cat_dir = txt_out / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        limit = args.limit if args.limit > 0 else len(items)
        for i, article in enumerate(items[:limit]):
            title = article.get('title', 'Untitled')
            url = article.get('url')
            
            if not url:
                print(f"No URL for article: {title}")
                continue

            print(f"\nArticle {i+1}/{limit}: {title}")
            print(f"Original URL: {url}")
            
            # Try to resolve Google News URL
            if 'news.google' in url:
                print("Resolving Google News URL...")
                resolved = await resolve_google_news_url(url)
                print(f"Resolved URL: {resolved}")
                if resolved and resolved != url:
                    url = resolved
                else:
                    print("Could not resolve URL, will attempt to scrape the Google News page itself.")

            # Scrape the content
            print(f"Scraping content from: {url}")
            content = await scrape_url(url)
            
            if content:
                print(f"Successfully scraped {len(content)} characters from {url}")
                article['content_scraped'] = content
                
                # Save to text file
                safe_title = re.sub(r'[^0-9a-zA-Z_\-]', '_', title)[:120]
                filename = f"{i:03d}_{safe_title}.txt"
                txt_path = cat_dir / filename
                
                with txt_path.open('w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Company: {category}\n")
                    f.write(f"Source: {article.get('publisher', '')}\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(content)
                print(f"Saved to {txt_path.name}")
            else:
                print(f"Failed to scrape content from {url}")

    # Save updated JSON
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated JSON written to: {output_path}")
    print(f"Article text files written to: {txt_out}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Scrape article content from URLs in a JSON file")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--output", help="Path for output JSON (default: input_updated.json)")
    parser.add_argument("--txt-out", help="Directory for text files", default="articles")
    parser.add_argument("--limit", type=int, help="Limit articles per category", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    sys.exit(asyncio.run(process_articles(args)))