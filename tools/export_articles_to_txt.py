"""
Export articles from a collected JSON into per-article .txt files.
Usage:
  python tools/export_articles_to_txt.py path/to/collected_data.json [output_dir]

Default output dir: collected_txt
"""
import json
import os
import re
import sys
from pathlib import Path


def safe_filename(s: str, max_len: int = 120) -> str:
    s = s or "untitled"
    s = s.strip()
    # remove characters that are unsafe for filenames
    s = re.sub(r"[\\/:*?\"<>|]", "", s)
    s = re.sub(r"\s+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def export(json_path: str, out_dir: str):
    json_path = Path(json_path)
    out_dir = Path(out_dir)

    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return 1

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    for category, items in data.items():
        cat_dir = out_dir / safe_filename(category)
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i, article in enumerate(items, start=1):
            title = article.get("title") or article.get("headline") or "untitled"
            company = article.get("company") or "unknown_company"
            published = article.get("published_date") or article.get("date") or ""
            source = article.get("source") or article.get("source_name") or ""
            url = article.get("url") or article.get("link") or ""

            body = article.get("content") or article.get("summary") or article.get("text") or ""
            # fallback to entire article dict string if nothing else
            if not body:
                body = json.dumps(article, ensure_ascii=False, indent=2)

            filename = f"{i:03d}_{safe_filename(company)}_{safe_filename(title)}.txt"
            filepath = cat_dir / filename

            with filepath.open("w", encoding="utf-8") as out:
                out.write(f"Title: {title}\n")
                out.write(f"Company: {company}\n")
                out.write(f"Published: {published}\n")
                out.write(f"Source: {source}\n")
                out.write(f"URL: {url}\n")
                out.write('\n')
                out.write(body)

            total += 1

    print(f"Exported {total} articles to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/export_articles_to_txt.py path/to/collected.json [output_dir]")
        sys.exit(2)

    json_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "collected_txt"
    sys.exit(export(json_path, out_dir))
