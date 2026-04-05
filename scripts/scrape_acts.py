"""
Scrape BNSS and BSA sections from advocatekhoj.com into JSON files.
Usage: python scripts/scrape_acts.py
"""

import json
import re
import time
import logging
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bronze" / "laws"

# Act configurations
ACTS = {
    # "BNSS": {
    #     "slug": "bharatiyanagarik2023",
    #     "title": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    # },
    # "BSA": {
    #     "slug": "bharatiyaaakshya2023",
    #     "title": "Bharatiya Sakshya Adhiniyam, 2023",
    # },
    # "BNS": {
    #     "slug": "bharatiyanyaya2023",
    #     "title": "Bharatiya Nyaya Sanhita, 2023",
    # },
    "CONSTITUTION": {
        "slug": "constitutionofindia",
        "title": "Constitution of India, 1949",
    },
}

BASE_URL = "https://www.advocatekhoj.com/library/bareacts/{slug}/{section_id}.php"
INDEX_URL = "https://www.advocatekhoj.com/library/bareacts/{slug}/index.php?Title={title}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}

DELAY = 1.0  # seconds between requests


def fetch_index(slug: str, title: str) -> list[dict]:
    """Fetch the index page and extract section links."""
    url = INDEX_URL.format(slug=slug, title=quote(title))
    base_dir = f"https://www.advocatekhoj.com/library/bareacts/{slug}/"
    logger.info("Fetching index: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    sections = []

    # Links are RELATIVE like "1.php?Title=...", "a.php?Title=..."
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match relative links: "1.php?...", "482.php?...", "a.php?..."
        match = re.match(r'^([^/?]+)\.php\?', href)
        if not match:
            continue
        sec_id = match.group(1)
        if sec_id == "index":
            continue
        sec_title = a.get_text(strip=True)
        full_url = base_dir + href
        sections.append({
            "section_id": sec_id,
            "title": sec_title,
            "url": full_url,
        })

    # Deduplicate by section_id
    seen = set()
    unique = []
    for s in sections:
        if s["section_id"] not in seen:
            seen.add(s["section_id"])
            unique.append(s)

    logger.info("Found %d unique sections in index", len(unique))
    return unique


def fetch_section(url: str, act_title: str) -> str:
    """Fetch a single section page and extract text content."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to find the main content area
    # advocatekhoj uses various div structures
    content_div = None
    for selector in [
        "div.judgmark",
        "div.card-body",
        "div.panel-body",
        "td.nopad",
    ]:
        content_div = soup.select_one(selector)
        if content_div:
            break

    if not content_div:
        # Fallback: get all text from body, removing nav/header/footer
        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        content_div = soup.find("body")

    if content_div:
        text = content_div.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    return ""


def scrape_act(act_code: str):
    """Scrape all sections for one act."""
    cfg = ACTS[act_code]
    slug = cfg["slug"]
    title = cfg["title"]

    out_dir = DATA_DIR / act_code / "json"
    txt_dir = DATA_DIR / act_code / "txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Get section list from index
    sections = fetch_index(slug, title)
    if not sections:
        logger.error("No sections found for %s", act_code)
        return

    success = 0
    for i, sec in enumerate(sections):
        sec_id = sec["section_id"]
        sec_title = sec["title"]
        url = sec["url"]

        # Build filename
        safe_title = re.sub(r'[^\w\s-]', '', sec_title.lower())
        safe_title = re.sub(r'\s+', '-', safe_title.strip())[:80]
        if sec_id.isdigit():
            fname = f"{int(sec_id):03d}_{safe_title}"
        else:
            fname = f"x_{safe_title}" if safe_title else f"x_{sec_id}"

        json_path = out_dir / f"{fname}.json"
        txt_path = txt_dir / f"{fname}.txt"

        # Skip if already scraped
        if json_path.exists():
            logger.debug("Skipping %s §%s (already exists)", act_code, sec_id)
            success += 1
            continue

        logger.info("[%d/%d] %s §%s — %s", i + 1, len(sections), act_code, sec_id, sec_title)

        try:
            content = fetch_section(url, title)
            if not content:
                logger.warning("Empty content for %s §%s", act_code, sec_id)
                continue

            # Determine section_number
            section_number = sec_id if sec_id.isdigit() or re.match(r'\d+[a-zA-Z]', sec_id) else sec_id

            data = {
                "section_number": section_number,
                "section_id": sec_id,
                "title": sec_title,
                "source_url": url,
                "content": f"{title}\n{content}",
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n{sec_id}. {sec_title}\n{content}")

            success += 1

        except Exception as e:
            logger.error("Failed to fetch %s §%s: %s", act_code, sec_id, e)

        time.sleep(DELAY)

    logger.info("✅ %s: scraped %d/%d sections", act_code, success, len(sections))


def main():
    for act_code in ACTS:
        logger.info("=" * 60)
        logger.info("Scraping %s (%s)", act_code, ACTS[act_code]["title"])
        logger.info("=" * 60)
        scrape_act(act_code)


if __name__ == "__main__":
    main()
