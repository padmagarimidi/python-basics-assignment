
import csv
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/index.html"
OUTPUT_CSV = "data/raw_books.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Zepto Capstone Scraper; educational use)"}

MIN_BOOKS = 60          # required minimum rows for the final dataset
MIN_CATEGORIES = 3      # required minimum number of distinct categories
MAX_CATEGORIES = 8      # safety cap so a full run finishes in reasonable time
REQUEST_DELAY_SECS = 0.3  # be polite to the practice server


def get_soup(url):
    """Fetch a URL and return (BeautifulSoup, resolved_url)."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser"), resp.url


def discover_categories(base_url):
    """Return a list of (category_name, category_url) from the sidebar nav."""
    soup, real_url = get_soup(base_url)
    links = soup.select("div.side_categories ul.nav-list ul li a")
    categories = []
    for a in links:
        name = a.get_text(strip=True)
        href = urljoin(real_url, a["href"])
        if name:
            categories.append((name, href))
    return categories


def scrape_category(category_name, category_url):
    """Scrape every book across every paginated page of one category."""
    books = []
    url = category_url
    while url:
        soup, real_url = get_soup(url)
        for article in soup.select("article.product_pod"):
            try:
                title = article.h3.a["title"].strip()
                price_text = article.select_one("p.price_color").get_text(strip=True)
                rating_classes = article.select_one("p.star-rating")["class"]
                rating_word = [c for c in rating_classes if c != "star-rating"][0]
                availability_text = article.select_one(
                    "p.instock.availability"
                ).get_text(strip=True)
                books.append(
                    {
                        "title": title,
                        "price": price_text,
                        "star_rating": rating_word,
                        "availability": availability_text,
                        "category": category_name,
                    }
                )
            except (AttributeError, TypeError, KeyError, IndexError) as exc:
                # A malformed/unexpected product card -- skip it, keep the
                # pipeline running rather than crashing on one bad row.
                print(f"  [warn] skipped a malformed book card in {category_name}: {exc}")
                continue

        next_link = soup.select_one("li.next a")
        url = urljoin(real_url, next_link["href"]) if next_link else None
        time.sleep(REQUEST_DELAY_SECS)
    return books


def main():
    print("Discovering categories from the site navigation...")
    categories = discover_categories(BASE_URL)
    if not categories:
        raise RuntimeError("Could not discover any categories - site structure may have changed.")

    all_books = []
    used_categories = 0

    for name, href in categories:
        print(f"Scraping category: {name}")
        books = scrape_category(name, href)
        all_books.extend(books)
        used_categories += 1
        print(f"  -> {len(books)} books (running total: {len(all_books)})")

        if used_categories >= MIN_CATEGORIES and len(all_books) >= MIN_BOOKS:
            break
        if used_categories >= MAX_CATEGORIES:
            break

    if len(all_books) < MIN_BOOKS or used_categories < MIN_CATEGORIES:
        print(
            f"[warn] Only collected {len(all_books)} books across {used_categories} "
            "categories. Increase MAX_CATEGORIES if this is below the required minimum."
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "price", "star_rating", "availability", "category"]
        )
        writer.writeheader()
        writer.writerows(all_books)

    print(f"\nSaved {len(all_books)} raw rows across {used_categories} categories to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
