# Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes `books.toscrape.com` (a public scraping-practice site — no login, no
API key, no paid tier), cleans and converts the data, loads it into a
normalized SQLite database, and queries it with both SQL and pandas.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No API keys or paid services are needed anywhere in
this module.

## Run

Run the whole pipeline in one go:

```bash
python pipeline.py
```

Or run each step individually (each stage writes a file the next stage reads):

```bash
python scrape.py     # -> data/raw_books.csv
python clean.py      # -> data/cleaned_books.csv
python load_db.py    # -> zepto_books.db
python queries.py    # prints SQL + pandas query output
```

## Files

| File            | Purpose                                                             |
|-----------------|----------------------------------------------------------------------|
| `scrape.py`     | Scrapes book title/price/rating/availability/category from ≥3 categories on books.toscrape.com |
| `clean.py`      | Cleans and types the raw fields; applies the fixed GBP→INR conversion |
| `load_db.py`    | Creates the two-table SQLite schema and loads the cleaned data       |
| `queries.py`    | Runs ≥5 SQL queries and verifies them against equivalent pandas operations |
| `pipeline.py`   | Orchestrates all four steps end to end                               |
| `data/`         | Intermediate CSVs (raw and cleaned)                                   |
| `zepto_books.db`| Generated SQLite database (created by `load_db.py`)                   |

## Design decisions

**Scraping strategy.** Category names are discovered dynamically from the
site's sidebar navigation (rather than hard-coded), and books are scraped
category-by-category by following each category's "next page" link until
exhausted. This gives the category label for free from the listing page
itself, with no need to visit each book's individual detail page. The
scraper stops once it has covered at least 3 categories **and** at least 60
books (capped at 8 categories as a safety limit), which in practice pulls
in well over 100 books.

**Currency conversion.** `price_inr = price_gbp * 105.50`. This is the
fixed, project-defined baseline rate specified in the assignment — not a
live or historical market rate, so it needs no date reference or API call.
No paid or keyless external currency API is used for the graded path.

**Cleaning / missing-data policy.**
- `price_gbp` is treated as an **essential** field: if it can't be parsed
  (e.g. malformed text), the row is **dropped**, since a book with no
  usable price isn't a usable row for this dataset.
- `rating` is a **secondary** numeric field: if it can't be parsed, it is
  **median-imputed** from the other valid ratings rather than discarding an
  otherwise-good row.
- `in_stock`: availability text that doesn't clearly say "in stock" or "out
  of stock" is conservatively treated as `False` (not confirmed in stock).

**Database schema.** Two tables, `categories` (PK `category_id`) and
`books` (PK `book_id`, FK `category_id` → `categories.category_id`), so the
category name is stored once and referenced by id — a standard normalized
1:N design.

**SQL queries (`queries.py`).** Five queries collectively cover
`SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and a
`JOIN` across the two tables. The join query's result is read back with
`pd.read_sql(...)` and independently reproduced with `pd.merge(...)` on
in-memory DataFrames (no SQL); both include `book_id` as a tie-breaker in
the sort so the two results are directly comparable — the script asserts
they are identical (`DataFrame.equals`).

## Note on the project's Git workflow requirement

This repository's overall commit history (not just this module) includes a
feature branch created, committed to at least twice, and merged back into
`main` — visible via `git log --graph --all`. That requirement is scored
once against the whole repo, per the assignment brief, so it isn't
duplicated in each module's own history.
