"""
load_db.py -- Module 1 (Data Pipeline), Step 3.

"""

import sqlite3

import pandas as pd

CLEAN_CSV = "data/cleaned_books.csv"
DB_PATH = "zepto_books.db"

SCHEMA = """
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
"""


def build_database():
    df = pd.read_csv(CLEAN_CSV)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;")
    cur.executescript(SCHEMA)

    for name in sorted(df["category"].unique()):
        cur.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (name,)
        )
    conn.commit()

    category_ids = {
        name: cid
        for cid, name in cur.execute("SELECT category_id, category_name FROM categories")
    }

    rows = [
        (
            r.title,
            r.price_gbp,
            r.price_inr,
            int(r.rating),
            int(bool(r.in_stock)),
            category_ids[r.category],
        )
        for r in df.itertuples()
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"Loaded {len(rows)} books across {len(category_ids)} categories into {DB_PATH}")


if __name__ == "__main__":
    build_database()
