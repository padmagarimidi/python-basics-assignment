
import sqlite3

import pandas as pd

DB_PATH = "zepto_books.db"

QUERIES = {
    "Q1_top10_priciest_in_stock (SELECT/WHERE, ORDER BY, LIMIT)": """
        SELECT title, price_inr, rating
        FROM books
        WHERE in_stock = 1
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    "Q2_distinct_categories (DISTINCT)": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "Q3_high_rated_books (IN)": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC
        LIMIT 15;
    """,
    "Q4_midrange_price_books (BETWEEN)": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    "Q5_books_per_category (JOIN)": """
        SELECT c.category_name, b.title, b.rating, b.price_inr, b.book_id
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.book_id
        LIMIT 20;
    """,
}


def run_sql_queries(conn):
    for label, sql in QUERIES.items():
        print(f"\n--- {label} ---")
        print(sql.strip())
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        print(cols)
        for row in rows:
            print(row)


def pandas_verification(conn):
    print("\n=== pandas verification ===")

    q1_df = pd.read_sql(QUERIES["Q1_top10_priciest_in_stock (SELECT/WHERE, ORDER BY, LIMIT)"], conn)
    print("\npd.read_sql -> Q1 (top 10 priciest in-stock books):")
    print(q1_df)

    q3_df = pd.read_sql(QUERIES["Q3_high_rated_books (IN)"], conn)
    print("\npd.read_sql -> Q3 (rating IN (4,5)), first 5 rows:")
    print(q3_df.head())

    # SQL JOIN result, via pandas
    sql_join_df = pd.read_sql(QUERIES["Q5_books_per_category (JOIN)"], conn)
    sql_join_df = sql_join_df.drop(columns=["book_id"]).reset_index(drop=True)

    # Same result reproduced with pd.merge on in-memory DataFrames, no SQL.
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged_df = pd.merge(books_df, categories_df, on="category_id")
    merged_df = merged_df.sort_values(
        ["category_name", "rating", "book_id"], ascending=[True, False, True]
    )
    merged_df = merged_df[["category_name", "title", "rating", "price_inr"]].head(20)
    merged_df = merged_df.reset_index(drop=True)

    print("\nSQL JOIN result (read back via pd.read_sql):")
    print(sql_join_df)
    print("\npd.merge on in-memory DataFrames (no SQL):")
    print(merged_df)

    match = sql_join_df.equals(merged_df)
    print(f"\nDo the SQL JOIN and the pandas merge produce the same result? {match}")


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        run_sql_queries(conn)
        pandas_verification(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
