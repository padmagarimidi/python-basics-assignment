import os

import scrape
import clean
import load_db
import queries


def main():
    os.makedirs("data", exist_ok=True)

    print("=== STEP 1: SCRAPE ===")
    scrape.main()

    print("\n=== STEP 2: CLEAN ===")
    clean.main()

    print("\n=== STEP 3: LOAD INTO SQLITE ===")
    load_db.build_database()

    print("\n=== STEP 4: SQL + PANDAS QUERIES ===")
    queries.main()


if __name__ == "__main__":
    main()
