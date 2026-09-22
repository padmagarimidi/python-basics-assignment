
import numpy as np
import pandas as pd

RAW_CSV = "data/raw_books.csv"
CLEAN_CSV = "data/cleaned_books.csv"

# Fixed, project-defined baseline conversion rate for this assignment.

GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(raw_price):
    try:
        cleaned = str(raw_price).replace("£", "").replace("Â", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan


def parse_rating(raw_rating):
    return RATING_WORDS.get(str(raw_rating).strip(), np.nan)


def parse_availability(raw_availability):
    text = str(raw_availability).lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return np.nan  # unrecognised text - treated as missing, not guessed


def main():
    df = pd.read_csv(RAW_CSV)
    rows_before = len(df)

    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_availability)

    # Drop rows with an unparseable price (essential field).
    dropped = df["price_gbp"].isna().sum()
    df = df.dropna(subset=["price_gbp"]).copy()

    # Median-impute any unparseable ratings (secondary numeric field).
    imputed = int(df["rating"].isna().sum())
    if imputed:
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)
    df["rating"] = df["rating"].round().astype(int)

    # Conservative default for unrecognised availability text.
    df["in_stock"] = df["in_stock"].fillna(False).astype(bool)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    df.to_csv(CLEAN_CSV, index=False)

    print(f"Rows in:  {rows_before}")
    print(f"Rows dropped (unparseable price): {dropped}")
    print(f"Ratings median-imputed: {imputed}")
    print(f"Rows out: {len(df)} -> {CLEAN_CSV}")
    print(f"Conversion rate used: 1 GBP = {GBP_TO_INR} INR (fixed project constant)")


if __name__ == "__main__":
    main()
