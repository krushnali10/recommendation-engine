import pandas as pd
import numpy as np
import sqlite3
import os

DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "amazon-product-reviews",
                         "ratings_Electronics (1).csv")

def load_and_prepare(sample_size=200000, min_interactions=5, seed=42):
    print("Loading raw dataset...")
    df = pd.read_csv(DATA_PATH, header=None,
                     names=["user_id", "product_id", "rating", "timestamp"])

    print(f"  Raw rows: {len(df):,}")

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Filter users and products with at least min_interactions
    print("Filtering sparse users and products...")
    user_counts   = df["user_id"].value_counts()
    product_counts = df["product_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= min_interactions].index)]
    df = df[df["product_id"].isin(product_counts[product_counts >= min_interactions].index)]
    print(f"  After filtering: {len(df):,} rows")

    # Sample for performance
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
        print(f"  Sampled down to: {len(df):,} rows")

    df = df.reset_index(drop=True)
    df["interaction_id"] = df.index + 1

    # Map event_type from rating
    def rating_to_event(r):
        if r >= 4.5: return "purchase"
        if r >= 3.5: return "add_to_cart"
        if r >= 2.5: return "wishlist"
        return "view"

    df["event_type"] = df["rating"].apply(rating_to_event)

    # Build products table from unique product_ids
    print("Building products table...")
    unique_products = df["product_id"].unique()
    np.random.seed(seed)

    CATEGORIES = ["Electronics", "Computers", "Camera", "Audio",
                  "Mobile", "Gaming", "Networking", "Wearables"]
    BRANDS = ["Sony","Samsung","Apple","Logitech","Bose",
              "Canon","LG","Dell","HP","Asus","JBL","Anker"]

    products = pd.DataFrame({
        "product_id":   unique_products,
        "name":         [f"Product {pid[-6:]}" for pid in unique_products],
        "category":     np.random.choice(CATEGORIES, len(unique_products)),
        "subcategory":  np.random.choice(["Accessories","Gadgets","Devices",
                                          "Components","Peripherals"],
                                         len(unique_products)),
        "brand":        np.random.choice(BRANDS, len(unique_products)),
        "price":        np.round(np.random.uniform(10, 1500, len(unique_products)), 2),
        "rating":       df.groupby("product_id")["rating"].mean().reindex(unique_products).values,
        "num_reviews":  df.groupby("product_id")["rating"].count().reindex(unique_products).values,
        "in_stock":     np.random.choice([True, False], len(unique_products), p=[0.9, 0.1]),
    })
    products["tags"] = (products["category"].str.lower() + "," +
                        products["subcategory"].str.lower() + "," +
                        products["brand"].str.lower())

    # Build users table from unique user_ids
    print("Building users table...")
    unique_users = df["user_id"].unique()
    users = pd.DataFrame({
        "user_id":            unique_users,
        "preferred_category": np.random.choice(CATEGORIES, len(unique_users)),
        "secondary_category": np.random.choice(CATEGORIES, len(unique_users)),
        "age_group":          np.random.choice(["18-24","25-34","35-44","45-54","55+"],
                                               len(unique_users)),
        "loyalty_tier":       np.random.choice(["Bronze","Silver","Gold","Platinum"],
                                               len(unique_users)),
    })

    # Final interactions table
    interactions = df[["interaction_id","user_id","product_id",
                        "event_type","rating","timestamp"]].copy()

    print(f"\n  Products:     {len(products):,}")
    print(f"  Users:        {len(users):,}")
    print(f"  Interactions: {len(interactions):,}")
    return products, users, interactions


def save_to_sqlite(products, users, interactions, db_path):
    conn = sqlite3.connect(db_path)
    products.to_sql("products",         conn, if_exists="replace", index=False)
    users.to_sql("users",               conn, if_exists="replace", index=False)
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            rec_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT,
            product_id TEXT,
            score      REAL,
            method     TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"\nDatabase saved: {db_path}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    db   = os.path.join(base, "recommendation.db")

    products, users, interactions = load_and_prepare(sample_size=200000)

    products.to_csv(os.path.join(base, "products.csv"),         index=False)
    users.to_csv(os.path.join(base, "users.csv"),               index=False)
    interactions.to_csv(os.path.join(base, "interactions.csv"), index=False)

    save_to_sqlite(products, users, interactions, db)
    print("\nDone! Ready for Phase 3.")