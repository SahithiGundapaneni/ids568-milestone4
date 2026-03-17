import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_data(rows, seed, output):
    np.random.seed(seed)

    users = np.random.randint(1, 10000, rows)
    products = np.random.randint(1, 5000, rows)
    categories = np.random.choice(["electronics", "clothing", "home", "beauty"], rows)
    event_types = np.random.choice(["view", "cart", "purchase"], rows, p=[0.6, 0.2, 0.2])
    prices = np.round(np.random.uniform(5, 500, rows), 2)
    quantity = np.random.randint(1, 5, rows)
    regions = np.random.choice(["US", "EU", "APAC"], rows)

    base_time = datetime(2024, 1, 1)
    timestamps = [
        base_time + timedelta(seconds=int(x))
        for x in np.random.randint(0, 1000000, rows)
    ]

    df = pd.DataFrame({
        "user_id": users,
        "product_id": products,
        "category": categories,
        "event_type": event_types,
        "price": prices,
        "quantity": quantity,
        "region": regions,
        "event_ts": timestamps
    })

    os.makedirs(output, exist_ok=True)
    df.to_csv(f"{output}/data.csv", index=False)
    print(f"Saved {rows} rows to {output}/data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")

    args = parser.parse_args()
    generate_data(args.rows, args.seed, args.output)