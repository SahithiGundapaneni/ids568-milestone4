"""
generate_data.py — Synthetic Transaction Data Generator
Milestone 4 | MLOps Course | Module 5

Produces a Parquet dataset of synthetic e-commerce transaction events.

Features
--------
- Configurable row count, seed, output directory
- Reproducible via seeded NumPy RNG (SHA-256 hash printed for verification)
- Optional data skew (Zipf distribution on user_id) to stress distributed pipelines
- Generates 10M+ rows in under 60s on a modern laptop

Columns
-------
user_id        : string  (U000001 … U{n_users})
merchant_id    : string  (M00001 … M{n_merchants})
category       : string  (one of 8 categories)
event_type     : string  (view / add_to_cart / purchase / refund)
amount         : float   (purchase value; 0 for non-purchase events)
timestamp      : datetime (random within 2024-01-01 – 2024-12-31)
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CATEGORIES = [
    "electronics", "clothing", "home", "sports",
    "books", "beauty", "toys", "groceries",
]
EVENT_TYPES = ["view", "add_to_cart", "purchase", "refund"]
EVENT_WEIGHTS = [0.55, 0.25, 0.15, 0.05]   # realistic funnel distribution

# 2024 full year in Unix seconds
EPOCH_START = int(pd.Timestamp("2024-01-01").timestamp())
EPOCH_END   = int(pd.Timestamp("2024-12-31 23:59:59").timestamp())


def generate_data(
    n_rows: int,
    seed: int,
    n_users: int = 100_000,
    n_merchants: int = 10_000,
    skew: bool = False,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction DataFrame.

    Parameters
    ----------
    n_rows      : total events to generate
    seed        : RNG seed for full reproducibility
    n_users     : distinct user IDs
    n_merchants : distinct merchant IDs
    skew        : if True, apply Zipf(a=1.5) distribution to user selection
                  to simulate hot-user data skew

    Returns
    -------
    pd.DataFrame with columns:
        user_id, merchant_id, category, event_type, amount, timestamp
    """
    rng = np.random.default_rng(seed)
    log.info("Generating %d rows | seed=%d | users=%d | merchants=%d | skew=%s",
             n_rows, seed, n_users, n_merchants, skew)

    # --- user_id ---
    if skew:
        # Zipf distribution → top 20% of users generate ~80% of events
        zipf_probs = 1.0 / (np.arange(1, n_users + 1) ** 1.5)
        zipf_probs /= zipf_probs.sum()
        user_indices = rng.choice(n_users, size=n_rows, replace=True, p=zipf_probs)
    else:
        user_indices = rng.integers(0, n_users, size=n_rows)
    user_ids = np.array([f"U{i+1:06d}" for i in user_indices])

    # --- merchant_id ---
    merchant_indices = rng.integers(0, n_merchants, size=n_rows)
    merchant_ids = np.array([f"M{i+1:05d}" for i in merchant_indices])

    # --- category ---
    categories = rng.choice(CATEGORIES, size=n_rows)

    # --- event_type ---
    event_types = rng.choice(EVENT_TYPES, size=n_rows, p=EVENT_WEIGHTS)

    # --- amount (only non-zero for purchase/refund) ---
    raw_amounts = np.exp(rng.normal(loc=3.5, scale=1.2, size=n_rows)).clip(0.01, 10_000)
    is_monetary = np.isin(event_types, ["purchase", "refund"])
    amounts = np.where(is_monetary, raw_amounts, 0.0).round(2)

    # --- timestamp (seconds since epoch, uniform over 2024) ---
    ts_unix = rng.integers(EPOCH_START, EPOCH_END, size=n_rows).astype("int64")
    timestamps = pd.to_datetime(ts_unix, unit="s")

    df = pd.DataFrame({
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "category": categories,
        "event_type": event_types,
        "amount": amounts,
        "timestamp": timestamps,
    })
    return df


def sha256_hash(df: pd.DataFrame) -> str:
    """Compute a short SHA-256 hash over the raw Parquet bytes for reproducibility verification."""
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--rows",       type=int,   default=10_000_000, help="Number of rows (default: 10M)")
    parser.add_argument("--seed",       type=int,   default=42,          help="RNG seed (default: 42)")
    parser.add_argument("--output",     type=str,   required=True,       help="Output directory")
    parser.add_argument("--users",      type=int,   default=100_000,     help="Distinct users (default: 100K)")
    parser.add_argument("--merchants",  type=int,   default=10_000,      help="Distinct merchants (default: 10K)")
    parser.add_argument("--skew",       action="store_true",              help="Apply Zipf user skew")
    parser.add_argument("--format",     choices=["parquet", "csv"],       default="parquet",
                        help="Output format (default: parquet)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    t0 = time.perf_counter()
    df = generate_data(
        n_rows=args.rows,
        seed=args.seed,
        n_users=args.users,
        n_merchants=args.merchants,
        skew=args.skew,
    )
    gen_elapsed = time.perf_counter() - t0

    # --- Write output ---
    if args.format == "parquet":
        out_path = os.path.join(args.output, "transactions.parquet")
        df.to_parquet(out_path, index=False)
    else:
        out_path = os.path.join(args.output, "transactions.csv")
        df.to_csv(out_path, index=False)

    write_elapsed = time.perf_counter() - t0
    file_size_mb = Path(out_path).stat().st_size / (1024 ** 2)

    # --- Compute hash for reproducibility ---
    log.info("Computing reproducibility hash …")
    data_hash = sha256_hash(df)

    print()
    print("=" * 55)
    print(f"  Generated in   : {gen_elapsed:.1f}s (total: {write_elapsed:.1f}s)")
    print(f"  Shape          : {df.shape}")
    print(f"  Output         : {out_path}  ({file_size_mb:.1f} MB)")
    print(f"  Hash (sha256)  : {data_hash}")
    print(f"  Unique users   : {df['user_id'].nunique():,}")
    print(f"  Unique merchants: {df['merchant_id'].nunique():,}")
    print(f"  Amount range   : ${df['amount'].min():.2f} – ${df['amount'].max():.2f}")
    print(f"  Event dist     : {dict(df['event_type'].value_counts())}")
    print("=" * 55)
    print()
    print("Reproducibility: run the same command again and the hash should match.")


if __name__ == "__main__":
    main()
