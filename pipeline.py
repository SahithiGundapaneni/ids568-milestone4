"""
pipeline.py - Distributed Feature Engineering Pipeline
Milestone 4 | MLOps Course | Module 5
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import ray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _row_transforms(df):
    """Per-row feature transforms."""
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    mu = df["amount"].mean()
    sigma = df["amount"].std(ddof=1) + 1e-9
    df["amount_z_score"] = (df["amount"] - mu) / sigma
    df["amount_bucket"] = pd.cut(
        df["amount"], bins=[-np.inf, 25.0, 100.0, 500.0, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    df["amount_log_bin"] = pd.cut(
        np.log1p(df["amount"]), bins=5, labels=False
    ).fillna(0).astype(int)
    event_map = {"view": 0, "add_to_cart": 1, "purchase": 2, "refund": 3}
    df["event_type_enc"] = df["event_type"].map(event_map).fillna(0).astype(int)
    return df


def _per_user_aggregation(df):
    """Per-user aggregation."""
    agg = (
        df.groupby("user_id", sort=False)
        .agg(
            total_spend=("amount", "sum"),
            mean_spend=("amount", "mean"),
            median_spend=("amount", "median"),
            std_spend=("amount", "std"),
            tx_count=("amount", "count"),
            max_amount=("amount", "max"),
            min_amount=("amount", "min"),
            unique_merchants=("merchant_id", "nunique"),
            unique_categories=("category", "nunique"),
            pct_weekend=("is_weekend", "mean"),
            pct_purchase=("event_type_enc", lambda x: (x == 2).mean()),
        )
        .reset_index()
    )
    agg["std_spend"] = agg["std_spend"].fillna(0.0)
    return agg


def run_local(input_path, output_path):
    """Single-machine pandas execution."""
    log.info("[LOCAL] Starting pandas baseline ...")
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    t0 = time.perf_counter()

    df = pd.read_parquet(input_path)
    log.info("[LOCAL] Loaded %d rows, %d columns", len(df), df.shape[1])
    df = _row_transforms(df)
    features = _per_user_aggregation(df)

    os.makedirs(output_path, exist_ok=True)
    features.to_parquet(os.path.join(output_path, "local_features.parquet"), index=False)

    elapsed = time.perf_counter() - t0
    peak_gb = round((proc.memory_info().rss - mem_before) / (1024 ** 3), 3)
    log.info("[LOCAL] Done in %.3fs | peak delta-mem %.2f GB | %d output rows",
             elapsed, peak_gb, len(features))
    return {
        "mode": "local", "runtime_s": round(elapsed, 3),
        "output_rows": int(len(features)), "peak_memory_gb": peak_gb,
        "partitions": 1, "shuffle_mb": None,
    }


@ray.remote
def _ray_process_partition(records):
    """Remote task: process one partition."""
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(records)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    mu = df["amount"].mean()
    sigma = df["amount"].std(ddof=1) + 1e-9
    df["amount_z_score"] = (df["amount"] - mu) / sigma
    df["amount_bucket"] = pd.cut(
        df["amount"], bins=[-np.inf, 25.0, 100.0, 500.0, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    df["amount_log_bin"] = pd.cut(
        np.log1p(df["amount"]), bins=5, labels=False
    ).fillna(0).astype(int)
    event_map = {"view": 0, "add_to_cart": 1, "purchase": 2, "refund": 3}
    df["event_type_enc"] = df["event_type"].map(event_map).fillna(0).astype(int)

    agg = (
        df.groupby("user_id", sort=False)
        .agg(
            total_spend=("amount", "sum"),
            mean_spend=("amount", "mean"),
            median_spend=("amount", "median"),
            std_spend=("amount", "std"),
            tx_count=("amount", "count"),
            max_amount=("amount", "max"),
            min_amount=("amount", "min"),
            unique_merchants=("merchant_id", "nunique"),
            unique_categories=("category", "nunique"),
            pct_weekend=("is_weekend", "mean"),
            pct_purchase=("event_type_enc", lambda x: (x == 2).mean()),
        )
        .reset_index()
    )
    agg["std_spend"] = agg["std_spend"].fillna(0.0)
    return agg.to_dict(orient="records")


def run_distributed(input_path, output_path, n_workers):
    """Distributed Ray execution: MAP -> SHUFFLE -> REDUCE."""
    log.info("[RAY] Initialising Ray with %d workers ...", n_workers)
    ray.init(
        num_cpus=n_workers,
        object_store_memory=int(200 * 1024 ** 2),
        _memory=int(500 * 1024 ** 2),
        ignore_reinit_error=True,
        logging_level=logging.WARNING,
    )

    t0 = time.perf_counter()
    df = pd.read_parquet(input_path)
    n_rows = len(df)
    n_partitions = max(n_workers * 2, 4)
    log.info("[RAY] %d rows -> %d partitions (%d workers)", n_rows, n_partitions, n_workers)

    # Split into list of DataFrames then convert each to records (list of dicts)
    partition_size = max(1, n_rows // n_partitions)
    partitions = [
        df.iloc[i: i + partition_size].reset_index(drop=True).to_dict(orient="records")
        for i in range(0, n_rows, partition_size)
    ]

    # MAP + SHUFFLE phase
    t_map_start = time.perf_counter()
    futures = [_ray_process_partition.remote(part) for part in partitions]
    partial_results = ray.get(futures)
    t_map_s = round(time.perf_counter() - t_map_start, 3)
    log.info("[RAY] MAP+SHUFFLE done in %.3fs", t_map_s)

    shuffle_mb = round(sum(len(str(r)) for batch in partial_results for r in batch) / (1024 ** 2), 2)

    # REDUCE phase
    t_reduce_start = time.perf_counter()
    partial_dfs = [pd.DataFrame(batch) for batch in partial_results]
    combined = pd.concat(partial_dfs, ignore_index=True)
    features = (
        combined.groupby("user_id", sort=False)
        .agg(
            total_spend=("total_spend", "sum"),
            mean_spend=("mean_spend", "mean"),
            median_spend=("median_spend", "mean"),
            std_spend=("std_spend", "mean"),
            tx_count=("tx_count", "sum"),
            max_amount=("max_amount", "max"),
            min_amount=("min_amount", "min"),
            unique_merchants=("unique_merchants", "max"),
            unique_categories=("unique_categories", "max"),
            pct_weekend=("pct_weekend", "mean"),
            pct_purchase=("pct_purchase", "mean"),
        )
        .reset_index()
    )
    t_reduce_s = round(time.perf_counter() - t_reduce_start, 3)
    log.info("[RAY] REDUCE done in %.3fs", t_reduce_s)

    os.makedirs(output_path, exist_ok=True)
    features.to_parquet(os.path.join(output_path, "distributed_features.parquet"), index=False)

    elapsed = time.perf_counter() - t0
    log.info("[RAY] Done in %.3fs | shuffle ~%.1f MB", elapsed, shuffle_mb)
    ray.shutdown()

    return {
        "mode": "distributed", "runtime_s": round(elapsed, 3),
        "output_rows": int(len(features)), "partitions": n_partitions,
        "shuffle_mb": shuffle_mb, "map_s": t_map_s, "reduce_s": t_reduce_s,
    }


def verify_outputs(output_path):
    """Verify local and distributed outputs match."""
    local_file = os.path.join(output_path, "local_features.parquet")
    dist_file = os.path.join(output_path, "distributed_features.parquet")
    if not (os.path.exists(local_file) and os.path.exists(dist_file)):
        log.warning("[VERIFY] Output files missing - skipping")
        return False

    local_df = pd.read_parquet(local_file).sort_values("user_id").reset_index(drop=True)
    dist_df = pd.read_parquet(dist_file).sort_values("user_id").reset_index(drop=True)

    checks = {"same_users": set(local_df["user_id"]) == set(dist_df["user_id"])}
    if checks["same_users"]:
        dist_aligned = dist_df.set_index("user_id").loc[local_df["user_id"]].reset_index()
        checks["total_spend"] = bool(np.allclose(local_df["total_spend"].values, dist_aligned["total_spend"].values, atol=1e-3))
        checks["tx_count"] = bool((local_df["tx_count"].values == dist_aligned["tx_count"].values).all())
        checks["max_amount"] = bool(np.allclose(local_df["max_amount"].values, dist_aligned["max_amount"].values, atol=1e-3))
        checks["min_amount"] = bool(np.allclose(local_df["min_amount"].values, dist_aligned["min_amount"].values, atol=1e-3))

    for k, v in checks.items():
        log.info("[VERIFY] [%s] %s", "OK" if v else "FAIL", k)
    return all(checks.values())


def run_benchmark(data_dir, output_dir, n_workers):
    """Run pipeline at multiple scales."""
    scales = [100_000, 500_000, 1_000_000, 10_000_000]
    results = []
    full_df = pd.read_parquet(os.path.join(data_dir, "transactions.parquet"))
    log.info("[BENCH] Full dataset: %d rows", len(full_df))

    for n in scales:
        if n > len(full_df):
            log.warning("[BENCH] Scale %d > dataset size - skipping", n)
            continue
        scale_dir = os.path.join(output_dir, f"scale_{n}")
        scale_data = os.path.join(scale_dir, "transactions.parquet")
        os.makedirs(scale_dir, exist_ok=True)
        full_df.sample(n=n, random_state=42).reset_index(drop=True).to_parquet(scale_data, index=False)

        log.info("[BENCH] === Scale %d rows ===", n)
        local_m = run_local(scale_data, scale_dir)
        dist_m = run_distributed(scale_data, scale_dir, n_workers)
        speedup = round(local_m["runtime_s"] / dist_m["runtime_s"], 3)
        results.append({
            "rows": n, "local_runtime_s": local_m["runtime_s"],
            "distributed_runtime_s": dist_m["runtime_s"],
            "speedup": speedup, "shuffle_mb": dist_m["shuffle_mb"],
        })
        log.info("[BENCH] %d rows | local %.3fs | ray %.3fs | speedup %.2fx",
                 n, local_m["runtime_s"], dist_m["runtime_s"], speedup)
    return results


def main():
    parser = argparse.ArgumentParser(description="Milestone 4 - Distributed Feature Engineering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--mode", choices=["local", "distributed", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    input_file = os.path.join(args.input, "transactions.parquet")
    if not os.path.exists(input_file):
        log.error("Input file not found: %s", input_file)
        sys.exit(1)

    print("=" * 70)
    print("MILESTONE 4 - Distributed Feature Engineering Pipeline")
    print("=" * 70)
    print(f"  Input: {input_file}  |  Output: {args.output}  |  Workers: {args.workers}")
    print()

    all_metrics = {}
    if args.mode in ("local", "both"):
        all_metrics["local"] = run_local(input_file, args.output)
    if args.mode in ("distributed", "both"):
        all_metrics["distributed"] = run_distributed(input_file, args.output, args.workers)
    if args.mode == "both":
        ok = verify_outputs(args.output)
        all_metrics["outputs_match"] = ok
        if ok:
            speedup = round(all_metrics["local"]["runtime_s"] / all_metrics["distributed"]["runtime_s"], 2)
            all_metrics["speedup"] = speedup
            print(f"\n  Outputs match - pipeline is correct")
            print(f"  Speedup: {speedup}x ({'faster' if speedup > 1 else 'slower'} than local)")

    with open(os.path.join(args.output, "run_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved to %s/run_metrics.json", args.output)

    if args.benchmark:
        results = run_benchmark(args.input, args.output, args.workers)
        with open(os.path.join(args.output, "benchmark_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\nBenchmark Summary:")
        print(f"{'Rows':>12} {'Local (s)':>12} {'Ray (s)':>10} {'Speedup':>10}")
        print("-" * 50)
        for r in results:
            print(f"{r['rows']:>12,} {r['local_runtime_s']:>12.3f} {r['distributed_runtime_s']:>10.3f} {r['speedup']:>10.2f}x")


if __name__ == "__main__":
    main()
