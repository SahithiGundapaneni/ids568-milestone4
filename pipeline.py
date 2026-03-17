import argparse
import os
import shutil
import time
import pandas as pd
import ray


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

    df["revenue"] = df["price"] * df["quantity"]
    df["is_purchase"] = (df["event_type"] == "purchase").astype(int)
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    df["hour_of_day"] = df["event_ts"].dt.hour.fillna(0).astype(int)

    return df


def run_pipeline(input_path: str, output_path: str) -> None:
    start_time = time.time()

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    ray.init(ignore_reinit_error=True)

    ds = ray.data.read_csv(os.path.join(input_path, "data.csv")).materialize()
    print(f"Initial partitions: {ds.num_blocks()}")

    ds = ds.repartition(4).materialize()
    print(f"Partitions after repartition: {ds.num_blocks()}")

    ds = ds.map_batches(add_features, batch_format="pandas")
    ds = ds.sort(["user_id", "product_id", "event_ts"])
    ds.write_csv(output_path)

    end_time = time.time()
    print(f"Output written to: {output_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    run_pipeline(args.input, args.output)