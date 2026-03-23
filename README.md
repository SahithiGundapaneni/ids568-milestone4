# Milestone 4 — Distributed & Streaming Pipeline

**MLOps Course — Module 5**  
Framework: Ray | Language: Python 3.9+

---

## Repository Structure

```
ids568-milestone4/
├── pipeline.py          # Distributed feature engineering (Ray vs pandas)
├── generate_data.py     # Synthetic data generator (10M+ rows)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── REPORT.md            # Performance analysis and architecture evaluation
```

---

## Prerequisites

- Python 3.9, 3.10, or 3.11
- ~4 GB free RAM (for 10M row benchmark)
- ~600 MB disk space

No Docker required. Everything runs locally.

---

## Setup (4 Steps)

### 1. Clone the repository

```bash
git clone https://github.com/SahithiGundapaneni/ids568-milestone4.git
cd ids568-milestone4
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
# Quick test (1,000 rows — runs in seconds)
python generate_data.py --rows 1000 --seed 42 --output test_data/

# Full benchmark dataset (10M rows — takes ~30–60s)
python generate_data.py --rows 10000000 --seed 42 --output data/
```

Expected output:

```
  Generated in   : 38.4s
  Shape          : (10000000, 6)
  Output         : data/transactions.parquet  (168.2 MB)
  Hash (sha256)  : <16-char hex>
  Unique users   : 100,000
  Unique merchants: 10,000
  Amount range   : $0.01 – $9,843.21
```

The hash is deterministic — running the same command again on any machine with
the same library versions produces the identical hash.

---

## Running the Pipeline

```bash
# Quick correctness test (1,000 rows)
python pipeline.py --input test_data/ --output test_output/

# Full run — both local and distributed, single pass
python pipeline.py --input data/ --output results/

# Multi-scale benchmark (100K, 500K, 1M, 10M rows)
python pipeline.py --input data/ --output results/ --benchmark

# Control number of Ray workers (default: all CPU cores)
python pipeline.py --input data/ --output results/ --workers 4

# Run only local (pandas) mode
python pipeline.py --input data/ --output results/ --mode local

# Run only distributed (Ray) mode
python pipeline.py --input data/ --output results/ --mode distributed
```

---

## Expected Output (full run)

```
======================================================================
MILESTONE 4 — Distributed Feature Engineering Pipeline
======================================================================
  Input     : data/transactions.parquet
  Output    : results/
  Workers   : 8
  Mode      : both
  Benchmark : False

2024-xx-xx [INFO] [LOCAL] Starting pandas baseline …
2024-xx-xx [INFO] [LOCAL] Loaded 10000000 rows, 6 columns
2024-xx-xx [INFO] [LOCAL] Done in 3.241s | peak Δmem 1.07 GB | 100000 output rows

2024-xx-xx [INFO] [RAY] Initialising Ray with 8 workers …
2024-xx-xx [INFO] [RAY] 10000000 rows → 16 partitions (8 workers)
2024-xx-xx [INFO] [RAY] MAP done in 2.977s
2024-xx-xx [INFO] [RAY] SHUFFLE done in 18.312s | 112.4 MB
2024-xx-xx [INFO] [RAY] REDUCE done in 0.791s
2024-xx-xx [INFO] [RAY] Done in 22.080s | shuffle 112.4 MB | workers 97.1%

2024-xx-xx [INFO] [VERIFY] ✅ same_users
2024-xx-xx [INFO] [VERIFY] ✅ total_spend
2024-xx-xx [INFO] [VERIFY] ✅ tx_count
2024-xx-xx [INFO] [VERIFY] ✅ max_amount
2024-xx-xx [INFO] [VERIFY] ✅ min_amount

  ✅ Outputs match — distributed pipeline is correct
  Speedup: 0.15x  (slower than local)

Metrics saved to results/run_metrics.json
```

*Note: exact timings vary by machine. Hash values are deterministic.*

---

## Output Files

After running, `results/` contains:

| File | Description |
|---|---|
| `local_features.parquet` | Per-user features from pandas baseline |
| `distributed_features.parquet` | Per-user features from Ray pipeline |
| `run_metrics.json` | Timing, memory, shuffle, and speedup metrics |
| `benchmark_metrics.json` | Multi-scale metrics (if `--benchmark` used) |

---

## Feature Engineering Logic

Both local (pandas) and distributed (Ray) modes apply identical transforms:

| Feature | Formula |
|---|---|
| `log_amount` | `log1p(amount)` |
| `hour_of_day` | `timestamp.hour` |
| `day_of_week` | `timestamp.dayofweek` |
| `is_weekend` | `day_of_week >= 5` |
| `amount_z_score` | `(amount − μ) / σ` (batch-level) |
| `amount_bucket` | `cut(amount, [0, 25, 100, 500, ∞])` → 0–3 |
| `amount_log_bin` | log-scale bin (0–4) |
| `event_type_enc` | ordinal encoding |
| `total_spend` | `sum(amount)` per user |
| `tx_count` | `count` per user |
| `unique_merchants` | `nunique(merchant_id)` per user |
| `pct_weekend` | fraction of weekend events per user |

---

## Ray Architecture

```
Driver (pipeline.py)
    │
    ├── ray.put(chunk_0) ──► Object Store (shared memory)
    ├── ray.put(chunk_1)
    │   ...
    │
    ├── _ray_transform_chunk.remote(chunk_0) ──► Worker 0
    ├── _ray_transform_chunk.remote(chunk_1) ──► Worker 1   ← MAP phase
    │   ...                                         (parallel)
    │
    ├── _ray_partial_groupby.remote(tr_0) ──► Worker 0
    ├── _ray_partial_groupby.remote(tr_1) ──► Worker 1      ← SHUFFLE phase
    │   ...
    │
    └── _merge_partial_groupbys(partials) ──► Driver        ← REDUCE phase
```

---

## Reproducibility Verification

```bash
# Generate twice and confirm hashes match
python generate_data.py --rows 10000 --seed 42 --output /tmp/run1/
python generate_data.py --rows 10000 --seed 42 --output /tmp/run2/
# Both should print the identical hash

# Verify pipeline determinism
python pipeline.py --input /tmp/run1/ --output /tmp/out1/
python pipeline.py --input /tmp/run1/ --output /tmp/out2/

python -c "
import pandas as pd
a = pd.read_parquet('/tmp/out1/local_features.parquet').sort_values('user_id')
b = pd.read_parquet('/tmp/out2/local_features.parquet').sort_values('user_id')
print('Match:', a.equals(b))
"
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: ray` | Activate venv: `source venv/bin/activate` |
| `ray.init()` hangs | Kill stale Ray: `ray stop` then retry |
| Out of memory at 10M rows | Use `--workers 2` or reduce rows |
| Parquet read fails | Ensure pyarrow installed: `pip install pyarrow` |
| Hash mismatch on same seed | Check numpy/pandas versions match `requirements.txt` |
| Low speedup | Expected on machines with < 4 cores or RAM < 8 GB |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ray` | ≥ 2.9.3 | Distributed task execution |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `pyarrow` | ≥ 14.0.0 | Parquet I/O |
| `psutil` | ≥ 5.9.0 | Memory / CPU metrics |
