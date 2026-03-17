# Milestone 4: Distributed Feature Engineering with Ray

## Overview

This project implements a distributed feature engineering pipeline using Ray Data on synthetic event data. The pipeline demonstrates scalable data processing, parallel execution, and reproducibility.

---

## Setup

1. Create and activate a virtual environment:

python -m venv venv  
venv\Scripts\activate  

2. Install dependencies:

pip install -r requirements.txt  

---

## Data Generation

Generate synthetic data using:

python generate_data.py --rows 1000 --seed 42 --output test_data  

---

## Running the Pipeline

Execute the pipeline:

python pipeline.py --input test_data --output output_data  

---

## Pipeline Steps

- Read dataset  
- Repartition data  
- Apply feature engineering using `map_batches`  
- Sort data  
- Write output  

---

## Distributed Processing

The pipeline uses Ray Data for parallel execution. Operations are distributed across multiple partitions, enabling efficient large-scale data processing.

---

## Reproducibility

The pipeline is deterministic when using the same seed. Running the data generation multiple times with the same seed produces identical outputs.

---

## Dependencies

- pandas==2.2.3  
- pyarrow==15.0.2  
- ray[data]==2.31.0  

---

## Output

The processed dataset is written to the specified output directory (`output_data`).

---

## Conclusion

This project demonstrates how distributed feature engineering using Ray improves scalability and performance while maintaining reproducibility and efficiency.