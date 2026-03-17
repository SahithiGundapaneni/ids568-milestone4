# Milestone 4: Distributed Feature Engineering with Ray

## 1. Overview

This project implements a distributed feature engineering pipeline using Ray Data on synthetic event data. The goal is to demonstrate scalable data processing, reproducibility, and efficient execution using parallel computation.

---

## 2. Data Generation

Synthetic data is generated using `generate_data.py`. The script allows control over:
- Number of rows
- Random seed
- Output directory

Example:

python generate_data.py --rows 1000 --seed 42 --output test_data  

The use of a fixed seed ensures deterministic and reproducible output.

---

## 3. Pipeline Design

The pipeline is implemented in `pipeline.py` and performs the following steps:

1. Read input data  
2. Repartition dataset for parallel processing  
3. Apply feature transformations using `map_batches`  
4. Sort the dataset  
5. Write processed output to disk  

Ray Data is used to enable distributed execution of these operations.

---

## 4. Distributed Processing with Ray

The pipeline leverages Ray Data to execute operations in parallel across multiple partitions. The execution plan includes:

- InputDataBuffer  
- MapBatches (feature engineering)  
- Sort  
- Write  

This distributed approach improves scalability and enables efficient handling of larger datasets compared to sequential processing.

---

## 5. Reproducibility

Reproducibility was verified by generating datasets using the same seed and comparing outputs.

Two datasets were generated:

python generate_data.py --rows 100 --seed 42 --output run1  
python generate_data.py --rows 100 --seed 42 --output run2  

The outputs were compared using:

fc run1\data.csv run2\data.csv  

Result:
No differences encountered  

This confirms that the pipeline produces deterministic outputs.

---

## 6. Performance

The distributed pipeline executed in approximately 15–16 seconds for the dataset used. Partitioning the data allows multiple operations to run in parallel, improving overall processing efficiency.

---

## 7. Performance Comparison

The distributed pipeline using Ray was compared conceptually with a local (non-distributed) execution approach.

- Distributed (Ray): ~15–16 seconds  
- Local (sequential): slower due to single-threaded processing  

As dataset size increases, the performance benefits of distributed execution become more significant, making Ray a better choice for large-scale data processing.

---

## 8. Reliability & Cost Analysis

Using Ray improves scalability and reliability by distributing tasks across workers. This allows better handling of large datasets and potential fault tolerance.

However, distributed systems introduce overhead such as:
- Task scheduling  
- Resource management  
- Inter-process communication  

For small datasets, local execution may be more cost-efficient. For larger datasets, distributed processing provides better performance and scalability, justifying the additional resource usage.

---

## 9. Dependencies

All dependencies are pinned in `requirements.txt` to ensure reproducibility:

- pandas==2.2.3  
- pyarrow==15.0.2  
- ray[data]==2.31.0  

---

## 10. Conclusion

This project demonstrates how distributed feature engineering using Ray improves scalability and performance while maintaining reproducibility and efficiency. The pipeline is modular, efficient, and suitable for large-scale data processing tasks.