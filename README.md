# Milestone 4: Distributed Feature Engineering with Ray

## Overview
This project implements a distributed feature engineering pipeline using **Ray Data** on synthetic event data.

The repository includes:
- `generate_data.py` for seeded synthetic data generation
- `pipeline.py` for distributed feature engineering
- `README.md` for setup and execution
- `REPORT.md` for performance analysis and architecture discussion
- `requirements.txt` for reproducibility

## Environment
- OS: Windows
- Language: Python
- Distributed framework: Ray Data
- Data: Synthetic only
- Reproducibility: Seeded randomness supported

## Setup

### 1. Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1