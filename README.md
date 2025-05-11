# Bitcoin Time Series Modeling Project

## Overview
An automated system for Bitcoin price prediction using time series analysis. The project includes data extraction, model training, and automated predictions using MLflow for experiment tracking and Prefect for workflow orchestration.

## Features
- Automated BTC price data extraction from Yahoo Finance
- ARIMA time series modeling with automated parameter tuning
- MLflow integration for experiment tracking and model versioning
- Automated predictions with confidence intervals
- PostgreSQL database for data storage
- Prefect workflows for task orchestration

## Prerequisites
- Python 3.8+
- PostgreSQL database
- MLflow tracking server
- Prefect 2.0+

## Installation
1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```env
POSTGRES_URL
MLFLOW_TRACKING_URI
```

## Usage

### Data Extraction
Run the extraction flow to fetch Bitcoin price data:
```bash
python src/prefect_flows/extraction_flow.py
```

### Model Training
Train a new ARIMA model:
```bash
python src/modeling/train.py
```

### Making Predictions
Generate predictions using the latest model:
```bash
python src/inference/inference.py
```

## Workflows
1. **Data Extraction Flow**
   - Extracts 5-minute interval BTC price data
   - Runs every 12 hours
   - Implements locking mechanism to prevent duplicate runs

2. **Training Flow**
   - Trains ARIMA model on historical data
   - Logs experiments to MLflow
   - Generates confidence intervals

3. **Inference Flow**
   - Loads latest production model
   - Generates predictions with confidence intervals
   - Stores results in database

## Database Schema

### BTC Prices Table
```sql
CREATE TABLE btc_usd_prices (
    intervals TIMESTAMP PRIMARY KEY,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT
);
```

### Predictions Table
```sql
CREATE TABLE btc_usd_predictions (
    datetime TIMESTAMP PRIMARY KEY,
    forecast FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT
);
```

## MLflow Integration
- Experiments tracked at `mlruns/`
- Models stored with metadata and artifacts
- Automated model versioning and deployment

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

