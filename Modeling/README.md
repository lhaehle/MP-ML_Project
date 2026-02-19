# MP -- US New Car Market Nowcasting

Estimates non-reporting brand car sales from reporting brand sales data using statistical and ML models. Reads sales/registration data from SQL Server, performs nowcasting (short-term estimation) and forecasting with hierarchical reconciliation.

## Tech Stack

- **Language**: Python 3.x (Conda env `mp_stable`)
- **Data**: Pandas, NumPy, SQLAlchemy + pyodbc (SQL Server)
- **ML**: scikit-learn, LightGBM, CatBoost, XGBoost
- **Visualization**: matplotlib, plotly

## Pipeline

The system runs in two stages:

1. **`main_1.py`** -- loads raw data from SQL Server, joins reference tables, cleans/transforms, and writes `tmp_20260205.csv`
2. **`main_2.py`** -- reads `tmp_20260205.csv`, runs nowcasting models, reconciliation, and reporting

## Files Required to Run

### Stage 1: `main_1.py`

| File | Type | Description |
|------|------|-------------|
| `main_1.py` | Script | Data loading, SQL connection, data preparation |
| `.env` | Config | DB credentials (`DB_USERNAME`, `DB_PASSWORD`) |
| `List of US Census Tract Centroids_20251003.xlsx` | Reference data | Census tract centroids for geographic mapping |
| `HYU_MODEL_KEY_SHORT_5YR_202506.txt` | Reference data | Model key / vehicle segment reference |

Both reference files are auto-cached as `.csv` on first load (e.g. `List of US Census Tract Centroids_20251003.xlsx.csv`).

**Output**: `tmp_20260205.csv` (cleaned transaction-level data)

### Stage 2: `main_2.py`

| File | Type | Description |
|------|------|-------------|
| `main_2.py` | Script | Nowcasting models, reconciliation, reporting |
| `tmp_20260205.csv` | Input data | Output of `main_1.py` (Stage 1) |

**Output**: Console reporting (results registry, bin inspection, reconciliation matrices)

## Directory Structure

```
code/
  main_1.py                                  Stage 1: data extraction
  main_2.py                                  Stage 2: nowcasting & reporting
  .env                                       DB credentials (not in VCS)
  List of US Census Tract Centroids_20251003.xlsx   Reference data
  HYU_MODEL_KEY_SHORT_5YR_202506.txt                Reference data
  tmp_20260205.csv                           Intermediate data (Stage 1 output)
  plots/                                     Generated PNG visualizations
  catboost_info/                             CatBoost training artifacts
  archive/                                   Older/experimental versions
    matrix_1*.py, multimodel*.py,            Previous model iterations
    tabular_*.py, car_sales_nowcasting*.py
    *.csv                                    Archived intermediate data
  co0_df.csv .. co3_df.csv                   Cached intermediate DataFrames
  us5b_df.csv                                Cached intermediate DataFrame
  *.xlsx                                     Large working datasets (not in VCS)
```

## Setup

### Prerequisites

Install [ODBC Driver 17 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) (required for database access).

### Create Conda Environment

```bash
conda env create -f environment.yml
conda activate mp_stable
```

To recreate an existing environment:

```bash
conda deactivate
conda env remove -n mp_stable
conda env create -f environment.yml
```

### Configure Database Credentials

Create a `.env` file in the project root (or set environment variables):

```
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

### Run

```bash
conda activate mp_stable
python main_1.py                              # Stage 1: data extraction
python main_2.py                              # Stage 2: nowcasting & reporting
```

## Data Source

SQL Server database `RTID_SourceData`, tables `SPGM_Weekly_INV_NVI_SLS_*`. Connection at `10.0.30.16:1433` via ODBC Driver 17.
