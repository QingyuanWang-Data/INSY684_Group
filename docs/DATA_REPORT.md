# Data Report

## Dataset Source

We use the Home Credit Default Risk dataset from Kaggle. The goal is to predict whether a loan applicant will experience repayment difficulty, represented by the binary target variable `TARGET`.

## Raw Data Files

| File | Role |
|---|---|
| `application_train.csv` | Main labeled training table with `TARGET` |
| `application_test.csv` | Kaggle scoring table without `TARGET` |
| `bureau.csv` | Applicant credit bureau history |
| `bureau_balance.csv` | Monthly status history for bureau records |
| `previous_application.csv` | Previous Home Credit applications |
| `POS_CASH_balance.csv` | Monthly POS/cash loan status |
| `installments_payments.csv` | Installment payment history |
| `credit_card_balance.csv` | Credit card monthly balance history |

## Data Processing Pipeline

The raw Kaggle dataset contains multiple relational tables. The preprocessing pipeline converts these tables into one applicant-level modeling frame using `SK_ID_CURR` as the primary key.

Main steps:

1. Load raw CSV files from `data/raw`.
2. Validate that all required files exist.
3. Clean sentinel values such as `DAYS_EMPLOYED = 365243`.
4. Create ratio features from the main application table.
5. Filter auxiliary records using relative-time columns to reduce temporal leakage risk.
6. Aggregate auxiliary tables by applicant ID using summary statistics.
7. Left-join aggregate features back to the main application table.
8. Generate data quality and profiling outputs.

## Final Training Frame

After preprocessing:

| Metric | Value |
|---|---:|
| Rows | 307,511 |
| Columns | 452 |
| Duplicate applicants | 0 |
| Missing target values | 0 |
| Positive class count | 24,825 |
| Negative class count | 282,686 |
| Positive class rate | 8.07% |

## Class Imbalance

The positive class represents only 8.07% of the training data. This means accuracy alone is not a reliable metric. Model evaluation should include ranking, threshold-sensitive, and calibration metrics such as ROC-AUC, PR-AUC, recall, precision, and calibration curves.

## Missing Values

High missingness is expected for some historical product-specific features because not every applicant has records in every auxiliary table. For example, some applicants do not have credit card balance records or detailed previous-application interest rate fields. We preserve these missing values instead of dropping applicants, since missingness itself may carry predictive information in credit risk modeling.

## Data Leakage Prevention

The preprocessing pipeline applies a temporal leakage guard before aggregating historical records. Relative-time columns such as `DAYS_CREDIT`, `DAYS_DECISION`, `MONTHS_BALANCE`, `DAYS_ENTRY_PAYMENT`, and `DAYS_INSTALMENT` are filtered so that only non-future records are retained. This helps ensure that the model does not use information that would not have been available at the time of application.

## Production-Oriented Data Checks

The script `scripts/build_dataset.py` generates reproducible data profiling outputs under `artifacts/data_profile/`:

- `class_balance.csv`
- `data_quality_checks.json`
- `training_frame_profile.csv`

These outputs support data validation, monitoring, and future drift checks in a production ML workflow.