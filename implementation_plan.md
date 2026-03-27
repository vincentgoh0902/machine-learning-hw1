# Stock Price Prediction System (S&P 500)

This document outlines the approach for developing a stock price prediction system to compare XGBoost and Random Forest models using S&P 500 data.

## Proposed Changes

### Data Pipeline
- Process for downloading the S&P 500 (^GSPC) data using `yfinance`.
- The dataset will cover '2021-01-01' to '2025-12-31'.

#### [NEW] `data_loader.py`
- Contains functions to fetch the data.
- Handles missing values using forward fill (if any).
- Ensures that multi-level columns from `yfinance` are appropriately flattened.

### Feature Engineering
- Construct features ensuring **no look-ahead bias**.
- Features will strictly include lagged prices (1-day, 3-day) and rolling 5-day / 20-day moving averages of the shifted closing prices.
- Strict condition: no current day's testing variables will leak into the current day's prediction. The target remains the actual closing stock price `Close`.

#### [NEW] `features.py`
- Implements `Close_Lag1` and `Close_Lag3` by shifting the `Close` price.
- Implements `SMA_5` and `SMA_20` based on `Close_Lag1` to strictly ensure the rolling average relies only on past data.

### Modeling & Splitting
- Train/Test Split logic: Time-based split.
  - Train: 2021-01-01 to 2024-12-31.
  - Test: 2025-01-01 to 2025-12-31.
  - No random shuffling to maintain temporal integrity.

#### [NEW] `models.py`
- Instantiates `RandomForestRegressor` and `XGBRegressor`.
- Fits models strictly on the training set.

### Evaluation & Visualization
- Compute Mean Squared Error (MSE) for both models.
- Generate comparative line charts detailing Actual vs. Predicted values.

#### [NEW] `evaluate.py`
- Calculates MSE metrics.
- Uses `matplotlib` to chart the actual prices versus predicted prices over 2025.

#### [NEW] `main.py`
- The driver script to orchestrate data loading, feature engineering, modeling, and evaluation.
- All code will be heavily commented as requested.

## Verification Plan

### Automated Tests
- Run `python main.py` and ensure the output prints the MSEs.
- Verify `matplotlib` outputs plots correctly (we can save them to files).

### Manual Verification
- Manually inspect the dataset's chronological split ensuring dates 2021-2024 fall in Train, and 2025 falls in Test.
- Confirm features used for $T$ rely only on data up to $T-1$.
