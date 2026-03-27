# S&P 500 Stock Price Prediction (HW1)

**Course:** Machine Learning in Engineering Science (National Cheng Kung University)
**Instructor:** Chi-Hua Yu
**Environment:** Google Antigravity IDE (Agentic Software Development)

## Project Objective
This project focuses on predicting the actual closing prices of the S&P 500 (`^GSPC`) using machine learning. Rather than manual coding, this system was developed using an **Agentic Software Development** workflow within the Google Antigravity IDE, where AI agents were orchestrated to plan, write, and evaluate the code while maintaining strict human oversight for financial common sense and academic integrity.

## Project Structure
* `main.py`: The orchestrator script that runs the end-to-end pipeline.
* `data_loader.py`: Fetches S&P 500 data via `yfinance` and handles API multi-index formatting.
* `features.py`: Engineers time-series features and splits the data chronologically.
* `models.py`: Initialises and trains the XGBoost and Random Forest regressors.
* `evaluate.py`: Calculates Mean Squared Error (MSE) and generates the actual vs. predicted visualisation.
* `sp500_predictions.png`: The final plot showing the model results on the 2025 test set.

## Methodology & Financial Integrity

### 1. Data Collection & Splitting
* **Dataset:** S&P 500 (^GSPC) daily data from `2021-01-01` to `2025-12-31`.
* **Chronological Split:** To maintain temporal integrity, the data was split sequentially without random shuffling.
  * **Training Set:** 2021-01-01 to 2024-12-31
  * **Testing Set:** 2025-01-01 to 2025-12-31

### 2. Preventing Look-Ahead Bias (The $T-1$ Rule)
A critical requirement of this project was to avoid data leakage. To predict the price at day $T$, the models strictly rely on data up to day $T-1$:
* **Lagged Features:** `Close_Lag1` and `Close_Lag3` were created using Pandas `.shift()`.
* **Rolling Averages:** 5-day and 20-day Simple Moving Averages (`SMA_5`, `SMA_20`) were calculated *only* using the `Close_Lag1` feature, ensuring today's price never influences today's moving average.

## Results & Analysis
The models were evaluated using **Mean Squared Error (MSE)** on the 2025 testing set. 

**Observation on Extrapolation:**
The visualisation (`sp500_predictions.png`) reveals a known mathematical limitation of tree-based algorithms (Random Forest and XGBoost). In 2025, the S&P 500 experienced a massive rally, reaching unprecedented all-time highs. 

Because decision trees calculate predictions by routing data to terminal "leaves" and outputting the mean of the target values within those leaves, they are fundamentally incapable of extrapolation. When the 2025 features exceeded all known split thresholds from the 2021-2024 training data, every subsequent data point was routed into the exact same "highest" leaf nodes. 

Consequently, the models "flatlined" not at the absolute peak of the training data, but at the **ensemble average of their highest terminal leaves**. Furthermore, XGBoost plateaued at a slightly lower price point than Random Forest; this is due to XGBoost's sequential boosting architecture and learning rate regularisation (`learning_rate=0.1`), which applies shrinkage to its predictions, making it more conservative than a standard Random Forest ensemble. This overall flatlining behavior confirms that no look-ahead bias occurred, as the models successfully remained "blind" to the future highs.

## How to Run
1. Ensure dependencies are installed: `pip install yfinance pandas scikit-learn xgboost matplotlib`
2. Run the pipeline: `python main.py`
