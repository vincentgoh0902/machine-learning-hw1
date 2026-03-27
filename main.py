import data_loader
import features
import models
import evaluate

def main():
    """
    The orchestrator module unifying all steps of the Machine Learning pipeline:
    1. Fetch
    2. Engineer Features & Split
    3. Train
    4. Predict & Evaluate
    """
    
    # ---> Phase 1. Data Collection
    print(">>> 1. Fetching S&P 500 data from Yahoo Finance ('^GSPC')...")
    df_raw = data_loader.load_data()
    print(f"Successfully loaded {len(df_raw)} records.\n")
    
    # ---> Phase 2. Feature Generation & Filtering
    print(">>> 2. Engineering time-series features (Lags: 1-day, 3-day | SMAs: 5-day, 20-day)...")
    X, y, df_processed = features.engineer_features(df_raw)
    
    # ---> Phase 3. Splitting Without Data Leakage
    print(">>> 3. Splitting dataset chronologically: Train (2021-2024), Test (2025)...")
    X_train, X_test, y_train, y_test, test_dates = features.split_data(X, y, df_processed)
    
    print(f"  Training Dimension: X={X_train.shape}, y={y_train.shape}")
    print(f"  Testing Dimension:  X={X_test.shape}, y={y_test.shape}\n")
    
    # ---> Phase 4. Regressor Optimization & Fit
    print(">>> 4. Training Random Forest Regressor model...")
    rf_model = models.train_rf(X_train, y_train)
    rf_preds = models.predict(rf_model, X_test)
    
    print(">>> 5. Training XGBoost Regressor model...")
    xgb_model = models.train_xgb(X_train, y_train)
    xgb_preds = models.predict(xgb_model, X_test)
    
    # ---> Phase 5. Aggregation & Reporting 
    print("\n>>> 6. Scoring Models against Validation Set / Exporting Plot...")
    evaluate.evaluate_and_plot(y_test, rf_preds, xgb_preds, test_dates, filename='sp500_predictions.png')
    
    print("Pipeline execution fully completed.")

if __name__ == "__main__":
    main()
