import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_and_plot(y_true, y_pred_rf, y_pred_xgb, dates, filename='sp500_predictions.png'):
    """
    Evaluates model performance using Mean Squared Error (MSE) and 
    generates an overlay plot contrasting the true price against our model predictions.
    
    Args:
        y_true (pd.Series): The actual target values for the test set.
        y_pred_rf (np.array): Predicted values from Random Forest.
        y_pred_xgb (np.array): Predicted values from XGBoost.
        dates (pd.DatetimeIndex): The dates associated with the test set for the x-axis.
        filename (str): The destination path for saving the PNG plot.
    """
    # 1. Calculation phase
    # Quantitatively score both models by calculating MSE
    mse_rf = mean_squared_error(y_true, y_pred_rf)
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)
    
    print("-" * 40)
    print("EVALUATION RESULTS (Mean Squared Error):")
    print(f"Random Forest MSE: {mse_rf:,.2f}")
    print(f"XGBoost MSE:       {mse_xgb:,.2f}")
    print("-" * 40)
    
    # 2. Rendering phase
    # Instantiate a clean matplotlib figure
    plt.figure(figsize=(14, 7))
    
    # Plot true values
    plt.plot(dates, y_true, label='Actual S&P 500 Price (2025)', color='black', linewidth=2)
    
    # Plot algorithm predictions
    plt.plot(dates, y_pred_rf, label='Random Forest', color='green', alpha=0.7, linestyle='--')
    plt.plot(dates, y_pred_xgb, label='XGBoost', color='red', alpha=0.7, linestyle='-.')
    
    # Cosmetic enhancements (Title, Labels, Legend)
    plt.title('S&P 500 Price Prediction: Actual vs Predicted (2025 Test Set)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Closing Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout() # Ensures no margins are clipped
    
    # Finalize and export the visualization
    plt.savefig(filename, dpi=300)
    print(f"Plot visually saved to '{filename}' using 300 DPI.")
    plt.show() # Attempt to display if a graphic backend exists overlay
    
    return mse_rf, mse_xgb
