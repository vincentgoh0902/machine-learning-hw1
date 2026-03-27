import pandas as pd

def engineer_features(df):
    """
    Engineers custom indicators and lag features to prevent look-ahead bias.
    This ensures we only use data up to day T-1 to predict the price for day T.
    
    Args:
        df (pd.DataFrame): The raw historical stock data.
        
    Returns:
        pd.DataFrame (X): The feature set.
        pd.Series (y): The target variable (actual closing price).
        pd.DataFrame (df): The merged dataframe with dropped NaNs.
    """
    # Create a copy to prevent any SettingWithCopyWarnings
    df = df.copy()
    
    # 1. Target Variable
    # Our target is to predict today's real closing price.
    y = df['Close']
    
    # 2. Lag Features
    # To predict today's price, we absolutely cannot use today's open/high/low/volume.
    # Therefore, we shift the 'Close' prices downwards so that day T sees day T-1's closing price.
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag3'] = df['Close'].shift(3)
    
    # 3. Rolling Average Features
    # We calculate the 5-day and 20-day Simple Moving Averages (SMA).
    # To prevent look-ahead bias, this must be calculated on the Lag1 price.
    df['SMA_5'] = df['Close_Lag1'].rolling(window=5).mean()
    df['SMA_20'] = df['Close_Lag1'].rolling(window=20).mean()
    
    # Drop rows that have NaN values because of the shift and rolling window functions.
    # The first 20 days will be dropped.
    df = df.dropna()
    
    # Re-assign target y and define our final feature space X
    y = df['Close']
    X = df[['Close_Lag1', 'Close_Lag3', 'SMA_5', 'SMA_20']]
    
    return X, y, df

def split_data(X, y, df, train_end_date='2024-12-31', test_start_date='2025-01-01'):
    """
    Splits the datasets into Training and Testing components chronologically.
    No random shuffling is performed.
    
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        df (pd.DataFrame): Entire dataframe (used to access index dates).
        train_end_date (str): The cutoff date for the training set.
        test_start_date (str): The starting date for the test set.
        
    Returns:
        X_train, X_test, y_train, y_test, test_dates
    """
    # Convert index to datetime format just to be safe
    df.index = pd.to_datetime(df.index)
    
    # Create boolean masks to index our splits
    train_mask = df.index <= pd.to_datetime(train_end_date)
    test_mask = df.index >= pd.to_datetime(test_start_date)
    
    # Generate splits based on the boolean masks
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    
    # Store the test dates for plotting purposes along the x-axis later
    test_dates = df.index[test_mask]
    
    return X_train, X_test, y_train, y_test, test_dates
