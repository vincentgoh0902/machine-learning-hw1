from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_rf(X_train, y_train):
    """
    Initializes and trains a Random Forest Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        
    Returns:
        RandomForestRegressor: A trained RF model.
    """
    # n_estimators=100 creates an ensemble of 100 decision trees.
    # random_state=42 ensures reproducibility across multiple runs.
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgb(X_train, y_train):
    """
    Initializes and trains an XGBoost Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        
    Returns:
        XGBRegressor: A trained XGB model.
    """
    # n_estimators=100 limits boosting rounds.
    # random_state=42 guarantees consistent splits and sampling.
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def predict(model, X_test):
    """
    Generates target predictions using the provided trained model.
    
    Args:
        model: MLa trained regressor (RF or XGB).
        X_test (pd.DataFrame): Testing feature dataset.
        
    Returns:
        np.array: Array of predicted prices.
    """
    return model.predict(X_test)
