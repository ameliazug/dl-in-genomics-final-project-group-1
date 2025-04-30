import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

'''
Three baseline models were selected to evaluate M2A2's performance.

    1) Random Forest: Ensemble of decision trees capturing non-linear interactions; robust to noise.

    2) Ridge Regression: Linear regression with L2 regularization to reduce overfitting and handle multicollinearity.
    
    3) Original M2A Model: CNN extracts spatial patterns; FC layers map to promoter activity.
'''

# Use: X shape = (num_promoters, 2 window sizes × 20 windows × 3 stats (120?)), y shape = (num_promoters,)
#   X is a 2D NumPy array or Pandas DataFrame containing DNA methylation features per promoter region.
#   y is a 1D NumPy array or Series containing the ground truth values for each promoter’s activity

# We need the same train/test split used for M2A2
def load_training_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = np.array(f["FeatureInput"])[:, :, :, [0, 1, 3]]  # Use features: Ave, Var, FracSSD
        X = X.reshape(X.shape[0], -1)
        y = np.array(f["log2_ChipDivInput"]).astype(float)
    return X, y

# ---- Evaluation ----
def evaluate_model(preds, y_true, name):
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

def run_baselines(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- Baseline 1: Random Forest ----
    print("Running Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    evaluate_model(rf_preds, y_test, "Random Forest")

    # ---- Baseline 2: Ridge Regression ----
    print("Running Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    evaluate_model(ridge_preds, y_test, "Ridge Regression")
    
    return rf_preds, ridge_preds, y_test

# ---- Visualization ----
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal fit')
    plt.xlabel("Actual Promoter Activity")
    plt.ylabel("Predicted Promoter Activity")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------- Run --------------- #

if __name__ == "__main__":

    h5_input_path = "data/A549/methyl_enrichment.h5"  # <<<<=== Replace
    X, y = load_training_data(h5_file=h5_input_path)
    rf_preds, ridge_preds, y_test = run_baselines(X, y)

    plot_predictions(y_test, rf_preds, "Random Forest")
    plot_predictions(y_test, ridge_preds, "Ridge Regression")
