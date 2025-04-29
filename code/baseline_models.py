import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth  # MARS
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

'''
Three baseline models were selected to evaluate M2A2's performance.

    1) Random Forest: Ensemble of decision trees capturing non-linear interactions; robust to noise.

    2) MARS: Fits piecewise linear splines to model non-linear feature effects.
    
    3) Original M2A Model: CNN extracts spatial patterns; FC layers map to promoter activity.
'''

# Use: X shape = (num_promoters, 2 window sizes × 20 windows × 3 stats (120?)), y shape = (num_promoters,)
#   X is a 2D NumPy array or Pandas DataFrame containing DNA methylation features per promoter region.
#   y is a 1D NumPy array or Series containing the ground truth values for each promoter’s activity

# We need the same train/test split used for M2A2, so we should just save these dfs
# TODO: Access X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    # These params can change slash we don't need to use split() we should just save and load them depending how big

# ---- Baseline 1: Random Forest ----
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# ---- Baseline 2: MARS ----
mars_model = Earth()
mars_model.fit(X_train, y_train)
mars_preds = mars_model.predict(X_test)

# ---- Evaluation ----
def evaluate_model(preds, y_true, name):
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

evaluate_model(rf_preds, y_test, "Random Forest")
evaluate_model(mars_preds, y_test, "MARS")
