# Day 70: House Price Project — Modeling + Final Evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Full pipeline from Days 67-69
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df['SalePrice'] = df['MedHouseVal'] * 100000

def winsorize(series, lower=0.05, upper=0.95):
    return series.clip(series.quantile(lower), series.quantile(upper))

for col in ['AveRooms','AveBedrms','Population','AveOccup']:
    df[col] = winsorize(df[col])

df['LogSalePrice']      = np.log1p(df['SalePrice'])
df['rooms_per_person']  = df['AveRooms']  / df['AveOccup']
df['income_per_person'] = df['MedInc']    / df['AveOccup']
df['bedrooms_ratio']    = df['AveBedrms'] / df['AveRooms']

feature_cols = [c for c in df.columns
                if c not in ['SalePrice','LogSalePrice','MedHouseVal']]
X = df[feature_cols]
y = df['LogSalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Baseline
baseline_pred  = np.full(len(y_test), y_train.mean())
baseline_rmse  = rmse(y_test, baseline_pred)
print(f"Baseline RMSE: {baseline_rmse:.4f}")

# Compare models
models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200,
                            learning_rate=0.1, max_depth=3, random_state=42),
}

results = {}
print(f"\n{'Model':<25} {'Test RMSE':>10}")
print("-" * 38)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    test_rmse = rmse(y_test, model.predict(X_test_s))
    results[name] = test_rmse
    print(f"{name:<25} {test_rmse:>10.4f}")

# Tune best model
param_grid = {
    'n_estimators'  : [100, 200, 300],
    'learning_rate' : [0.05, 0.1, 0.15],
    'max_depth'     : [2, 3, 4]
}
grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=0
)
grid.fit(X_train_s, y_train)
tuned_rmse = rmse(y_test, grid.best_estimator_.predict(X_test_s))

print(f"\nBest params : {grid.best_params_}")
print(f"Tuned RMSE  : {tuned_rmse:.4f}")
improvement = ((baseline_rmse - tuned_rmse) / baseline_rmse) * 100
print(f"Improvement : {improvement:.1f}% better than baseline")

# Feature importance
feat_imp = pd.Series(
    grid.best_estimator_.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

print("\nTop 3 most important features:")
print(feat_imp.tail(3))

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names  = list(results.keys()) + ['Tuned GBM']
rmses  = list(results.values()) + [tuned_rmse]
colors = ['#888780','#888780','#1D9E75','#1D9E75','#7F77DD']

bars = axes[0].bar(range(len(names)), rmses, color=colors, alpha=0.8)
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
axes[0].axhline(y=baseline_rmse, color='red',
                linestyle='--', label=f'Baseline: {baseline_rmse:.3f}')
axes[0].set_title('Model Comparison')
axes[0].set_ylabel('Test RMSE (lower is better)')
axes[0].legend()
for bar, r in zip(bars, rmses):
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.005,
                 f'{r:.3f}', ha='center', fontsize=9)

feat_imp.tail(8).plot(kind='barh', ax=axes[1], color='steelblue', alpha=0.8)
axes[1].set_title('Feature Importance — Tuned GBM')
axes[1].set_xlabel('Importance score')

plt.suptitle('House Price Project — Final Results', fontsize=13)
plt.tight_layout()
plt.show()