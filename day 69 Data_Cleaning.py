# Day 69: House Price Project — Cleaning + Feature Engineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df['SalePrice'] = df['MedHouseVal'] * 100000

# Winsorize outliers
def winsorize(series, lower=0.05, upper=0.95):
    return series.clip(series.quantile(lower), series.quantile(upper))

for col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']:
    before = df[col].max()
    df[col] = winsorize(df[col])
    print(f"{col}: max {before:.1f} → {df[col].max():.1f}")

# Log transform target
df['LogSalePrice'] = np.log1p(df['SalePrice'])
print(f"\nSkewness before: {df['SalePrice'].skew():.4f}")
print(f"Skewness after : {df['LogSalePrice'].skew():.4f}")

# Feature engineering
df['rooms_per_person']  = df['AveRooms']  / df['AveOccup']
df['income_per_person'] = df['MedInc']    / df['AveOccup']
df['bedrooms_ratio']    = df['AveBedrms'] / df['AveRooms']

new_features = ['rooms_per_person', 'income_per_person', 'bedrooms_ratio']
print("\nNew feature correlations:")
for feat in new_features:
    corr = df[feat].corr(df['SalePrice'])
    print(f"  {feat:<22}: {corr:.4f}")

# Split and scale
feature_cols = [c for c in df.columns
                if c not in ['SalePrice','LogSalePrice','MedHouseVal']]
X = df[feature_cols]
y = df['LogSalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nX_train shape: {X_train_s.shape}")
print(f"X_test shape : {X_test_s.shape}")
print(f"Features     : {feature_cols}")