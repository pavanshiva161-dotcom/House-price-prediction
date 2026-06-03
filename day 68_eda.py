# Day 68: House Price Project — Deep EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from scipy import stats

np.random.seed(42)

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df['SalePrice'] = df['MedHouseVal'] * 100000

# Target analysis
skewness = df['SalePrice'].skew()
print(f"Skewness before log: {skewness:.4f}")

df['LogSalePrice'] = np.log1p(df['SalePrice'])
log_skew = df['LogSalePrice'].skew()
print(f"Skewness after log : {log_skew:.4f}")

# Correlations
feature_cols = [c for c in df.columns
                if c not in ['SalePrice','LogSalePrice','MedHouseVal']]
corr = df[feature_cols + ['SalePrice']].corr()['SalePrice']
corr = corr.drop('SalePrice').sort_values(ascending=False)
print("\nAll feature correlations:")
print(corr)

# Outlier detection
def count_outliers(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()
    return outliers, outliers/len(series)*100

print("\nOutlier analysis:")
for col in feature_cols:
    n, pct = count_outliers(df[col])
    print(f"  {col:<15}: {n:>5} outliers ({pct:.1f}%)")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['SalePrice'], bins=50,
             color='steelblue', edgecolor='white')
axes[0].set_title(f'SalePrice (skew={skewness:.2f})')
axes[1].hist(df['LogSalePrice'], bins=50,
             color='coral', edgecolor='white')
axes[1].set_title(f'log(SalePrice) (skew={log_skew:.2f})')
plt.tight_layout()
plt.show()