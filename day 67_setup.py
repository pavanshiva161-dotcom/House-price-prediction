# Day 67: House Price Project — Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

np.random.seed(42)

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df['SalePrice'] = df['MedHouseVal'] * 100000

# Basic info
print(f"Shape: {df.shape}")
print(f"Mean price : ${df['SalePrice'].mean():,.0f}")
print(f"Median price: ${df['SalePrice'].median():,.0f}")

# Top correlations
corr = df.corr()['SalePrice'].sort_values(ascending=False)
print("\nTop 4 correlations with SalePrice:")
print(corr.head(4))

# Plot
plt.figure(figsize=(8, 4))
plt.hist(df['SalePrice'], bins=50,
         color='steelblue', edgecolor='white')
plt.title('SalePrice Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()