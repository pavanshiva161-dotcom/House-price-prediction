# 🏠 House Price Prediction

End-to-end Data Science project predicting California house prices using Machine Learning.

## Results
| Model | RMSE |
|-------|------|
| Baseline (predict mean) | ~0.60 |
| Linear Regression | ~0.72 |
| Random Forest | ~0.49 |
| Gradient Boosting (tuned) | ~0.46 |

23% improvement over baseline**

## Project Workflow
1. **EDA** — explored distributions, correlations, outliers
2. **Cleaning** — winsorized outliers, log-transformed target
3. **Feature Engineering** — created 3 new ratio features
4. **Modeling** — compared 4 algorithms with cross-validation
5. **Tuning** — GridSearchCV on best model

## Technologies
Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn

## How to Run
1. Clone this repo
2. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run any .py file in VS Code or Google Colab
