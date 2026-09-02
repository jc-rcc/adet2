import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

X = df[['milesTravelled', 'numDeliveries', 'gasPrice']]

for i, col in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    interp = "Serious" if vif > 10 else ("Moderate" if vif >= 5 else "Low")
    print(f"{col:<16}{vif:>12.4f}{interp:>20}")