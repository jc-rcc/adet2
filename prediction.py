import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

y = df['travelTime']

X_red = sm.add_constant(df[['milesTravelled', 'gasPrice']])
red = sm.OLS(y, X_red).fit()

new = pd.DataFrame({'const': [1], 'milesTravelled': [90], 'gasPrice': [3.60]})
pred = red.predict(new)

print(f"Predicted travelTime: {pred[0]:.4f} hours")