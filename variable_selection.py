import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

y = df['travelTime']

X_full = sm.add_constant(df[['milesTravelled', 'numDeliveries', 'gasPrice']])
full = sm.OLS(y, X_full).fit()

X_red = sm.add_constant(df[['milesTravelled', 'gasPrice']])
red = sm.OLS(y, X_red).fit()

print("=" * 70)
print("FULL MODEL (3 predictors)")
print("=" * 70)
print(f"R-squared      = {full.rsquared:.4f}")
print(f"Adjusted R^2   = {full.rsquared_adj:.4f}")
print(f"{'Term':<16}{'Coefficient':>14}{'p-value':>12}{'Sig?':>8}")
for term in ['const', 'milesTravelled', 'numDeliveries', 'gasPrice']:
    sig = "Yes" if full.pvalues[term] < 0.05 else "No"
    print(f"{term:<16}{full.params[term]:>14.4f}{full.pvalues[term]:>12.4f}{sig:>8}")

print()
print("=" * 70)
print("REDUCED MODEL (drop numDeliveries)")
print("=" * 70)
print(f"R-squared      = {red.rsquared:.4f}")
print(f"Adjusted R^2   = {red.rsquared_adj:.4f}")
print(f"{'Term':<16}{'Coefficient':>14}{'p-value':>12}{'Sig?':>8}")
for term in ['const', 'milesTravelled', 'gasPrice']:
    sig = "Yes" if red.pvalues[term] < 0.05 else "No"
    print(f"{term:<16}{red.params[term]:>14.4f}{red.pvalues[term]:>12.4f}{sig:>8}")