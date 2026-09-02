import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

y = df['travelTime']
X = sm.add_constant(df[['milesTravelled', 'numDeliveries', 'gasPrice']])

model = sm.OLS(y, X).fit()

predictors = ['milesTravelled', 'numDeliveries', 'gasPrice']

lines = []
lines.append("="*70)
lines.append("MULTIPLE LINEAR REGRESSION RESULTS")
lines.append("="*70)

lines.append("\n--- Coefficient Table ---")
header = f"{'Term':<16}{'Coefficient':>14}{'p-value':>12}{'Significant?':>16}"
lines.append(header)
lines.append("-"*len(header))

rows = [('Intercept', model.params['const'], model.pvalues['const'])]
for p in predictors:
    rows.append((p, model.params[p], model.pvalues[p]))

for term, coef, pval in rows:
    sig = "Yes" if pval < 0.05 else "No"
    lines.append(f"{term:<16}{coef:>14.4f}{pval:>12.4f}{sig:>16}")

b0 = model.params['const']
b1 = model.params['milesTravelled']
b2 = model.params['numDeliveries']
b3 = model.params['gasPrice']

lines.append("\n--- 1. Complete Regression Equation ---")
lines.append(f"travelTime = {b0:.4f} + "
             f"({b1:.4f} * milesTravelled) + "
             f"({b2:.4f} * numDeliveries) + "
             f"({b3:.4f} * gasPrice)")

lines.append("\n--- 2. Interpret coefficient of milesTravelled ---")
lines.append(f"Holding numDeliveries and gasPrice constant, kada dagdag ng 1 mile sa "
             f"milesTravelled, tumataas/bumababa ang travelTime ng {b1:.4f} hours.")

lines.append("\n--- 3. Interpret coefficient of numDeliveries ---")
lines.append(f"Holding milesTravelled and gasPrice constant, kada dagdag ng 1 delivery sa "
             f"numDeliveries, tumataas/bumababa ang travelTime ng {b2:.4f} hours.")

lines.append("\n--- 4. Interpret coefficient of gasPrice ---")
lines.append(f"Holding milesTravelled and numDeliveries constant, kada dagdag ng $1 sa "
             f"gasPrice, tumataas/bumababa ang travelTime ng {b3:.4f} hours.")

lines.append("\n--- 5. R-squared and Adjusted R-squared ---")
lines.append(f"R-squared       = {model.rsquared:.4f}")
lines.append(f"Adjusted R2     = {model.rsquared_adj:.4f}")

lines.append("\n--- 6. What does R-squared tell you? ---")
lines.append(f"{model.rsquared*100:.1f}% ng variation sa travelTime ay nae-explain ng "
             f"kombinasyon ng milesTravelled, numDeliveries, at gasPrice.")

lines.append("\n--- 7. Overall model F-test significance ---")
f_pval = model.f_pvalue
if f_pval < 0.05:
    lines.append(f"Yes, overall model is statistically significant (F-test p-value = {f_pval:.4f} < 0.05).")
else:
    lines.append(f"No, overall model is not statistically significant (F-test p-value = {f_pval:.4f} > 0.05).")

output = "\n".join(lines)
print(output)

with open("multiple_regression_results.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")

print("\nSaved to: multiple_regression_results.txt")