import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

y = df['travelTime']
predictors = ['milesTravelled', 'numDeliveries', 'gasPrice']

results = []

for pred in predictors:
    X = sm.add_constant(df[pred])
    model = sm.OLS(y, X).fit()

    intercept = model.params['const']
    slope = model.params[pred]
    r_squared = model.rsquared
    p_value = model.pvalues[pred]
    significant = "Yes" if p_value < 0.05 else "No"

    results.append({
        'Predictor': pred,
        'Intercept': round(intercept, 4),
        'Slope': round(slope, 4),
        'R²': round(r_squared, 4),
        'p-value': round(p_value, 4),
        'Significant?': significant
    })

results_df = pd.DataFrame(results)
lines = []
lines.append("\n" + "="*70)
lines.append("SIMPLE LINEAR REGRESSION RESULTS")
lines.append("="*70)
lines.append(results_df.to_string(index=False))

lines.append("\n" + "="*70)
lines.append("INTERPRETATION OF STRONGEST MODEL (milesTravelled)")
lines.append("="*70)

strongest = results_df.loc[results_df['R²'].idxmax()]
lines.append(f"\n1. Regression Equation:")
lines.append(f"   travelTime = {strongest['Intercept']} + {strongest['Slope']} * milesTravelled")

lines.append(f"\n2. Slope Interpretation:")
lines.append(f"   Kada dagdag ng 1 mile sa milesTravelled, tumataas ang travelTime ng {strongest['Slope']} hours.")

lines.append(f"\n3. R-squared Interpretation:")
lines.append(f"   {strongest['R²']*100:.1f}% ng variation sa travelTime ay nae-explain ng milesTravelled.")

lines.append(f"\n4. Statistical Significance:")
if strongest['Significant?'] == 'Yes':
    lines.append(f"   Yes, statistically significant at alpha = 0.05 (p-value = {strongest['p-value']} < 0.05).")
else:
    lines.append(f"   No, not statistically significant at alpha = 0.05 (p-value = {strongest['p-value']} > 0.05).")

output = "\n".join(lines)
print(output)

with open("simple_regression_results.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")

print("\nSaved to: simple_regression_results.txt")
