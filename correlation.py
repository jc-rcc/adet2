import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("delivery.csv")
corr_matrix = df.corr(numeric_only=True)
print("--- Correlation Matrix (Numbers) ---")
print(corr_matrix)
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()