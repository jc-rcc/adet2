import pandas as pd
from itertools import combinations

df = pd.read_csv("delivery.csv")
corr = df.corr(numeric_only=True)

lines = ["--- 6 Relationships ---"]
for i, (a, b) in enumerate(combinations(corr.columns, 2), 1):
    c = corr.loc[a, b]
    if abs(c) >= 0.7:
        strength = "Strong"
    elif abs(c) >= 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"
    lines.append(f"{i}. {a} vs {b}: {c:.2f} ({strength})")

output = "\n".join(lines)
print(output)

with open("relationships.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")

print("\nSaved to: relationships.txt")
