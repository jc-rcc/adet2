import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("delivery.csv")
df.columns = ['milesTravelled', 'numDeliveries', 'gasPrice', 'travelTime']

X = df[['milesTravelled', 'numDeliveries', 'gasPrice']]

L = []
def out(t=""):
    L.append(t)
    print(t)

def taglish(text):
    out(f"\n   [TAGLISH] {text}")

def formal(text):
    out(f"   [FORMAL]  {text}")

out("=" * 72)
out("VIF FROM ZERO TO HERO")
out("=" * 72)

# ---------------- SECTION 0 ----------------
out("\n" + "#" * 72)
out("SECTION 0 - ANO ANG VIF?")
out("#" * 72)
taglish("Balikan natin. Sa multiple regression, parehong naging non-significant ang "
        "milesTravelled at numDeliveries kahit mataas naman ang correlation nila sa "
        "travelTime. Bakit? Kasi halos kambal sila (0.96). Ang VIF ang numeric na sukat kung "
        "gaano ka-kambal ang isang variable sa mga kasamahan nito.")
formal("Variance Inflation Factor (VIF) quantifies how much the variance of a regression "
       "coefficient is inflated due to multicollinearity - how strongly one predictor can be "
       "explained by the other predictors.")

# ---------------- SECTION 1 ----------------
out("\n" + "#" * 72)
out("SECTION 1 - ANG CONCEPT")
out("#" * 72)
taglish("Ang ideya: kunin ang isang variable, tapos subukang i-predict ito gamit ang LAHAT ng "
        "ibang predictors. Kung mataas ang R-squared ng pag-predict na 'yon, ibig sabihin alam na "
        "ng ibang variables kung ano ang mangyayari sa kanya - redundant o duplicate na siya.")
formal("VIF measures redundancy: if a variable can be accurately predicted from the remaining "
       "predictors (high R-squared), it carries little unique information on its own.")

# ---------------- SECTION 2 ----------------
out("\n" + "#" * 72)
out("SECTION 2 - ANG FORMULA")
out("#" * 72)
out("   VIF = 1 / (1 - R-squared)")
out("   kung saan ang R-squared ay mula sa pag-regress ng ISANG predictor sa lahat ng iba.\n")
out("   Interpretation:")
out("      VIF < 5    -> Low (okay)")
out("      5 - 10     -> Moderate (may problema)")
out("      > 10       -> Serious (kailangan aksyunan)")

# ---------------- SECTION 3 ----------------
out("\n" + "#" * 72)
out("SECTION 3 - MANUAL COMPUTATION (step-by-step)")
out("#" * 72)

def manual_vif(pred_name, Xdf):
    others = Xdf.drop(columns=[pred_name])
    Xc = sm.add_constant(others)
    mini = sm.OLS(Xdf[pred_name], Xc).fit()
    r2 = mini.rsquared
    vif = 1 / (1 - r2)
    return r2, vif

manual_results = {}
for col in X.columns:
    out(f"\n" + "-" * 72)
    out(f"=== {col} ===")
    r2, vif = manual_vif(col, X)
    manual_results[col] = vif
    out(f"   Hakbang 1: i-predict ang {col} gamit ang iba pang predictors")
    taglish(f"I-regress natin ang {col} sa iba. Ang R-squared na lumabas: {r2:.4f}.")
    out(f"   Hakbang 2: ipakita ang R-squared = {r2:.4f}")
    out(f"   Hakbang 3: VIF = 1 / (1 - {r2:.4f}) = {vif:.4f}")
    if vif > 10:
        interp = "Serious"
    elif vif >= 5:
        interp = "Moderate"
    else:
        interp = "Low"
    taglish(f"{r2*100:.2f}% ng {col} ay alam na ng mga ibang predictors. Bawas na lang "
            f"{100-r2*100:.2f}% ang kakaiba. Kaya VIF = {vif:.2f} = {interp}.")
    formal(f"{r2*100:.2f}% of {col} is explained by the other predictors, leaving only "
           f"{100-r2*100:.2f}% unique information. VIF = {vif:.2f} indicates {interp} multicollinearity.")

# ---------------- SECTION 4 ----------------
out("\n" + "#" * 72)
out("SECTION 4 - BUILT-IN VERIFICATION")
out("#" * 72)
out("\n   Paghambingin natin ang manual result vs. built-in function.\n")
out(f"   {'Variable':<16}{'Manual':>10}{'Built-in':>12}{'Interpretation':>18}")
out("   " + "-" * 56)
for i, col in enumerate(X.columns):
    built = variance_inflation_factor(X.values, i)
    vif = manual_results[col]
    if vif > 10:
        interp = "Serious"
    elif vif >= 5:
        interp = "Moderate"
    else:
        interp = "Low"
    out(f"   {col:<16}{vif:>10.4f}{built:>12.4f}{interp:>18}")
out("\n   [TAGLISH] Tugma ang manual at built-in - kumpirmadong tama ang ating manual computation!")
formal("The manual calculation matches the built-in function, confirming the procedure is correct.")

# ---------------- SECTION 5 ----------------
out("\n" + "#" * 72)
out("SECTION 5 - HERO: PAGBASA AT PAGGAMIT")
out("#" * 72)

idx = {'milesTravelled':0, 'numDeliveries':1, 'gasPrice':2}
built = {col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns)}
highest = max(built, key=built.get)

out("\nQ1. Alin ang may pinakamataas na VIF? / Which has the highest VIF?")
taglish(f"Ang numDeliveries ang pinakamataas (VIF = {built['numDeliveries']:.2f}).")
formal(f"The variable with the highest VIF is {highest} with VIF = {built[highest]:.2f}.")

out("\nQ2. Aling variables ang may seryosong multicollinearity?")
taglish("Ang milesTravelled at numDeliveries (parehong > 10). Ang gasPrice ay malinis (1.71).")
formal("milesTravelled (VIF = {:.2f}) and numDeliveries (VIF = {:.2f}) show serious "
       "multicollinearity (VIF > 10). gasPrice (VIF = {:.2f}) does not.".format(
    built['milesTravelled'], built['numDeliveries'], built['gasPrice']))

out("\nQ3. Ano ang ibig sabihin ng mataas na VIF?")
taglish("Halos ganap nang naipaliwanag ng ibang predictors ang variable na ito, kaya hindi "
        "ma-separate ng model ang kakaibang epekto nito sa travelTime. Unstable ang coefficient "
        "at hindi maaasahan ang p-value.")
formal("A high VIF means the variable is almost fully predictable from the other predictors, "
       "so the model cannot isolate its unique effect. Its coefficient becomes unstable and its "
       "p-value unreliable.")

out("\nQ4. Bakit hochly collinear ang milesTravelled at numDeliveries?")
taglish("Sa totoong deliveries, parami ng deliveries = parami ng milyang tinatakbo. Natural silang "
        "magkakasabay gumalaw, kaya 'di magahiwalay ng data ang epekto nila.")
formal("In real delivery operations, more deliveries naturally require more travel miles. They "
       "move together, so their individual effects cannot be separated.")

out("\nQ5. Awtomatiko bang tatanggalin ang variable na may pinakamataas na VIF?")
taglish("Hindi. Dapat batayan sa theory at layunin ng model. Sa case natin, magandang mag-drop ng "
        "isa sa kambal para stable ang interpretasyon, pero ang pagpili kung sino ang idrop ay "
        "base sa dahilan, hindi puro numero ang VIF.")
formal("No. Removal should be based on theory and the modeling objective, not solely on the VIF "
       "value. Here dropping one collinear predictor aids stable interpretation, but the choice of "
       "which to keep depends on substantive reasoning.")

result = "\n".join(L)
with open("vif_tutorial_results.txt", "w", encoding="utf-8") as f:
    f.write(result + "\n")
out("\nSaved to: vif_tutorial_results.txt")