import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset as reset_test
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns

# --- AUTOMATIC WORKING DIRECTORY SETUP ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 1. ENVIRONMENT SETUP ---
print("--- 1. ENVIRONMENT & DATA PREPARATION ---")
np.set_printoptions(suppress=True, precision=4)

# --- 2. DATA IMPORT ---
y_growth_raw = pd.read_csv("Y_1990-2020.csv", sep=";", decimal=",", na_values="..")
rents_raw = pd.read_csv("Rents_1990-2020.csv", sep=";", decimal=",", na_values="..")
inst_raw = pd.read_csv("Inst_1996-2020.csv", sep=";", decimal=",", na_values="..")
y_1990_raw = pd.read_csv("Y_const2015_1990.csv", sep=";", decimal=",", na_values="..")
invest_raw = pd.read_csv("Invest_1990-2020.csv", sep=";", decimal=",", na_values="..")
trade_raw = pd.read_csv("Trade_1990-2020.csv", sep=";", decimal=",", na_values="..")
educ_raw = pd.read_csv("Educ_1990-2020.csv", sep=";", decimal=",", na_values="..")


# --- 3. DATA TRANSFORMATION & CLEANING ---
def clean_average(df, start_year, end_year=2020, max_nas=8):
    year_cols = [c for c in df.columns if c.isdigit() and start_year <= int(c) <= end_year]
    valid_mask = df[year_cols].isna().sum(axis=1) <= max_nas
    res = df[valid_mask][['Country Name']].copy()
    res['avg_val'] = df[year_cols].mean(axis=1)
    return res


GROWTH = clean_average(y_growth_raw, 1990).rename(columns={'avg_val': 'GROWTH'})
RENTS = clean_average(rents_raw, 1990).rename(columns={'avg_val': 'RENTS'})

RENTS_b = rents_raw[['Country Name', '1990']].dropna(subset=['1990']).copy()
RENTS_b['1990'] = pd.to_numeric(RENTS_b['1990'], errors='coerce')
RENTS_b.rename(columns={'1990': 'RENTS_b'}, inplace=True)

INST = clean_average(inst_raw, 1996, max_nas=4).rename(columns={'avg_val': 'INST'})

INST_b = inst_raw[['Country Name', '1996']].dropna(subset=['1996']).copy()
INST_b['1996'] = pd.to_numeric(INST_b['1996'], errors='coerce')
INST_b.rename(columns={'1996': 'INST_b'}, inplace=True)

ln_GDP1990 = y_1990_raw[['Country Name', '1990']].copy()
ln_GDP1990['temp_y'] = pd.to_numeric(ln_GDP1990['1990'], errors='coerce')
ln_GDP1990 = ln_GDP1990.dropna(subset=['temp_y'])
ln_GDP1990['ln_GDP1990'] = np.log(ln_GDP1990['temp_y'])
ln_GDP1990 = ln_GDP1990[['Country Name', 'ln_GDP1990']]

inv_avg = clean_average(invest_raw, 1990)
inv_avg['ln_INV'] = np.log(inv_avg['avg_val'])
ln_INV = inv_avg[['Country Name', 'ln_INV']]

inv_b = invest_raw.copy()
inv_b['temp_1990'] = pd.to_numeric(inv_b['1990'], errors='coerce')
inv_b = inv_b.dropna(subset=['temp_1990'])
inv_b['ln_INV_b'] = np.log(inv_b['temp_1990'])
ln_INV_b = inv_b[['Country Name', 'ln_INV_b']]

trade_avg = clean_average(trade_raw, 1990)
trade_avg['ln_TRADE'] = np.log(trade_avg['avg_val'])
ln_TRADE = trade_avg[['Country Name', 'ln_TRADE']]

trade_b = trade_raw.copy()
trade_b['Trade_1990'] = pd.to_numeric(trade_b['1990'], errors='coerce')
trade_b = trade_b.dropna(subset=['Trade_1990'])
trade_b['ln_TRADE_b'] = np.log(trade_b['Trade_1990'])
ln_TRADE_b = trade_b[['Country Name', 'ln_TRADE_b']]

EDUC = clean_average(educ_raw, 1990, max_nas=15).rename(columns={'avg_val': 'EDUC'})

EDUC_b = educ_raw[['Country Name', '1990']].dropna(subset=['1990']).copy()
EDUC_b['1990'] = pd.to_numeric(EDUC_b['1990'], errors='coerce')
EDUC_b.rename(columns={'1990': 'EDUC_b'}, inplace=True)


# --- 4. DESCRIPTIVE STATISTICS ---
print("\n--- 4. DESCRIPTIVE STATISTICS ---")
desc_vars = [GROWTH['GROWTH'], RENTS['RENTS'], INST['INST'], ln_GDP1990['ln_GDP1990']]
desc_names = ["GROWTH", "RENTS", "INST", "ln_GDP1990"]
for name, var in zip(desc_names, desc_vars):
    print(f"Variable: {name} | Mean: {var.mean():.3f} | SD: {var.std():.3f}")


# --- 5. MERGING DATA & SAMPLE STABILIZATION (N=49) ---
print("\n--- 5. MERGING AND STABLE SAMPLE SETUP ---")
dfs = [GROWTH, ln_GDP1990, RENTS, RENTS_b, INST, INST_b, ln_INV, ln_INV_b, ln_TRADE, ln_TRADE_b, EDUC, EDUC_b]
final_df = dfs[0]
for d in dfs[1:]:
    final_df = pd.merge(final_df, d, on='Country Name', how='inner')

final_df = final_df.dropna()
final_df['Interaction'] = final_df['RENTS'] * final_df['INST']
final_df['Interaction_b'] = final_df['RENTS_b'] * final_df['INST_b']
print(f"Final sample size (N): {len(final_df)}")


# --- 6. OLS REGRESSION MODELS (M1 - M4) ---
print("\n--- 6. OLS REGRESSION MODELS ---")
m1 = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + EDUC', data=final_df).fit()
m2 = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + EDUC + RENTS', data=final_df).fit()
m3 = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + EDUC + RENTS + INST', data=final_df).fit()
m4 = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + EDUC + RENTS + INST + Interaction', data=final_df).fit()

print("\n>>> MODEL 1 (Base Model) Summary:")
print(m1.summary())

print("\n>>> MODEL 2 (+ Rents) Summary:")
print(m2.summary())

print("\n>>> MODEL 3 (+ Inst) Summary:")
print(m3.summary())

print("\n>>> MODEL 4 (Full Model with Interaction) Summary:")
print(m4.summary())


# --- 7. ROBUSTNESS & SENSITIVITY ANALYSIS ---
print("\n--- 7. ROBUSTNESS & SENSITIVITY ANALYSIS ---")

# Base year model
m4_base = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV_b + ln_TRADE_b + EDUC_b + RENTS_b + INST_b + Interaction_b', data=final_df).fit()
print("\n>>> Robustness: Base Year (1990/1996) Model Summary:")
print(m4_base.summary())

# Larger sample model (N=124, without EDUC)
dfs_large = [GROWTH, ln_GDP1990, RENTS, RENTS_b, INST, INST_b, ln_INV, ln_INV_b, ln_TRADE, ln_TRADE_b]
final_df_large = dfs_large[0]
for d in dfs_large[1:]:
    final_df_large = pd.merge(final_df_large, d, on='Country Name', how='inner')
final_df_large = final_df_large.dropna()
final_df_large['Interaction'] = final_df_large['RENTS'] * final_df_large['INST']

m_large = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + RENTS + INST + Interaction', data=final_df_large).fit()
print(f"\n>>> Robustness: Larger Sample Size (N = {len(final_df_large)}) Model Summary:")
print(m_large.summary())


# --- 8. ECONOMETRIC DIAGNOSTICS ---
print("\n--- 8. ECONOMETRIC DIAGNOSTICS ---")
reset_p = reset_test(m4, power=3).pvalue
print(f"RESET Test p-value: {reset_p:.4f}")

jb_stat, jb_p, _, _ = jarque_bera(m4.resid)
print(f"Jarque-Bera Test p-value: {jb_p:.4f}")

exog_vars = m4.model.exog
vif_data = pd.DataFrame()
vif_data["Feature"] = m4.model.exog_names
vif_data["VIF"] = [variance_inflation_factor(exog_vars, i) for i in range(exog_vars.shape[1])]
print("\nVIF Results:")
print(vif_data)

bp_stat, bp_p, _, _ = het_breuschpagan(m4.resid, m4.model.exog)
print(f"Breusch-Pagan Test p-value: {bp_p:.4f}")


# --- 9. ADVANCED DIAGNOSTICS (HC1 & NON-LINEARITY) ---
print("\n--- 9. HC1 ROBUST STANDARD ERRORS & NON-LINEARITY ---")
m4_hc1 = smf.ols('GROWTH ~ ln_GDP1990 + ln_INV + ln_TRADE + EDUC + RENTS + INST + Interaction', data=final_df).fit(cov_type='HC1')
print(m4_hc1.summary())


# --- 10. MARGINAL EFFECTS ANALYSIS ---
print("\n--- 10. MARGINAL EFFECTS CALCULATIONS ---")
b_rents = m4.params['RENTS']
b_inter = m4.params['Interaction']

for inst_val in [-2.5, 0.0, 2.5]:
    marginal_effect = b_rents + b_inter * inst_val
    print(f"Marginal effect of RENTS at INST = {inst_val}: {marginal_effect:.4f}")


# --- 11. SHORT PERIOD MODEL (1995-2020) ---
print("\n--- 11. SHORT PERIOD MODEL (1995-2020) ---")
GROWTH_95 = clean_average(y_growth_raw, 1995).rename(columns={'avg_val': 'GROWTH_95'})
RENTS_95 = clean_average(rents_raw, 1995).rename(columns={'avg_val': 'RENTS_95'})
ln_INV_95 = clean_average(invest_raw, 1995)
ln_INV_95['ln_INV_95'] = np.log(ln_INV_95['avg_val'])
ln_INV_95 = ln_INV_95[['Country Name', 'ln_INV_95']]
ln_TRADE_95 = clean_average(trade_raw, 1995)
ln_TRADE_95['ln_TRADE_95'] = np.log(ln_TRADE_95['avg_val'])
ln_TRADE_95 = ln_TRADE_95[['Country Name', 'ln_TRADE_95']]
EDUC_95 = clean_average(educ_raw, 1995, max_nas=15).rename(columns={'avg_val': 'EDUC_95'})

dfs_95 = [GROWTH_95, ln_GDP1990, RENTS_95, INST, ln_INV_95, ln_TRADE_95, EDUC_95]
final_df_95 = dfs_95[0]
for d in dfs_95[1:]:
    final_df_95 = pd.merge(final_df_95, d, on='Country Name', how='inner')
final_df_95 = final_df_95.dropna()
final_df_95['Interaction_95'] = final_df_95['RENTS_95'] * final_df_95['INST']

m4_short = smf.ols('GROWTH_95 ~ ln_GDP1990 + ln_INV_95 + ln_TRADE_95 + EDUC_95 + RENTS_95 + INST + Interaction_95', data=final_df_95).fit()

print(m4_short.summary())
print("Full pipeline execution completed successfully.")