# Empirical Institutional Analysis & The Resource Curse

This repository contains the complete econometric data analysis pipeline developed for my bachelor's thesis (Szkoła Główna Handlowa w Warszawie - SGH). The project investigates the conditional impact of natural resource rents on long-term economic growth, emphasizing the moderating role of institutional quality across a cross-country sample.

## Project Structure

* **`bsc_thesis_analysis.py`**: Main Python script executing data cleaning, transformation, OLS regressions (models M1–M4), robustness checks, econometric diagnostics, and marginal effects calculations.
* **`*.csv`**: Raw macroeconomic and institutional datasets (World Bank, WGI indicators, 1990–2020).

## Methodology & Econometric Models

1. **Conditional Convergence Model**: Ordinary Least Squares (OLS) estimation based on an augmented Solow-Swan growth framework.
2. **Interaction Analysis**: Investigating whether high institutional quality mitigates the "resource curse" associated with natural resource rents.
3. **Robustness Checks**: 
   - Base-year specifications ($1990/1996$).
   - Extended sample size ($N = 124$).
   - Shortened period comparison ($1995–2020$).
4. **Diagnostics**: Heteroscedasticity-robust standard errors (HC1), Jarque-Bera normality tests, Ramsey RESET tests, and Variance Inflation Factor (VIF) analysis for multicollinearity.

## Tech Stack

* **Language**: Python 3.13
* **Libraries**: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `seaborn`

## Usage

Ensure all CSV data files and `bsc_thesis_analysis.py` are in the same working directory, then run:

```bash
python bsc_thesis_analysis.py