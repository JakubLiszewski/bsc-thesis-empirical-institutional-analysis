# Quality of State Institutions and the Resource Curse
**Empirical analysis using economic growth regressions (1990–2020)**

This repository contains the Python implementation of my Bachelor's Thesis, originally defended at the SGH Warsaw School of Economics. The project empirically verifies the hypothesis concerning the role of institutional quality in shaping the relationship between natural resource abundance and economic growth.

## 📊 Project Overview
The "resource curse" is a paradoxical phenomenon where countries with abundant natural resources tend to experience slower economic growth than those with fewer natural resources. This analysis aims to determine whether the quality of state institutions acts as a conditional factor that dictates how resource wealth impacts long-term GDP growth.

**Key features of this analysis:**
* **Cross-sectional data analysis** for a sample of 49 countries over a 30-year horizon (1990–2020).
* **Multiple OLS regression models** incorporating interaction variables to test conditional dependencies.
* **Advanced econometric diagnostics**, including White's robust standard errors (HC1), the Delta method for marginal effects, and tests for multicollinearity (VIF) and heteroskedasticity (Breusch-Pagan).

## 🛠️ Tech Stack
* **Language:** Python 
* **Data Manipulation:** `pandas`, `numpy`
* **Statistical Modeling:** `statsmodels`, `scipy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Reporting:** `stargazer` (for LaTeX/text regression tables)

## 📈 Key Findings
1. **No Unconditional Curse:** The study did not find sufficient evidence for an unconditional negative relationship between resource abundance and economic growth in the modern economy (1990-2020).
2. **Institutions Matter:** High-quality state institutions (measured by the control of corruption index) have a significant, positive impact on long-term GDP dynamics.
3. **The Conditional Nature of the Curse:** The relationship between resource wealth and economic growth is highly dependent on institutional quality. Efficient, transparent institutions allow states to leverage resource rents for growth, effectively reversing the negative effects of the resource curse. 

## 📂 Data Sources
* **World Bank (WDI):** GDP per capita, natural resource rents, gross capital formation, trade openness, and school enrollment.
* **World Bank (WGI):** Control of Corruption index.
