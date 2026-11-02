## Summary

This project analyses 8 years (2019–2026) of daily Sri Lankan port activity data from IMF PortWatch, covering the ports of Colombo, Hambantota, Trincomalee and Galle (10,824 daily records). It follows a complete data science workflow: data cleaning, feature engineering, exploratory analysis, hypothesis testing and machine learning.

**Key findings**
- Colombo handles about 82% of all imports.
- Weekday imports are significantly higher than weekend imports (Welch t-test, p = 0.017), but there is no significant difference across quarters (ANOVA, p = 0.88).
- Linear Regression (R² = 0.619) and Random Forest (R² = 0.609) perform similarly. Random Forest has the lower MAE (8,982 vs 9,673).
- The number of daily port calls is by far the strongest predictor of import volume.

**Tools:** Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn.
