Lowest MAE overall: **Mean of last 4 quarters**. Best learned model: **XGBoost (Poisson)**.

- **Mean of last 4 quarters** vs **Naive (last quarter)**: mean change in absolute error -3.23 deaths per state-quarter (95% state-bootstrap interval -4.33 to -2.18); Mean of last 4 quarters was closer on 61% of 222 state-quarters.
- **XGBoost (Poisson)** vs **Naive (last quarter)**: mean change in absolute error -3.01 deaths per state-quarter (95% state-bootstrap interval -4.12 to -1.91); XGBoost (Poisson) was closer on 62% of 222 state-quarters.
- **XGBoost (Poisson)** vs **Mean of last 4 quarters**: mean change in absolute error +0.22 deaths per state-quarter (95% state-bootstrap interval -0.53 to +0.95); XGBoost (Poisson) was closer on 43% of 222 state-quarters.
