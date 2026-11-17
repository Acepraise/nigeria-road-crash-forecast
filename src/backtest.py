"""Rolling-origin backtest: baselines vs Ridge vs Poisson GLM vs XGBoost.

For each test quarter T (the last N target quarters), every model is trained
only on rows whose target quarter is strictly before T, then predicts all 37
states for T. This mimics using the model for real: you never get to train on
the future.

Hyperparameters are fixed up front (not tuned on the test quarters), so the
reported scores are not optimistically biased by model selection.

Outputs (in results/):
- predictions.csv      every prediction, every model, every fold
- metrics.csv          MAE / RMSE / bias per model
- fold_metrics.csv     MAE per model per test quarter
- comparison.md        paired comparisons (baselines vs best learned model)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

from features import FEATURES, add_state_level, build_feature_table

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crashes_state_quarter.csv"
RESULTS = ROOT / "results"
N_TEST_QUARTERS = 6
SEED = 42

COUNT_FEATURES = [f for f in FEATURES if not f.startswith("is_q") and f != "killed_per_fatal_crash_lag1"]
OTHER_FEATURES = [f for f in FEATURES if f not in COUNT_FEATURES]


def linear_preprocessor():
    # log1p tames the heavy right tail (FCT and Kaduna dwarf Bayelsa);
    # imputer and scaler are fit inside the pipeline, so on training rows only.
    return ColumnTransformer(
        [
            (
                "counts",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                COUNT_FEATURES,
            ),
            (
                "other",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                OTHER_FEATURES,
            ),
        ]
    )


class LogRidge:
    """Ridge on log1p(deaths), predictions mapped back to counts."""

    def __init__(self):
        self.model = Pipeline(
            [("prep", linear_preprocessor()), ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30)))]
        )

    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.model.predict(X)), 0, None)


def make_models():
    return {
        "Poisson GLM": Pipeline(
            [("prep", linear_preprocessor()), ("glm", PoissonRegressor(alpha=1.0, max_iter=1000))]
        ),
        "Ridge (log)": LogRidge(),
        "XGBoost (Poisson)": XGBRegressor(
            objective="count:poisson",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=1,
        ),
    }


BASELINES = {
    "Naive (last quarter)": lambda X: X["killed_lag1"].to_numpy(),
    "Seasonal naive (same quarter last year)": lambda X: X["killed_lag4"].to_numpy(),
    "Mean of last 4 quarters": lambda X: X["killed_mean4"].to_numpy(),
}


def run():
    panel = pd.read_csv(DATA)
    feats = build_feature_table(panel)
    target_quarters = sorted(feats.target_qi.unique())
    test_quarters = target_quarters[-N_TEST_QUARTERS:]

    preds = []
    for T in test_quarters:
        train = feats[feats.target_qi < T]
        test = feats[feats.target_qi == T]
        # Guard against leakage: no training target may be at or after T.
        assert train.target_qi.max() < T
        train, test = add_state_level(train, test)

        base = test[["state", "target_quarter", "y"]].copy()
        for name, fn in BASELINES.items():
            preds.append(base.assign(model=name, pred=fn(test)))
        for name, model in make_models().items():
            model.fit(train[FEATURES], train["y"])
            preds.append(base.assign(model=name, pred=model.predict(test[FEATURES])))

    preds = pd.concat(preds, ignore_index=True)
    preds["abs_err"] = (preds.pred - preds.y).abs()
    preds["sq_err"] = (preds.pred - preds.y) ** 2
    preds["err"] = preds.pred - preds.y

    metrics = (
        preds.groupby("model")
        .agg(MAE=("abs_err", "mean"), RMSE=("sq_err", lambda s: np.sqrt(s.mean())), bias=("err", "mean"))
        .sort_values("MAE")
        .round(2)
    )
    naive_mae = metrics.loc["Naive (last quarter)", "MAE"]
    metrics["MAE vs naive"] = ((metrics.MAE / naive_mae - 1) * 100).round(1).astype(str) + "%"

    fold = preds.pivot_table(index="target_quarter", columns="model", values="abs_err", aggfunc="mean").round(2)

    # Paired comparisons on the same state-quarters, with a bootstrap over
    # states (errors within one state are correlated across quarters).
    def paired(model_a, model_b):
        a = preds[preds.model == model_a].set_index(["state", "target_quarter"]).abs_err
        b = preds[preds.model == model_b].set_index(["state", "target_quarter"]).abs_err
        diff = (a - b.loc[a.index]).reset_index(name="d")
        rng = np.random.default_rng(SEED)
        states = diff.state.unique()
        by_state = diff.groupby("state").d.apply(np.array)
        boots = [
            np.concatenate([by_state[s] for s in rng.choice(states, size=len(states), replace=True)]).mean()
            for _ in range(5000)
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return (
            f"- **{model_a}** vs **{model_b}**: mean change in absolute error "
            f"{diff.d.mean():+.2f} deaths per state-quarter (95% state-bootstrap interval "
            f"{lo:+.2f} to {hi:+.2f}); {model_a} was closer on {(diff.d < 0).mean():.0%} "
            f"of {len(diff)} state-quarters.\n"
        )

    best_ml = [m for m in metrics.index if m not in BASELINES][0]
    comparison = (
        f"Lowest MAE overall: **{metrics.index[0]}**. Best learned model: **{best_ml}**.\n\n"
        + paired("Mean of last 4 quarters", "Naive (last quarter)")
        + paired(best_ml, "Naive (last quarter)")
        + paired(best_ml, "Mean of last 4 quarters")
    )

    RESULTS.mkdir(exist_ok=True)
    preds.to_csv(RESULTS / "predictions.csv", index=False)
    metrics.to_csv(RESULTS / "metrics.csv")
    fold.to_csv(RESULTS / "fold_metrics.csv")
    (RESULTS / "comparison.md").write_text(comparison)

    print(f"Training rows available: {len(feats)}; test quarters: "
          f"{feats[feats.target_qi.isin(test_quarters)].target_quarter.unique().tolist()}")
    print(metrics.to_string())
    print()
    print(fold.to_string())
    print()
    print(comparison)
    return preds, metrics, fold


if __name__ == "__main__":
    run()
