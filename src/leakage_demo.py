"""What the score would look like if the validation were done carelessly.

Two common mistakes on panel time series, done here on purpose:
1. Training on the future: to score quarter T, train on every other quarter,
   including ones after T (this is what shuffled K-fold CV does).
2. Target leakage through an aggregate: the state's typical death level is
   computed from ALL rows, including the very rows being scored.

To keep the comparison fair, every variant is scored on exactly the same
rows as the honest backtest (all 37 states, the last 6 target quarters),
with the same XGBoost settings. Writes results/leakage_demo.csv.
"""

import numpy as np
import pandas as pd

from backtest import DATA, N_TEST_QUARTERS, RESULTS, make_models
from features import FEATURES, add_state_level, build_feature_table


def xgb():
    return make_models()["XGBoost (Poisson)"]


def score(feats, test_quarters, train_filter, leak_level):
    errs = []
    full_level = feats.groupby("state").y.mean()
    for T in test_quarters:
        train = feats[train_filter(feats.target_qi, T)]
        test = feats[feats.target_qi == T]
        if leak_level:
            train = train.assign(state_level=train.state.map(full_level))
            test = test.assign(state_level=test.state.map(full_level))
        else:
            train, test = add_state_level(train, test)
        m = xgb().fit(train[FEATURES], train.y)
        errs.append(np.abs(m.predict(test[FEATURES]) - test.y))
    return np.concatenate(errs).mean()


def main():
    feats = build_feature_table(pd.read_csv(DATA))
    test_quarters = sorted(feats.target_qi.unique())[-N_TEST_QUARTERS:]
    past_only = lambda qi, T: qi < T  # noqa: E731
    all_but_T = lambda qi, T: qi != T  # noqa: E731

    rows = [
        ("Honest: train on past quarters only", score(feats, test_quarters, past_only, False)),
        ("Leak 1: also train on future quarters", score(feats, test_quarters, all_but_T, False)),
        ("Leak 2: past only, state level from all rows", score(feats, test_quarters, past_only, True)),
        ("Leaks 1 + 2 together", score(feats, test_quarters, all_but_T, True)),
    ]
    out = pd.DataFrame(rows, columns=["validation", "MAE"])
    honest = out.MAE.iloc[0]
    out["vs honest"] = ((out.MAE / honest - 1) * 100).round(1).astype(str) + "%"
    out["MAE"] = out.MAE.round(2)
    out.to_csv(RESULTS / "leakage_demo.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
