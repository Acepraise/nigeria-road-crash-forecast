"""Feature engineering for next-quarter road deaths by state.

Every row answers one question: standing at the end of quarter t, how many
people will be killed on this state's roads in quarter t+1?

Leakage rules this module enforces:
- Every feature for target quarter T is computed only from quarters <= T-1.
- The only thing known about T itself is its calendar quarter (Q1 to Q4),
  which is known in advance.
- Anything that summarises a state across time (the state's typical death
  level) is NOT computed here, because it would see the test period. It is
  computed inside each backtest fold from training rows only; see
  `add_state_level`.
"""

import numpy as np
import pandas as pd

LAG_SOURCES = ["killed", "fatal_crashes", "total_crashes", "injured", "people_involved"]


def quarter_index(q: str) -> int:
    """'2021Q3' -> a running integer so quarters can be shifted safely."""
    return int(q[:4]) * 4 + int(q[-1]) - 1


def build_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["qi"] = df["quarter"].map(quarter_index)
    df = df.sort_values(["state", "qi"]).reset_index(drop=True)

    out = []
    for state, g in df.groupby("state", sort=True):
        g = g.set_index("qi")
        for T in g.index:
            prev = [T - k for k in (1, 2, 3, 4)]
            if not all(p in g.index for p in prev):
                continue  # need four full quarters of history
            row = {
                "state": state,
                "target_qi": T,
                "target_quarter": g.loc[T, "quarter"],
                "target_q_of_year": T % 4 + 1,
                "y": g.loc[T, "killed"],
            }
            for col in LAG_SOURCES:
                row[f"{col}_lag1"] = g.loc[T - 1, col]
            row["killed_lag2"] = g.loc[T - 2, "killed"]
            # T-4 is the same calendar quarter one year earlier.
            row["killed_lag4"] = g.loc[T - 4, "killed"]
            row["killed_mean4"] = np.mean([g.loc[p, "killed"] for p in prev])
            row["killed_per_fatal_crash_lag1"] = (
                g.loc[T - 1, "killed"] / g.loc[T - 1, "fatal_crashes"]
                if g.loc[T - 1, "fatal_crashes"] > 0
                else np.nan
            )
            out.append(row)

    feats = pd.DataFrame(out)
    # Calendar-quarter dummies for the target quarter (known in advance).
    for q in (2, 3, 4):
        feats[f"is_q{q}"] = (feats.target_q_of_year == q).astype(int)
    return feats.sort_values(["target_qi", "state"]).reset_index(drop=True)


def add_state_level(train: pd.DataFrame, test: pd.DataFrame):
    """State's typical quarterly deaths, learned from TRAINING targets only.

    A state never seen in training (does not happen here, but the code guards
    it) falls back to the training median.
    """
    level = train.groupby("state")["y"].mean()
    fallback = train["y"].median()
    train = train.assign(state_level=train["state"].map(level))
    test = test.assign(state_level=test["state"].map(level).fillna(fallback))
    return train, test


FEATURES = [
    "killed_lag1",
    "killed_lag2",
    "killed_lag4",
    "killed_mean4",
    "fatal_crashes_lag1",
    "total_crashes_lag1",
    "injured_lag1",
    "people_involved_lag1",
    "killed_per_fatal_crash_lag1",
    "state_level",
    "is_q2",
    "is_q3",
    "is_q4",
]
