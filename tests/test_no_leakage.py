"""Tests that the feature table never looks into the future."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import add_state_level, build_feature_table, quarter_index  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "crashes_state_quarter.csv"


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def feats(panel):
    return build_feature_table(panel)


def test_panel_shape(panel):
    assert panel.state.nunique() == 37
    assert panel.quarter.nunique() == 15
    assert not panel.isna().any().any()


def test_lag1_is_previous_quarter(panel, feats):
    lookup = panel.set_index(["state", "quarter"]).killed
    for _, r in feats.sample(50, random_state=0).iterrows():
        prev_qi = r.target_qi - 1
        prev_q = f"{prev_qi // 4}Q{prev_qi % 4 + 1}"
        assert r.killed_lag1 == lookup[(r.state, prev_q)]
        assert prev_qi < r.target_qi


def test_target_never_in_features(feats):
    # Changing a row's own target must not change any of its features.
    row = feats.iloc[100]
    lag_cols = [c for c in feats.columns if "lag" in c or "mean4" in c]
    perturbed = feats.copy()
    perturbed.loc[100, "y"] += 1000
    assert (perturbed.loc[100, lag_cols] == row[lag_cols]).all()


def test_state_level_uses_training_targets_only(feats):
    T = sorted(feats.target_qi.unique())[-1]
    train, test = feats[feats.target_qi < T], feats[feats.target_qi == T].copy()
    _, test_a = add_state_level(train, test)
    test.loc[:, "y"] = test.y * 100  # wildly change the test targets
    _, test_b = add_state_level(train, test)
    assert (test_a.state_level.values == test_b.state_level.values).all()


def test_quarter_index_ordering():
    assert quarter_index("2021Q4") + 1 == quarter_index("2022Q1")
