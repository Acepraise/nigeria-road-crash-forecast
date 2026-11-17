"""Turn the raw FRSC state-by-quarter sheet export into a tidy panel.

Input:  data_raw/aggregate_values.json  (Google Sheets values export of the
        "Aggregate" tab, rows 3 to 41)
Output: data/crashes_state_quarter.csv  (one row per state per quarter)
        data/data_quality_report.md     (every issue found and how it was handled)

The sheet lays out 19 side-by-side blocks. The first 15 are quarters
(Q4 2020 to Q2 2024). The last 4 are yearly totals (2021, 2022, 2023 and
H1 2024); the script proves this by checking that each one equals the sum of
its quarters, then drops them so no quarter is counted twice.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "aggregate_values.json"
OUT = ROOT / "data" / "crashes_state_quarter.csv"
REPORT = ROOT / "data" / "data_quality_report.md"

QUARTERS = (
    ["2020Q4"]
    + [f"{y}Q{q}" for y in (2021, 2022, 2023) for q in (1, 2, 3, 4)]
    + ["2024Q1", "2024Q2"]
)
FIELDS = [
    "fatal_crashes",
    "serious_crashes",
    "minor_crashes",
    "total_crashes",
    "injured",
    "killed",
    "total_casualties",
    "people_involved",
]
ANNUAL_BLOCKS = {15: QUARTERS[1:5], 16: QUARTERS[5:9], 17: QUARTERS[9:13], 18: QUARTERS[13:15]}


def to_num(cell):
    s = (cell or "").replace(",", "").strip()
    return float(s) if s else float("nan")


def load_blocks():
    payload = json.loads(RAW.read_text())
    values = payload.get("values") or payload["content"]["values"]
    header = values[0]
    starts = [i for i, x in enumerate(header) if x == "STATE"]
    assert len(starts) == 19, f"expected 19 blocks, found {len(starts)}"

    rows = []
    for raw in values[1:]:
        state = raw[0].strip()
        for b, start in enumerate(starts):
            cells = raw[start + 1 : start + 1 + len(FIELDS)]
            cells += [""] * (len(FIELDS) - len(cells))
            rec = {"state": state, "block": b}
            rec.update({f: to_num(c) for f, c in zip(FIELDS, cells)})
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = load_blocks()
    notes = []

    # 1. Prove the last four blocks are yearly totals, then drop them.
    quarterly = df[df.block < 15].copy()
    quarterly["quarter"] = quarterly.block.map(dict(enumerate(QUARTERS)))
    for b, qs in ANNUAL_BLOCKS.items():
        annual = df[(df.block == b) & (df.state == "TOTAL")].total_crashes.iloc[0]
        summed = quarterly[(quarterly.quarter.isin(qs)) & (quarterly.state == "TOTAL")].total_crashes.sum()
        assert annual == summed, (b, annual, summed)
    notes.append(
        "Blocks 16 to 19 of the sheet are yearly totals (2021, 2022, 2023, H1 2024). "
        "Each matches the sum of its quarters exactly, so they were dropped."
    )

    # 2. Separate the national TOTAL row; keep it only for a cross-check.
    national = quarterly[quarterly.state == "TOTAL"].set_index("quarter")
    panel = quarterly[quarterly.state != "TOTAL"].drop(columns="block")
    assert panel.state.nunique() == 37, panel.state.nunique()

    # 3. Missing values.
    missing = panel[panel[FIELDS].isna().any(axis=1)]
    for _, r in missing.iterrows():
        gaps = [f for f in FIELDS if pd.isna(r[f])]
        notes.append(f"{r.state} {r.quarter}: blank {', '.join(gaps)} in the source.")

    # Borno 2021Q3 is blank for killed and total_casualties. Killed can be
    # recovered from the national total: national killed minus every other state.
    for _, r in missing.iterrows():
        if pd.isna(r.killed):
            others = panel[(panel.quarter == r.quarter) & (panel.state != r.state)].killed.sum()
            recovered = national.loc[r.quarter, "killed"] - others
            mask = (panel.state == r.state) & (panel.quarter == r.quarter)
            panel.loc[mask, "killed"] = recovered
            panel.loc[mask, "total_casualties"] = r.injured + recovered
            notes.append(
                f"  Recovered {r.state} {r.quarter} killed = {recovered:.0f} from the national "
                "total minus all other states."
            )

    # 4. Internal consistency checks (reported, not silently changed).
    bad_cases = panel[
        panel.total_crashes != panel.fatal_crashes + panel.serious_crashes + panel.minor_crashes
    ]
    for _, r in bad_cases.iterrows():
        notes.append(
            f"{r.state} {r.quarter}: total crashes {r.total_crashes:.0f} does not equal "
            f"fatal + serious + minor ({r.fatal_crashes + r.serious_crashes + r.minor_crashes:.0f})."
        )
    bad_cas = panel[panel.total_casualties != panel.injured + panel.killed]
    for _, r in bad_cas.iterrows():
        notes.append(
            f"{r.state} {r.quarter}: total casualties {r.total_casualties:.0f} does not equal "
            f"injured + killed ({r.injured + r.killed:.0f}). Casualties are recomputed as "
            "injured + killed; the model does not use this column."
        )
    panel["total_casualties"] = panel.injured + panel.killed

    # 5. Cross-check: states add up to the national row for killed.
    for q in QUARTERS:
        s = panel[panel.quarter == q].killed.sum()
        n = national.loc[q, "killed"]
        if s != n:
            notes.append(f"{q}: states sum to {s:.0f} killed but the national row says {n:.0f}.")

    panel = panel.sort_values(["state", "quarter"]).reset_index(drop=True)
    for f in FIELDS:
        panel[f] = panel[f].astype(int)

    OUT.parent.mkdir(exist_ok=True)
    panel.to_csv(OUT, index=False)

    lines = [
        "# Data quality report",
        "",
        f"Source: Federal Road Safety Corps (FRSC) state crash returns, {QUARTERS[0]} to {QUARTERS[-1]}.",
        f"Rows written: {len(panel)} ({panel.state.nunique()} states including FCT x {len(QUARTERS)} quarters).",
        "",
        "## Issues found",
        "",
    ] + [f"- {n}" if not n.startswith("  ") else f"  - {n.strip()}" for n in notes]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
