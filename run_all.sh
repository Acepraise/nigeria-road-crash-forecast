#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/prepare_data.py
(cd src && python backtest.py && python leakage_demo.py && python make_charts.py)
python -m pytest -q tests
