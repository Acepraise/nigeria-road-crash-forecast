"""Charts for the README. Reads results/, writes figures/*.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def finish(fig, ax, name, note):
    ax.grid(axis="y" if ax.get_ylabel() else "x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)


def model_mae():
    m = pd.read_csv(RESULTS / "metrics.csv", index_col=0).sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(m.index, m.MAE, color=BLUE, height=0.6)
    for b, v in zip(bars, m.MAE):
        ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=INK_2)
    ax.set_xlabel("Mean absolute error (deaths per state per quarter, lower is better)")
    ax.set_title("A 4-quarter average forecasts as well as XGBoost")
    ax.set_xlim(0, m.MAE.max() * 1.12)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    finish(fig, ax, "model_mae.png", "Rolling backtest, 37 states x 6 test quarters (2023Q1 to 2024Q2). Source: FRSC.")


def mae_by_quarter():
    f = pd.read_csv(RESULTS / "fold_metrics.csv", index_col=0)
    series = [
        ("Naive (last quarter)", BLUE, "Naive"),
        ("Mean of last 4 quarters", ORANGE, "4-quarter mean"),
        ("XGBoost (Poisson)", AQUA, "XGBoost"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(f.index))
    for col, color, label in series:
        ax.plot(x, f[col], color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(label, (len(f.index) - 1, f[col].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
    ax.set_xticks(list(x), f.index)
    ax.set_xlim(-0.3, len(f.index) + 0.9)
    ax.set_ylim(0, f[[s[0] for s in series]].to_numpy().max() * 1.15)
    ax.set_ylabel("Mean absolute error (deaths)")
    ax.set_title("Error by test quarter")
    ax.legend(frameon=False, loc="lower left", ncol=3, labelcolor=INK_2)
    finish(fig, ax, "mae_by_quarter.png", "Each point: all 37 states forecast for that quarter, trained only on earlier quarters.")


def national_forecast():
    panel = pd.read_csv(ROOT / "data" / "crashes_state_quarter.csv")
    actual = panel.groupby("quarter").killed.sum()
    p = pd.read_csv(RESULTS / "predictions.csv")
    fc = p.pivot_table(index="target_quarter", columns="model", values="pred", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = list(actual.index)
    pos = {q: i for i, q in enumerate(xs)}
    ax.plot([pos[q] for q in xs], actual.values, color=BLUE, linewidth=2, marker="o", markersize=6, label="Actual")
    for col, color, label in [("Mean of last 4 quarters", ORANGE, "4-quarter mean"), ("XGBoost (Poisson)", AQUA, "XGBoost")]:
        ax.plot([pos[q] for q in fc.index], fc[col].values, color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.annotate("Actual", (pos[xs[-1]], actual.iloc[-1]), xytext=(8, -4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(xs)), xs, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(xs) + 0.5)
    ax.set_ylim(0, actual.max() * 1.15)
    ax.set_ylabel("People killed, all states")
    ax.set_title("Every approach missed the sharp 2023Q3 dip")
    ax.legend(frameon=False, loc="lower left", ncol=3, labelcolor=INK_2)
    finish(fig, ax, "national_forecast.png", "State forecasts summed to a national total. Source: FRSC.")


def leakage():
    d = pd.read_csv(RESULTS / "leakage_demo.csv")
    base = pd.read_csv(RESULTS / "metrics.csv", index_col=0).loc["Mean of last 4 quarters", "MAE"]
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # Dots, not bars: the axis does not start at zero, so bar length would mislead.
    ax.scatter(d.MAE, d.validation, color=BLUE, s=70, zorder=3)
    for i, v in enumerate(d.MAE):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", color=INK_2)
    ax.axvline(base, color=INK_2, linewidth=1, linestyle="--")
    ax.text(base, len(d) - 0.35, f" 4-quarter mean baseline: {base:.2f}", color=INK_2, fontsize=9)
    ax.set_xlim(12.2, max(d.MAE.max(), base) + 0.35)
    ax.set_ylim(-0.6, len(d) - 0.1)
    ax.set_xlabel("XGBoost MAE on the same 222 state-quarters (axis does not start at zero)")
    ax.set_title("Leaky validation would flip the verdict")
    ax.tick_params(axis="y", colors=INK_2, length=0)
    finish(fig, ax, "leakage_demo.png", "Same model and settings; only the validation setup changes.")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    model_mae()
    mae_by_quarter()
    national_forecast()
    leakage()
    print("charts written to", FIGS)
