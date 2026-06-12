"""
10_figure_portfolio_spreads.py
--------------------------------
Replaces: output/figures/portfolio_cumret_spreads.png

Figure 2 — Ranking diagnostic (Block 3 RF and XGB):
    - RF Block 3 long-short spread   blue solid   #4e79a7
    - XGB Block 3 long-short spread  orange solid #f28e2b

Input:  output/intermediate/portfolio_weekly_returns.csv
Output: output/figures/portfolio_cumret_spreads.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

matplotlib.use("Agg")

plt.style.use("default")
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "black",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
})

WEEKLY_PATH = "output/intermediate/portfolio_weekly_returns.csv"
OUT_FIG     = "output/figures/portfolio_cumret_spreads.png"
os.makedirs("output/figures", exist_ok=True)

weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["date"])

fig, ax = plt.subplots(figsize=(10, 5))

# Long-short spread — solid lines only
for col, color, label in [
    ("pred_rf_b3",  "#4e79a7", "RF Block 3 long-short spread"),
    ("pred_xgb_b3", "#f28e2b", "XGB Block 3 long-short spread"),
]:
    sub    = weekly[weekly["pred_col"] == col].sort_values("date")
    cumret = np.exp(sub["long_short_return"].cumsum()) - 1
    ax.plot(sub["date"], cumret, color=color, lw=1.5, linestyle="-", label=label)

ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.yaxis.grid(True, zorder=0, color="#cccccc", linewidth=0.5, linestyle="--")
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

ax.set_xlabel("Date", fontsize=9)
ax.set_ylabel("Cumulative market-adjusted spread", fontsize=9)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG}")
