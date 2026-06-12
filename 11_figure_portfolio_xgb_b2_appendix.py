"""
11_figure_portfolio_xgb_b2_appendix.py
----------------------------------------
New appendix figure.

Figure 3 — XGB Block 2 best-performing specification (appendix):
    - XGB Block 2 top-quintile       orange solid  #f28e2b
    - XGB Block 2 exclude-bottom     blue solid    #4e79a7
    - Equal-weight benchmark         grey dotted   #bab0ac

Input:  output/intermediate/portfolio_weekly_returns.csv
Output: output/figures/portfolio_cumret_xgb_b2.png
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
OUT_FIG     = "output/figures/portfolio_cumret_xgb_b2.png"
os.makedirs("output/figures", exist_ok=True)

weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["date"])

sub = (
    weekly[weekly["pred_col"] == "pred_xgb_b2"]
    .sort_values("date")
    .reset_index(drop=True)
)

cumret_top         = np.exp(sub["top_return"].cumsum()) - 1
cumret_excl_bottom = np.exp(sub["exclude_bottom_return"].cumsum()) - 1
cumret_ew          = np.exp(sub["equal_weight_return"].cumsum()) - 1

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(sub["date"], cumret_top,
        color="#f28e2b", lw=1.5, linestyle="-",
        label="XGB Block 2 top quintile")

ax.plot(sub["date"], cumret_excl_bottom,
        color="#4e79a7", lw=1.5, linestyle="-",
        label="XGB Block 2 exclude bottom")

ax.plot(sub["date"], cumret_ew,
        color="#bab0ac", lw=1.5, linestyle=":",
        label="Equal-weight benchmark")

ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.yaxis.grid(True, zorder=0, color="#cccccc", linewidth=0.5, linestyle="--")
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

ax.set_xlabel("Date", fontsize=9)
ax.set_ylabel("Cumulative market-adjusted return", fontsize=9)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG}")
