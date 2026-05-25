import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

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

# Load and compute rolling volatility
df = pd.read_csv("output/intermediate/vnindex_clean.csv")
df["date"]    = pd.to_datetime(df["date"])
df            = df.sort_values("date")
df            = df[(df["date"] >= "2016-01-01") & (df["date"] <= "2025-12-31")]
df["ret"]     = np.log(df["vnindex"] / df["vnindex"].shift(1))
df["vol_20d"] = df["ret"].rolling(20).std()
df            = df.dropna(subset=["vol_20d"])

CALM_START   = pd.Timestamp("2016-01-01")
CALM_END     = pd.Timestamp("2019-12-31")
STRESS_START = pd.Timestamp("2022-01-01")
STRESS_END   = pd.Timestamp("2023-12-31")

fig, ax = plt.subplots(figsize=(6.5, 2.8))

# Shaded regions
ax.axvspan(CALM_START,   CALM_END,   alpha=0.15, color="#2166ac", zorder=0)
ax.axvspan(STRESS_START, STRESS_END, alpha=0.18, color="#d73027", zorder=0)

# Volatility line
ax.plot(df["date"], df["vol_20d"], color="#111111", linewidth=0.75, zorder=2)

# Grid
ax.yaxis.grid(True, zorder=0, color="#cccccc", linewidth=0.5, linestyle="--")
ax.set_axisbelow(True)

# Axes labels — smaller font on y-axis label
ax.set_xlabel("Year", fontsize=9)
ax.set_ylabel("Rolling 20-day volatility\n(log-return std)", fontsize=8.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31"))
ax.tick_params(colors="black", which="both", labelsize=8.5)

# ── Direct annotations on shaded regions (academic style) ──────────────────
# Place text label at top of each shaded band
y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else df["vol_20d"].max() * 0.97

# Calm label: centered in calm band
calm_mid = CALM_START + (CALM_END - CALM_START) / 2
ax.text(calm_mid, df["vol_20d"].max() * 0.96,
        "Calm\n(2016–2019)",
        ha="center", va="top", fontsize=8,
        color="#1a5276",
        fontstyle="italic")

# Stress label: centered in stress band
stress_mid = STRESS_START + (STRESS_END - STRESS_START) / 2
ax.text(stress_mid, df["vol_20d"].max() * 0.96,
        "Stress\n(2022–2023)",
        ha="center", va="top", fontsize=8,
        color="#922b21",
        fontstyle="italic")

plt.tight_layout(pad=0.4)
plt.savefig("output/figures/regime_volatility_apa.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
plt.show()
print("Saved: regime_volatility_apa.png")