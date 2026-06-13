# Notebook 5: Sample Description and Descriptive Statistics

**Input:** `output/intermediate/descriptive_ready.csv` (saved by NB2 — post-winsorization, pre-rank-normalization)  
**Output:** `output/results/descriptive_ready.csv`, `desc_summary_stats.csv`, `desc_table.tex`

**Pipeline:**
```
descriptive_ready.csv  (from NB2: winsorized, pre-rank-norm, original scale)
  → return vars  *= 100   (convert to %, reporting only)
  → f_buy_5d     *= 1e6   (unit correction, reporting only)
  → f_sell_5d    *= 1e6   (unit correction, reporting only)
  → amihud_5d    *= 1e6   (readability scaling)
  → save to output/results/ → compute tables
```


```
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TARGET_COL   = "target_5d_mktadj"
FEATURE_COLS = [
    "ret_1d_lag", "ret_5d_lag", "momentum_1m", "volatility_1m", "log_size",
    "turnover_5d", "amihud_5d", "f_buy_5d", "f_sell_5d"
]
FLOW_COLS   = ["f_buy_5d", "f_sell_5d"]
RETURN_COLS = [TARGET_COL, "ret_1d_lag", "ret_5d_lag", "momentum_1m", "volatility_1m"]
```

## 5.1 Load Data


```
desc_df = pd.read_csv("output/intermediate/descriptive_ready.csv", parse_dates=["date"])

print("Shape:", desc_df.shape)
print("Columns:", list(desc_df.columns))
print("Date range:", desc_df["date"].min(), "–", desc_df["date"].max())
print("Sample periods:", desc_df["sample_period"].value_counts().to_dict())
```

    Shape: (111858, 13)
    Columns: ['ticker', 'date', 'target_5d_mktadj', 'ret_1d_lag', 'ret_5d_lag', 'momentum_1m', 'volatility_1m', 'log_size', 'turnover_5d', 'amihud_5d', 'f_buy_5d', 'f_sell_5d', 'sample_period']
    Date range: 2014-02-17 00:00:00 – 2025-12-29 00:00:00
    Sample periods: {'oos': 97976, 'development': 13882}


## 5.2 Rescale for Readability (Reporting Only)

**Return variables × 100:** expressed as percentages for reporting.  
**f_buy_5d, f_sell_5d × 10⁶:** corrects unit difference between CaFeF (billions VND) and Datastream (thousands VND).  
**amihud_5d × 10⁶:** raw values ~10⁻¹⁰ to 10⁻⁵, scaling for readability.

Applied after winsorization (done in NB2) so outliers are already clipped.  
Does NOT affect model results — rank normalization preserves ordering.


```
# Convert returns to percentage (reporting only)
for col in RETURN_COLS:
    desc_df[col] = desc_df[col] * 100

# Unit correction for foreign flows
for col in FLOW_COLS:
    desc_df[col] = desc_df[col] * 1e6

# Readability scaling for Amihud
desc_df["amihud_5d"] = desc_df["amihud_5d"] * 1e6

print("Rescaling applied.")
print(f"  Returns ×100: {RETURN_COLS}")
print(f"  Flows ×1e6:   {FLOW_COLS}")
print(f"  Amihud ×1e6")
print()
print("Return sanity check (target std should be ~2–10% weekly):")
for col in RETURN_COLS:
    print(f"  {col}: mean={desc_df[col].mean():.4f}%, std={desc_df[col].std():.4f}%")
```

    Rescaling applied.
      Returns ×100: ['target_5d_mktadj', 'ret_1d_lag', 'ret_5d_lag', 'momentum_1m', 'volatility_1m']
      Flows ×1e6:   ['f_buy_5d', 'f_sell_5d']
      Amihud ×1e6
    
    Return sanity check (target std should be ~2–10% weekly):
      target_5d_mktadj: mean=-0.0298%, std=4.9341%
      ret_1d_lag: mean=0.0604%, std=2.2999%
      ret_5d_lag: mean=0.2404%, std=5.3065%
      momentum_1m: mean=0.5795%, std=9.9677%
      volatility_1m: mean=2.2194%, std=1.0641%


## 5.3 Save descriptive_ready.csv


```
desc_df.to_csv("output/results/descriptive_ready.csv", index=False)
print("Saved: output/results/descriptive_ready.csv")
print(f"Shape: {desc_df.shape}")
```

    Saved: output/results/descriptive_ready.csv
    Shape: (111858, 13)


## 5.4 Panel A: Sample Structure


```
dev = desc_df[desc_df["sample_period"] == "development"]
oos = desc_df[desc_df["sample_period"] == "oos"]

def panel_stats(df, label):
    return {
        "Period":           label,
        "Date range":       f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}",
        "Weeks":            df["date"].nunique(),
        "Observations":     len(df),
        "Unique tickers":   df["ticker"].nunique(),
        "Avg tickers/week": round(df.groupby("date")["ticker"].nunique().mean(), 1),
    }

panel_a = pd.DataFrame([
    panel_stats(dev,     "Development"),
    panel_stats(oos,     "OOS"),
    panel_stats(desc_df, "Full sample"),
])

print("Panel A: Sample Structure")
print(panel_a.to_string(index=False))
```

    Panel A: Sample Structure
         Period          Date range  Weeks  Observations  Unique tickers  Avg tickers/week
    Development Feb 2014 – Dec 2015     96         13882             165             144.6
            OOS Jan 2016 – Dec 2025    514         97976             216             190.6
    Full sample Feb 2014 – Dec 2025    610        111858             219             183.4


## 5.5 Panel B: Summary Statistics

Post-winsorization | pre-rank-normalization | returns in % | flows + Amihud ×10⁶


```
LABEL_MAP = {
    TARGET_COL:      "5-day mkt-adj return (%)",
    "ret_1d_lag":    "1-day lagged return (%)",
    "ret_5d_lag":    "5-day lagged return (%)",
    "momentum_1m":   "1-month momentum (%)",
    "volatility_1m": "1-month realized volatility (%)",
    "log_size":      "Log market capitalization",
    "turnover_5d":   "5-day avg turnover",
    "amihud_5d":     "Amihud illiquidity (\u00d710\u2076)",
    "f_buy_5d":      "Foreign buy flow (\u00d710\u2076)",
    "f_sell_5d":     "Foreign sell flow (\u00d710\u2076)",
}
BLOCK_MAP = {
    TARGET_COL:      "Target",
    "ret_1d_lag":    "Block 1", "ret_5d_lag":    "Block 1",
    "momentum_1m":   "Block 1", "volatility_1m": "Block 1",
    "log_size":      "Block 1",
    "turnover_5d":   "Block 2", "amihud_5d":     "Block 2",
    "f_buy_5d":      "Block 3", "f_sell_5d":     "Block 3",
}

rows = []
for col in [TARGET_COL] + FEATURE_COLS:
    series      = desc_df[col].dropna()
    pct_nonzero = (desc_df[col] > 0).mean() * 100 if col in FLOW_COLS else None
    rows.append({
        "Block":      BLOCK_MAP[col],
        "Variable":   LABEL_MAP[col],
        "Mean":       series.mean(),
        "Std":        series.std(),
        "P10":        series.quantile(0.10),
        "P50":        series.quantile(0.50),
        "P90":        series.quantile(0.90),
        "% Non-zero": pct_nonzero,
        "N":          len(series),
    })

panel_b = pd.DataFrame(rows)

print("Panel B: Summary Statistics")
print("(post-winsorization | pre-rank-norm | returns in % | flows+Amihud \u00d710\u2076)")
print()
for _, r in panel_b.iterrows():
    nz = f"  [{r['% Non-zero']:.1f}% non-zero]" if pd.notna(r["% Non-zero"]) else ""
    print(f"{r['Block']:<8} {r['Variable']:<42} "
          f"mean={r['Mean']:>9.4f}  std={r['Std']:>9.4f}  "
          f"p10={r['P10']:>9.4f}  p50={r['P50']:>9.4f}  "
          f"p90={r['P90']:>9.4f}{nz}")

panel_b.to_csv("output/results/desc_summary_stats.csv", index=False)
print("\nSaved: desc_summary_stats.csv")
```

    Panel B: Summary Statistics
    (post-winsorization | pre-rank-norm | returns in % | flows+Amihud ×10⁶)
    
    Target   5-day mkt-adj return (%)                   mean=  -0.0298  std=   4.9341  p10=  -5.3730  p50=  -0.3044  p90=   5.8057
    Block 1  1-day lagged return (%)                    mean=   0.0604  std=   2.2999  p10=  -2.3651  p50=   0.0000  p90=   2.7697
    Block 1  5-day lagged return (%)                    mean=   0.2404  std=   5.3065  p10=  -5.3162  p50=   0.0000  p90=   6.2932
    Block 1  1-month momentum (%)                       mean=   0.5795  std=   9.9677  p10=  -9.9912  p50=   0.2557  p90=  12.0079
    Block 1  1-month realized volatility (%)            mean=   2.2194  std=   1.0641  p10=   0.9938  p50=   2.0505  p90=   3.7068
    Block 1  Log market capitalization                  mean=  15.1949  std=   1.5643  p10=  13.4170  p50=  14.9344  p90=  17.5901
    Block 2  5-day avg turnover                         mean=   0.0072  std=   0.0116  p10=   0.0002  p50=   0.0027  p90=   0.0197
    Block 2  Amihud illiquidity (×10⁶)                  mean=   0.2359  std=   1.3599  p10=   0.0001  p50=   0.0019  p90=   0.1406
    Block 3  Foreign buy flow (×10⁶)                    mean=   0.1782  std=   4.1237  p10=   0.0000  p50=   0.0305  p90=   0.3330  [84.4% non-zero]
    Block 3  Foreign sell flow (×10⁶)                   mean=   0.1632  std=   1.8788  p10=   0.0000  p50=   0.0304  p90=   0.3069  [83.0% non-zero]
    
    Saved: desc_summary_stats.csv


## 5.6 Export LaTeX Table

Requires `\usepackage{booktabs}` in LaTeX preamble.


```
def fmt(x, decimals=2):
    if pd.isna(x) or x is None:
        return ""
    return f"{x:.{decimals}f}"

def fmt4(x):
    return fmt(x, 4)

# Return rows use 2 decimal places; others use 4
RETURN_LABELS = {LABEL_MAP[c] for c in RETURN_COLS}

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Sample Description and Summary Statistics}")
lines.append(r"\label{tab:descriptive}")

# Panel A
lines.append(r"\vspace{0.5em}")
lines.append(r"\textbf{Panel A: Sample Structure}")
lines.append(r"\vspace{0.3em}")
lines.append(r"\begin{tabular}{lrrrr}")
lines.append(r"\toprule")
lines.append(r"Period & Date range & Weeks & Obs. & Avg tickers/week \\")
lines.append(r"\midrule")
for _, row in panel_a.iterrows():
    lines.append(
        f"{row['Period']} & {row['Date range']} & "
        f"{int(row['Weeks']):,} & {int(row['Observations']):,} & "
        f"{row['Avg tickers/week']:.1f} \\\\"
    )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

# Panel B
lines.append(r"\vspace{1em}")
lines.append(r"\textbf{Panel B: Summary Statistics}")
lines.append(r"\vspace{0.3em}")
lines.append(r"\begin{tabular}{llrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Block & Variable & Mean & Std & P10 & P50 & P90 & \% Non-zero \\")
lines.append(r"\midrule")

current_block = None
for _, row in panel_b.iterrows():
    block_label   = row["Block"] if row["Block"] != current_block else ""
    current_block = row["Block"]
    nz_str = f"{row['% Non-zero']:.1f}" if pd.notna(row["% Non-zero"]) else ""
    f = fmt if row["Variable"] in RETURN_LABELS else fmt4
    lines.append(
        f"{block_label} & {row['Variable']} & "
        f"{f(row['Mean'])} & {f(row['Std'])} & "
        f"{f(row['P10'])} & {f(row['P50'])} & "
        f"{f(row['P90'])} & {nz_str} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

# Note
lines.append(r"\vspace{0.3em}")
lines.append(
    r"\begin{minipage}{\linewidth}"
    r"\footnotesize \textit{Note.} "
    r"Return variables are expressed as percentages. "
    r"Amihud illiquidity and foreign flow variables are multiplied by $10^6$ for readability."
    r"\end{minipage}"
)
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
with open("output/results/desc_table.tex", "w") as f:
    f.write(latex_str)

print("Saved: desc_table.tex")
print()
print(latex_str)
```

    Saved: desc_table.tex
    
    \begin{table}[ht]
    \centering
    \small
    \caption{Sample Description and Summary Statistics}
    \label{tab:descriptive}
    \vspace{0.5em}
    \textbf{Panel A: Sample Structure}
    \vspace{0.3em}
    \begin{tabular}{lrrrr}
    \toprule
    Period & Date range & Weeks & Obs. & Avg tickers/week \\
    \midrule
    Development & Feb 2014 – Dec 2015 & 96 & 13,882 & 144.6 \\
    OOS & Jan 2016 – Dec 2025 & 514 & 97,976 & 190.6 \\
    Full sample & Feb 2014 – Dec 2025 & 610 & 111,858 & 183.4 \\
    \bottomrule
    \end{tabular}
    \vspace{1em}
    \textbf{Panel B: Summary Statistics}
    \vspace{0.3em}
    \begin{tabular}{llrrrrrr}
    \toprule
    Block & Variable & Mean & Std & P10 & P50 & P90 & \% Non-zero \\
    \midrule
    Target & 5-day mkt-adj return (%) & -0.03 & 4.93 & -5.37 & -0.30 & 5.81 &  \\
    Block 1 & 1-day lagged return (%) & 0.06 & 2.30 & -2.37 & 0.00 & 2.77 &  \\
     & 5-day lagged return (%) & 0.24 & 5.31 & -5.32 & 0.00 & 6.29 &  \\
     & 1-month momentum (%) & 0.58 & 9.97 & -9.99 & 0.26 & 12.01 &  \\
     & 1-month realized volatility (%) & 2.22 & 1.06 & 0.99 & 2.05 & 3.71 &  \\
     & Log market capitalization & 15.1949 & 1.5643 & 13.4170 & 14.9344 & 17.5901 &  \\
    Block 2 & 5-day avg turnover & 0.0072 & 0.0116 & 0.0002 & 0.0027 & 0.0197 &  \\
     & Amihud illiquidity (×10⁶) & 0.2359 & 1.3599 & 0.0001 & 0.0019 & 0.1406 &  \\
    Block 3 & Foreign buy flow (×10⁶) & 0.1782 & 4.1237 & 0.0000 & 0.0305 & 0.3330 & 84.4 \\
     & Foreign sell flow (×10⁶) & 0.1632 & 1.8788 & 0.0000 & 0.0304 & 0.3069 & 83.0 \\
    \bottomrule
    \end{tabular}
    \vspace{0.3em}
    \begin{minipage}{\linewidth}\footnotesize \textit{Note.} Return variables are expressed as percentages. Amihud illiquidity and foreign flow variables are multiplied by $10^6$ for readability.\end{minipage}
    \end{table}

