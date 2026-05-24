# Notebook 5: Sample Description and Descriptive Statistics

**Input:** `features_panel.csv`, `model_ready.csv`  
**Output:** `descriptive_ready.csv`, `desc_summary_stats.csv`, `desc_table.tex`

**Pipeline:**
```
features_panel.csv
  → filter exact (ticker, date) pairs from model_ready
  → cross-sectional winsorization P1/P99 WITHIN EACH WEEK  [replicate NB2]
  → STOP before rank normalization
  → f_buy_5d  *= 1e6  (unit correction, reporting only)
  → f_sell_5d *= 1e6  (unit correction, reporting only)
  → amihud_5d *= 1e6  (readability scaling)
  → save descriptive_ready.csv → compute tables
```

**Critical:** model_ready.csv, predictions.csv, SHAP files are NEVER touched.


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
WINSOR_COLS = FEATURE_COLS + [TARGET_COL]
```

## 5.1 Load Data


```
model_ready    = pd.read_csv("model_ready.csv",    parse_dates=["date"])
features_panel = pd.read_csv("features_panel.csv", parse_dates=["date"])

print("model_ready shape:    ", model_ready.shape)
print("features_panel shape: ", features_panel.shape)

missing = [c for c in WINSOR_COLS if c not in features_panel.columns]
print("Missing columns:", missing if missing else "None — all OK")
```

    model_ready shape:     (111858, 13)
    features_panel shape:  (568513, 35)
    Missing columns: None — all OK


## 5.2 Filter to Exact Modeling Sample

Keep only (ticker, date) pairs present in model_ready.


```
base = model_ready[["ticker", "date", "sample_period"]].copy()

desc_df = base.merge(
    features_panel[["ticker", "date"] + FEATURE_COLS + [TARGET_COL]],
    on=["ticker", "date"],
    how="left"
)

print("Shape:", desc_df.shape)
print("Missing per column:")
print(desc_df[WINSOR_COLS].isna().sum())
assert len(desc_df) == len(model_ready), "Row count mismatch"
print("\nRow count matches model_ready: OK")
```

    Shape: (111858, 13)
    Missing per column:
    ret_1d_lag          0
    ret_5d_lag          0
    momentum_1m         0
    volatility_1m       0
    log_size            0
    turnover_5d         0
    amihud_5d           0
    f_buy_5d            0
    f_sell_5d           0
    target_5d_mktadj    0
    dtype: int64
    
    Row count matches model_ready: OK


## 5.3 Cross-Sectional Winsorization (Replicate NB2 Exactly)

Winsorize at P1/P99 WITHIN EACH WEEK. Identical to NB2.
Stopping here = closest snapshot to actual modeling environment.


```
def cross_sectional_winsorize(df, cols, lower=0.01, upper=0.99):
    df = df.copy()
    for col in cols:
        def _clip(group):
            lo = group.quantile(lower)
            hi = group.quantile(upper)
            return group.clip(lo, hi)
        df[col] = df.groupby("date")[col].transform(_clip)
    return df

desc_df = cross_sectional_winsorize(desc_df, WINSOR_COLS)
print("Winsorization complete.")
print("\nStats after winsorization (pre-rescaling):")
print(desc_df[WINSOR_COLS].describe().loc[["mean","std","min","max"]].round(6))
```

    Winsorization complete.
    
    Stats after winsorization (pre-rescaling):
          ret_1d_lag  ret_5d_lag  momentum_1m  volatility_1m   log_size  \
    mean    0.000604    0.002404     0.004578       0.022194  15.194916   
    std     0.022999    0.053065     0.105180       0.010641   1.564310   
    min    -0.091604   -0.357565    -1.030005       0.002227  11.351666   
    max     0.125691    0.332002     0.579087       0.079189  20.045885   
    
          turnover_5d  amihud_5d  f_buy_5d  f_sell_5d  target_5d_mktadj  
    mean     0.007196   0.000000  0.000000   0.000000         -0.000298  
    std      0.011601   0.000001  0.000004   0.000002          0.049341  
    min      0.000002   0.000000  0.000000   0.000000         -0.376129  
    max      0.198732   0.000040  0.000895   0.000299          0.298420  


## 5.4 Rescale for Readability (Reporting Only)

**f_buy_5d, f_sell_5d × 10⁶:** corrects unit difference between
CaFeF (billions VND) and Datastream (thousands VND).
Expresses foreign flow as proportion of total trading value.

**amihud_5d × 10⁶:** raw values ~10⁻¹⁰ to 10⁻⁵, scaling for readability.

Applied AFTER winsorization so outliers are already clipped.
Does NOT affect model results — rank normalization preserves ordering.


```
# Unit correction for foreign flows
for col in FLOW_COLS:
    desc_df[col] = desc_df[col] * 1e6

# Readability scaling for Amihud
desc_df["amihud_5d"] = desc_df["amihud_5d"] * 1e6

print("Rescaling applied.")
print()
print("Foreign flow sanity check (should be in [0, ~1] for most obs):")
for col in FLOW_COLS:
    pct_zero     = (desc_df[col] == 0).mean() * 100
    pct_above_1  = (desc_df[col] > 1).mean() * 100
    pct_nonzero  = (desc_df[col] > 0).mean() * 100
    print(f"  {col}: mean={desc_df[col].mean():.4f}, "
          f"P99={desc_df[col].quantile(0.99):.4f}, "
          f"max={desc_df[col].max():.4f}, "
          f"{pct_zero:.1f}% zero, {pct_nonzero:.1f}% non-zero, "
          f"{pct_above_1:.2f}% above 1.0")
```

    Rescaling applied.
    
    Foreign flow sanity check (should be in [0, ~1] for most obs):
      f_buy_5d: mean=0.1782, P99=1.4679, max=894.7444, 15.6% zero, 84.4% non-zero, 1.51% above 1.0
      f_sell_5d: mean=0.1632, P99=1.3736, max=298.5240, 17.0% zero, 83.0% non-zero, 1.44% above 1.0


## 5.5 Save descriptive_ready.csv


```
desc_df.to_csv("descriptive_ready.csv", index=False)
print("Saved: descriptive_ready.csv")
print(f"Shape: {desc_df.shape}")
```

    Saved: descriptive_ready.csv
    Shape: (111858, 13)


## 5.6 Panel A: Sample Structure


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


## 5.7 Panel B: Summary Statistics

Post-winsorization | pre-rank-normalization | flows + Amihud rescaled ×10⁶


```
LABEL_MAP = {
    TARGET_COL:      "5-day mkt-adj return (target)",
    "ret_1d_lag":    "1-day lagged return",
    "ret_5d_lag":    "5-day lagged return",
    "momentum_1m":   "1-month momentum",
    "volatility_1m": "1-month realized volatility",
    "log_size":      "Log market capitalization",
    "turnover_5d":   "5-day avg turnover",
    "amihud_5d":     "Amihud illiquidity (×10⁶)",
    "f_buy_5d":      "Foreign buy flow (×10⁶)",
    "f_sell_5d":     "Foreign sell flow (×10⁶)",
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
print("(post-winsorization | pre-rank-normalization | flows + Amihud ×10⁶)")
print()
for _, r in panel_b.iterrows():
    nz = f"  [{r['% Non-zero']:.1f}% non-zero]" if pd.notna(r["% Non-zero"]) else ""
    print(f"{r['Block']:<8} {r['Variable']:<42} "
          f"mean={r['Mean']:>9.4f}  std={r['Std']:>9.4f}  "
          f"p10={r['P10']:>9.4f}  p50={r['P50']:>9.4f}  "
          f"p90={r['P90']:>9.4f}{nz}")

panel_b.to_csv("desc_summary_stats.csv", index=False)
print("\nSaved: desc_summary_stats.csv")
```

    Panel B: Summary Statistics
    (post-winsorization | pre-rank-normalization | flows + Amihud ×10⁶)
    
    Target   5-day mkt-adj return (target)              mean=  -0.0003  std=   0.0493  p10=  -0.0537  p50=  -0.0030  p90=   0.0581
    Block 1  1-day lagged return                        mean=   0.0006  std=   0.0230  p10=  -0.0237  p50=   0.0000  p90=   0.0277
    Block 1  5-day lagged return                        mean=   0.0024  std=   0.0531  p10=  -0.0532  p50=   0.0000  p90=   0.0629
    Block 1  1-month momentum                           mean=   0.0046  std=   0.1052  p10=  -0.1086  p50=   0.0015  p90=   0.1254
    Block 1  1-month realized volatility                mean=   0.0222  std=   0.0106  p10=   0.0099  p50=   0.0205  p90=   0.0371
    Block 1  Log market capitalization                  mean=  15.1949  std=   1.5643  p10=  13.4170  p50=  14.9344  p90=  17.5901
    Block 2  5-day avg turnover                         mean=   0.0072  std=   0.0116  p10=   0.0002  p50=   0.0027  p90=   0.0197
    Block 2  Amihud illiquidity (×10⁶)                  mean=   0.2359  std=   1.3599  p10=   0.0001  p50=   0.0019  p90=   0.1406
    Block 3  Foreign buy flow (×10⁶)                    mean=   0.1782  std=   4.1237  p10=   0.0000  p50=   0.0305  p90=   0.3330  [84.4% non-zero]
    Block 3  Foreign sell flow (×10⁶)                   mean=   0.1632  std=   1.8788  p10=   0.0000  p50=   0.0304  p90=   0.3069  [83.0% non-zero]
    
    Saved: desc_summary_stats.csv


## 5.8 Export LaTeX Table

Requires `\usepackage{booktabs}` in LaTeX preamble.


```
def fmt(x, decimals=4):
    if pd.isna(x) or x is None:
        return ""
    return f"{x:.{decimals}f}"

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
    lines.append(
        f"{block_label} & {row['Variable']} & "
        f"{fmt(row['Mean'])} & {fmt(row['Std'])} & "
        f"{fmt(row['P10'])} & {fmt(row['P50'])} & "
        f"{fmt(row['P90'])} & {nz_str} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

# Table note
lines.append(r"\vspace{0.5em}")
lines.append(r"\begin{minipage}{0.95\textwidth}")
lines.append(r"\footnotesize")
lines.append(
    r"\textit{Notes:} Panel A reports the sample structure after all filters "
    r"described in Section~\ref{sec:data}. "
    r"Panel B reports summary statistics on the full sample. "
    r"All predictors are reported after cross-sectional winsorization at the "
    r"1st and 99th percentile within each week and before rank normalization, "
    r"representing the data environment immediately prior to the final "
    r"transformation step in the modeling pipeline. "
    r"The prediction target is reported at the same winsorized scale as it "
    r"enters model estimation. "
    r"Foreign buy and sell flow variables are rescaled by $10^6$ for reporting "
    r"purposes only, correcting for a unit difference between the CaFeF and "
    r"Datastream data sources; this rescaling does not affect model estimation "
    r"because all predictors are subsequently transformed into weekly "
    r"cross-sectional ranks, and any positive scalar rescaling preserves "
    r"cross-sectional ordering \citep{gu_2020_empirical}. "
    r"Amihud illiquidity is scaled by $10^6$ for readability. "
    r"Return variables are expressed as decimals (e.g., 0.01 = 1\%). "
    r"The percentage of non-zero weekly observations is reported for "
    r"foreign flow variables."
)
lines.append(r"\end{minipage}")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
with open("desc_table.tex", "w") as f:
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
    Target & 5-day mkt-adj return (target) & -0.0003 & 0.0493 & -0.0537 & -0.0030 & 0.0581 &  \\
    Block 1 & 1-day lagged return & 0.0006 & 0.0230 & -0.0237 & 0.0000 & 0.0277 &  \\
     & 5-day lagged return & 0.0024 & 0.0531 & -0.0532 & 0.0000 & 0.0629 &  \\
     & 1-month momentum & 0.0046 & 0.1052 & -0.1086 & 0.0015 & 0.1254 &  \\
     & 1-month realized volatility & 0.0222 & 0.0106 & 0.0099 & 0.0205 & 0.0371 &  \\
     & Log market capitalization & 15.1949 & 1.5643 & 13.4170 & 14.9344 & 17.5901 &  \\
    Block 2 & 5-day avg turnover & 0.0072 & 0.0116 & 0.0002 & 0.0027 & 0.0197 &  \\
     & Amihud illiquidity (×10⁶) & 0.2359 & 1.3599 & 0.0001 & 0.0019 & 0.1406 &  \\
    Block 3 & Foreign buy flow (×10⁶) & 0.1782 & 4.1237 & 0.0000 & 0.0305 & 0.3330 & 84.4 \\
     & Foreign sell flow (×10⁶) & 0.1632 & 1.8788 & 0.0000 & 0.0304 & 0.3069 & 83.0 \\
    \bottomrule
    \end{tabular}
    \vspace{0.5em}
    \begin{minipage}{0.95\textwidth}
    \footnotesize
    \textit{Notes:} Panel A reports the sample structure after all filters described in Section~\ref{sec:data}. Panel B reports summary statistics on the full sample. All predictors are reported after cross-sectional winsorization at the 1st and 99th percentile within each week and before rank normalization, representing the data environment immediately prior to the final transformation step in the modeling pipeline. The prediction target is reported at the same winsorized scale as it enters model estimation. Foreign buy and sell flow variables are rescaled by $10^6$ for reporting purposes only, correcting for a unit difference between the CaFeF and Datastream data sources; this rescaling does not affect model estimation because all predictors are subsequently transformed into weekly cross-sectional ranks, and any positive scalar rescaling preserves cross-sectional ordering \citep{gu_2020_empirical}. Amihud illiquidity is scaled by $10^6$ for readability. Return variables are expressed as decimals (e.g., 0.01 = 1\%). The percentage of non-zero weekly observations is reported for foreign flow variables.
    \end{minipage}
    \end{table}

