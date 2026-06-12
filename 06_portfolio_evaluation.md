# Notebook 6: Portfolio-Sorting Economic Evaluation

**Input:** `output/intermediate/predictions.csv`, `output/intermediate/regime_labels.csv`

**Output:**
- `output/intermediate/portfolio_weekly_returns.csv`
- `output/results/portfolio_results.csv`
- `output/results/portfolio_regime_results.csv`
- `output/figures/portfolio_cumret_long_only.png`
- `output/figures/portfolio_cumret_spreads.png`

**Purpose:** Evaluate the economic meaning of model forecasts through portfolio sorting.
The `actual` column is the five-day-ahead market-adjusted log return target (`target_5d_mktadj`).
All portfolio returns are gross market-adjusted log returns; short-selling is not assumed.
The long-short spread is reported as a ranking diagnostic.
No model re-training occurs here.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

PRED_PATH    = "output/intermediate/predictions.csv"
REGIME_PATH  = "output/intermediate/regime_labels.csv"

OUT_WEEKLY   = "output/intermediate/portfolio_weekly_returns.csv"
OUT_FULL     = "output/results/portfolio_results.csv"
OUT_REGIME   = "output/results/portfolio_regime_results.csv"
OUT_TABLE    = "output/results/portfolio_sorting_table_apa.tex"
FIG_LONG     = "output/figures/portfolio_cumret_long_only.png"
FIG_SPREAD   = "output/figures/portfolio_cumret_spreads.png"

os.makedirs("output/intermediate", exist_ok=True)
os.makedirs("output/results",      exist_ok=True)
os.makedirs("output/figures",      exist_ok=True)

PRED_COLS = [
    "pred_fe_b1",  "pred_fe_b2",  "pred_fe_b3",
    "pred_rf_b1",  "pred_rf_b2",  "pred_rf_b3",
    "pred_xgb_b1", "pred_xgb_b2", "pred_xgb_b3",
]
```


```python
df     = pd.read_csv(PRED_PATH,  parse_dates=["date"])
regime = pd.read_csv(REGIME_PATH, parse_dates=["date"])

# merge regime label
df = df.merge(regime, on="date", how="left")

print("Prediction panel shape:", df.shape)
print("OOS weeks             :", df["date"].nunique())
print("Model-block combos    :", len(PRED_COLS))
print("Regime value counts:")
print(df[["date","regime"]].drop_duplicates()["regime"].value_counts())
```

    Prediction panel shape: (97976, 13)
    OOS weeks             : 514
    Model-block combos    : 9
    Regime value counts:
    regime
    calm      207
    other     204
    stress    103
    Name: count, dtype: int64



```python
# Rank stocks within each week; pct=True gives [0,1], method='first' breaks ties
for col in PRED_COLS:
    df[col + "_rank"] = df.groupby("date")[col].rank(method="first", pct=True)

# Top quintile: rank >= 0.80; bottom quintile: rank <= 0.20
for col in PRED_COLS:
    r = col + "_rank"
    df[col + "_top"]    = df[r] >= 0.80
    df[col + "_bottom"] = df[r] <= 0.20
```


```python
records = []

for col in PRED_COLS:
    top_flag    = col + "_top"
    bottom_flag = col + "_bottom"

    for date, g in df.groupby("date"):
        actual = g["actual"]
        is_top    = g[top_flag]
        is_bottom = g[bottom_flag]

        top_ret    = actual[is_top].mean()
        bot_ret    = actual[is_bottom].mean()
        ew_ret     = actual.mean()
        excl_ret   = actual[~is_bottom].mean()

        records.append({
            "pred_col"                      : col,
            "date"                          : date,
            "regime"                        : g["regime"].iloc[0],
            "top_return"                    : top_ret,
            "bottom_return"                 : bot_ret,
            "equal_weight_return"           : ew_ret,
            "exclude_bottom_return"         : excl_ret,
            "long_short_return"             : top_ret - bot_ret,
            "top_minus_equal_weight"        : top_ret - ew_ret,
            "exclude_bottom_minus_equal_weight": excl_ret - ew_ret,
            "n_top"                         : int(is_top.sum()),
            "n_bottom"                      : int(is_bottom.sum()),
            "n_total"                       : len(g),
        })

weekly = pd.DataFrame(records)
weekly.to_csv(OUT_WEEKLY, index=False)
print("Saved:", OUT_WEEKLY)
print(weekly.head())
```

    Saved: output/intermediate/portfolio_weekly_returns.csv
         pred_col       date regime  top_return  bottom_return  \
    0  pred_fe_b1 2016-01-04   calm    0.004813      -0.015289   
    1  pred_fe_b1 2016-01-11   calm    0.026382       0.014200   
    2  pred_fe_b1 2016-01-18   calm   -0.012484      -0.016824   
    3  pred_fe_b1 2016-01-25   calm    0.007939       0.006115   
    4  pred_fe_b1 2016-02-01   calm    0.005819      -0.013988   
    
       equal_weight_return  exclude_bottom_return  long_short_return  \
    0            -0.005326              -0.002895           0.020101   
    1             0.020345               0.021844           0.012182   
    2            -0.000530               0.003512           0.004339   
    3             0.004151               0.003668           0.001824   
    4             0.002245               0.006303           0.019806   
    
       top_minus_equal_weight  exclude_bottom_minus_equal_weight  n_top  n_bottom  \
    0                0.010139                           0.002432     32        31   
    1                0.006037                           0.001499     31        30   
    2               -0.011954                           0.004042     33        32   
    3                0.003788                          -0.000483     32        31   
    4                0.003574                           0.004058     32        31   
    
       n_total  
    0      158  
    1      153  
    2      161  
    3      157  
    4      155  



```python
RETURN_SERIES = [
    "top_return",
    "bottom_return",
    "exclude_bottom_return",
    "equal_weight_return",
    "long_short_return",
    "top_minus_equal_weight",
    "exclude_bottom_minus_equal_weight",
]

def max_drawdown(log_ret_series):
    """Maximum drawdown from cumulative wealth index (log-return basis)."""
    cumlog = log_ret_series.cumsum()
    wealth = np.exp(cumlog)
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return dd.min()

def performance_metrics(series):
    """Return a dict of performance metrics for a weekly log-return series."""
    s = series.dropna()
    mean_w   = s.mean()
    vol_w    = s.std(ddof=1)
    ann_mean = mean_w * 52
    ann_vol  = vol_w  * np.sqrt(52)
    sharpe   = (mean_w / vol_w) * np.sqrt(52) if vol_w > 0 else np.nan
    hit      = (s > 0).mean()
    mdd      = max_drawdown(s)
    return {
        "mean_weekly"       : mean_w,
        "ann_mean"          : ann_mean,
        "weekly_vol"        : vol_w,
        "ann_vol"           : ann_vol,
        "sharpe"            : sharpe,
        "hit_rate"          : hit,
        "max_drawdown"      : mdd,
        "n_weeks"           : len(s),
    }
```


```python
full_rows = []

for col in PRED_COLS:
    sub = weekly[weekly["pred_col"] == col].sort_values("date")
    for series_name in RETURN_SERIES:
        m = performance_metrics(sub[series_name])
        m["pred_col"]    = col
        m["series_name"] = series_name
        full_rows.append(m)

full_results = pd.DataFrame(full_rows)
cols_order = ["pred_col", "series_name", "mean_weekly", "ann_mean",
              "weekly_vol", "ann_vol", "sharpe", "hit_rate", "max_drawdown", "n_weeks"]
full_results = full_results[cols_order]
full_results.to_csv(OUT_FULL, index=False)
print("Saved:", OUT_FULL)
print(full_results.head(10).round(4).to_string())
```

    Saved: output/results/portfolio_results.csv
         pred_col                        series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0  pred_fe_b1                         top_return       0.0003    0.0154      0.0159   0.1145  0.1340    0.5214       -0.3279      514
    1  pred_fe_b1                      bottom_return      -0.0022   -0.1150      0.0213   0.1537 -0.7483    0.4825       -0.6808      514
    2  pred_fe_b1              exclude_bottom_return      -0.0003   -0.0158      0.0137   0.0987 -0.1597    0.5311       -0.3092      514
    3  pred_fe_b1                equal_weight_return      -0.0007   -0.0354      0.0144   0.1037 -0.3416    0.5078       -0.3498      514
    4  pred_fe_b1                  long_short_return       0.0025    0.1304      0.0201   0.1451  0.8984    0.5564       -0.1790      514
    5  pred_fe_b1             top_minus_equal_weight       0.0010    0.0508      0.0103   0.0743  0.6835    0.5331       -0.1281      514
    6  pred_fe_b1  exclude_bottom_minus_equal_weight       0.0004    0.0197      0.0029   0.0206  0.9535    0.5700       -0.0279      514
    7  pred_fe_b2                         top_return       0.0002    0.0108      0.0155   0.1117  0.0963    0.5350       -0.3496      514
    8  pred_fe_b2                      bottom_return      -0.0024   -0.1270      0.0213   0.1533 -0.8282    0.4747       -0.7177      514
    9  pred_fe_b2              exclude_bottom_return      -0.0002   -0.0128      0.0137   0.0985 -0.1299    0.5389       -0.3061      514



```python
# APA-style table for thesis reporting
model_order = [
    "pred_fe_b1", "pred_fe_b2", "pred_fe_b3",
    "pred_rf_b1", "pred_rf_b2", "pred_rf_b3",
    "pred_xgb_b1", "pred_xgb_b2", "pred_xgb_b3",
]

model_label = {
    "pred_fe_b1":  "FE B1",
    "pred_fe_b2":  "FE B2",
    "pred_fe_b3":  "FE B3",
    "pred_rf_b1":  "RF B1",
    "pred_rf_b2":  "RF B2",
    "pred_rf_b3":  "RF B3",
    "pred_xgb_b1": "XGB B1",
    "pred_xgb_b2": "XGB B2",
    "pred_xgb_b3": "XGB B3",
}

def get_metric(pred_col, series_name, metric):
    return full_results.loc[
        (full_results["pred_col"] == pred_col) &
        (full_results["series_name"] == series_name),
        metric
    ].iloc[0]

table_rows = []
for col in model_order:
    table_rows.append({
        "model": model_label[col],
        "top_mean": get_metric(col, "top_return", "ann_mean") * 100,
        "ew_mean": get_metric(col, "equal_weight_return", "ann_mean") * 100,
        "top_ew": get_metric(col, "top_minus_equal_weight", "ann_mean") * 100,
        "excl_bottom_ew": get_metric(col, "exclude_bottom_minus_equal_weight", "ann_mean") * 100,
        "long_short_mean": get_metric(col, "long_short_return", "ann_mean") * 100,
        "long_short_sharpe": get_metric(col, "long_short_return", "sharpe"),
    })

portfolio_table = pd.DataFrame(table_rows)

def fmt_pct(x):
    return f"{x:.2f}"

def fmt_num(x):
    return f"{x:.2f}"

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Portfolio Sorting Performance over the Full Out-of-Sample Period}")
lines.append(r"\label{tab:portfolio_sorting}")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Model & Top mean & EW mean & Top EW & Excl. bottom EW & Long short mean & Long short Sharpe \\")
lines.append(r"\midrule")

for _, row in portfolio_table.iterrows():
    lines.append(
        f"{row['model']} & "
        f"{fmt_pct(row['top_mean'])} & "
        f"{fmt_pct(row['ew_mean'])} & "
        f"{fmt_pct(row['top_ew'])} & "
        f"{fmt_pct(row['excl_bottom_ew'])} & "
        f"{fmt_pct(row['long_short_mean'])} & "
        f"{fmt_num(row['long_short_sharpe'])} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\vspace{0.5em}")
lines.append(r"\begin{minipage}{0.95\textwidth}")
lines.append(r"\footnotesize")
lines.append(
    r"\textit{Note.} Values are based on weekly five-day market-adjusted log returns. "
    r"Mean returns are annualized and reported in percent. "
    r"EW refers to the equal-weight benchmark across all stocks in the weekly prediction universe. "
    r"Top EW is the annualized spread between the top-quintile portfolio and the equal-weight benchmark. "
    r"Excl. bottom EW is the annualized spread between the portfolio that excludes the bottom quintile and the equal-weight benchmark. "
    r"The long-short portfolio is reported as a ranking diagnostic."
)
lines.append(r"\end{minipage}")
lines.append(r"\end{table}")

latex_table = "\n".join(lines)

with open(OUT_TABLE, "w") as f:
    f.write(latex_table)

print("Saved:", OUT_TABLE)
print()
print(latex_table)
```

    Saved: output/results/portfolio_sorting_table_apa.tex
    
    \begin{table}[H]
    \centering
    \small
    \caption{Portfolio Sorting Performance over the Full Out-of-Sample Period}
    \label{tab:portfolio_sorting}
    \begin{tabular}{lrrrrrr}
    \toprule
    Model & Top mean & EW mean & Top EW & Excl. bottom EW & Long short mean & Long short Sharpe \\
    \midrule
    FE B1 & 1.54 & -3.54 & 5.08 & 1.97 & 13.04 & 0.90 \\
    FE B2 & 1.08 & -3.54 & 4.62 & 2.26 & 13.77 & 0.98 \\
    FE B3 & 1.03 & -3.54 & 4.58 & 2.07 & 12.98 & 0.93 \\
    RF B1 & 4.69 & -3.54 & 8.24 & 2.97 & 20.31 & 1.47 \\
    RF B2 & 4.28 & -3.54 & 7.83 & 3.07 & 20.29 & 1.38 \\
    RF B3 & 4.31 & -3.54 & 7.85 & 2.95 & 19.78 & 1.43 \\
    XGB B1 & 7.20 & -3.54 & 10.75 & 3.02 & 22.97 & 1.66 \\
    XGB B2 & 7.43 & -3.54 & 10.97 & 3.62 & 25.66 & 1.88 \\
    XGB B3 & 3.66 & -3.54 & 7.20 & 3.32 & 20.64 & 1.53 \\
    \bottomrule
    \end{tabular}
    \vspace{0.5em}
    \begin{minipage}{0.95\textwidth}
    \footnotesize
    \textit{Note.} Values are based on weekly five-day market-adjusted log returns. Mean returns are annualized and reported in percent. EW refers to the equal-weight benchmark across all stocks in the weekly prediction universe. Top EW is the annualized spread between the top-quintile portfolio and the equal-weight benchmark. Excl. bottom EW is the annualized spread between the portfolio that excludes the bottom quintile and the equal-weight benchmark. The long-short portfolio is reported as a ranking diagnostic.
    \end{minipage}
    \end{table}



```python
regime_rows = []

for col in PRED_COLS:
    sub = weekly[weekly["pred_col"] == col].sort_values("date")
    for reg in sub["regime"].dropna().unique():
        sub_r = sub[sub["regime"] == reg]
        for series_name in RETURN_SERIES:
            m = performance_metrics(sub_r[series_name])
            m["regime"]      = reg
            m["pred_col"]    = col
            m["series_name"] = series_name
            regime_rows.append(m)

regime_results = pd.DataFrame(regime_rows)
cols_order_r = ["regime", "pred_col", "series_name", "mean_weekly", "ann_mean",
                "weekly_vol", "ann_vol", "sharpe", "hit_rate", "max_drawdown", "n_weeks"]
regime_results = regime_results[cols_order_r]
regime_results.to_csv(OUT_REGIME, index=False)
print("Saved:", OUT_REGIME)
print(regime_results.head(10).round(4).to_string())
```

    Saved: output/results/portfolio_regime_results.csv
      regime    pred_col                        series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0   calm  pred_fe_b1                         top_return      -0.0012   -0.0622      0.0149   0.1071 -0.5810    0.4928       -0.3279      207
    1   calm  pred_fe_b1                      bottom_return      -0.0025   -0.1290      0.0167   0.1207 -1.0695    0.4493       -0.4124      207
    2   calm  pred_fe_b1              exclude_bottom_return      -0.0014   -0.0729      0.0128   0.0925 -0.7884    0.5121       -0.3092      207
    3   calm  pred_fe_b1                equal_weight_return      -0.0016   -0.0841      0.0129   0.0928 -0.9058    0.4928       -0.3280      207
    4   calm  pred_fe_b1                  long_short_return       0.0013    0.0668      0.0149   0.1072  0.6234    0.5314       -0.1213      207
    5   calm  pred_fe_b1             top_minus_equal_weight       0.0004    0.0219      0.0079   0.0568  0.3849    0.5314       -0.0949      207
    6   calm  pred_fe_b1  exclude_bottom_minus_equal_weight       0.0002    0.0112      0.0023   0.0166  0.6742    0.5169       -0.0183      207
    7  other  pred_fe_b1                         top_return       0.0005    0.0267      0.0167   0.1202  0.2222    0.5049       -0.3015      204
    8  other  pred_fe_b1                      bottom_return      -0.0016   -0.0850      0.0183   0.1316 -0.6458    0.4853       -0.4354      204
    9  other  pred_fe_b1              exclude_bottom_return       0.0003    0.0149      0.0139   0.1003  0.1486    0.5196       -0.2914      204



```python
fig, ax = plt.subplots(figsize=(10, 5))

for col, color, label in [
    ("pred_rf_b3",  "#4e79a7", "RF Block 3 top quintile"),
    ("pred_xgb_b3", "#f28e2b", "XGB Block 3 top quintile"),
]:
    sub = weekly[weekly["pred_col"] == col].sort_values("date")
    cumret = np.exp(sub["top_return"].cumsum()) - 1
    ax.plot(sub["date"], cumret, color=color, lw=1.5, label=label)

# Equal-weight benchmark is the same across model-block combinations
sub_ew = weekly[weekly["pred_col"] == "pred_rf_b3"].sort_values("date")
cumret_ew = np.exp(sub_ew["equal_weight_return"].cumsum()) - 1
ax.plot(
    sub_ew["date"], cumret_ew,
    color="#bab0ac", lw=1.5, linestyle="--",
    label="Equal-weight benchmark"
)

ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.set_title("Cumulative Market-Adjusted Return of Top-Quintile Portfolios", fontsize=11)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative market-adjusted return")
ax.legend(frameon=False)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_LONG, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", FIG_LONG)
```


    
![png](06_portfolio_evaluation_files/06_portfolio_evaluation_9_0.png)
    


    Saved: output/figures/portfolio_cumret_long_only.png



```python
fig, ax = plt.subplots(figsize=(10, 5))

for col, color, ls, label in [
    ("pred_rf_b3",  "#4e79a7", "-",  "RF Block 3 top minus bottom"),
    ("pred_xgb_b3", "#f28e2b", "-",  "XGB Block 3 top minus bottom"),
    ("pred_rf_b3",  "#4e79a7", "--", "RF Block 3 exclude bottom vs equal weight"),
    ("pred_xgb_b3", "#f28e2b", "--", "XGB Block 3 exclude bottom vs equal weight"),
]:
    sub = weekly[weekly["pred_col"] == col].sort_values("date")
    series_col = "long_short_return" if ls == "-" else "exclude_bottom_minus_equal_weight"
    cumret = np.exp(sub[series_col].cumsum()) - 1
    ax.plot(sub["date"], cumret, color=color, lw=1.5, linestyle=ls, label=label)

ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.set_title("Cumulative Market-Adjusted Portfolio Spreads", fontsize=11)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative market-adjusted spread")
ax.legend(frameon=False, fontsize=9)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG_SPREAD, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", FIG_SPREAD)
```


    
![png](06_portfolio_evaluation_files/06_portfolio_evaluation_10_0.png)
    


    Saved: output/figures/portfolio_cumret_spreads.png



```python
print("=== Validation Summary ===")
print(f"Prediction panel shape       : {df.shape}")
print(f"OOS weeks                    : {df['date'].nunique()}")
print(f"Model-block combinations     : {len(PRED_COLS)}")
print()
print("--- portfolio_weekly_returns (first 5 rows) ---")
print(pd.read_csv(OUT_WEEKLY).head().to_string())
print()
print("--- portfolio_results (first 5 rows) ---")
print(pd.read_csv(OUT_FULL).head().round(4).to_string())
print()
print("--- portfolio_regime_results (first 5 rows) ---")
print(pd.read_csv(OUT_REGIME).head().round(4).to_string())
```

    === Validation Summary ===
    Prediction panel shape       : (97976, 40)
    OOS weeks                    : 514
    Model-block combinations     : 9
    
    --- portfolio_weekly_returns (first 5 rows) ---
         pred_col        date regime  top_return  bottom_return  equal_weight_return  exclude_bottom_return  long_short_return  top_minus_equal_weight  exclude_bottom_minus_equal_weight  n_top  n_bottom  n_total
    0  pred_fe_b1  2016-01-04   calm    0.004813      -0.015289            -0.005326              -0.002895           0.020101                0.010139                           0.002432     32        31      158
    1  pred_fe_b1  2016-01-11   calm    0.026382       0.014200             0.020345               0.021844           0.012182                0.006037                           0.001499     31        30      153
    2  pred_fe_b1  2016-01-18   calm   -0.012484      -0.016824            -0.000530               0.003512           0.004339               -0.011954                           0.004042     33        32      161
    3  pred_fe_b1  2016-01-25   calm    0.007939       0.006115             0.004151               0.003668           0.001824                0.003788                          -0.000483     32        31      157
    4  pred_fe_b1  2016-02-01   calm    0.005819      -0.013988             0.002245               0.006303           0.019806                0.003574                           0.004058     32        31      155
    
    --- portfolio_results (first 5 rows) ---
         pred_col            series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0  pred_fe_b1             top_return       0.0003    0.0154      0.0159   0.1145  0.1340    0.5214       -0.3279      514
    1  pred_fe_b1          bottom_return      -0.0022   -0.1150      0.0213   0.1537 -0.7483    0.4825       -0.6808      514
    2  pred_fe_b1  exclude_bottom_return      -0.0003   -0.0158      0.0137   0.0987 -0.1597    0.5311       -0.3092      514
    3  pred_fe_b1    equal_weight_return      -0.0007   -0.0354      0.0144   0.1037 -0.3416    0.5078       -0.3498      514
    4  pred_fe_b1      long_short_return       0.0025    0.1304      0.0201   0.1451  0.8984    0.5564       -0.1790      514
    
    --- portfolio_regime_results (first 5 rows) ---
      regime    pred_col            series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0   calm  pred_fe_b1             top_return      -0.0012   -0.0622      0.0149   0.1071 -0.5810    0.4928       -0.3279      207
    1   calm  pred_fe_b1          bottom_return      -0.0025   -0.1290      0.0167   0.1207 -1.0695    0.4493       -0.4124      207
    2   calm  pred_fe_b1  exclude_bottom_return      -0.0014   -0.0729      0.0128   0.0925 -0.7884    0.5121       -0.3092      207
    3   calm  pred_fe_b1    equal_weight_return      -0.0016   -0.0841      0.0129   0.0928 -0.9058    0.4928       -0.3280      207
    4   calm  pred_fe_b1      long_short_return       0.0013    0.0668      0.0149   0.1072  0.6234    0.5314       -0.1213      207

