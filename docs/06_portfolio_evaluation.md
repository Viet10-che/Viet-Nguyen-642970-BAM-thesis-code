# Notebook 6: Portfolio-Sorting Economic Evaluation

**Input:** `output/intermediate/predictions.csv`, `output/intermediate/regime_labels.csv`

**Output:** `output/intermediate/portfolio_weekly_returns.csv`, `output/results/portfolio_results.csv`, `output/results/portfolio_regime_results.csv`

**Purpose:** evaluate the economic meaning of model forecasts by sorting stocks into predicted top and bottom quintiles.
Portfolio returns are gross market-adjusted log returns, and the long-short spread is reported only as a ranking diagnostic.


```python
import numpy as np
import pandas as pd
import os

PRED_PATH    = "output/intermediate/predictions.csv"
REGIME_PATH  = "output/intermediate/regime_labels.csv"

OUT_WEEKLY   = "output/intermediate/portfolio_weekly_returns.csv"
OUT_FULL     = "output/results/portfolio_results.csv"
OUT_REGIME   = "output/results/portfolio_regime_results.csv"

os.makedirs("output/intermediate", exist_ok=True)
os.makedirs("output/results",      exist_ok=True)

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

# Build weekly portfolio returns by sorting stocks into top and bottom prediction quintiles
for col in PRED_COLS:
    top_flag    = col + "_top"
    bottom_flag = col + "_bottom"

    # Calculate portfolio returns separately for each OOS week
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
# Portfolio strategy
RETURN_SERIES = [
    "top_return",
    "bottom_return",
    "exclude_bottom_return",
    "equal_weight_return",
    "long_short_return",
    "top_minus_equal_weight",
    "exclude_bottom_minus_equal_weight",
]

# Compute maximum drawdown from cumulative weekly log returns
def max_drawdown(log_ret_series):
    cumlog = log_ret_series.cumsum()
    wealth = np.exp(cumlog)
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return dd.min()

# Compute annualized return, risk, Sharpe, hit rate, and drawdown
def performance_metrics(series):
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
# Store full-sample portfolio performance results
full_rows = []

# Calculate performance metrics for each model and return series
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
# Store portfolio performance results separately by market regime
regime_rows = []

# Calculate performance metrics for each model, regime, and return series
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
print(regime_results.head(27).round(4).to_string())
```

    Saved: output/results/portfolio_regime_results.csv
        regime    pred_col                        series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0     calm  pred_fe_b1                         top_return      -0.0012   -0.0622      0.0149   0.1071 -0.5810    0.4928       -0.3279      207
    1     calm  pred_fe_b1                      bottom_return      -0.0025   -0.1290      0.0167   0.1207 -1.0695    0.4493       -0.4124      207
    2     calm  pred_fe_b1              exclude_bottom_return      -0.0014   -0.0729      0.0128   0.0925 -0.7884    0.5121       -0.3092      207
    3     calm  pred_fe_b1                equal_weight_return      -0.0016   -0.0841      0.0129   0.0928 -0.9058    0.4928       -0.3280      207
    4     calm  pred_fe_b1                  long_short_return       0.0013    0.0668      0.0149   0.1072  0.6234    0.5314       -0.1213      207
    5     calm  pred_fe_b1             top_minus_equal_weight       0.0004    0.0219      0.0079   0.0568  0.3849    0.5314       -0.0949      207
    6     calm  pred_fe_b1  exclude_bottom_minus_equal_weight       0.0002    0.0112      0.0023   0.0166  0.6742    0.5169       -0.0183      207
    7    other  pred_fe_b1                         top_return       0.0005    0.0267      0.0167   0.1202  0.2222    0.5049       -0.3015      204
    8    other  pred_fe_b1                      bottom_return      -0.0016   -0.0850      0.0183   0.1316 -0.6458    0.4853       -0.4354      204
    9    other  pred_fe_b1              exclude_bottom_return       0.0003    0.0149      0.0139   0.1003  0.1486    0.5196       -0.2914      204
    10   other  pred_fe_b1                equal_weight_return      -0.0001   -0.0049      0.0141   0.1016 -0.0481    0.4902       -0.3201      204
    11   other  pred_fe_b1                  long_short_return       0.0021    0.1117      0.0169   0.1219  0.9162    0.6029       -0.1710      204
    12   other  pred_fe_b1             top_minus_equal_weight       0.0006    0.0316      0.0093   0.0668  0.4726    0.5441       -0.1281      204
    13   other  pred_fe_b1  exclude_bottom_minus_equal_weight       0.0004    0.0198      0.0024   0.0171  1.1613    0.6275       -0.0186      204
    14  stress  pred_fe_b1                         top_return       0.0029    0.1488      0.0161   0.1161  1.2818    0.6117       -0.1194      103
    15  stress  pred_fe_b1                      bottom_return      -0.0028   -0.1463      0.0325   0.2343 -0.6245    0.5437       -0.4553      103
    16  stress  pred_fe_b1              exclude_bottom_return       0.0007    0.0383      0.0148   0.1070  0.3580    0.5922       -0.1857      103
    17  stress  pred_fe_b1                equal_weight_return       0.0000    0.0018      0.0175   0.1265  0.0146    0.5728       -0.2480      103
    18  stress  pred_fe_b1                  long_short_return       0.0057    0.2951      0.0318   0.2290  1.2888    0.5146       -0.1790      103
    19  stress  pred_fe_b1             top_minus_equal_weight       0.0028    0.1469      0.0153   0.1101  1.3342    0.5146       -0.0796      103
    20  stress  pred_fe_b1  exclude_bottom_minus_equal_weight       0.0007    0.0365      0.0044   0.0316  1.1544    0.5631       -0.0279      103
    21    calm  pred_fe_b2                         top_return      -0.0015   -0.0756      0.0145   0.1049 -0.7210    0.5217       -0.3496      207
    22    calm  pred_fe_b2                      bottom_return      -0.0030   -0.1538      0.0167   0.1205 -1.2763    0.4348       -0.4708      207
    23    calm  pred_fe_b2              exclude_bottom_return      -0.0013   -0.0668      0.0128   0.0921 -0.7254    0.5169       -0.3061      207
    24    calm  pred_fe_b2                equal_weight_return      -0.0016   -0.0841      0.0129   0.0928 -0.9058    0.4928       -0.3280      207
    25    calm  pred_fe_b2                  long_short_return       0.0015    0.0782      0.0139   0.1004  0.7794    0.5459       -0.1248      207
    26    calm  pred_fe_b2             top_minus_equal_weight       0.0002    0.0085      0.0077   0.0556  0.1524    0.5266       -0.1090      207



```python
print("Validation summary")
print(f"Weekly portfolio returns: {weekly.shape}")
print(f"Full performance results: {full_results.shape}")
print(f"Regime performance results: {regime_results.shape}")
print(f"OOS weeks: {weekly['date'].nunique()}")
print(f"Model-block combinations: {weekly['pred_col'].nunique()}")
```

    Validation Summary
    Prediction panel shape       : (97976, 40)
    OOS weeks                    : 514
    Model-block combinations     : 9
    
    portfolio_weekly_returns (first 5 rows)
         pred_col        date regime  top_return  bottom_return  equal_weight_return  exclude_bottom_return  long_short_return  top_minus_equal_weight  exclude_bottom_minus_equal_weight  n_top  n_bottom  n_total
    0  pred_fe_b1  2016-01-04   calm    0.004813      -0.015289            -0.005326              -0.002895           0.020101                0.010139                           0.002432     32        31      158
    1  pred_fe_b1  2016-01-11   calm    0.026382       0.014200             0.020345               0.021844           0.012182                0.006037                           0.001499     31        30      153
    2  pred_fe_b1  2016-01-18   calm   -0.012484      -0.016824            -0.000530               0.003512           0.004339               -0.011954                           0.004042     33        32      161
    3  pred_fe_b1  2016-01-25   calm    0.007939       0.006115             0.004151               0.003668           0.001824                0.003788                          -0.000483     32        31      157
    4  pred_fe_b1  2016-02-01   calm    0.005819      -0.013988             0.002245               0.006303           0.019806                0.003574                           0.004058     32        31      155
    
    portfolio_results (first 5 rows)
         pred_col            series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0  pred_fe_b1             top_return       0.0003    0.0154      0.0159   0.1145  0.1340    0.5214       -0.3279      514
    1  pred_fe_b1          bottom_return      -0.0022   -0.1150      0.0213   0.1537 -0.7483    0.4825       -0.6808      514
    2  pred_fe_b1  exclude_bottom_return      -0.0003   -0.0158      0.0137   0.0987 -0.1597    0.5311       -0.3092      514
    3  pred_fe_b1    equal_weight_return      -0.0007   -0.0354      0.0144   0.1037 -0.3416    0.5078       -0.3498      514
    4  pred_fe_b1      long_short_return       0.0025    0.1304      0.0201   0.1451  0.8984    0.5564       -0.1790      514
    
    portfolio_regime_results (first 5 rows)
      regime    pred_col            series_name  mean_weekly  ann_mean  weekly_vol  ann_vol  sharpe  hit_rate  max_drawdown  n_weeks
    0   calm  pred_fe_b1             top_return      -0.0012   -0.0622      0.0149   0.1071 -0.5810    0.4928       -0.3279      207
    1   calm  pred_fe_b1          bottom_return      -0.0025   -0.1290      0.0167   0.1207 -1.0695    0.4493       -0.4124      207
    2   calm  pred_fe_b1  exclude_bottom_return      -0.0014   -0.0729      0.0128   0.0925 -0.7884    0.5121       -0.3092      207
    3   calm  pred_fe_b1    equal_weight_return      -0.0016   -0.0841      0.0129   0.0928 -0.9058    0.4928       -0.3280      207
    4   calm  pred_fe_b1      long_short_return       0.0013    0.0668      0.0149   0.1072  0.6234    0.5314       -0.1213      207

