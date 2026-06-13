# Notebook 1: Data Preparation

**Input:** `Dataset/sheet_final_raw.csv`, `Dataset/cafef_foreignflow_222tickers_2014_2026.csv`, `Dataset/VNindex_raw.csv`

**Output:** `features_panel.csv`, `vnindex_clean.csv`

**Purpose:** Clean and merge raw data sources, construct the prediction target, and engineer all features used in modeling.


```python
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
```

## 1.1 Data Cleaning and Merging

We construct the base daily panel from two sources: Datastream (price and market data)
and CafeF (foreign investor flow). VNIndex is loaded separately as a market-level series.

The Datastream file is exported in wide format with metadata rows. We extract the
time-series data, parse ticker-variable codes, reshape to long panel format, then merge
with foreign flow data using a left join on (ticker, date).

Zero values in foreign flow are treated as valid observations. Missing rows indicate
non-trading days and remain as NaN after the merge.


```python
# File paths
PRICE_FILE   = "Dataset/sheet_final_raw.csv"
FLOW_FILE    = "Dataset/cafef_foreignflow_222tickers_2014_2026.csv"
VNINDEX_FILE = "Dataset/VNindex_raw.csv"


def parse_ds_code(col):
    """Parse Datastream column code into (ticker, variable) pair.
    Example: 'VT:AAA(RI)' -> ('AAA', 'total_return_index')
    """
    col = str(col).strip()
    m = re.match(r"VT:([A-Z0-9]+)\(([^)]+)\)", col)
    if not m:
        return None, None

    ticker = m.group(1)
    suffix = m.group(2)

    mapping = {
        "RI":    "total_return_index",
        "P":     "close",
        "PO":    "open",
        "PH":    "high",
        "PL":    "low",
        "VO":    "volume",
        "VA":    "trading_value",
        "MV":    "market_cap",
        "NOSH":  "shares_outstanding",
        "NOSHFR":"foreign_holdings"
    }

    return ticker, mapping.get(suffix)


# 1.1.1 Load and clean Datastream
# Row 0 = variable codes, row 1 = currency labels, row 2+ = data
ds_raw   = pd.read_csv(PRICE_FILE, sep=";", engine="python", encoding="utf-8-sig")
code_row = ds_raw.iloc[0].copy()
ds       = ds_raw.iloc[2:].copy().reset_index(drop=True)
ds.columns = code_row
ds = ds.rename(columns={ds.columns[0]: "date"})

date_col   = pd.to_datetime(ds.iloc[:, 0], format="%d/%m/%y", errors="coerce")
valid_mask = date_col.notna()
date_col   = date_col.loc[valid_mask].reset_index(drop=True)
body       = ds.iloc[valid_mask.values, 1:].reset_index(drop=True).copy()

body = body.fillna("").astype(str)
body = body.apply(lambda col: col.str.replace(",", ".", regex=False).str.strip())
body = body.replace({"": np.nan, "NA": np.nan, "nan": np.nan, "None": np.nan})
body = body.apply(pd.to_numeric, errors="coerce")

ds_clean = pd.concat([date_col.rename("date"), body], axis=1)


# 1.1.2 Reshape Datastream to long panel
rows = []
for j in range(1, ds_clean.shape[1]):
    ticker, var = parse_ds_code(ds_clean.columns[j])
    if ticker is None or var is None:
        continue
    rows.append(pd.DataFrame({
        "date":     ds_clean.iloc[:, 0].values,
        "value":    ds_clean.iloc[:, j].values,
        "ticker":   ticker,
        "variable": var
    }))

ds_long = pd.concat(rows, ignore_index=True)

ds_panel = (
    ds_long
    .pivot_table(index=["ticker", "date"], columns="variable",
                 values="value", aggfunc="first")
    .reset_index()
)
ds_panel.columns.name = None
ds_panel = ds_panel.sort_values(["ticker", "date"]).reset_index(drop=True)


# 1.1.3 Load and clean foreign flow (CafeF)
flow = pd.read_csv(FLOW_FILE)
flow.columns = flow.columns.str.lower().str.strip()
flow["ticker"] = flow["ticker"].astype(str).str.upper().str.strip()
flow["date"]   = pd.to_datetime(flow["date"], errors="coerce")

keep_cols = [
    "ticker", "date",
    "buy_value_bil_vnd", "sell_value_bil_vnd",
    "buy_vol", "sell_vol",
    "net_value_bil_vnd", "net_vol",
    "foreign_room_left", "foreign_own_pct",
    "close_price"
]
flow = flow[[c for c in keep_cols if c in flow.columns]].copy()
flow = flow.dropna(subset=["ticker", "date"])
flow = flow.drop_duplicates(subset=["ticker", "date"])


# 1.1.4 Merge Datastream + foreign flow
core_ds_cols = [
    "close", "open", "high", "low",
    "volume", "trading_value",
    "market_cap", "shares_outstanding",
    "total_return_index"
]

final_df = ds_panel.merge(flow, on=["ticker", "date"], how="left")
final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)
final_df = final_df.drop_duplicates(subset=["ticker", "date"])

existing_core = [c for c in core_ds_cols if c in final_df.columns]
final_df = final_df[final_df[existing_core].notna().any(axis=1)].copy()


# 1.1.5 Load and clean VNIndex
vn_raw = pd.read_csv(VNINDEX_FILE, sep=";", engine="python", encoding="utf-8-sig")
vn = vn_raw.iloc[2:].copy()
vn.columns = ["date", "vnindex"]
vn["date"] = pd.to_datetime(vn["date"], format="%d/%m/%y", errors="coerce")
vn["vnindex"] = (
    vn["vnindex"]
    .fillna("").astype(str)
    .str.replace(",", ".", regex=False).str.strip()
    .replace({"": np.nan, "NA": np.nan, "nan": np.nan, "None": np.nan})
)
vn["vnindex"] = pd.to_numeric(vn["vnindex"], errors="coerce")
vn = vn.dropna(subset=["date"]).reset_index(drop=True)

# Save VNIndex — used in later notebooks
vn.to_csv("output/intermediate/vnindex_clean.csv", index=False)
```


```python
# 1.1.6 Validation
print("Shape:           ", final_df.shape)
print("Tickers:         ", final_df["ticker"].nunique())
print("Date range:      ", final_df["date"].min().date(), "->", final_df["date"].max().date())
print("Duplicates:      ", final_df.duplicated(["ticker", "date"]).sum())

flow_check = ["buy_value_bil_vnd", "sell_value_bil_vnd", "buy_vol", "sell_vol"]
existing_flow = [c for c in flow_check if c in final_df.columns]
print("Foreign flow match rate:", round(final_df[existing_flow].notna().any(axis=1).mean(), 4))

print("\nVNIndex:")
print(vn.shape, "|", vn["date"].min().date(), "->", vn["date"].max().date())
```

    Shape:            (640954, 20)
    Tickers:          221
    Date range:       2014-01-01 -> 2026-04-08
    Duplicates:       0
    Foreign flow match rate: 0.9111
    
    VNIndex:
    (3201, 2) | 2014-01-01 -> 2026-04-08


## 1.2 Target Construction

The prediction target is the 5-day-ahead market-adjusted return, constructed under
a strict no-look-ahead-bias framework.

Daily log returns are computed from the Total Return Index within each ticker group.
Forward returns at time `t` are defined over the window `[t+1, t+5]` using trading-day
ordering via `groupby().shift(-k)`, not calendar-day subtraction.

Non-trading days are removed using the joint condition of missing `trading_value` and
`volume`. The final target is:

`target_5d_mktadj = fwd_stock_ret_5d - fwd_mkt_ret_5d`

Future information is used only to construct the label. All predictors will be built
using information available up to time `t` only.


```python
# 1.2.1 Filter to actual trading days
# Trading days identified by non-missing trading_value and volume
df = final_df[
    final_df["trading_value"].notna() &
    final_df["volume"].notna()
].copy()
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)


# 1.2.2 Daily stock log return (within ticker)
ticker_grp = df.groupby("ticker")

df["stock_log_ret"] = (
    ticker_grp["total_return_index"]
    .transform(lambda x: np.log(x / x.shift(1)))
)


# 1.2.3 Daily market log return
vn["mkt_log_ret"] = np.log(vn["vnindex"] / vn["vnindex"].shift(1))

# Align VNIndex to trading days in stock panel only
trading_dates = pd.DataFrame({"date": sorted(df["date"].unique())})
vn = trading_dates.merge(vn[["date", "vnindex", "mkt_log_ret"]], on="date", how="left")
vn = vn.sort_values("date").reset_index(drop=True)


# 1.2.4 5-day forward market return (computed on market date series)
vn["fwd_mkt_ret_5d"] = sum(vn["mkt_log_ret"].shift(-k) for k in range(1, 6))

# Merge market variables into stock panel
df = df.merge(
    vn[["date", "vnindex", "mkt_log_ret", "fwd_mkt_ret_5d"]],
    on="date",
    how="left"
)


# 1.2.5 5-day forward stock return (within ticker, trading-day ordering)
ticker_grp = df.groupby("ticker")

df["fwd_stock_ret_5d"] = sum(ticker_grp["stock_log_ret"].shift(-k) for k in range(1, 6))


# 1.2.6 Final target
df["target_5d_mktadj"] = df["fwd_stock_ret_5d"] - df["fwd_mkt_ret_5d"]
```


```python
# 1.2.7 Validation
print("Shape:          ", df.shape)
print("Tickers:        ", df["ticker"].nunique())
print("Date range:     ", df["date"].min().date(), "->", df["date"].max().date())
print("Missing target: ", df["target_5d_mktadj"].isna().sum())

# Missing target by ticker — expected: last 5 rows per ticker + HAI/HVG (delisted)
missing_by_ticker = (
    df[df["target_5d_mktadj"].isna()]
    .groupby("ticker").size()
    .sort_values(ascending=False)
)
print("\nTop 10 tickers with missing target:")
print(missing_by_ticker.head(10))

# All last-5-rows should be missing (no future data available)
tail_missing = (
    df.sort_values(["ticker", "date"])
      .groupby("ticker")
      .tail(5)["target_5d_mktadj"]
      .isna().mean()
)
print("\nMissing rate in last 5 rows per ticker (expected 1.0):", tail_missing)

# Sanity check: fwd_mkt_ret_5d matches manual recalculation
mkt = df[["date", "mkt_log_ret", "fwd_mkt_ret_5d"]].drop_duplicates().sort_values("date")
print("Max abs diff (expected ~0):",
      (mkt["fwd_mkt_ret_5d"] -
       sum(mkt["mkt_log_ret"].shift(-k) for k in range(1, 6))).abs().max())
```

    Shape:           (568513, 26)
    Tickers:         221
    Date range:      2014-01-02 -> 2026-04-08
    Missing target:  4903
    
    Top 10 tickers with missing target:
    ticker
    HAI    2100
    HVG    1708
    AAA       5
    PHR       5
    PME       5
    PNJ       5
    POM       5
    POW       5
    PPC       5
    PSH       5
    dtype: int64
    
    Missing rate in last 5 rows per ticker (expected 1.0): 1.0
    Max abs diff (expected ~0): 0.0


## 1.3 Feature Engineering

Three blocks of predictive features are constructed, all using only information
available up to time `t` to ensure no look-ahead bias.

- **Block 1** — baseline return dynamics, momentum, volatility, and firm size
- **Block 2** — liquidity measures (turnover and Amihud illiquidity)
- **Block 3** — foreign investor activity (normalized buy and sell flows)

This nested structure allows the incremental contribution of each information set
to be evaluated in the forecasting step.


```python
# 1.3.1 Compute raw columns and define ticker_grp
df["log_size"]  = np.log(df.groupby("ticker")["market_cap"].shift(1))  # groupby prevents cross-ticker contamination at ticker boundaries
df["turnover"]  = df["volume"] / df["shares_outstanding"]
df["amihud"]    = (np.abs(df["stock_log_ret"]) / df["trading_value"]).replace([np.inf, -np.inf], np.nan)  # replace Inf from trading_value=0 edge cases
df["f_buy"]     = (df["buy_value_bil_vnd"] / df["trading_value"]).replace([np.inf, -np.inf], np.nan)
df["f_sell"]    = (df["sell_value_bil_vnd"] / df["trading_value"]).replace([np.inf, -np.inf], np.nan)

# Define once after all columns exist
ticker_grp = df.groupby("ticker")
```


```python
# 1.3.2 Block 1: Baseline features
# 1-day lagged return: captures short-term reversal (bid-ask bounce, microstructure noise)
df["ret_1d_lag"] = ticker_grp["stock_log_ret"].shift(1)

# 5-day cumulative lagged return: captures weekly reversal pattern
df["ret_5d_lag"] = (
    ticker_grp["stock_log_ret"].shift(1).rolling(5).sum()
    .reset_index(level=0, drop=True)
)

# 1-month momentum: cumulative return over t-21 to t-6, skipping the most recent week
df["momentum_1m"] = (
    ticker_grp["stock_log_ret"].shift(6).rolling(16).sum()
    .reset_index(level=0, drop=True)
)

# Realized volatility: rolling 20-day standard deviation of daily returns
df["volatility_1m"] = (
    ticker_grp["stock_log_ret"].shift(1).rolling(20).std()
    .reset_index(level=0, drop=True)
)
```

### Block 2 — Liquidity Features

Turnover (volume / shares outstanding) reflects trading activity. The Amihud
illiquidity ratio (|return| / trading value) captures price impact. Both are
averaged over 5 trading days to reduce noise.


```python
# 1.3.3 Block 2: Liquidity features

# 5-day average turnover (volume / shares outstanding)
df["turnover_5d"] = (
    ticker_grp["turnover"].shift(1).rolling(5).mean()
    .reset_index(level=0, drop=True)
)

# 5-day average Amihud illiquidity ratio (|return| / trading value)
df["amihud_5d"] = (
    ticker_grp["amihud"].shift(1).rolling(5).mean()
    .reset_index(level=0, drop=True)
)
```

### Block 3 — Foreign Flow Features

Foreign buy and sell values are normalized by trading value to ensure
comparability across stocks. A lag-1 is applied before the 5-day average
to ensure no same-day information enters the predictor.


```python
# 1.3.4 Block 3: Foreign flow features
# 5-day average normalized foreign buy value (foreign buy / total trading value)
df["f_buy_5d"] = (
    ticker_grp["f_buy"].shift(1).rolling(5).mean()
    .reset_index(level=0, drop=True)
)

# 5-day average normalized foreign sell value (foreign sell / total trading value)
df["f_sell_5d"] = (
    ticker_grp["f_sell"].shift(1).rolling(5).mean()
    .reset_index(level=0, drop=True)
)
```


```python
# 1.3.5 Drop intermediates, save and validate
df = df.drop(columns=["turnover", "amihud", "f_buy", "f_sell"])
df.to_csv("output/intermediate/features_panel.csv", index=False)

feature_cols = [
    "ret_1d_lag", "ret_5d_lag", "momentum_1m", "volatility_1m", "log_size",
    "turnover_5d", "amihud_5d", "f_buy_5d", "f_sell_5d"
]
print("Missing rates:")
print(df[feature_cols].isna().mean().sort_values())

print("\nForeign flow max values:")
print(df[["f_buy_5d", "f_sell_5d"]].max())
print("Share > 10 (expected 0):", (df[["f_buy_5d", "f_sell_5d"]] > 10).any(axis=1).mean())

print("\nSaved: features_panel.csv |", df.shape)
```

    Missing rates:
    log_size         0.000389
    turnover_5d      0.001944
    ret_1d_lag       0.007469
    ret_5d_lag       0.009009
    amihud_5d        0.009009
    volatility_1m    0.014788
    momentum_1m      0.015173
    f_buy_5d         0.027813
    f_sell_5d        0.027813
    dtype: float64
    
    Foreign flow max values:
    f_buy_5d     0.003720
    f_sell_5d    0.021603
    dtype: float64
    Share > 10 (expected 0): 0.0
    
    Saved: features_panel.csv | (568513, 35)

