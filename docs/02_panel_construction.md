# Notebook 2: Panel Construction

**Input:** `features_panel.csv`
**Output:** `model_ready.csv`
**Purpose:** Convert the daily feature panel to a weekly forecasting panel, apply
coverage filters, and define the development and out-of-sample periods used in modeling.


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

## 2.1 Weekly Sampling

The daily feature panel is converted to a weekly forecasting panel by selecting
one forecast origin per calendar week — the first available trading day of that week.
This handles Vietnamese market holidays naturally without hardcoding Monday as origin.

The sample window is restricted to **2014-01-01 → 2025-12-31** based on forecast-origin
date. Observations with missing target are dropped at this stage.


```python
# 2.1.1 Load and restrict sample window
df = pd.read_csv("output/intermediate/features_panel.csv")
df["date"] = pd.to_datetime(df["date"])

df = df[
    (df["date"] >= "2014-01-01") &
    (df["date"] <= "2025-12-31")
].copy()

df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
```


```python
# 2.1.2 Build weekly origins
# First available trading day of each ISO week

market_dates = pd.DataFrame({"date": sorted(df["date"].dropna().unique())})
iso = market_dates["date"].dt.isocalendar()
market_dates["iso_year"] = iso.year
market_dates["iso_week"] = iso.week

weekly_origins = (
    market_dates
    .groupby(["iso_year", "iso_week"], as_index=False)["date"]
    .min()
    .rename(columns={"date": "weekly_origin"})
)

weekly_dates = set(weekly_origins["weekly_origin"])
```


```python
# 2.1.3 Filter to weekly panel and drop missing target
df_weekly = df[df["date"].isin(weekly_dates)].copy()
df_weekly = df_weekly[df_weekly["target_5d_mktadj"].notna()].copy()
df_weekly = df_weekly.sort_values(["ticker", "date"]).reset_index(drop=True)
```


```python
# 2.1.4 Validation
print("Shape:          ", df_weekly.shape)
print("Tickers:        ", df_weekly["ticker"].nunique())
print("Date range:     ", df_weekly["date"].min().date(), "->", df_weekly["date"].max().date())
print("Weekly origins: ", df_weekly["date"].nunique())
print("Missing target: ", df_weekly["target_5d_mktadj"].isna().sum())

# Tickers in daily panel but missing from weekly panel
tickers_daily  = set(df["ticker"].dropna().unique())
tickers_weekly = set(df_weekly["ticker"].dropna().unique())
print("\nTickers dropped:", sorted(tickers_daily - tickers_weekly))

# Weekly coverage distribution
print("\nWeekly ticker coverage:")
print(df_weekly.groupby("date")["ticker"].nunique().describe())
```

    Shape:           (114610, 35)
    Tickers:         219
    Date range:      2014-01-02 -> 2025-12-29
    Weekly origins:  622
    Missing target:  0
    
    Tickers dropped: ['HAI', 'HVG']
    
    Weekly ticker coverage:
    count    622.000000
    mean     184.260450
    std       21.628015
    min      132.000000
    25%      162.000000
    50%      195.000000
    75%      200.000000
    max      210.000000
    Name: ticker, dtype: float64


## 2.2 Model-Ready Dataset

The weekly panel is filtered to the columns needed for modeling. Predictors are
organized into three nested blocks. Rows with any missing value across target or
predictors are dropped. Weeks with fewer than 100 usable tickers are removed to
avoid unstable forecasting periods.

The sample is split into two periods:
- **Development (2014–2015):** reserved for hyperparameter tuning
- **OOS (2016–2025):** reserved for expanding-window forecasting


```python
# 2.2.1 Define target and feature blocks
TARGET_COL = "target_5d_mktadj"

block1_features = [
    "ret_1d_lag", "ret_5d_lag", "momentum_1m", "volatility_1m", "log_size"
]
block2_features = block1_features + ["turnover_5d", "amihud_5d"]
block3_features = block2_features + ["f_buy_5d", "f_sell_5d"]

feature_sets = {
    "b1": block1_features,
    "b2": block2_features,
    "b3": block3_features
}
```


```python
# 2.2.2 Filter to model-ready columns and drop missing values
keep_cols = ["ticker", "date", TARGET_COL] + block3_features

df_model = df_weekly[keep_cols].dropna().copy()


# 2.2.3 Coverage filter
# Remove weeks with fewer than 100 tickers to avoid unstable forecasting periods
MIN_TICKERS = 100

weekly_coverage = (
    df_model.groupby("date")["ticker"]
    .nunique()
    .reset_index(name="n_tickers")
)

valid_weeks   = weekly_coverage[weekly_coverage["n_tickers"] >= MIN_TICKERS]["date"]
dropped_weeks = weekly_coverage[weekly_coverage["n_tickers"] <  MIN_TICKERS].copy()

df_model = df_model[df_model["date"].isin(valid_weeks)].copy()


# 2.2.4 Sample period flags
df_model["sample_period"] = "oos"
df_model.loc[df_model["date"] < "2016-01-01", "sample_period"] = "development"

df_model = df_model.sort_values(["ticker", "date"]).reset_index(drop=True)
```


```python
# 2.2.5 Winsorization and rank normalization
# Applied after column selection and NaN drop, before saving

# Winsorize features and target at 1%/99% each week
# Clips extreme cross-sectional values; protects FE (OLS) from outlier influence
cols_to_winsorize = block3_features + [TARGET_COL]
df_model[cols_to_winsorize] = df_model.groupby("date")[cols_to_winsorize].transform(
    lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
)

# Save descriptive_ready for Notebook 5
# Post-winsorization, pre-rank-normalization: preserves original scale for descriptive statistics
df_model.to_csv("output/intermediate/descriptive_ready.csv", index=False)
print("Saved: output/intermediate/descriptive_ready.csv")

# Cross-sectional rank normalization of features only (not target)
# Following Gu et al. (2020): transform each feature to [-0.5, 0.5] within each week
# Target (target_5d_mktadj) is retained in its original scale to preserve economic meaning
def rank_normalize(x):
    r = x.rank(method="average", na_option="keep")
    n = r.count()
    if n <= 1:
        return x * 0
    return (r - 1) / (n - 1) - 0.5

df_model[block3_features] = df_model.groupby("date")[block3_features].transform(rank_normalize)

print("Winsorization + rank normalization applied.")
print(f"Target range after winsorization: {df_model[TARGET_COL].min():.4f} -> {df_model[TARGET_COL].max():.4f}")
print("Feature range after normalization (expected near [-0.5, 0.5]):")
print(df_model[block3_features].agg(["min", "max"]).round(4).to_string())
```

    Saved: output/intermediate/descriptive_ready.csv
    Winsorization + rank normalization applied.
    Target range after winsorization: -0.3761 -> 0.2984
    Feature range after normalization (expected near [-0.5, 0.5]):
         ret_1d_lag  ret_5d_lag  momentum_1m  volatility_1m  log_size  turnover_5d  amihud_5d  f_buy_5d  f_sell_5d
    min     -0.4975     -0.4975      -0.4975        -0.4975   -0.4975      -0.4975    -0.4975   -0.4829    -0.4874
    max      0.4975      0.4975       0.4975         0.4975    0.4975       0.4975     0.4975    0.4975     0.4975



```python
# 2.2.6 Save and validation
df_model.to_csv("output/intermediate/model_ready.csv", index=False)

if len(dropped_weeks) > 0:
    dropped_weeks.to_csv("output/intermediate/dropped_weeks_log.csv", index=False)

dev = df_model[df_model["sample_period"] == "development"]
oos = df_model[df_model["sample_period"] == "oos"]

print("Shape:    ", df_model.shape)
print("Tickers:  ", df_model["ticker"].nunique())
print("Missing:  ", df_model.isna().sum().sum())

print("\nDevelopment:", dev["date"].min().date(), "->", dev["date"].max().date(),
      "| rows:", len(dev))
print("OOS:        ", oos["date"].min().date(), "->", oos["date"].max().date(),
      "| rows:", len(oos))

print("\nWeekly coverage:")
print(df_model.groupby("date")["ticker"].nunique().describe())

print("\nDropped weeks (< 100 tickers):")
print(dropped_weeks if len(dropped_weeks) > 0 else "None")

print("\nSaved: model_ready.csv |", df_model.shape)
```

    Shape:     (111858, 13)
    Tickers:   219
    Missing:   0
    
    Development: 2014-02-17 -> 2015-12-28 | rows: 13882
    OOS:         2016-01-04 -> 2025-12-29 | rows: 97976
    
    Weekly coverage:
    count    610.000000
    mean     183.373770
    std       22.106281
    min      131.000000
    25%      161.000000
    50%      195.000000
    75%      200.000000
    max      210.000000
    Name: ticker, dtype: float64
    
    Dropped weeks (< 100 tickers):
              date  n_tickers
    37  2014-11-03          6
    538 2024-07-08          3
    560 2024-12-09          7
    561 2024-12-16          2
    563 2024-12-30          4
    
    Saved: model_ready.csv | (111858, 13)



```python
# ============================================================
# OVERTHINK VALIDATION — delete before submission
# Traces model_ready → features_panel → raw source files
# ============================================================
import random, textwrap
import numpy as np
import pandas as pd

print("=" * 65)
print("OVERTHINK VALIDATION — full chain check")
print("=" * 65)

ERRORS = []

# ── 0. reload all files fresh ────────────────────────────────
raw   = pd.read_csv("output/intermediate/features_panel.csv",  parse_dates=["date"])
ready = pd.read_csv("output/intermediate/model_ready.csv",     parse_dates=["date"])

PRICE_FILE   = "Dataset/sheet_final_raw.csv"
FLOW_FILE    = "Dataset/cafef_foreignflow_222tickers_2014_2026.csv"
VNINDEX_FILE = "Dataset/VNindex_raw.csv"

# ── 1. all dates are ISO-week origins ────────────────────────
print("\n[1] Verifying all dates are ISO-week origins ...")
raw_trading_days = raw["date"].drop_duplicates().sort_values()

def iso_week_origin(d):
    yr, wk, _ = d.isocalendar()
    candidates = raw_trading_days[
        raw_trading_days.apply(lambda x: x.isocalendar()[:2] == (yr, wk))
    ]
    return candidates.min() if len(candidates) else pd.NaT

bad_dates = []
for d in ready["date"].unique():
    origin = iso_week_origin(d)
    if pd.isna(origin) or d != origin:
        bad_dates.append((d, origin))
if bad_dates:
    ERRORS.append(f"[1] {len(bad_dates)} dates are NOT iso-week origins: {bad_dates[:5]}")
    print(f"  ✗ FAIL — {len(bad_dates)} bad dates")
else:
    print(f"  ✓ all {ready['date'].nunique()} unique dates are valid iso-week origins")

# ── 2. coverage filter ────────────────────────────────────────
print(f"\n[2] Every week has >= {MIN_TICKERS} tickers ...")
week_counts = ready.groupby("date")["ticker"].nunique()
under = week_counts[week_counts < MIN_TICKERS]
if len(under):
    ERRORS.append(f"[2] {len(under)} weeks below {MIN_TICKERS} tickers")
    print(f"  ✗ FAIL — {len(under)} weeks below threshold")
else:
    print(f"  ✓ all {len(week_counts)} weeks >= {MIN_TICKERS} "
          f"(min={week_counts.min()}, max={week_counts.max()})")

# ── 3. no NaN in features + target ───────────────────────────
print(f"\n[3] No NaN in features + target ...")
check_cols = block3_features + [TARGET_COL]
bad_cols   = ready[check_cols].isna().sum()
bad_cols   = bad_cols[bad_cols > 0]
if len(bad_cols):
    ERRORS.append(f"[3] NaN found: {bad_cols.to_dict()}")
    print(f"  ✗ FAIL — {bad_cols.to_dict()}")
else:
    print(f"  ✓ zero NaN across {len(check_cols)} columns, {len(ready):,} rows")

# ── 4. sample_period flags ────────────────────────────────────
print(f"\n[4] sample_period flags ...")
dev_mask  = ready["sample_period"] == "development"
oos_mask  = ready["sample_period"] == "oos"
dev_dates = ready.loc[dev_mask, "date"]
oos_dates = ready.loc[oos_mask, "date"]
flag_ok   = True

bad_vals = set(ready["sample_period"].unique()) - {"development", "oos"}
if bad_vals:
    ERRORS.append(f"[4] unexpected sample_period values: {bad_vals}")
    flag_ok = False
if ready["sample_period"].isna().any():
    ERRORS.append("[4] NaN in sample_period")
    flag_ok = False
if len(dev_dates) and len(oos_dates) and dev_dates.max() >= oos_dates.min():
    ERRORS.append(f"[4] dev/oos overlap: dev max={dev_dates.max().date()}, "
                  f"oos min={oos_dates.min().date()}")
    flag_ok = False
if flag_ok:
    print(f"  ✓ flags clean, no overlap")
    print(f"    dev: {dev_dates.min().date()} → {dev_dates.max().date()} "
          f"({dev_mask.sum():,} rows)")
    print(f"    oos: {oos_dates.min().date()} → {oos_dates.max().date()} "
          f"({oos_mask.sum():,} rows)")
else:
    print(f"  ✗ FAIL")

# ── 5. no duplicate (ticker, date) ───────────────────────────
print(f"\n[5] No duplicate (ticker, date) pairs ...")
dups = ready.duplicated(subset=["ticker", "date"]).sum()
if dups:
    ERRORS.append(f"[5] {dups} duplicate (ticker, date) pairs")
    print(f"  ✗ FAIL — {dups} duplicates")
else:
    print(f"  ✓ no duplicates across {len(ready):,} rows")

# ── 6. dropped weeks do not appear in model_ready ────────────
print(f"\n[6] Dropped weeks log spot-check ...")
try:
    dropped = pd.read_csv("output/intermediate/dropped_weeks_log.csv", parse_dates=["date"])
    overlap = set(dropped["date"]) & set(ready["date"])
    if overlap:
        ERRORS.append(f"[6] {len(overlap)} dropped weeks still in model_ready")
        print(f"  ✗ FAIL — {len(overlap)} dropped weeks leaked in")
    else:
        print(f"  ✓ none of {len(dropped)} dropped weeks appear in model_ready")
        if "reason" in dropped.columns:
            print(f"    reasons: {dropped['reason'].value_counts().to_dict()}")
except FileNotFoundError:
    print("  ⚠ dropped_weeks_log.csv not found — skipping")

# ── 7. column set completeness ────────────────────────────────
print(f"\n[7] Column set completeness ...")
expected = {"ticker", "date", TARGET_COL, "sample_period"} | set(block3_features)
missing  = expected - set(ready.columns)
extra    = set(ready.columns) - expected
if missing:
    ERRORS.append(f"[7] Missing columns: {missing}")
    print(f"  ✗ FAIL — missing: {missing}")
else:
    print(f"  ✓ all expected columns present")
if extra:
    print(f"  ⚠ extra columns: {extra}")

# ── 8. random row cross-check: model_ready ↔ features_panel ──
# NOTE: after winsorization and rank normalization, feature values in
# model_ready intentionally differ from features_panel. This check
# verifies structural integrity only (ticker/date presence), not values.
print(f"\n[8] Structural cross-check: model_ready → features_panel ...")
N_SAMPLE   = 30
sample_idx = random.sample(range(len(ready)), min(N_SAMPLE, len(ready)))
row_errors = []
for i in sample_idx:
    row = ready.iloc[i]
    d, tk = row["date"], row["ticker"]
    src = raw[(raw["date"] == d) & (raw["ticker"] == tk)]
    if len(src) == 0:
        row_errors.append(f"({tk}, {d.date()}) missing in features_panel")
if row_errors:
    ERRORS.extend([f"[8] {e}" for e in row_errors])
    print(f"  ✗ FAIL — {len(row_errors)} rows missing from features_panel")
else:
    print(f"  ✓ {N_SAMPLE} sampled (ticker, date) pairs found in features_panel")

# ══════════════════════════════════════════════════════════════
# RAW FILE CHECKS — traces features_panel back to source CSVs
# ══════════════════════════════════════════════════════════════

# ── 9. parse Datastream raw ───────────────────────────────────
print(f"\n[9] Parsing Datastream raw file ...")
ds_raw    = pd.read_csv(PRICE_FILE, sep=";", header=None, low_memory=False)
code_row  = ds_raw.iloc[1]
data_part = ds_raw.iloc[3:].copy().reset_index(drop=True)
data_part.columns = ds_raw.iloc[0]

METRIC_MAP = {
    "P": "close", "PH": "high", "PL": "low", "PO": "open",
    "VO": "volume", "VA": "trading_value",
    "MV": "market_cap", "NOSH": "shares_outstanding", "RI": "total_return_index",
}
col_meta = {}
for col_idx, code in enumerate(code_row):
    if pd.isna(code) or not str(code).startswith("VT:"):
        continue
    inner = str(code).replace("VT:", "")
    if "(" not in inner:
        continue
    tk_code = inner.split("(")[0]
    mt_code = inner.split("(")[1].rstrip(")")
    if mt_code in METRIC_MAP:
        col_meta[col_idx] = (tk_code, METRIC_MAP[mt_code])

data_part["_date"] = pd.to_datetime(
    data_part.iloc[:, 0].astype(str).str.strip(),
    dayfirst=True, format="%d/%m/%y", errors="coerce"
)
data_part = data_part.dropna(subset=["_date"])
print(f"  parsed {len(data_part):,} date-rows, {len(col_meta)} ticker-metric cols")

def get_ds_value(ticker, metric, date):
    for idx, (tk, mt) in col_meta.items():
        if tk == ticker and mt == metric:
            row = data_part[data_part["_date"] == date]
            if len(row) == 0:
                return None
            val = row.iloc[0, idx]
            if pd.isna(val) or str(val).strip() in ("", "NA", "nan"):
                return np.nan
            return float(str(val).replace(",", "."))
    return None  # ticker not in Datastream

# ── 10. random cross-check: features_panel ↔ Datastream ──────
print(f"\n[10] Random cross-check: features_panel → Datastream raw ...")
DS_COLS   = ["close", "high", "low", "open", "volume",
             "trading_value", "market_cap", "shares_outstanding",
             "total_return_index"]
ds_sample = raw.sample(min(25, len(raw)), random_state=42)
ds_errors = []
for _, row in ds_sample.iterrows():
    d, tk = row["date"], row["ticker"]
    for metric in DS_COLS:
        v_panel = row[metric]
        v_raw   = get_ds_value(tk, metric, d)
        if v_raw is None:
            continue  # ticker not in Datastream — skip
        if pd.isna(v_panel) and np.isnan(v_raw):
            continue
        if pd.isna(v_panel) != np.isnan(v_raw):
            ds_errors.append(f"({tk}, {d.date()}) {metric}: panel={v_panel} raw={v_raw} [NaN]")
            continue
        if not (pd.isna(v_panel) or np.isnan(v_raw)):
            if not np.isclose(float(v_panel), float(v_raw), rtol=1e-4, atol=1e-6):
                ds_errors.append(f"({tk}, {d.date()}) {metric}: "
                                 f"panel={v_panel:.4f} ≠ raw={v_raw:.4f}")
if ds_errors:
    ERRORS.extend([f"[10] {e}" for e in ds_errors])
    print(f"  ✗ FAIL — {len(ds_errors)} Datastream mismatches:")
    for e in ds_errors[:8]: print(f"    {e}")
else:
    print(f"  ✓ 25 rows — Datastream raw values match features_panel")

# ── 11. random cross-check: features_panel ↔ CafeF flow ──────
print(f"\n[11] Random cross-check: features_panel → CafeF raw ...")
flow_raw  = pd.read_csv(FLOW_FILE, parse_dates=["date"])
FLOW_COLS = {
    "buy_vol": "buy_vol", "sell_vol": "sell_vol", "net_vol": "net_vol",
    "buy_value_bil_vnd": "buy_value_bil_vnd",
    "sell_value_bil_vnd": "sell_value_bil_vnd",
    "net_value_bil_vnd": "net_value_bil_vnd",
}
flow_errors = []
for _, row in ds_sample.iterrows():  # reuse same 25 rows
    d, tk = row["date"], row["ticker"]
    fr = flow_raw[(flow_raw["date"] == d) & (flow_raw["ticker"] == tk)]
    if len(fr) == 0:
        for fp_col in FLOW_COLS:
            if not pd.isna(row[fp_col]):
                flow_errors.append(f"({tk}, {d.date()}) {fp_col}: "
                                   f"panel={row[fp_col]} but not in CafeF")
        continue
    fr = fr.iloc[0]
    for fp_col, raw_col in FLOW_COLS.items():
        v_p, v_r = row[fp_col], fr[raw_col]
        if pd.isna(v_p) and pd.isna(v_r):
            continue
        if pd.isna(v_p) != pd.isna(v_r):
            flow_errors.append(f"({tk}, {d.date()}) {fp_col}: NaN mismatch")
            continue
        if not np.isclose(float(v_p), float(v_r), rtol=1e-4, atol=1e-6):
            flow_errors.append(f"({tk}, {d.date()}) {fp_col}: "
                               f"panel={v_p:.4f} ≠ raw={v_r:.4f}")
if flow_errors:
    ERRORS.extend([f"[11] {e}" for e in flow_errors])
    print(f"  ✗ FAIL — {len(flow_errors)} CafeF mismatches:")
    for e in flow_errors[:8]: print(f"    {e}")
else:
    print(f"  ✓ 25 rows — CafeF raw values match features_panel")

# ── 12. VNIndex spot-check ────────────────────────────────────
print(f"\n[12] VNIndex spot-check: vnindex_clean → VNindex_raw ...")
try:
    vn_raw = pd.read_csv(VNINDEX_FILE, sep=";", skiprows=3,
                         header=None, names=["date_str", "price_str"])
    vn_raw["date"]  = pd.to_datetime(vn_raw["date_str"].str.strip(),
                                     dayfirst=True, format="%d/%m/%y", errors="coerce")
    vn_raw["close"] = vn_raw["price_str"].str.replace(",", ".").astype(float)
    vn_raw = vn_raw.dropna(subset=["date", "close"])

    vn_clean  = pd.read_csv("output/intermediate/vnindex_clean.csv", parse_dates=["date"])
    vn_sample = vn_clean.sample(min(15, len(vn_clean)), random_state=7)
    vn_errors = []
    for _, r in vn_sample.iterrows():
        match = vn_raw[vn_raw["date"] == r["date"]]
        if len(match) == 0:
            vn_errors.append(f"  {r['date'].date()} not found in VNIndex raw")
            continue
        if not np.isclose(float(r["vnindex"]), float(match.iloc[0]["close"]), rtol=1e-4):
            vn_errors.append(f"  {r['date'].date()}: "
                             f"clean={r['vnindex']:.2f} ≠ raw={match.iloc[0]['close']:.2f}")
    if vn_errors:
        ERRORS.extend([f"[12] {e}" for e in vn_errors])
        print(f"  ✗ FAIL — {len(vn_errors)} VNIndex mismatches:")
        for e in vn_errors: print(e)
    else:
        print(f"  ✓ 15 VNIndex dates — close prices match raw exactly")
except Exception as ex:
    print(f"  ⚠ VNIndex check skipped: {ex}")

# ── FINAL VERDICT ─────────────────────────────────────────────
print("\n" + "=" * 65)
if ERRORS:
    print(f"  ✗ VALIDATION FAILED — {len(ERRORS)} issue(s):")
    for e in ERRORS:
        print(textwrap.indent(str(e), "    "))
else:
    print("  ✓ ALL 12 CHECKS PASSED — full chain raw → model_ready clean")
print("=" * 65)

```

    =================================================================
    OVERTHINK VALIDATION — full chain check
    =================================================================
    
    [1] Verifying all dates are ISO-week origins ...
      ✓ all 610 unique dates are valid iso-week origins
    
    [2] Every week has >= 100 tickers ...
      ✓ all 610 weeks >= 100 (min=131, max=210)
    
    [3] No NaN in features + target ...
      ✓ zero NaN across 10 columns, 111,858 rows
    
    [4] sample_period flags ...
      ✓ flags clean, no overlap
        dev: 2014-02-17 → 2015-12-28 (13,882 rows)
        oos: 2016-01-04 → 2025-12-29 (97,976 rows)
    
    [5] No duplicate (ticker, date) pairs ...
      ✓ no duplicates across 111,858 rows
    
    [6] Dropped weeks log spot-check ...
      ✓ none of 5 dropped weeks appear in model_ready
    
    [7] Column set completeness ...
      ✓ all expected columns present
    
    [8] Structural cross-check: model_ready → features_panel ...
      ✓ 30 sampled (ticker, date) pairs found in features_panel
    
    [9] Parsing Datastream raw file ...
      parsed 3,201 date-rows, 1987 ticker-metric cols
    
    [10] Random cross-check: features_panel → Datastream raw ...
      ✓ 25 rows — Datastream raw values match features_panel
    
    [11] Random cross-check: features_panel → CafeF raw ...
      ✓ 25 rows — CafeF raw values match features_panel
    
    [12] VNIndex spot-check: vnindex_clean → VNindex_raw ...
      ✓ 15 VNIndex dates — close prices match raw exactly
    
    =================================================================
      ✓ ALL 12 CHECKS PASSED — full chain raw → model_ready clean
    =================================================================

