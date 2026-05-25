# Predicting Stock Returns Using Foreign Institutional Flow on the Vietnamese Market

This repository contains the full replication code for the BAM thesis by Viet Nguyen (student ID 642970). The study examines whether foreign institutional trading flow carries predictive information about future stock returns on the Vietnamese equity market, using machine learning models alongside a fixed effects panel regression baseline.

## How to Run

Run the five notebooks in order from 01 to 05. Each notebook picks up where the previous one left off. The two Python scripts (06 and 07) can be run at any point after notebook 01 finishes.

All output files are saved automatically into the output folder, which is already in place.

## Data

The three raw input files go inside the **Dataset** folder.

**sheet_final_raw.csv** contains weekly closing prices for VN100 constituent stocks, sourced from Datastream.

**cafef_foreignflow_222tickers_2014_2026_FIXED.csv** contains weekly foreign buy and sell volume for each stock, scraped from CafeF.

**VNindex_raw.csv** contains raw VN-Index level data used to compute market-wide volatility regimes.

## Notebooks

**01 data preparation** loads the raw price and foreign flow data, aligns dates, handles missing values, and computes the core features for each stock and week.

**02 panel construction** takes the feature panel and applies additional filters, including removing weeks with insufficient coverage and dropping delisted tickers. The result is a cleaned panel ready for modelling.

**03 forecasting** trains and evaluates three model types across three feature blocks using an expanding walk-forward window. The model types are a Fixed Effects panel regression, a Random Forest, and XGBoost. All predictions and intermediate results are saved for use in the next step.

**04 evaluation** reads the saved predictions and computes all hypothesis test results reported in the thesis. This includes predictability tests (H2), feature importance via SHAP values (H3), and regime-conditional performance (H4). Charts and summary tables are saved into the output folder.

**05 descriptive statistics** generates the descriptive statistics tables shown in the thesis, including a LaTeX-formatted table ready for inclusion in the written report.

## Figure Scripts

**06 figure.py** produces a publication-style chart showing the volatility regime classification over the full sample period.

**07 CV fold figure.py** produces a diagram illustrating the expanding window cross-validation design used in the forecasting step.

## Output Structure

The output folder contains three subfolders. The **intermediate** subfolder holds files passed between notebooks and is repopulated on each run. The **results** subfolder holds the final CSVs and tables reported in the thesis. The **figures** subfolder holds all charts and figures.

## Requirements

The code runs on Python 3.9 or later. The main packages are pandas, numpy, scikit-learn, xgboost, shap, statsmodels, and matplotlib. A standard Anaconda environment includes all of these.
