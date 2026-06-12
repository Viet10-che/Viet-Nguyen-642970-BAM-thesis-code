# Liquidity, Foreign Flows, and Machine Learning in Frontier Market Return Prediction: Evidence from Vietnam

This repository contains the full replication code for the BAM thesis by Viet Nguyen (student ID 642970). The study examines whether foreign institutional trading flow and liquidity carry predictive information about future stock returns on the Vietnamese equity market, using machine learning models alongside a fixed effects panel regression baseline.

## How to Run

Run notebooks 01 to 06 in order. Each notebook reads from the outputs of the previous one. Notebooks 02 to 06 can be run without the licensed datasets as long as the intermediate files in the output folder are present. Figure scripts 07 to 11 can be run independently once the relevant intermediate outputs exist.

All output files are saved automatically into the output folder.

## Data

Three raw input files are required inside the **Dataset** folder.

**sheet_final_raw.csv** contains weekly price and market data for VN100 constituent stocks, sourced from LSEG Datastream. This file is not included in the repository due to licensing restrictions. Please contact the author at 642970bn@student.eur.nl if you need this file to run Notebook 1.

**cafef_foreignflow_222tickers_2014_2026.csv** contains weekly foreign buy and sell volume for each stock, scraped from CafeF.

**VNindex_raw.csv** contains raw VN-Index level data used to compute market-wide volatility regimes, sourced from LSEG Datastream. This file is also not included due to licensing restrictions; please contact the author.

Notebooks 02 to 06 do not require these raw files and can be run using the intermediate outputs already saved in the output folder.

## Notebooks

**01 data preparation** loads and merges the raw price and foreign flow data, constructs the five-day-ahead market-adjusted return target, and engineers all predictive features across three blocks.

**02 panel construction** converts the daily feature panel to a weekly forecasting panel, applies coverage filters, and defines the development and out-of-sample periods.

**03 forecasting** runs all model training and prediction: hyperparameter tuning, expanding walk-forward forecasting for nine model–block combinations, SHAP value computation, and regime labelling.

**04 evaluation** reads the saved predictions and tests all eight hypotheses, including predictability tests, SHAP-based feature importance, and regime-conditional performance.

**05 descriptive statistics** computes and exports the sample description and summary statistics tables used in the thesis.

**06 portfolio evaluation** evaluates the economic significance of model forecasts through portfolio sorting and computes regime-conditional portfolio performance.

## Figure Scripts

**07 regime\_volatility.py** plots the rolling volatility regime classification over the full sample period.

**08 CV fold figure.py** produces the expanding window cross-validation diagram used in the methodology section.

**09 figure\_portfolio\_long\_only.py** plots cumulative returns for the top-quintile and exclude-bottom long-only strategies for RF and XGB Block 3.

**10 figure\_portfolio\_spreads.py** plots the cumulative long-short spread as a ranking diagnostic for RF and XGB Block 3.

**11 figure\_portfolio\_xgb\_b2\_appendix.py** plots the XGB Block 2 portfolio performance included in the appendix.

## Output Structure

The output folder contains three subfolders. The **intermediate** subfolder holds files passed between notebooks and is repopulated on each run. The **results** subfolder holds the final CSVs and tables reported in the thesis. The **figures** subfolder holds all charts and figures.

## Requirements

The code runs on Python 3.9 or later. The main packages are pandas, numpy, scikit-learn, xgboost, shap, statsmodels, and matplotlib. A standard Anaconda environment includes all of these.

## License

The code in this repository is released under the MIT License. You are free to use, modify, and distribute it with attribution. The datasets are not covered by this license; please refer to the original data providers (LSEG Datastream and CafeF) for their respective terms of use.
