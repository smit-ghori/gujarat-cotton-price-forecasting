# Gujarat Cotton Price Forecasting

A machine-learning and time-series analysis of Gujarat cotton wholesale prices. The notebook cleans and explores historical AGMARKNET data, engineers market and date features, compares forecasting models, tunes a HistGradientBoosting model, blends it with Ridge regression, and produces district-level forecasts.

The notebook reports coverage across 126 APMC mandis and 30 Gujarat districts. It includes a time-ordered evaluation and an indicative five-month forecast for **21 October 2026 to 20 March 2027**.

## Repository contents

| Path | Description |
| --- | --- |
| `notebooks/gujarat_cotton_price_forecasting.ipynb` | Full analysis, charts, model evaluation, and forecast workflow |
| `src/gujarat_cotton_price_forecasting.py` | Runnable Python export of the notebook analysis |
| `data/gujarat_cotton_10years_historical.csv` | Historical cotton market-price data used by the analysis |
| `outputs/gujarat_cotton_5months_district_forecast.xlsx` | Multi-sheet district forecast workbook |
| `requirements.txt` | Python package dependencies |

## Setup

Python 3.10 or later is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/gujarat_cotton_price_forecasting.ipynb` in Jupyter and run its cells from top to bottom. To run the exported script from the repository root:

```powershell
python src/gujarat_cotton_price_forecasting.py
```

The script writes its refreshed workbook under `outputs/`. Running the notebook also writes the workbook there.

## Modeling and reported results

The analysis compares a default HistGradientBoosting model, tuned HistGradientBoosting and Ridge regressors, a 75%/25% blended ensemble, and a persistence baseline. The notebook's saved evaluation reports the blended ensemble at **R-squared 0.8211**, **RMSE Rs 323.75/quintal**, **MAE Rs 177.10/quintal**, and **MAPE 2.48%**. Results can vary if the input data, package versions, or training period changes.

The committed forecast is a dated model output, not a live quote or financial advice. Its input data and forecast cutoff are from October 2026; refresh the data and rerun the analysis before using it for current decisions. Forecasts are uncertain and should not be treated as guaranteed prices.

## Data

The CSV is the historical data snapshot used by this project and is identified in the notebook as AGMARKNET data. Please consult the data provider's terms and attribution requirements before redistributing or reusing it. This repository does not include an automated data-download workflow; refresh data only through a permitted, official source.

## License

No license has been specified. All rights remain with their respective owners unless a license is added.
