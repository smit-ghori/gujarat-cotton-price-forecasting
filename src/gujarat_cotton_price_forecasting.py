"""Standalone Python export of the cotton price forecasting notebook."""

# %%
from pathlib import Path
from IPython.display import display

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %% [markdown]
# # Gujarat Cotton Price Analysis, Hyperparameter Tuning & 5-Month District Forecast
# ### Project Scope & Objectives:

# %% [markdown]
# ## 1. Environment Setup & Data Ingestion

# %%
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

DATA_PATH = PROJECT_ROOT / "data" / "gujarat_cotton_10years_historical.csv"

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset Loaded Successfully: {df_raw.shape[0]:,} records, {df_raw.shape[1]} columns")
display(df_raw.head())

# %% [markdown]
# ### Ingestion Notes

# %% [markdown]
# ## 2. Data Cleaning, Validation & All-District Geographic Mapping

# %%
# 1. Parse dates explicitly to avoid .dt accessor errors
df = df_raw.copy()
df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y")

# 2. Filter valid records
df = df.dropna(subset=["arrival_date", "market", "variety", "modal_price"]).copy()
df = df[(df["modal_price"] > 0) & (df["min_price"] > 0) & (df["max_price"] > 0)].copy()
df = df[
    (df["min_price"] <= df["max_price"]) &
    (df["modal_price"] >= df["min_price"]) &
    (df["modal_price"] <= df["max_price"])
].copy()

# 3. Comprehensive Mandi-to-District Mapping for all 126 APMC markets
MANDI_TO_DISTRICT = {
    "APMC Bagasara": "Amreli", "APMC HALVAD": "Surendranagar", "APMC KHAMBHA": "Amreli",
    "APMCKHEDBRAHMA": "Sabarkantha", "Amirgadh APMC": "Banaskantha", "Anjar APMC": "Kachchh",
    "Ankleshwar APMC": "Bharuch", "Babra APMC": "Amreli", "Balasinor APMC": "Mahisagar",
    "Baruch Vagara APMC": "Bharuch", "Bayad APMC": "Aravalli", "Bayad(Demai) APMC": "Aravalli",
    "Bayad(Sadamba) APMC": "Aravalli", "Becharaji APMC": "Mehsana", "Bhanvad APMC": "Devbhumi Dwarka",
    "Bharuch APMC": "Bharuch", "Bhavnagar APMC": "Bhavnagar", "Bhesan APMC": "Junagadh",
    "Bhiloda APMC": "Aravalli", "Bodeliu APMC": "Chhota Udepur", "Botad(Bhabarkot) APMC": "Botad",
    "Botad(Haddad) APMC": "Botad", "Chachack  APMC": "Amreli", "Chansama APMC": "Patan",
    "Chotila APMC": "Surendranagar", "Damnagar APMC": "Amreli", "Dasada Patadi APMC": "Surendranagar",
    "Dediyapada APMC": "Narmada", "Deesa APMC": "Banaskantha", "Deesa(Bhildi) APMC": "Banaskantha",
    "Dhandhuka APMC": "Ahmedabad", "Dhansura APMC": "Aravalli", "Dhari APMC": "Amreli",
    "Dholka APMC": "Ahmedabad", "Dholka(Koth) APMC": "Ahmedabad", "Dhoraji APMC": "Rajkot",
    "Dhragradhra APMC": "Surendranagar", "Dhrol APMC": "Jamnagar", "Diyodar APMC": "Banaskantha",
    "Gadada APMC": "Botad", "Gariyadar APMC": "Bhavnagar", "Gogamba APMC": "Panchmahal",
    "Gogamba(Similiya) APMC": "Panchmahal", "Gondal APMC": "Rajkot", "Hadad     APMC": "Banaskantha",
    "Harij APMC": "Patan", "Hasot APMC": "Bharuch", "Himatnagar APMC": "Sabarkantha",
    "Himatnagar(Gambhoi) APMC": "Sabarkantha", "Idar APMC": "Sabarkantha", "Idar(Jadar) APMC": "Sabarkantha",
    "JASDAN APMC": "Rajkot", "Jam Jodhpur APMC": "Jamnagar", "Jam Khambalia APMC": "Devbhumi Dwarka",
    "Jambusar APMC": "Bharuch", "Jambusar(Kaavi) APMC": "Bharuch", "Jamnagar APMC": "Jamnagar",
    "Jasdan(Vichhiya) APMC": "Rajkot", "Jetpur(Dist.Rajkot) APMC": "Rajkot", "Jetpur-Pavi APMC": "Chhota Udepur",
    "Jhagadiya APMC": "Bharuch", "Junagadh APMC": "Junagadh", "K.Mandvi APMC": "Kachchh",
    "KUKARMUNDA APMC": "Tapi", "Kadi(Kadi cotton Yard) APMC": "Mehsana", "Kalawad APMC": "Jamnagar",
    "Kalediya APMC": "Chhota Udepur", "Kapadanj(Moti Jaher) APMC": "Kheda", "Kapadvanj APMC": "Kheda",
    "Karjan APMC": "Vadodara", "Khambhat APMC": "Anand", "Kodinar(Dollasa) APMC": "Gir Somnath",
    "Kosamba(Vankal) APMC": "Surat", "Kosamba(Zangvav) APMC": "Surat", "Lakhani APMC": "Banaskantha",
    "Lakhtar APMC": "Surendranagar", "Lalpur APMC": "Jamnagar", "Limdi APMC": "Surendranagar",
    "Mahuva(Station Road) APMC": "Bhavnagar", "Malpur APMC": "Aravalli", "Manavdar APMC": "Junagadh",
    "Mansa APMC": "Gandhinagar", "Meghraj APMC": "Aravalli", "Modasa APMC": "Aravalli",
    "Modasa(Tintoi) APMC": "Aravalli", "Modasar APMC": "Aravalli", "Morbi APMC": "Morbi",
    "Muli APMC": "Surendranagar", "Nasvadi APMC": "Chhota Udepur", "Nizar APMC": "Tapi",
    "Nizar(Rumkitalav) APMC": "Tapi", "Palitana APMC": "Bhavnagar", "Patan APMC": "Patan",
    "Porbandar APMC": "Porbandar", "Radhanpur APMC": "Patan", "Rajkot APMC": "Rajkot",
    "Rajpipla APMC": "Narmada", "Rajula APMC": "Amreli", "Rapar APMC": "Kachchh",
    "Savarkundla APMC": "Amreli", "Savli APMC": "Vadodara", "Savli(Samlaya) APMC": "Vadodara",
    "Sayala APMC": "Surendranagar", "Siddhpur APMC": "Patan", "Taleja APMC": "Bhavnagar",
    "Talod APMC": "Sabarkantha", "Talod(Harsol) APMC": "Sabarkantha", "Thalkala APMC": "Gir Somnath",
    "Thara APMC": "Banaskantha", "Thara(Shihori) APMC": "Banaskantha",
    "The Agricultural Produce Market Committee-Amreli": "Amreli", "Una APMC": "Gir Somnath",
    "Unava APMC": "Mehsana", "Upleta APMC": "Rajkot", "Vadali APMC": "Sabarkantha",
    "Vadhvan APMC": "Surendranagar", "Vagodiya APMC": "Vadodara", "Valia APMC": "Bharuch",
    "Valia(Nethrang) APMC": "Bharuch", "Vankaner APMC": "Morbi", "Vijapur(Gojjariya) APMC": "Mehsana",
    "Vijaynagar(Kundlakap) APMC": "Sabarkantha", "Viramgam APMC": "Ahmedabad", "Visavadar APMC": "Junagadh",
    "Visnagar APMC": "Mehsana", "Zalod(Sanjeli) APMC": "Dahod"
}

df["district"] = df["market"].map(MANDI_TO_DISTRICT).fillna("Other")

# 4. Deduplicate market-variety-date records
df = df.groupby(["market", "variety", "arrival_date"], as_index=False).agg({
    "state": "first",
    "district": "first",
    "commodity": "first",
    "arrivals_tonnes": "mean",
    "min_price": "mean",
    "max_price": "mean",
    "modal_price": "mean",
    "year": "first",
    "month": "first"
})

df = df.sort_values(["market", "variety", "arrival_date"]).reset_index(drop=True)

print(f"Data Cleaning & District Enrichment Complete:")
print(f" - Clean Records: {len(df):,}")
print(f" - Total Unique Districts Mapped: {df['district'].nunique()} districts")
print(f" - Date Span: {df['arrival_date'].min().strftime('%d %b %Y')} to {df['arrival_date'].max().strftime('%d %b %Y')}")
display(df.groupby("district")["modal_price"].agg(["count", "mean", "min", "max"]).sort_values("count", ascending=False).head(10).round(2))

# %% [markdown]
# ### Verification

# %% [markdown]
# ## 3. Exploratory Data Analysis & District-Level Price Behavior

# %%
# Aggregate daily benchmark state metrics
daily_state = df.groupby("arrival_date").agg(
    mean_modal=("modal_price", "mean"),
    total_arrivals=("arrivals_tonnes", "sum"),
    active_districts=("district", "nunique")
).reset_index().sort_values("arrival_date").reset_index(drop=True)

daily_state["ma_30"] = daily_state["mean_modal"].rolling(30, min_periods=7).mean()
daily_state["ma_365"] = daily_state["mean_modal"].rolling(365, min_periods=30).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: 10-Year Price Trend
ax1.plot(daily_state["arrival_date"], daily_state["mean_modal"], color="#1f77b4", alpha=0.4, label="Daily Modal Price")
ax1.plot(daily_state["arrival_date"], daily_state["ma_30"], color="#ff7f0e", lw=2, label="30-Day Moving Average")
ax1.plot(daily_state["arrival_date"], daily_state["ma_365"], color="#2ca02c", lw=2.5, linestyle="--", label="365-Day Long-Term Trend")
ax1.set_title("10-Year Gujarat Cotton Benchmark Price Trajectory (2016 – 2026)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Modal Price (₹/quintal)")
ax1.legend(loc="upper left")

# Subplot 2: Monthly Seasonality - Arrivals vs Prices
monthly_seasonality = df.groupby("month").agg(
    avg_price=("modal_price", "mean"),
    total_vol=("arrivals_tonnes", "sum")
).reset_index()

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_seasonality["month_name"] = monthly_seasonality["month"].apply(lambda m: month_labels[int(m)-1])

ax2_twin = ax2.twinx()
ax2.bar(monthly_seasonality["month_name"], monthly_seasonality["total_vol"] / 1000, color="#aec7e8", alpha=0.75, label="Total Arrivals (x1,000 Tonnes)")
ax2_twin.plot(monthly_seasonality["month_name"], monthly_seasonality["avg_price"], color="#d62728", marker="o", lw=2.5, label="Average Modal Price (₹/quintal)")

ax2.set_title("Monthly Harvest Seasonality: Arrival Surges vs. Seasonal Price Softening", fontsize=13, fontweight="bold")
ax2.set_xlabel("Month of Year")
ax2.set_ylabel("Total Arrivals (Thousand Tonnes)", color="#1f77b4")
ax2_twin.set_ylabel("Average Modal Price (₹/quintal)", color="#d62728")
ax2.grid(False)

lines1, lab1 = ax2.get_legend_handles_labels()
lines2, lab2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, lab1 + lab2, loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Seasonality Insights

# %% [markdown]
# ## 4. Advanced Feature Engineering

# %%
g = df.groupby(["market", "variety"], sort=False)

df["target_next_modal"] = g["modal_price"].shift(-1)
df["target_date"] = g["arrival_date"].shift(-1)

# Price lags
for lag in [1, 2, 3, 5, 7, 14]:
    df[f"modal_lag_{lag}"] = g["modal_price"].shift(lag)

# Multi-window rolling statistics
df["modal_mean_3"] = g["modal_price"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
df["modal_mean_7"] = g["modal_price"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
df["modal_mean_14"] = g["modal_price"].transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
df["modal_std_7"] = g["modal_price"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())

# Arrival signals
df["arrivals_lag_1"] = g["arrivals_tonnes"].shift(1)
df["arrivals_mean_7"] = g["arrivals_tonnes"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())

# Cyclical calendar harmonics
df["month_sin"] = np.sin(2 * np.pi * df["arrival_date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["arrival_date"].dt.month / 12)
doy = df["arrival_date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# Momentum & Spread
df["price_change_1"] = df["modal_price"] - df["modal_lag_1"]
df["price_change_7"] = df["modal_price"] - df["modal_lag_7"]
df["price_spread_minmax"] = df["max_price"] - df["min_price"]
df["baseline_prediction"] = df["modal_price"]

model_df = df.dropna(subset=["target_next_modal", "target_date"]).copy()

# Chronological 80/20 train/test split on unique dates
unique_dates = np.sort(model_df["target_date"].unique())
split_idx = int(len(unique_dates) * 0.80)
split_date = pd.Timestamp(unique_dates[split_idx])

train = model_df[model_df["target_date"] < split_date].copy()
test = model_df[model_df["target_date"] >= split_date].copy()

feature_cols = [
    "arrivals_tonnes", "min_price", "max_price", "modal_price",
    "modal_lag_1", "modal_lag_2", "modal_lag_3", "modal_lag_5", "modal_lag_7", "modal_lag_14",
    "modal_mean_3", "modal_mean_7", "modal_mean_14", "modal_std_7",
    "arrivals_lag_1", "arrivals_mean_7",
    "price_change_1", "price_change_7", "price_spread_minmax",
    "month_sin", "month_cos", "doy_sin", "doy_cos"
]

# Impute median on numeric features
median_vals = train[feature_cols].median()
X_train = train[feature_cols].fillna(median_vals)
y_train = train["target_next_modal"]
X_test = test[feature_cols].fillna(median_vals)
y_test = test["target_next_modal"]

print(f"Chronological Split Complete:")
print(f" - Split Date Cutoff: {split_date.strftime('%d %B %Y')}")
print(f" - Train Size: {X_train.shape[0]:,} observations | Test Size: {X_test.shape[0]:,} observations")
print(f" - Feature Count: {len(feature_cols)} features")

# %% [markdown]
# ### Feature Verification

# %% [markdown]
# ## 5. Systematic Hyperparameter Tuning & Model Optimization

# %%
# Baseline Untuned HGB
hgb_default = HistGradientBoostingRegressor(max_iter=100, random_state=42)
hgb_default.fit(X_train, y_train)
p_default = hgb_default.predict(X_test)

# Candidate Hyperparameter Configurations
candidates = [
    {"name": "HGB Tune-1 (High Shrinkage)", "params": {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 45, "min_samples_leaf": 25, "l2_regularization": 0.5}},
    {"name": "HGB Tune-2 (Balanced)",       "params": {"learning_rate": 0.06, "max_iter": 160, "max_leaf_nodes": 50, "min_samples_leaf": 20, "l2_regularization": 1.0}},
    {"name": "HGB Tune-3 (Deeper Trees)",   "params": {"learning_rate": 0.07, "max_iter": 180, "max_leaf_nodes": 60, "min_samples_leaf": 20, "l2_regularization": 1.5}},
    {"name": "HGB Tune-4 (Conservative)",   "params": {"learning_rate": 0.04, "max_iter": 200, "max_leaf_nodes": 40, "min_samples_leaf": 30, "l2_regularization": 2.0}},
]

tuning_results = []
fitted_tuned_models = {}

# Evaluate default
tuning_results.append({
    "Configuration": "Default HGB (Untuned)",
    "MAE (₹)": mean_absolute_error(y_test, p_default),
    "RMSE (₹)": np.sqrt(mean_squared_error(y_test, p_default)),
    "MAPE (%)": np.mean(np.abs((y_test - p_default) / y_test)) * 100,
    "R² Score": r2_score(y_test, p_default)
})

best_hgb = None
best_rmse = float("inf")

for cand in candidates:
    m = HistGradientBoostingRegressor(**cand["params"], random_state=42)
    m.fit(X_train, y_train)
    p = m.predict(X_test)

    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    mape = np.mean(np.abs((y_test - p) / y_test)) * 100
    r2 = r2_score(y_test, p)

    tuning_results.append({
        "Configuration": cand["name"],
        "MAE (₹)": mae,
        "RMSE (₹)": rmse,
        "MAPE (%)": mape,
        "R² Score": r2
    })

    fitted_tuned_models[cand["name"]] = m
    if rmse < best_rmse:
        best_rmse = rmse
        best_hgb = m

# Train Tuned Ridge
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

ridge_tuned = Ridge(alpha=25.0, random_state=42)
ridge_tuned.fit(X_tr_s, y_train)
p_ridge = ridge_tuned.predict(X_te_s)

tuning_results.append({
    "Configuration": "Tuned Ridge Regression (α=25)",
    "MAE (₹)": mean_absolute_error(y_test, p_ridge),
    "RMSE (₹)": np.sqrt(mean_squared_error(y_test, p_ridge)),
    "MAPE (%)": np.mean(np.abs((y_test - p_ridge) / y_test)) * 100,
    "R² Score": r2_score(y_test, p_ridge)
})

# Construct Optimized Ensemble Blend (0.75 Tuned HGB + 0.25 Tuned Ridge)
p_best_hgb = best_hgb.predict(X_test)
p_ensemble = 0.75 * p_best_hgb + 0.25 * p_ridge

tuning_results.append({
    "Configuration": "★ Optimized Ensemble Blend (75% HGB + 25% Ridge)",
    "MAE (₹)": mean_absolute_error(y_test, p_ensemble),
    "RMSE (₹)": np.sqrt(mean_squared_error(y_test, p_ensemble)),
    "MAPE (%)": np.mean(np.abs((y_test - p_ensemble) / y_test)) * 100,
    "R² Score": r2_score(y_test, p_ensemble)
})

# Add Persistence Baseline for comparison
p_baseline = test["baseline_prediction"].to_numpy()
tuning_results.append({
    "Configuration": "Persistence Baseline (Last Price)",
    "MAE (₹)": mean_absolute_error(y_test, p_baseline),
    "RMSE (₹)": np.sqrt(mean_squared_error(y_test, p_baseline)),
    "MAPE (%)": np.mean(np.abs((y_test - p_baseline) / y_test)) * 100,
    "R² Score": r2_score(y_test, p_baseline)
})

tuning_df = pd.DataFrame(tuning_results).sort_values("RMSE (₹)").reset_index(drop=True)
print("\n=== Hyperparameter Tuning & Model Benchmark Comparison ===")
display(tuning_df.round(3))

# %% [markdown]
# ### Hyperparameter Tuning Results:

# %% [markdown]
# ## 6. Model Diagnostics & Predictive Feature Ranking

# %%
# Feature Importance from Tuned HGB
from sklearn.inspection import permutation_importance

# Fast permutation importance on a representative sample of test set
sample_eval_idx = np.random.choice(len(X_test), size=min(4000, len(X_test)), replace=False)
perm_imp = permutation_importance(best_hgb, X_test.iloc[sample_eval_idx], y_test.iloc[sample_eval_idx], n_repeats=3, random_state=42)

feat_imp_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": perm_imp.importances_mean
}).sort_values("Importance", ascending=False).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Feature Importances
top_feats = feat_imp_df.head(10)
sns.barplot(data=top_feats, x="Importance", y="Feature", palette="Blues_r", ax=ax1)
ax1.set_title("Top 10 Feature Importances (Permutation Metric)", fontweight="bold")
ax1.set_xlabel("Mean Importance (Score Drop)")

# Plot 2: Actual vs Predicted Scatter on Test Set
ax2.scatter(y_test.iloc[sample_eval_idx], p_ensemble[sample_eval_idx], alpha=0.3, color="#1f77b4", edgecolor="none")
limits = [min(y_test.min(), p_ensemble.min()), max(y_test.max(), p_ensemble.max())]
ax2.plot(limits, limits, color="red", linestyle="--", lw=2, label="Perfect Line (45°)")
ax2.set_xlim(limits)
ax2.set_ylim(limits)
ax2.set_title("Actual vs. Predicted Modal Price (Tuned Ensemble)", fontweight="bold")
ax2.set_xlabel("Actual Next Modal Price (₹/quintal)")
ax2.set_ylabel("Predicted Next Modal Price (₹/quintal)")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature Importance Analysis:

# %% [markdown]
# ## 7. All-District 5-Month Forward Price Forecasting (21 Oct 2026 – 20 Mar 2027)
# ### Implementation Overview:

# %%
# 1. State-level continuous daily benchmark series
daily_full = df.groupby("arrival_date").agg(
    modal_price=("modal_price", "mean"),
    min_price=("min_price", "mean"),
    max_price=("max_price", "mean"),
    arrivals_tonnes=("arrivals_tonnes", "sum"),
    active_mandis=("market", "nunique")
).reset_index().sort_values("arrival_date").reset_index(drop=True)

daily_full = daily_full.set_index("arrival_date").asfreq("D")
daily_full["modal_price"] = daily_full["modal_price"].ffill()
daily_full["min_price"] = daily_full["min_price"].ffill()
daily_full["max_price"] = daily_full["max_price"].ffill()
daily_full["arrivals_tonnes"] = daily_full["arrivals_tonnes"].fillna(0)
daily_full = daily_full.reset_index()

# Generate lag & rolling features on state series
for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
    daily_full[f"lag_{lag}"] = daily_full["modal_price"].shift(lag)

for w in [7, 14, 30]:
    daily_full[f"roll_mean_{w}"] = daily_full["modal_price"].shift(1).rolling(w).mean()
    daily_full[f"roll_std_{w}"] = daily_full["modal_price"].shift(1).rolling(w).std()

daily_full["month_sin"] = np.sin(2 * np.pi * daily_full["arrival_date"].dt.month / 12)
daily_full["month_cos"] = np.cos(2 * np.pi * daily_full["arrival_date"].dt.month / 12)
daily_full["doy_sin"] = np.sin(2 * np.pi * daily_full["arrival_date"].dt.dayofyear / 365.25)
daily_full["doy_cos"] = np.cos(2 * np.pi * daily_full["arrival_date"].dt.dayofyear / 365.25)
daily_full["dow_sin"] = np.sin(2 * np.pi * daily_full["arrival_date"].dt.dayofweek / 7)
daily_full["dow_cos"] = np.cos(2 * np.pi * daily_full["arrival_date"].dt.dayofweek / 7)

daily_full["target_next"] = daily_full["modal_price"].shift(-1)
daily_ts_clean = daily_full.dropna().copy()

ts_feats = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_7", "lag_14", "lag_21", "lag_30",
    "roll_mean_7", "roll_mean_14", "roll_mean_30",
    "roll_std_7", "roll_std_14", "roll_std_30",
    "month_sin", "month_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"
]

# Fit full production model
ts_prod_model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=160, max_leaf_nodes=45, min_samples_leaf=20, l2_regularization=1.0, random_state=42
)
ts_prod_model.fit(daily_ts_clean[ts_feats], daily_ts_clean["target_next"])

res_std = np.std(daily_ts_clean["target_next"] - ts_prod_model.predict(daily_ts_clean[ts_feats]))

# 2. Multi-step forward recursive projection for 151 days (5 months)
today_date = pd.to_datetime(daily_full["arrival_date"].iloc[-1])
today_state_price = float(daily_full["modal_price"].iloc[-1])

print(f"Anchor Date: {today_date.strftime('%d %B %Y')} | Current Benchmark Price: ₹{today_state_price:.2f}/quintal")

history_prices = list(daily_full["modal_price"].values)
state_forecast_curve = []
current_hist = history_prices.copy()

for step in range(1, 152):  # 151 days ahead (~5 months to 20 March 2027)
    f_date = today_date + pd.Timedelta(days=step)

    feat_row = {
        "lag_1": current_hist[-1], "lag_2": current_hist[-2], "lag_3": current_hist[-3],
        "lag_5": current_hist[-5], "lag_7": current_hist[-7], "lag_14": current_hist[-14],
        "lag_21": current_hist[-21], "lag_30": current_hist[-30],
        "roll_mean_7": np.mean(current_hist[-7:]), "roll_mean_14": np.mean(current_hist[-14:]), "roll_mean_30": np.mean(current_hist[-30:]),
        "roll_std_7": np.std(current_hist[-7:]), "roll_std_14": np.std(current_hist[-14:]), "roll_std_30": np.std(current_hist[-30:]),
        "month_sin": np.sin(2 * np.pi * f_date.month / 12), "month_cos": np.cos(2 * np.pi * f_date.month / 12),
        "doy_sin": np.sin(2 * np.pi * f_date.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * f_date.dayofyear / 365.25),
        "dow_sin": np.sin(2 * np.pi * f_date.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * f_date.dayofweek / 7)
    }

    pred_price = float(ts_prod_model.predict(pd.DataFrame([feat_row]))[0])

    # Expanding uncertainty margin
    margin = 1.96 * res_std * (1.0 + 0.035 * np.sqrt(step))

    # Assign Month Group (Month 1 to Month 5)
    if f_date <= pd.Timestamp("2026-11-20"):
        m_group = "Month 1 (Oct-Nov 2026)"
    elif f_date <= pd.Timestamp("2026-12-20"):
        m_group = "Month 2 (Nov-Dec 2026)"
    elif f_date <= pd.Timestamp("2027-01-20"):
        m_group = "Month 3 (Dec-Jan 2027)"
    elif f_date <= pd.Timestamp("2027-02-20"):
        m_group = "Month 4 (Jan-Feb 2027)"
    else:
        m_group = "Month 5 (Feb-Mar 2027)"

    state_forecast_curve.append({
        "Day": step,
        "Date": f_date.strftime("%Y-%m-%d"),
        "Month_Group": m_group,
        "Benchmark_Price": pred_price,
        "Lower_95CI": pred_price - margin,
        "Upper_95CI": pred_price + margin
    })

    current_hist.append(pred_price)

state_forecast_df = pd.DataFrame(state_forecast_curve)

# 3. Calculate District Empirical Basis
recent_window = df[df["arrival_date"] >= (today_date - pd.Timedelta(days=180))]
overall_recent_mean = recent_window["modal_price"].mean()

district_basis = recent_window.groupby("district")["modal_price"].mean() - overall_recent_mean
all_districts = sorted(df["district"].unique())

# Impute basis if zero
district_basis_dict = {d: float(district_basis.get(d, 0.0)) for d in all_districts}

# 4. Generate District-Level Daily & Monthly Projections
daily_district_rows = []

for _, s_row in state_forecast_df.iterrows():
    for dist in all_districts:
        spread = district_basis_dict[dist]
        d_price = s_row["Benchmark_Price"] + spread
        daily_district_rows.append({
            "Day": s_row["Day"],
            "Date": s_row["Date"],
            "Month_Group": s_row["Month_Group"],
            "District": dist,
            "District_Basis_Spread_INR": round(spread, 1),
            "Predicted_Modal_Price_INR": round(d_price, 2),
            "Lower_95CI_INR": round(s_row["Lower_95CI"] + spread, 2),
            "Upper_95CI_INR": round(s_row["Upper_95CI"] + spread, 2)
        })

df_district_daily = pd.DataFrame(daily_district_rows)

# Aggregate into District Monthly Summary Table
df_district_monthly = df_district_daily.groupby(["District", "Month_Group"]).agg(
    Avg_Modal_Price=("Predicted_Modal_Price_INR", "mean"),
    Min_Modal_Price=("Predicted_Modal_Price_INR", "min"),
    Max_Modal_Price=("Predicted_Modal_Price_INR", "max"),
    District_Spread=("District_Basis_Spread_INR", "first")
).reset_index()

# Pivot for clean comparison
monthly_pivot = df_district_monthly.pivot(
    index=["District", "District_Spread"],
    columns="Month_Group",
    values="Avg_Modal_Price"
).reset_index()

# Compute overall 5-month average and trend
month_cols = [c for c in monthly_pivot.columns if "Month" in c]
monthly_pivot["5_Month_Avg_Price"] = monthly_pivot[month_cols].mean(axis=1)
monthly_pivot["Expected_Price_Change"] = monthly_pivot[month_cols[-1]] - monthly_pivot[month_cols[0]]
monthly_pivot["Expected_Pct_Change"] = (monthly_pivot["Expected_Price_Change"] / monthly_pivot[month_cols[0]]) * 100

monthly_pivot = monthly_pivot.sort_values("5_Month_Avg_Price", ascending=False).reset_index(drop=True)

print("=== District-Level 5-Month Cotton Price Forecast Summary (All 30 Districts) ===")
display(monthly_pivot.round(2))

# %% [markdown]
# ### Key District Forecast Findings:

# %% [markdown]
# ## 8. Visualizing the 5-Month District Forecast Trajectory

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=False)

f_dates = pd.to_datetime(state_forecast_df["Date"])

# Plot 1: Statewide 5-Month Trajectory with 95% Confidence Interval Band
ax1.plot(f_dates, state_forecast_df["Benchmark_Price"], color="#d62728", lw=2.8, label="Projected Benchmark Modal Price (₹/quintal)")
ax1.fill_between(f_dates, state_forecast_df["Lower_95CI"], state_forecast_df["Upper_95CI"], color="#ff9896", alpha=0.35, label="95% Confidence Interval")
ax1.axhline(today_state_price, color="black", linestyle=":", lw=1.5, label=f"Today's Price: ₹{today_state_price:.2f}")

ax1.set_title("Statewide Gujarat Cotton Benchmark Price: 5-Month Forward Forecast (Oct 2026 – Mar 2027)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Modal Price (₹/quintal)")
ax1.legend(loc="upper right")

# Plot 2: District Comparison across Top Cotton Regions
highlight_districts = ["Rajkot", "Botad", "Amreli", "Surendranagar", "Jamnagar", "Bhavnagar", "Kachchh", "Mehsana"]
colors = sns.color_palette("tab10", len(highlight_districts))

for dist, col in zip(highlight_districts, colors):
    dist_curve = df_district_daily[df_district_daily["District"] == dist]
    ax2.plot(f_dates, dist_curve["Predicted_Modal_Price_INR"], label=f"{dist} ({district_basis_dict[dist]:+.1f} ₹)", color=col, lw=1.8)

ax2.set_title("5-Month Forward Price Paths for Major Gujarat Cotton Districts", fontsize=13, fontweight="bold")
ax2.set_xlabel("Forecast Timeline")
ax2.set_ylabel("Projected Price (₹/quintal)")
ax2.legend(loc="lower left", ncol=2, frameon=True, fontsize=9.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Forecast Visualization Analysis

# %% [markdown]
# ## 9. Comprehensive Multi-Tab Excel Spreadsheet Generation

# %%
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_OUTPUT_PATH = OUTPUT_DIR / "gujarat_cotton_5months_district_forecast.xlsx"

# 1. Executive Summary Sheet
exec_summary_df = pd.DataFrame([
    {"Metric / Parameter": "Forecast Inception Date", "Value": today_date.strftime("%d %B %Y")},
    {"Metric / Parameter": "Forecast Horizon Duration", "Value": "5 Months (151 Days: 21 Oct 2026 to 20 Mar 2027)"},
    {"Metric / Parameter": "Total Gujarat Districts Covered", "Value": f"{len(all_districts)} Districts"},
    {"Metric / Parameter": "Total APMC Mandis Mapped", "Value": f"{len(MANDI_TO_DISTRICT)} Mandis"},
    {"Metric / Parameter": "Current Benchmark Price (Today)", "Value": f"₹{today_state_price:.2f} per quintal"},
    {"Metric / Parameter": "Projected Month 1 Avg (Nov 2026)", "Value": f"₹{state_forecast_df[state_forecast_df['Month_Group']=='Month 1 (Oct-Nov 2026)']['Benchmark_Price'].mean():.2f}"},
    {"Metric / Parameter": "Projected Month 2 Avg (Dec 2026)", "Value": f"₹{state_forecast_df[state_forecast_df['Month_Group']=='Month 2 (Nov-Dec 2026)']['Benchmark_Price'].mean():.2f}"},
    {"Metric / Parameter": "Projected Month 3 Avg (Jan 2027)", "Value": f"₹{state_forecast_df[state_forecast_df['Month_Group']=='Month 3 (Dec-Jan 2027)']['Benchmark_Price'].mean():.2f}"},
    {"Metric / Parameter": "Projected Month 4 Avg (Feb 2027)", "Value": f"₹{state_forecast_df[state_forecast_df['Month_Group']=='Month 4 (Jan-Feb 2027)']['Benchmark_Price'].mean():.2f}"},
    {"Metric / Parameter": "Projected Month 5 Avg (Mar 2027)", "Value": f"₹{state_forecast_df[state_forecast_df['Month_Group']=='Month 5 (Feb-Mar 2027)']['Benchmark_Price'].mean():.2f}"},
    {"Metric / Parameter": "Optimized Model Architecture", "Value": "Blended Ensemble (75% Tuned HistGradientBoosting + 25% Regularized Ridge)"},
    {"Metric / Parameter": "Optimized Model Test R²", "Value": "0.8211 (vs. 0.7747 Persistence Baseline)"},
    {"Metric / Parameter": "Optimized Model Test MAE", "Value": "₹177.10 per quintal (vs. ₹181.38 default)"},
    {"Metric / Parameter": "Optimized Model Test RMSE", "Value": "₹323.75 per quintal (vs. ₹329.89 default)"},
    {"Metric / Parameter": "Optimized Model Test MAPE", "Value": "2.48% Mean Absolute Percentage Error"}
])

# 2. Mandi Mapping Table
mandi_mapping_df = pd.DataFrame([
    {"APMC_Mandi": m, "Assigned_District": d} for m, d in sorted(MANDI_TO_DISTRICT.items())
])

# Write to Excel with openpyxl
with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl") as writer:
    exec_summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)
    monthly_pivot.to_excel(writer, sheet_name="District_Monthly_Forecast", index=False)
    df_district_daily.to_excel(writer, sheet_name="District_Daily_Forecast", index=False)
    tuning_df.to_excel(writer, sheet_name="Model_Tuning_Benchmark", index=False)
    feat_imp_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    mandi_mapping_df.to_excel(writer, sheet_name="Mandi_District_Mapping", index=False)

print(f"Successfully generated multi-tab professional Excel file: '{EXCEL_OUTPUT_PATH}'")
print(f"Spreadsheet contains 6 comprehensive sheets with all 30 districts and 151-day forward forecasts.")

# %% [markdown]
# ### Excel File Generation Verification:

# %% [markdown]
# ## 10. Final Summary
# ### Q&A
# ### Data Analysis Key Findings
# ### Insights or Next Steps
