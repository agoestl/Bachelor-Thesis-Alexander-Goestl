# Replication of Barber, Huang, Odean (2022)
#
# Sharp RDD at $300M market-cap cutoff using RDFlex (Noack, Olma, Rothe 2024).
# Adjustment methods:
#   1. No Covariates (DummyRegressor baseline)
#   2. Ridge regression
#   3. Lasso regression (with second-order interactions, cf. Olma 2024 p.27)
#   4. Random Forest (Optuna-tuned)
#   5. LightGBM (Optuna-tuned)
#   6. XGBoost (Optuna-tuned)
#   7. Neural Network (MLPRegressor, Optuna-tuned, cf. R nnet)
#   8. Super Learner (stacked ensemble of all above)

# Inputs:
#   rd_robintrack.csv                         (RobinTrack aggregated user counts)
#   robintrack-popularity-history/tmp/...     (per-ticker fallback)
#   datastream_daily_filtered.csv.gz / rd_datastream.csv  (Datastream equities)
#   F-F_Research_Data_5_Factors_2x3_daily.csv (Fama-French 5 factors)
#   abnormal_svi.csv                          (Google abnormal SVI, optional)
#   matched_sample_bw{50,75,100,125}.csv      (cached matched samples)
#   master_dataset.csv                        (cached master panel)
#   results.csv / results_combined.csv        (cached RDFlex results)
#   results_auto_bw.csv                       (cached auto-bw results)
#   results_rdrobust.csv                      (R rdrobust results, optional merge)
#   results_rdrobust_auto_bw.csv              (R rdrobust auto-bw, optional merge)
#   results_covariate_continuity.csv          (R covariate continuity, optional)
#   seed_robustness.csv / kernel_robustness.csv / tuned_hyperparameters.csv
# Output:
#   master_dataset.csv                        (built master panel)
#   matched_sample_bw{50,75,100,125}.csv      (matched samples for R script)
#   results.csv / results_combined.csv        (RDFlex results)
#   results_auto_bw.csv                       (RDFlex auto-bw results)
#   tuned_hyperparameters.csv                 (Optuna best params)
#   power_analysis.csv / density_test_results.csv / kernel_robustness.csv
#   seed_robustness.csv                       (cross-fitting seed stability)
#   *.png                                     (all plots and table images)

import gc
import inspect
import sys
import traceback
import warnings
import time
from pathlib import Path
import matplotlib
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import norm, ttest_ind, gaussian_kde
from doubleml import DoubleMLRDDData
from doubleml.rdd import RDFlex
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# paths — script lives in scripts/, data/results/figures are sibling folders
script_dir = Path(__file__).parent.absolute()
repo_dir = script_dir.parent
data_dir = repo_dir / "data"
output_dir = repo_dir / "results"
output_dir.mkdir(exist_ok=True)
figures_dir = repo_dir / "figures"
figures_dir.mkdir(exist_ok=True)
robintrack_dir = data_dir / "robintrack-popularity-history"
robintrack_individual_dir = robintrack_dir / "tmp" / "popularity_export"
robintrack_csv = data_dir / "rd_robintrack.csv"
datastream_csv_gz = data_dir / "datastream_daily_filtered.csv.gz"
datastream_csv = data_dir / "rd_datastream.csv"
svi_csv = data_dir / "abnormal_svi.csv"
ff_csv = data_dir / "F-F_Research_Data_5_Factors_2x3_daily.csv"

# Global Tuning Parameters
N_OPTUNA_TRIALS = 10  # Optuna trials per model
OPTUNA_CV = 5  # cross-validation folds for Optuna tuning
RDFLEX_N_Folds = 5  # cross-fitting folds for RDFlex estimation
RDFLEX_N_REP = 5  # repetitions for RDFlex

# paper values
PAPER_BENCHMARK_TABLE7 = {
    (50, 1): {"beta1": 95.71, "se": 39.472, "n": 1332},
    (50, 2): {"beta1": 115.8, "se": 53.933, "n": 1332},
    (50, 3): {"beta1": 247.3, "se": 78.782, "n": 1332},
    (75, 1): {"beta1": 90.57, "se": 37.123, "n": 2068},
    (75, 2): {"beta1": 109.8, "se": 61.131, "n": 2068},
    (75, 3): {"beta1": 97.18, "se": 80.490, "n": 2068},
    (100, 1): {"beta1": 77.55, "se": 33.300, "n": 2782},
    (100, 2): {"beta1": 131.0, "se": 53.648, "n": 2782},
    (100, 3): {"beta1": 174.8, "se": 72.724, "n": 2782},
    (125, 1): {"beta1": 49.13, "se": 26.718, "n": 3406},
    (125, 2): {"beta1": 102.3, "se": 41.947, "n": 3406},
    (125, 3): {"beta1": 147.7, "se": 59.496, "n": 3406},
}
for _k, _v in PAPER_BENCHMARK_TABLE7.items():
    _v["ci_lower"] = _v["beta1"] - 1.96 * _v["se"]
    _v["ci_upper"] = _v["beta1"] + 1.96 * _v["se"]
    _v["ci_width"] = _v["ci_upper"] - _v["ci_lower"]

# Constants
CUTOFF_MC = 300
BANDWIDTHS = [50, 75, 100, 125]
POLY_ORDERS = [1, 2, 3]
WINSOR_LOW = 0.002
WINSOR_HIGH = 0.998
RANK_MAX_TOTAL = 80
RANK_MAX_TREATED = 20
RET_DIFF_LO = 0.5
RET_DIFF_HI = 2.0
# to remove small samples from RDFlex estimation
MIN_RDFLEX_OBS = 50

# covariates used by RDFlex (all adjustment strategies share this set)
COVARIATES = [
    "abs_ret",
    "log_volume",
    "log_users",
    "mkt_ret",
    "log_users_lag",
]

# exclude abs_ret from covariates for estimation (keep for diagnostics)
ESTIMATION_COVARIATES = [c for c in COVARIATES if c != "abs_ret"]

# covariate labels for plots
COVARIATE_LABELS = {
    "abs_ret": "Absolute Return",
    "log_volume": "Log Volume",
    "log_users": "Log Users",
    "mkt_ret": "Market Return",
    "log_users_lag": "Log Users (lag)",
}

# visual parameters
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10
plt.ioff()

# 8 methods with colors and labels (Okabe-Ito colorblind-safe palette)
METHOD_COLORS = {
    "nocov": "#000000",
    "rdrobust": "#E69F00",
    "ridge": "#56B4E9",
    "lasso": "#009E73",
    "rf": "#F0E442",
    "lgbm": "#0072B2",
    "xgb": "#D55E00",
    "nnet": "#CC79A7",
    "sl": "#999999",
}
METHOD_LABELS = {
    "nocov": "No Covariates",
    "rdrobust": "Linear (rdrobust)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "nnet": "Neural Network",
    "sl": "Super Learner",
}

# hyperparameter search spaces
RF_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 400, "step": 50},
    "max_depth": {"type": "categorical", "choices": [3, 5, 7, 10, None]},
    "min_samples_split": {"type": "int", "low": 2, "high": 12},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 6},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8]},
}
LGBM_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 400, "step": 50},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "min_child_samples": {"type": "int", "low": 5, "high": 30},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
}
XGB_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 400, "step": 50},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "min_child_weight": {"type": "int", "low": 1, "high": 8},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
}
NNET_PARAM_SPACE = {
    "hidden_layer_sizes": {
        "type": "categorical",
        "choices": [
            (16,),
            (32,),
            (64,),
            (16, 8),
            (32, 16),
            (64, 32),
        ],
    },
    "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "max_iter": {"type": "categorical", "choices": [500, 1000]},
}

# labels for hyperparameters
HYPERPARAM_DESCRIPTIONS = {
    # Random Forest
    "rf_n_estimators": "Number of trees in the forest",
    "rf_max_depth": "Maximum depth of each tree (None = unlimited)",
    "rf_min_samples_split": "Minimum samples to split an internal node",
    "rf_min_samples_leaf": "Minimum samples at each leaf node",
    "rf_max_features": "Fraction of features considered per split",
    # LightGBM
    "lgbm_n_estimators": "Number of boosting rounds",
    "lgbm_max_depth": "Maximum depth of each tree",
    "lgbm_learning_rate": "Step size shrinkage per boosting round",
    "lgbm_min_child_samples": "Minimum samples in a leaf",
    "lgbm_subsample": "Fraction of rows sampled per tree",
    "lgbm_colsample_bytree": "Fraction of features sampled per tree",
    "lgbm_reg_alpha": "L1 regularisation on leaf weights",
    "lgbm_reg_lambda": "L2 regularisation on leaf weights",
    # XGBoost
    "xgb_n_estimators": "Number of boosting rounds",
    "xgb_max_depth": "Maximum depth of each tree",
    "xgb_learning_rate": "Step size shrinkage per boosting round",
    "xgb_min_child_weight": "Minimum sum of instance weight in a leaf",
    "xgb_subsample": "Fraction of rows sampled per tree",
    "xgb_colsample_bytree": "Fraction of features sampled per tree",
    "xgb_reg_alpha": "L1 regularisation on leaf weights",
    "xgb_reg_lambda": "L2 regularisation on leaf weights",
    # Neural Net
    "nnet_hidden_layer_sizes": "Architecture (neurons per hidden layer)",
    "nnet_alpha": "L2 regularisation penalty",
    "nnet_learning_rate_init": "Initial learning rate for Adam",
    "nnet_max_iter": "Maximum training iterations",
}

# Load Data


# load RobinTrack daily user counts from the aggregated export
def _load_robintrack():
    source_file = None
    if robintrack_csv.exists():
        source_file = robintrack_csv
    elif robintrack_individual_dir.exists():
        return _load_robintrack_individual()
    else:
        return None
    chunk_size = 2_000_000
    daily_parts = []
    for chunk in pd.read_csv(source_file, dtype={"ticker": str}, chunksize=chunk_size):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        chunk["date"] = chunk["timestamp"].dt.normalize()
        chunk = chunk.sort_values("timestamp")
        agg = chunk.groupby(["ticker", "date"])["users_holding"].last().reset_index()
        daily_parts.append(agg)
    df_daily = pd.concat(daily_parts, ignore_index=True)
    df_daily = (
        df_daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "date"])["users_holding"]
        .last()
        .reset_index()
    )
    df_daily = df_daily.rename(columns={"users_holding": "users_close"})
    df_daily = df_daily.sort_values(["ticker", "date"])
    df_daily["users_lag"] = df_daily.groupby("ticker")["users_close"].shift(1)
    df_daily["userchg"] = df_daily["users_close"] - df_daily["users_lag"]
    return df_daily


# load RobinTrack data from individual ticker CSV files if there is no aggregated export
def _load_robintrack_individual():
    all_frames = []
    for csv_file in robintrack_individual_dir.glob("*.csv"):
        ticker = csv_file.stem
        try:
            df_ticker = pd.read_csv(csv_file)
            df_ticker["ticker"] = ticker
            df_ticker["timestamp"] = pd.to_datetime(df_ticker["timestamp"])
            df_ticker["date"] = df_ticker["timestamp"].dt.normalize()
            df_ticker = df_ticker.sort_values("timestamp")
            agg = (
                df_ticker.groupby(["ticker", "date"])["users_holding"]
                .last()
                .reset_index()
            )
            all_frames.append(agg)
        except Exception:
            continue
    if not all_frames:
        return None
    df_daily = pd.concat(all_frames, ignore_index=True)
    df_daily = df_daily.rename(columns={"users_holding": "users_close"})
    df_daily = df_daily.sort_values(["ticker", "date"])
    df_daily["users_lag"] = df_daily.groupby("ticker")["users_close"].shift(1)
    df_daily["userchg"] = df_daily["users_close"] - df_daily["users_lag"]
    return df_daily


# load and clean Datastream equity data
def _load_datastream():
    if datastream_csv_gz.exists():
        src = datastream_csv_gz
    elif datastream_csv.exists():
        src = datastream_csv
    else:
        return None
    df = pd.read_csv(src)
    if "date" not in df.columns:
        if "datadate" in df.columns:
            df = df.rename(columns={"datadate": "date"})
        elif "marketdate" in df.columns:
            df = df.rename(columns={"marketdate": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["mktcap", "ret", "volume", "close", "open", "high", "numshrs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# load Fama-French 5 factors
def _load_fama_french():
    if not ff_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(ff_csv, skiprows=3)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date_str"})
    df["date_str"] = df["date_str"].astype(str).str.strip()
    df = df[df["date_str"].str.match(r"^\d{8}$")]
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace("-", "_")
        if cl == "mkt_rf":
            rename_map[c] = "mkt_ret"
        elif cl == "rf":
            rename_map[c] = "RF"
    df = df.rename(columns=rename_map)
    num_cols = [c for c in df.columns if c not in ("date", "date_str")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["date_str"], errors="ignore")
    return df


# load abnormal SVI data if available
def _load_svi():
    if not svi_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(svi_csv, parse_dates=["date"])
    if "asvi" not in df.columns and "svi" in df.columns:
        df["asvi"] = df["svi"]
    return df


# build the master dataset from all sources
def build_master_dataset():
    print("Reading Input ...")
    t0 = time.time()
    rt = _load_robintrack()
    rt_ind = _load_robintrack_individual()
    ds = _load_datastream()
    ff = _load_fama_french()
    svi = _load_svi()

    # optimize memory: convert float64 -> float32
    def optimize_dtypes(df):
        for col in df.columns:
            if df[col].dtype == "object":
                if col in ["ticker", "symbol"]:
                    df[col] = df[col].astype("category")
            elif df[col].dtype == "float64":
                if df[col].notna().any():
                    df[col] = df[col].astype("float32")
            elif df[col].dtype == "int64":
                if df[col].max() < 2**31 - 1 and df[col].min() > -(2**31):
                    df[col] = df[col].astype("int32")
        return df

    rt = optimize_dtypes(rt)
    ds = optimize_dtypes(ds)
    if len(ff) > 0:
        ff = optimize_dtypes(ff)
    if len(svi) > 0:
        svi = optimize_dtypes(svi)
    # merge RobinTrack with Datastream
    df = pd.merge(rt, ds, on=["ticker", "date"], how="inner", suffixes=("", "_ds"))
    del rt, ds
    gc.collect()
    if rt_ind is not None and len(rt_ind) > 0:
        rt_ind = optimize_dtypes(rt_ind)
        df = pd.merge(
            df, rt_ind, on=["ticker", "date"], how="left", suffixes=("", "_ind")
        )
        del rt_ind
        gc.collect()
    if len(ff) > 0:
        ff_cols = [
            c
            for c in ["date", "mkt_ret", "SMB", "HML", "RMW", "CMA", "RF"]
            if c in ff.columns
        ]
        df = pd.merge(df, ff[ff_cols], on="date", how="left", suffixes=("", "_ff"))
        del ff
        gc.collect()
    if len(svi) > 0:
        df = pd.merge(df, svi, on=["ticker", "date"], how="left", suffixes=("", "_svi"))
        del svi
        gc.collect()
    # final optimization pass
    df = optimize_dtypes(df)
    return df


# winsorize a series at given quantiles
def _winsorize(series, low=WINSOR_LOW, high=WINSOR_HIGH):
    lo_val = series.quantile(low)
    hi_val = series.quantile(high)
    return series.clip(lo_val, hi_val)


# compute all RDD-relevant variables from master dataset
def compute_rdd_variables(df):
    if df["mktcap"].max() > 1e10:
        df["mktcap_millions"] = df["mktcap"] / 1e6
    else:
        df["mktcap_millions"] = df["mktcap"]

    if "high" in df.columns and "numshrs" in df.columns:
        df["mktcap_hi"] = df["high"] * df["numshrs"] / 1e6
    else:
        df["mktcap_hi"] = df["mktcap_millions"]
    df["gt300"] = (df["mktcap_millions"] > CUTOFF_MC).astype(int)
    df["mktcap1_300"] = df["mktcap_millions"] - CUTOFF_MC
    df["mktcap2_300"] = df["mktcap1_300"] ** 2
    df["mktcap3_300"] = df["mktcap1_300"] ** 3
    if "abs_ret" not in df.columns and "ret" in df.columns:
        df["abs_ret"] = df["ret"].abs()
    df["abs_sortret"] = df["abs_ret"]
    # sort in-place instead of creating copies
    df.sort_values(["date", "abs_sortret"], ascending=[True, False], inplace=True)
    df["rank_absret"] = df.groupby("date")["abs_sortret"].rank(
        method="first", ascending=False
    )
    mask_gt = df["gt300"] == 1
    df["rank_absret_gt300"] = np.nan
    df.loc[mask_gt, "rank_absret_gt300"] = (
        df.loc[mask_gt]
        .groupby("date")["abs_sortret"]
        .rank(method="first", ascending=False)
    )
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "userchg" in df.columns:
        df["dayuserchgw"] = _winsorize(df["userchg"])
    else:
        df["dayuserchgw"] = np.nan
    return df


# construct the matched-pair sample for a given bandwidth
def construct_matched_sample(df, bandwidth):
    sub = df[
        (df["dayuserchgw"].notna())
        & (df["rank_absret"] <= RANK_MAX_TOTAL)
        & ((df["mktcap_millions"] - CUTOFF_MC).abs() < bandwidth)
    ].copy()
    if len(sub) == 0:
        return sub
    treated = sub[
        (sub["gt300"] == 1) & (sub["rank_absret_gt300"] <= RANK_MAX_TREATED)
    ].copy()
    control = sub[
        (sub["gt300"] == 0)
        & (sub["mktcap_hi"].notna())
        & (sub["mktcap_hi"] < CUTOFF_MC)
    ].copy()
    if len(treated) == 0 or len(control) == 0:
        return sub.iloc[0:0].copy()
    treated = treated.rename(
        columns={"abs_ret": "abs_ret_treated", "dayuserchgw": "dayuserchgw_treated"}
    )
    control = control.rename(
        columns={"abs_ret": "abs_ret_control", "dayuserchgw": "dayuserchgw_control"}
    )
    # cross-join treated and control within each date
    merged = pd.merge(treated, control, on="date", how="inner", suffixes=("_t", "_c"))
    if "abs_ret_treated" in merged.columns and "abs_ret_control" in merged.columns:
        ret_ratio = merged["abs_ret_treated"] / merged["abs_ret_control"].replace(
            0, np.nan
        )
        merged = merged[(ret_ratio >= RET_DIFF_LO) & (ret_ratio <= RET_DIFF_HI)]
    # keep only the best match per treated observation
    if len(merged) > 0:
        merged["abs_ret_diff"] = (
            merged["abs_ret_treated"] - merged["abs_ret_control"]
        ).abs()
        # identify each treated obs by date + rank_absret_gt300
        rank_col_t = (
            "rank_absret_gt300_t"
            if "rank_absret_gt300_t" in merged.columns
            else "rank_absret_gt300"
        )
        if rank_col_t in merged.columns:
            merged = merged.sort_values(["date", rank_col_t, "abs_ret_diff"])
            merged = merged.drop_duplicates(subset=["date", rank_col_t], keep="first")
        else:
            # fallback: just keep best match per treated ticker-date
            ticker_col = "ticker_t" if "ticker_t" in merged.columns else "ticker"
            merged = merged.sort_values(["date", ticker_col, "abs_ret_diff"])
            merged = merged.drop_duplicates(subset=["date", ticker_col], keep="first")
        merged = merged.drop(columns=["abs_ret_diff"])
    # reconstruct a long-format sample from matched pairs
    rows_t = merged.copy()
    rows_t["gt300"] = 1
    rows_t["dayuserchgw"] = rows_t.get(
        "dayuserchgw_treated", rows_t.get("dayuserchgw_t", np.nan)
    )
    rows_c = merged.copy()
    rows_c["gt300"] = 0
    rows_c["dayuserchgw"] = rows_c.get(
        "dayuserchgw_control", rows_c.get("dayuserchgw_c", np.nan)
    )
    pair_level_cols = [
        "mktcap_millions",
        "mktcap1_300",
        "mktcap2_300",
        "mktcap3_300",
        "mktcap_hi",
        "abs_ret",
        "ticker",
        "date",
        # Preserve columns used to construct RDFlex covariates after the treated/control merge
        "volume",
        "users_close",
        "users_lag",
        "mkt_ret",
        "asvi",
    ]
    for col in pair_level_cols:
        if col + "_t" in rows_t.columns:
            rows_t[col] = rows_t[col + "_t"]
        if col + "_c" in rows_c.columns:
            rows_c[col] = rows_c[col + "_c"]
    if "abs_ret_treated" in rows_t.columns:
        rows_t["abs_ret"] = rows_t["abs_ret_treated"]
    if "abs_ret_control" in rows_c.columns:
        rows_c["abs_ret"] = rows_c["abs_ret_control"]
    common_cols = list(
        set(rows_t.columns) & set(rows_c.columns) & set(df.columns)
        | {"gt300", "dayuserchgw"}
    )
    common_cols = [
        c for c in common_cols if c in rows_t.columns and c in rows_c.columns
    ]
    result = pd.concat([rows_t[common_cols], rows_c[common_cols]], ignore_index=True)
    result = result.dropna(subset=["dayuserchgw"])
    return result


# Super Learner pipeline
class SampleWeightPipeline(Pipeline):
    # fit forwarding sample_weight to the last step when accepted
    def fit(self, X, y=None, sample_weight=None, **params):
        if sample_weight is not None and self.steps:
            final_name, final_estimator = self.steps[-1]
            try:
                fit_params = inspect.signature(final_estimator.fit).parameters
            except (TypeError, ValueError):
                fit_params = {}
            if "sample_weight" in fit_params:
                params = dict(params)
                params[f"{final_name}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


# Super Learner Regressor
class SuperLearnerRegressor(BaseEstimator, RegressorMixin):
    # store base learners, meta-learner, and CV folds
    def __init__(self, estimators=None, final_estimator=None, cv=3):
        self.estimators = estimators or []
        self.final_estimator = final_estimator
        self.cv = cv

    # fit base learners out-of-fold, train meta-learner, refit on full data
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        oof = np.zeros((len(y), len(self.estimators)))
        # out-of-fold predictions for meta-learner
        for fold_train, fold_val in kf.split(X):
            for i, (nm, est) in enumerate(self.estimators):
                e = clone(est)
                if sample_weight is not None:
                    try:
                        e.fit(
                            X[fold_train],
                            y[fold_train],
                            sample_weight=sample_weight[fold_train],
                        )
                    except TypeError:
                        e.fit(X[fold_train], y[fold_train])
                else:
                    e.fit(X[fold_train], y[fold_train])
                oof[fold_val, i] = e.predict(X[fold_val])
        # meta-learner on out-of-fold predictions
        meta = clone(self.final_estimator) if self.final_estimator else RidgeCV(cv=3)
        meta.fit(oof, y)
        self.meta_ = meta
        # refit base learners on full data
        self.fitted_ = []
        for nm, est in self.estimators:
            e = clone(est)
            if sample_weight is not None:
                try:
                    e.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    e.fit(X, y)
            else:
                e.fit(X, y)
            self.fitted_.append((nm, e))
        return self

    # predict by stacking base-learner predictions through the meta-learner
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([e.predict(X) for _, e in self.fitted_])
        return self.meta_.predict(preds)


# suggest params from a search space dictionary
def _suggest_from_space(trial, space):
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif kind == "float":
            if spec.get("log", False):
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=True
                )
            else:
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], step=spec.get("step")
                )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


# tune a model with Optuna and cross-validated MSE
def _optuna_tune(base, space, Z, y, n_trials=N_OPTUNA_TRIALS, cv=OPTUNA_CV, tag=""):
    def objective(trial):
        params = _suggest_from_space(trial, space)
        model = clone(base)
        model.set_params(**params)
        scores = cross_val_score(
            model, Z, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=2
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# nnet pipeline
def _optuna_tune_pipeline(
    pipe, space, Z, y, step_name="model", n_trials=N_OPTUNA_TRIALS, cv=OPTUNA_CV, tag=""
):
    def objective(trial):
        params = _suggest_from_space(trial, space)
        p = clone(pipe)
        prefixed = {f"{step_name}__{k}": v for k, v in params.items()}
        p.set_params(**prefixed)
        scores = cross_val_score(
            p, Z, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=2
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# sample comparison
def diagnose_sample_sizes(sample_by_bw, label=""):
    pass


# Estimation
RDFLEX_RESULT_COLUMNS = [
    "bandwidth",
    "poly_order",
    "method",
    "beta1",
    "se",
    "t_stat",
    "pvalue",
    "ci_lower",
    "ci_upper",
    "ci_width",
    "n_obs",
    "covariates_used",
]


# prepare the covariate columns needed for RDFlex
def _prepare_rdflex_covariates(df):
    df = df.copy()
    if "volume" in df.columns:
        df["log_volume"] = np.log(df["volume"].clip(lower=0) + 1)
    if "users_close" in df.columns:
        df["log_users"] = np.log(df["users_close"].clip(lower=0) + 1)
    if "users_lag" in df.columns:
        df["log_users_lag"] = np.log(df["users_lag"].clip(lower=0) + 1)
    # excluding abs_ret
    available = [c for c in ESTIMATION_COVARIATES if c in df.columns]
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, available


# drop sparse covariates until enough complete-case rows remain
def _select_complete_rdflex_covariates(
    df, covars, required_cols, min_obs=MIN_RDFLEX_OBS
):
    selected = list(covars)
    dropped = []
    while selected:
        complete_n = len(df.dropna(subset=selected + required_cols))
        if complete_n >= min_obs:
            return selected, complete_n, dropped
        nonmissing = df[selected].notna().sum()
        drop_col = nonmissing.idxmin()
        dropped.append((drop_col, int(nonmissing[drop_col])))
        selected.remove(drop_col)
    complete_n = len(df.dropna(subset=required_cols))
    return selected, complete_n, dropped


# adjustment configs because of small 50M sample size
def _build_adjustment_configs(tuned, sample_n=1000):
    # adapt CV folds and validation fraction to sample size
    lasso_cv = min(5, max(2, sample_n // 80))
    sl_cv = min(3, max(2, sample_n // 100))
    ridge_cv = min(5, max(2, sample_n // 60))
    nnet_val_frac = 0.10 if sample_n < 500 else 0.15
    # Lasso with second-order interaction covariates
    lasso_pipe = SampleWeightPipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=2, interaction_only=False, include_bias=False
                ),
            ),
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=lasso_cv, n_jobs=2, max_iter=10000)),
        ]
    )
    # neural network with standardization
    nnet_params = dict(tuned.get("nnet", {}))
    # force smaller architectures for small samples to prevent convergence issues
    if sample_n < 300:
        nnet_params["hidden_layer_sizes"] = (8,)
    elif sample_n < 500:
        hl = nnet_params.get("hidden_layer_sizes", (64,))
        # cap total neurons to avoid overfitting on small samples
        total = sum(hl) if isinstance(hl, tuple) else hl
        if total > 32:
            nnet_params["hidden_layer_sizes"] = (16,)
    # ensure max_iter is high enough for convergence
    nnet_params["max_iter"] = max(nnet_params.get("max_iter", 1000), 2000)
    # force smaller learning rate for stability on small samples
    if sample_n < 500:
        nnet_params["learning_rate_init"] = min(
            nnet_params.get("learning_rate_init", 0.001), 0.001
        )
    nnet_pipe = SampleWeightPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=nnet_val_frac,
                    **nnet_params,
                ),
            ),
        ]
    )
    # base learners for super learner
    base_learners = [
        ("Ridge", RidgeCV(cv=ridge_cv)),
        ("Lasso", clone(lasso_pipe)),
        ("RF", RandomForestRegressor(random_state=42, n_jobs=2, **tuned["rf"])),
        ("LightGBM", LGBMRegressor(verbose=-1, n_jobs=2, **tuned["lgbm"])),
        (
            "XGBoost",
            XGBRegressor(verbosity=0, n_jobs=2, random_state=42, **tuned["xgb"]),
        ),
    ]
    # only include NNet in SL for samples large enough to handle nested CV
    if sample_n >= 400:
        base_learners.append(("NNet", clone(nnet_pipe)))
    sl = SuperLearnerRegressor(
        estimators=base_learners,
        final_estimator=RidgeCV(cv=min(3, sl_cv)),
        cv=sl_cv,
    )
    configs = [
        ("nocov", DummyRegressor(strategy="mean")),
        ("ridge", RidgeCV(cv=ridge_cv)),
        ("lasso", clone(lasso_pipe)),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=2, **tuned["rf"])),
        ("lgbm", LGBMRegressor(verbose=-1, n_jobs=2, **tuned["lgbm"])),
        ("xgb", XGBRegressor(verbosity=0, n_jobs=2, random_state=42, **tuned["xgb"])),
        ("nnet", clone(nnet_pipe)),
        ("sl", sl),
    ]
    return configs


# tune ML regressors via Optuna
def _tune_models(Z, y, tag=""):
    tuned = {}
    # RandomForest
    tuned["rf"] = _optuna_tune(
        RandomForestRegressor(random_state=42, n_jobs=2),
        RF_PARAM_SPACE,
        Z,
        y,
        n_trials=N_OPTUNA_TRIALS,
        cv=OPTUNA_CV,
        tag="RF",
    )
    # LightGBM
    tuned["lgbm"] = _optuna_tune(
        LGBMRegressor(verbose=-1, n_jobs=2, random_state=42),
        LGBM_PARAM_SPACE,
        Z,
        y,
        n_trials=N_OPTUNA_TRIALS,
        cv=OPTUNA_CV,
        tag="LGBM",
    )
    # XGBoost
    tuned["xgb"] = _optuna_tune(
        XGBRegressor(verbosity=0, n_jobs=2, random_state=42),
        XGB_PARAM_SPACE,
        Z,
        y,
        n_trials=N_OPTUNA_TRIALS,
        cv=OPTUNA_CV,
        tag="XGB",
    )
    # NeuralNet
    nnet_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=42, early_stopping=True, validation_fraction=0.15
                ),
            ),
        ]
    )
    nnet_best = _optuna_tune_pipeline(
        nnet_pipe,
        NNET_PARAM_SPACE,
        Z,
        y,
        step_name="model",
        n_trials=N_OPTUNA_TRIALS,
        cv=OPTUNA_CV,
        tag="NNet",
    )
    tuned["nnet"] = nnet_best
    return tuned


# record a single RDFlex result into a dict
def _record_rdflex(est, bandwidth, poly_order, adjustment, n, covars=None):
    coef = float(est.coef[0])
    se = float(est.se[0])
    pval = float(est.pval[0])
    ci = est.confint()
    return {
        "bandwidth": bandwidth,
        "poly_order": poly_order,
        "method": adjustment,
        "beta1": coef,
        "se": se,
        "t_stat": coef / se if se > 0 else np.nan,
        "pvalue": pval,
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
        "ci_width": float(ci.iloc[0, 1] - ci.iloc[0, 0]),
        "n_obs": int(n),
        "covariates_used": ", ".join(covars or []),
    }


# create and fit a single estimator
def _run_single_rdflex(df_prep, covars, ml_g, bw, poly_order):
    rdd_data = DoubleMLRDDData(
        data=df_prep,
        y_col="dayuserchgw",
        d_cols="gt300",
        score_col="mktcap1_300",
        x_cols=covars,
    )
    est = RDFlex(
        obj_dml_data=rdd_data,
        ml_g=ml_g,
        ml_m=None,
        fuzzy=False,
        cutoff=0,
        RDFLEX_N_Folds=RDFLEX_N_Folds,
        RDFLEX_N_REP=RDFLEX_N_REP,
        h_fs=float(bw),
        fs_kernel="triangular",
        p=poly_order,
    )
    est.fit()
    return est


# best hyperparameter values for each method
def save_tuned_hyperparameters(tuned, bandwidth):
    rows = []
    for method, params in tuned.items():
        for param_name, param_value in params.items():
            key = f"{method}_{param_name}"
            rows.append(
                {
                    "bandwidth": bandwidth,
                    "method": method,
                    "parameter": param_name,
                    "value": param_value,
                    "description": HYPERPARAM_DESCRIPTIONS.get(key, ""),
                }
            )
    return rows


# save all hyperparameter configs to CSV
def export_all_hyperparameters(all_hp_rows):
    if not all_hp_rows:
        return
    hp_df = pd.DataFrame(all_hp_rows)
    hp_csv = output_dir / "tuned_hyperparameters.csv"
    hp_df.to_csv(hp_csv, index=False)
    # also render a summary table image for the preferred bandwidth
    _plot_hyperparameter_table(hp_df)


# render a table image of best hyperparameters
def _plot_hyperparameter_table(hp_df):
    pref_bw = 125
    pref = hp_df[hp_df["bandwidth"] == pref_bw]
    if len(pref) == 0:
        # fallback to the largest bandwidth available
        pref_bw = hp_df["bandwidth"].max()
        pref = hp_df[hp_df["bandwidth"] == pref_bw]
    if len(pref) == 0:
        return
    methods = ["rf", "lgbm", "xgb", "nnet"]
    table_data = []
    for method in methods:
        mdf = pref[pref["method"] == method]
        for _, row in mdf.iterrows():
            val = row["value"]
            # format value nicely (guard against NaN from CSV round-trip)
            if isinstance(val, float) and np.isnan(val):
                val_str = "N/A"
            elif isinstance(val, float):
                if val < 0.01:
                    val_str = f"{val:.2e}"
                elif val == int(val):
                    val_str = f"{int(val)}"
                else:
                    val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            table_data.append(
                [
                    METHOD_LABELS.get(method, method),
                    row["parameter"],
                    val_str,
                    row["description"],
                ]
            )
    if not table_data:
        return
    # make table bigger relative to the PNG by using a tighter figure
    fig_h = len(table_data) * 0.35 + 0.8
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Parameter", "Best Value", "Description"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3])
    table.scale(1.0, 1.6)
    # style header
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # alternate row colours
    for i in range(1, len(table_data) + 1):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)
    plt.tight_layout(pad=0)
    plt.savefig(
        figures_dir / "tuned_hyperparameters_table.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()


# run estimation across all bandwidths, methods, and polynomial orders
def run_rdflex_path(sample_by_bw):
    rows = []
    all_hp_rows = []
    for bw in sorted(sample_by_bw.keys()):
        df_sample = sample_by_bw[bw]
        if len(df_sample) < 50:
            continue
        df_prep, covars = _prepare_rdflex_covariates(df_sample)
        required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
        covars, complete_n, dropped_covars = _select_complete_rdflex_covariates(
            df_prep, covars, required_cols
        )
        if len(covars) == 0:
            continue
        df_prep = df_prep.dropna(subset=covars + required_cols)
        if len(df_prep) < 50:
            continue
        # tune ML models on this bandwidth sample
        Z = df_prep[covars].values.astype(float)
        y = df_prep["dayuserchgw"].values.astype(float)
        tuned = _tune_models(Z, y, tag=f"bw${bw}M")
        all_hp_rows.extend(save_tuned_hyperparameters(tuned, bw))
        configs = _build_adjustment_configs(tuned, sample_n=len(df_prep))
        # on diffrent polynomial orders
        for poly_order in POLY_ORDERS:
            for name, ml_g in configs:
                success = False
                try:
                    est = _run_single_rdflex(
                        df_prep, covars, clone(ml_g), bw, poly_order
                    )
                    rec = _record_rdflex(
                        est, bw, poly_order, name, len(df_prep), covars
                    )
                    rows.append(rec)
                    success = True
                except Exception as exc:
                    pass
                # retry for nnet and sl with simpler fallback config
                if not success and name in ("nnet", "sl"):
                    try:
                        if name == "nnet":
                            # fallback nnet: tiny architecture, more iterations
                            fallback_nnet = SampleWeightPipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    (
                                        "model",
                                        MLPRegressor(
                                            hidden_layer_sizes=(8,),
                                            max_iter=2000,
                                            random_state=42,
                                            early_stopping=True,
                                            validation_fraction=0.10,
                                            learning_rate_init=0.001,
                                            solver="adam",
                                        ),
                                    ),
                                ]
                            )
                            fallback_ml = fallback_nnet
                        else:
                            # fallback sl
                            fallback_sl_base = [
                                (
                                    "Ridge",
                                    RidgeCV(cv=min(3, max(2, len(df_prep) // 80))),
                                ),
                                (
                                    "RF",
                                    RandomForestRegressor(
                                        random_state=42,
                                        n_jobs=2,
                                        n_estimators=100,
                                        max_depth=5,
                                        min_samples_leaf=3,
                                    ),
                                ),
                                (
                                    "LightGBM",
                                    LGBMRegressor(
                                        verbose=-1,
                                        n_jobs=2,
                                        n_estimators=100,
                                        max_depth=5,
                                        num_leaves=15,
                                    ),
                                ),
                            ]
                            fallback_ml = SuperLearnerRegressor(
                                estimators=fallback_sl_base,
                                final_estimator=RidgeCV(cv=2),
                                cv=2,
                            )
                        # use fewer reps to reduce failure chance
                        rdd_data = DoubleMLRDDData(
                            data=df_prep,
                            y_col="dayuserchgw",
                            d_cols="gt300",
                            score_col="mktcap1_300",
                            x_cols=covars,
                        )
                        est = RDFlex(
                            obj_dml_data=rdd_data,
                            ml_g=fallback_ml,
                            ml_m=None,
                            fuzzy=False,
                            cutoff=0,
                            RDFLEX_N_Folds=RDFLEX_N_Folds,
                            RDFLEX_N_REP=min(RDFLEX_N_REP, 10),
                            h_fs=float(bw),
                            fs_kernel="triangular",
                            p=poly_order,
                        )
                        est.fit()
                        rec = _record_rdflex(
                            est, bw, poly_order, name, len(df_prep), covars
                        )
                        rows.append(rec)
                        success = True
                    except Exception as exc2:
                        pass
                if not success:
                    rows.append(
                        {
                            "bandwidth": bw,
                            "poly_order": poly_order,
                            "method": name,
                            "beta1": np.nan,
                            "se": np.nan,
                            "t_stat": np.nan,
                            "pvalue": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "ci_width": np.nan,
                            "n_obs": int(len(df_prep)),
                            "covariates_used": ", ".join(covars),
                        }
                    )
            gc.collect()
        gc.collect()
    # export all tuned hyperparameters to CSV
    export_all_hyperparameters(all_hp_rows)
    return pd.DataFrame(rows, columns=RDFLEX_RESULT_COLUMNS)


# run estimation with automatic bandwidth selection
def run_rdflex_auto_bw(sample_by_bw):
    rows = []
    # use largest bandwidth sample for auto-bw estimation
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    if len(df_sample) < 50:
        return pd.DataFrame(rows, columns=RDFLEX_RESULT_COLUMNS + ["auto_bw"])
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
    covars, complete_n, dropped_covars = _select_complete_rdflex_covariates(
        df_prep, covars, required_cols
    )
    if len(df_prep) < 50 or len(covars) == 0:
        return pd.DataFrame(rows, columns=RDFLEX_RESULT_COLUMNS + ["auto_bw"])
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50:
        return pd.DataFrame(rows, columns=RDFLEX_RESULT_COLUMNS + ["auto_bw"])
    Z = df_prep[covars].values.astype(float)
    y = df_prep["dayuserchgw"].values.astype(float)
    tuned = _tune_models(Z, y, tag=f"auto-bw")
    configs = _build_adjustment_configs(tuned, sample_n=len(df_prep))
    for name, ml_g in configs:
        try:
            rdd_data = DoubleMLRDDData(
                data=df_prep,
                y_col="dayuserchgw",
                d_cols="gt300",
                score_col="mktcap1_300",
                x_cols=covars,
            )
            # h_fs=None means autmatic bandwidth selection
            est = RDFlex(
                obj_dml_data=rdd_data,
                ml_g=clone(ml_g),
                ml_m=None,
                fuzzy=False,
                cutoff=0,
                RDFLEX_N_Folds=RDFLEX_N_Folds,
                RDFLEX_N_REP=RDFLEX_N_REP,
                h_fs=None,
                fs_kernel="triangular",
                p=1,
            )
            est.fit()
            # h_fs property returns the bandwidth chosen by rdrobust
            chosen_bw = (
                float(est.h_fs)
                if hasattr(est, "h_fs") and est.h_fs is not None
                else np.nan
            )
            rec = _record_rdflex(est, chosen_bw, 1, name, len(df_prep), covars)
            rec["auto_bw"] = chosen_bw
            rows.append(rec)
        except Exception as exc:
            rows.append(
                {
                    "bandwidth": np.nan,
                    "poly_order": 1,
                    "method": name,
                    "beta1": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "ci_width": np.nan,
                    "n_obs": int(len(df_prep)),
                    "covariates_used": ", ".join(covars),
                    "auto_bw": np.nan,
                }
            )
    gc.collect()
    return pd.DataFrame(rows, columns=RDFLEX_RESULT_COLUMNS + ["auto_bw"])


# plot functions


# CI width comparison across adjustment methods
def plot_ci_width_comparison(results_all, poly_order=1, prefix="", title_suffix=""):
    df = results_all[results_all["poly_order"] == poly_order].copy()
    if len(df) == 0:
        return
    bandwidths = sorted(df["bandwidth"].unique())
    method_order = [m for m in METHOD_COLORS.keys() if m in df["method"].values]
    if not method_order:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(method_order) * 1.5), 7))
    n_methods = len(method_order)
    bar_width = 0.8 / max(n_methods, 1)
    x_base = np.arange(len(bandwidths))
    for j, method in enumerate(method_order):
        widths = []
        for bw in bandwidths:
            row = df[(df["method"] == method) & (df["bandwidth"] == bw)]
            widths.append(row["ci_width"].values[0] if len(row) > 0 else np.nan)
        offset = (j - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x_base + offset,
            widths,
            width=bar_width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"${bw}M" for bw in bandwidths])
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("95% Confidence Interval Width")
    # model labels below bars in legend
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{prefix}plot_ci_width_p{poly_order}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# bandwidth robustnesss
def plot_bandwidth_robustness(results_all, poly_order=1, prefix="", title_suffix=""):
    df = results_all[results_all["poly_order"] == poly_order].copy()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    for method in METHOD_COLORS.keys():
        df_m = df[df["method"] == method].sort_values("bandwidth")
        if len(df_m) == 0:
            continue
        ax.errorbar(
            df_m["bandwidth"],
            df_m["beta1"],
            yerr=1.96 * df_m["se"],
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    bw_grid = sorted(df["bandwidth"].unique())
    paper_points = [PAPER_BENCHMARK_TABLE7.get((bw, poly_order), {}) for bw in bw_grid]
    paper_beta = [p.get("beta1", np.nan) for p in paper_points]
    paper_se = [p.get("se", np.nan) for p in paper_points]
    if any(not np.isnan(b) for b in paper_beta):
        ax.errorbar(
            bw_grid,
            paper_beta,
            yerr=[1.96 * s for s in paper_se],
            marker="D",
            markersize=9,
            linestyle="--",
            linewidth=2,
            color="black",
            capsize=5,
            label="Paper Table VII",
        )
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Bandwidth ($M)")
    ax.set_ylabel("Treatment effect on dayuserchgw")
    ax.set_title(f"Bandwidth Robustness (poly {poly_order}){title_suffix}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{prefix}plot_bandwidth_robustness_p{poly_order}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# point estimates
def plot_point_estimates(
    results_all, bandwidth=50, poly_order=1, prefix="", title_suffix=""
):
    df = results_all[
        (results_all["bandwidth"] == bandwidth)
        & (results_all["poly_order"] == poly_order)
    ].copy()
    if len(df) == 0:
        return
    methods = [m for m in METHOD_COLORS.keys() if m in df["method"].values]
    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.1), 7))
    for i, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        mc = METHOD_COLORS.get(method, "gray")
        ax.errorbar(
            i,
            row["beta1"],
            yerr=1.96 * row["se"],
            fmt="o",
            markersize=9,
            color=mc,
            capsize=5,
            capthick=2,
            markerfacecolor=mc,
            markeredgecolor="black",
        )
    paper = PAPER_BENCHMARK_TABLE7.get((bandwidth, poly_order))
    if paper is not None:
        ax.axhline(
            y=paper["beta1"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Paper: {paper['beta1']:.2f} (SE={paper['se']:.2f})",
        )
        ax.fill_between(
            [-0.5, len(methods) - 0.5],
            paper["ci_lower"],
            paper["ci_upper"],
            color="black",
            alpha=0.08,
            label="Paper 95% CI",
        )
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_ylabel("dayuserchgw")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{prefix}plot_point_estimates_bw{bandwidth}_p{poly_order}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# covariate adjustment scatter
def plot_covariate_adjustment_scatter(
    results_all, sample_by_bw, prefix="", title_suffix=""
):
    for bw in sorted(sample_by_bw.keys()):
        df_sample = sample_by_bw[bw].copy()
        if len(df_sample) == 0:
            continue
        res_bw = results_all[
            (results_all["bandwidth"] == bw) & (results_all["poly_order"] == 1)
        ]
        methods = [m for m in METHOD_COLORS.keys() if m in res_bw["method"].values]
        if not methods:
            continue
        n_methods = len(methods)
        ncols = min(3, n_methods)
        nrows = (n_methods + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        x = df_sample["mktcap1_300"].values
        y = df_sample["dayuserchgw"].values
        for idx, method in enumerate(methods):
            ax = axes[idx]
            row = res_bw[res_bw["method"] == method]
            beta = row["beta1"].values[0] if len(row) > 0 else np.nan
            colors = np.where(x >= 0, "#d62728", "#1f77b4")
            ax.scatter(x, y, c=colors, alpha=0.25, s=8, edgecolors="none")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            if not np.isnan(beta):
                left_mean = np.nanmean(y[x < 0]) if np.sum(x < 0) > 0 else 0
                right_mean = left_mean + beta
                xlim = ax.get_xlim()
                ax.hlines(left_mean, xlim[0], 0, colors="navy", linewidth=2, alpha=0.7)
                ax.hlines(
                    right_mean, 0, xlim[1], colors="darkred", linewidth=2, alpha=0.7
                )
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
            ax.grid(True, alpha=0.3)
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        # single shared axis labels
        fig.supxlabel("Market Cap - \\$300M (\\$M)", fontsize=11)
        fig.supylabel("dayuserchgw", fontsize=11)
        plt.tight_layout(rect=[0.03, 0.03, 1, 1])
        plt.savefig(
            figures_dir / f"{prefix}plot_cov_adj_scatter_bw{bw}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# polynomial fits per bandwidth
def plot_rdd_paper_style(results_all, sample_by_bw, prefix="", title_suffix=""):
    for bw in sorted(sample_by_bw.keys()):
        df_sample = sample_by_bw[bw].copy()
        if len(df_sample) == 0:
            continue
        x = df_sample["mktcap1_300"].values
        y = df_sample["dayuserchgw"].values
        res_bw = results_all[
            (results_all["bandwidth"] == bw) & (results_all["poly_order"] == 1)
        ]
        nocov = res_bw[res_bw["method"] == "nocov"]
        beta_nocov = nocov["beta1"].values[0] if len(nocov) > 0 else np.nan
        se_nocov = nocov["se"].values[0] if len(nocov) > 0 else np.nan
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for pidx, poly_order in enumerate([1, 2, 3]):
            ax = axes[pidx]
            colors = np.where(x >= 0, "#d62728", "#1f77b4")
            ax.scatter(x, y, c=colors, alpha=0.2, s=10, edgecolors="none")
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            left_mask = x < 0
            right_mask = x >= 0
            for mask, side_color in [(left_mask, "navy"), (right_mask, "darkred")]:
                if np.sum(mask) < poly_order + 1:
                    continue
                xm = x[mask]
                ym = y[mask]
                try:
                    coeffs = np.polyfit(xm, ym, poly_order)
                    xfit = np.linspace(xm.min(), xm.max(), 200)
                    yfit = np.polyval(coeffs, xfit)
                    ax.plot(xfit, yfit, color=side_color, linewidth=2.5)
                except np.linalg.LinAlgError:
                    pass
            # binned means
            n_bins = 20
            bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_means = []
            for i in range(n_bins):
                mask_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
                bin_means.append(
                    np.nanmean(y[mask_bin]) if mask_bin.sum() > 0 else np.nan
                )
            bin_colors = ["#d62728" if c >= 0 else "#1f77b4" for c in bin_centers]
            ax.scatter(
                bin_centers,
                bin_means,
                c=bin_colors,
                s=50,
                edgecolors="black",
                linewidth=0.5,
                zorder=5,
            )
            paper = PAPER_BENCHMARK_TABLE7.get((bw, poly_order))
            paper_txt = ""
            if paper is not None:
                paper_txt = f"\nPaper: {paper['beta1']:.1f} (SE {paper['se']:.1f})"
            ax.set_title(f"Polynomial order {poly_order}", fontsize=11)
            ax.set_xlabel("Market cap − $300M")
            ax.set_ylabel("dayuserchgw")
            ax.grid(True, alpha=0.3)
            # show paper estimate for all panels
            if paper is not None:
                txt = f"Paper: {paper['beta1']:.1f} (SE {paper['se']:.1f})"
                ax.text(
                    0.03,
                    0.97,
                    txt,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"{prefix}plot_rdd_scatter_bw{bw}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# sample comparison
def plot_observation_comparison(sample_by_bw):
    bandwidths = sorted(sample_by_bw.keys())
    our_n = [len(sample_by_bw[bw]) for bw in bandwidths]
    paper_n = [PAPER_BENCHMARK_TABLE7.get((bw, 1), {}).get("n", 0) for bw in bandwidths]
    x = np.arange(len(bandwidths))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_w / 2, our_n, bar_w, label="Replication", color="#1f77b4")
    ax.bar(x + bar_w / 2, paper_n, bar_w, label="Original", color="#ff7f0e")
    for i, (ours, paper) in enumerate(zip(our_n, paper_n)):
        ax.text(
            x[i] - bar_w / 2,
            ours + 20,
            f"{ours:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x[i] + bar_w / 2,
            paper + 20,
            f"{paper:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"${bw}M" for bw in bandwidths])
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Number of Observations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "plot_observation_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# auto-bandwidth comparison
def plot_rdflex_bw_comparison(results_hardcoded, results_auto):
    if results_auto.empty:
        return
    # use poly_order=1 from hardcoded results at the $50M bandwidth for comparison
    hc_50 = results_hardcoded[
        (results_hardcoded["poly_order"] == 1) & (results_hardcoded["bandwidth"] == 50)
    ]
    methods = [m for m in METHOD_COLORS.keys() if m in results_auto["method"].values]
    if not methods:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    coef_auto = []
    coef_hard = []
    ci_auto = []
    ci_hard = []
    method_labels = []
    for m in methods:
        a = results_auto[results_auto["method"] == m]
        h = hc_50[hc_50["method"] == m]
        method_labels.append(METHOD_LABELS.get(m, m))
        coef_auto.append(a["beta1"].values[0] if len(a) > 0 else np.nan)
        coef_hard.append(h["beta1"].values[0] if len(h) > 0 else np.nan)
        ci_auto.append(a["ci_width"].values[0] if len(a) > 0 else np.nan)
        ci_hard.append(h["ci_width"].values[0] if len(h) > 0 else np.nan)
    x = np.arange(len(methods))
    w = 0.35
    # coefficient estimates
    ax1.bar(
        x - w / 2,
        coef_auto,
        w,
        label="Auto BW",
        color="#87CEEB",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.bar(
        x + w / 2,
        coef_hard,
        w,
        label="Hardcoded ($50M)",
        color="#2ca02c",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Treatment Effect")
    ax1.set_title("Coefficient Estimates")
    ax1.legend()
    ax1.axhline(0, color="gray", ls="--", alpha=0.5)
    # CI widths
    ax2.bar(
        x - w / 2,
        ci_auto,
        w,
        label="Auto BW",
        color="#87CEEB",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + w / 2,
        ci_hard,
        w,
        label="Hardcoded ($50M)",
        color="#2ca02c",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("CI Width")
    ax2.set_title("Confidence Interval Width")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "plot_auto_bw_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# Diagnostic plots


# covariate importance
def plot_covariate_importance(sample_by_bw):
    # use largest bandwidth for most data
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    if len(covars) == 0:
        return
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50:
        return
    X = df_prep[covars]
    y = df_prep["dayuserchgw"]
    methods = {
        "Ridge": RidgeCV(cv=5),
        "Lasso": LassoCV(cv=5, n_jobs=2, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=2
        ),
        "LightGBM": LGBMRegressor(n_estimators=300, verbose=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, verbosity=0, random_state=42),
    }
    for name, model in methods.items():
        model.fit(X, y)
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=covars)
        elif hasattr(model, "coef_"):
            imp = pd.Series(np.abs(model.coef_), index=covars)
        else:
            continue
        # skip if all zero
        if imp.abs().sum() < 1e-10:
            continue
        # normalize to relative importance
        total = imp.sum()
        if total > 0:
            imp = imp / total
        imp = imp.sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [COVARIATE_LABELS.get(c, c) for c in imp.index],
            imp.values,
            color="black",
            edgecolor="black",
        )
        ax.set_xlabel("Relative Feature Importance")
        fig.tight_layout()
        fig.savefig(figures_dir / f"covariate_importance_{name.lower()}.png", dpi=150)
        plt.close(fig)


# covariate balance
def plot_covariate_balance(sample_by_bw):
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["gt300"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    if len(covars) == 0:
        return
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50:
        return
    n = len(covars)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    for i, c in enumerate(covars):
        ax = axes[i]
        for val, color, label in [
            (1, "#d62728", "Above $300M"),
            (0, "#1f77b4", "Below $300M"),
        ]:
            vals = df_prep.loc[df_prep["gt300"] == val, c].dropna()
            if len(vals) > 0:
                ax.hist(
                    vals, bins=25, alpha=0.5, color=color, label=label, density=True
                )
        ax.set_title(COVARIATE_LABELS.get(c, c), fontsize=9)
        ax.legend(fontsize=7)
    for j in range(len(covars), len(axes)):
        axes[j].axis("off")
    # title removed per figure optimization
    fig.tight_layout()
    fig.savefig(
        figures_dir / "plot_covariate_balance.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# Super Learner diagnostics


# super learner weights and base learner performance
def plot_super_learner_weights(sample_by_bw):
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    if len(covars) == 0:
        return
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50:
        return
    X = df_prep[covars].values.astype(float)
    y = df_prep["dayuserchgw"].values.astype(float)
    # tune models for SL analysis
    tuned = _tune_models(X, y, tag="SL-weights")
    # Lasso with interactions
    lasso_pipe = SampleWeightPipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=2, interaction_only=False, include_bias=False
                ),
            ),
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=5, n_jobs=2, max_iter=5000)),
        ]
    )
    nnet_pipe = SampleWeightPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    **tuned.get("nnet", {}),
                ),
            ),
        ]
    )
    sl = SuperLearnerRegressor(
        estimators=[
            ("Ridge", RidgeCV(cv=5)),
            ("Lasso", clone(lasso_pipe)),
            ("RF", RandomForestRegressor(random_state=42, n_jobs=2, **tuned["rf"])),
            ("LightGBM", LGBMRegressor(verbose=-1, n_jobs=2, **tuned["lgbm"])),
            ("XGBoost", XGBRegressor(verbosity=0, n_jobs=2, **tuned["xgb"])),
            ("NNet", clone(nnet_pipe)),
        ],
        final_estimator=RidgeCV(cv=3),
        cv=3,
    )
    sl.fit(X, y)
    # extract meta-learner weights
    meta = sl.meta_
    names = [nm for nm, _ in sl.estimators]
    if hasattr(meta, "coef_"):
        weights = meta.coef_
    else:
        weights = np.ones(len(names)) / len(names)
    # base learner CV
    base_r2 = {}
    for nm, est in sl.estimators:
        r2 = cross_val_score(clone(est), X, y, cv=3, scoring="r2", n_jobs=2).mean()
        base_r2[nm] = r2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # map SL base-learner names to METHOD_COLORS keys
    _sl_method_map = {
        "Ridge": "ridge",
        "Lasso": "lasso",
        "RF": "rf",
        "LightGBM": "lgbm",
        "XGBoost": "xgb",
        "NNet": "nnet",
    }
    sl_bar_colors = [
        METHOD_COLORS.get(_sl_method_map.get(n, ""), "gray") for n in names
    ]
    # meta-learner weights
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, weights, color=sl_bar_colors, edgecolor="black", linewidth=0.5)
    for i, w in enumerate(weights):
        ax1.text(max(w, 0) + 0.002, i, f"{w:.3f}", va="center", fontsize=8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Meta-Learner Weight (Ridge coef)")
    ax1.set_title("Super Learner: Base Learner Weights")
    ax1.axvline(0, color="black", lw=0.5)
    # base learner CV
    r2_vals = [base_r2[n] for n in names]
    ax2.barh(y_pos, r2_vals, color=sl_bar_colors, edgecolor="black", linewidth=0.5)
    for i, r2 in enumerate(r2_vals):
        ax2.text(max(r2, 0) + 0.002, i, f"{r2:.4f}", va="center", fontsize=8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Cross-Validated R²")
    ax2.set_title("Base Learner Performance")
    ax2.axvline(0, color="black", lw=0.5)
    # main suptitle removed, subtitles on ax1/ax2 remain
    fig.tight_layout()
    fig.savefig(
        figures_dir / "plot_super_learner_weights.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    gc.collect()


# covariate vs running variable
def plot_covariate_running_var_correlation(sample_by_bw):
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["mktcap1_300"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    if len(covars) == 0:
        return
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50:
        return
    running_var = df_prep["mktcap1_300"].values
    n_covars = len(covars)
    ncols = min(3, n_covars)
    nrows = int(np.ceil(n_covars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    correlations = {}
    for i, cov in enumerate(covars):
        ax = axes[i]
        cov_vals = df_prep[cov].values.astype(float)
        valid = np.isfinite(running_var) & np.isfinite(cov_vals)
        x_v = running_var[valid]
        y_v = cov_vals[valid]
        # pearson correlation
        if len(x_v) > 2:
            corr = np.corrcoef(x_v, y_v)[0, 1]
        else:
            corr = np.nan
        correlations[cov] = corr
        # scatter with density coloring
        colors_side = np.where(x_v >= 0, "#d62728", "#1f77b4")
        ax.scatter(x_v, y_v, c=colors_side, alpha=0.15, s=5, edgecolors="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        # local polynomial fit on each side
        for mask, color in [
            (x_v < 0, "navy"),
            (x_v >= 0, "darkred"),
        ]:
            xm, ym = x_v[mask], y_v[mask]
            if len(xm) > 10:
                try:
                    coeffs = np.polyfit(xm, ym, min(3, len(xm) - 1))
                    xfit = np.linspace(xm.min(), xm.max(), 100)
                    yfit = np.polyval(coeffs, xfit)
                    ax.plot(xfit, yfit, color=color, linewidth=2)
                except np.linalg.LinAlgError:
                    pass
        ax.set_title(
            f"{COVARIATE_LABELS.get(cov, cov)}\nr = {corr:.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
    for j in range(n_covars, len(axes)):
        axes[j].axis("off")
    # single shared x-axis label
    fig.supxlabel("Market Cap - $300M ($M)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(
        figures_dir / "plot_covariate_running_var_corr.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    # summary bar plot of correlations
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sorted_covars = sorted(correlations.keys(), key=lambda c: abs(correlations[c]))
    labels = [COVARIATE_LABELS.get(c, c) for c in sorted_covars]
    vals = [correlations[c] for c in sorted_covars]
    colors = ["steelblue" if v >= 0 else "salmon" for v in vals]
    ax2.barh(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Pearson Correlation with Running Variable")
    for i, v in enumerate(vals):
        ax2.text(
            v + 0.005 * np.sign(v),
            i,
            f"{v:.3f}",
            va="center",
            fontsize=8,
        )
    fig2.tight_layout()
    fig2.savefig(
        figures_dir / "plot_covariate_corr_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2)


# lasso covariate selection diagnostic
def report_lasso_covariate_selection(sample_by_bw):
    pass


# standard error comparison
def plot_se_comparison(results_all):
    if results_all.empty:
        return
    for poly_order in POLY_ORDERS:
        df_p = results_all[results_all["poly_order"] == poly_order]
        if len(df_p) == 0:
            continue
        bandwidths = sorted(df_p["bandwidth"].unique())
        methods = [m for m in METHOD_COLORS.keys() if m in df_p["method"].values]
        n_groups = len(methods) + 1  # +1 for paper benchmark
        width = 0.8 / n_groups
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(bandwidths))
        # paper SE
        paper_ses = [
            PAPER_BENCHMARK_TABLE7.get((bw, poly_order), {}).get("se", np.nan)
            for bw in bandwidths
        ]
        ax.bar(
            x - 0.4 + width / 2,
            paper_ses,
            width,
            label="Paper Table VII",
            color="white",
            edgecolor="black",
            linewidth=1.0,
            hatch="//",
        )
        # methods SE
        for j, method in enumerate(methods):
            ses = []
            for bw in bandwidths:
                row = df_p[(df_p["method"] == method) & (df_p["bandwidth"] == bw)]
                ses.append(row["se"].values[0] if len(row) > 0 else np.nan)
            ax.bar(
                x - 0.4 + (j + 1.5) * width,
                ses,
                width,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "gray"),
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"${bw}M" for bw in bandwidths])
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("Standard Error")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"plot_se_comparison_p{poly_order}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# relative CI width reduction
def plot_relative_se_reduction(results_all):
    if results_all.empty:
        return
    for poly_order in POLY_ORDERS:
        df_p = results_all[results_all["poly_order"] == poly_order].copy()
        if len(df_p) == 0:
            continue
        bandwidths = sorted(df_p["bandwidth"].dropna().unique())
        # methods excluding nocov (it is the baseline = 1.0)
        ml_methods = [
            m
            for m in METHOD_COLORS.keys()
            if m in df_p["method"].values and m != "nocov"
        ]
        if not ml_methods:
            continue
        # compute relative CI width: method_ci_width / nocov_ci_width
        norm_rows = []
        for bw in bandwidths:
            nocov_row = df_p[(df_p["bandwidth"] == bw) & (df_p["method"] == "nocov")]
            if len(nocov_row) == 0:
                continue
            nocov_ci = nocov_row["ci_width"].values[0]
            if np.isnan(nocov_ci) or nocov_ci <= 0:
                continue
            for method in ml_methods:
                m_row = df_p[(df_p["bandwidth"] == bw) & (df_p["method"] == method)]
                if len(m_row) == 0:
                    continue
                m_ci = m_row["ci_width"].values[0]
                if np.isnan(m_ci):
                    continue
                norm_rows.append(
                    {
                        "bandwidth": bw,
                        "method": method,
                        "relative_ci": m_ci / nocov_ci,
                        "se_reduction_pct": (1 - m_ci / nocov_ci) * 100,
                    }
                )
        if not norm_rows:
            continue
        norm_df = pd.DataFrame(norm_rows)
        # grouped bar chart
        n_methods = len(ml_methods)
        bar_width = 0.8 / max(n_methods, 1)
        x_base = np.arange(len(bandwidths))
        fig, ax = plt.subplots(figsize=(max(12, n_methods * 1.5), 7))
        for j, method in enumerate(ml_methods):
            vals = []
            for bw in bandwidths:
                row = norm_df[
                    (norm_df["method"] == method) & (norm_df["bandwidth"] == bw)
                ]
                vals.append(row["relative_ci"].values[0] if len(row) > 0 else np.nan)
            offset = (j - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_base + offset,
                vals,
                width=bar_width,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "gray"),
                edgecolor="black",
                linewidth=0.5,
            )
            # annotate bars with CI change %
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    change_pct = (v - 1) * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{change_pct:+.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"${int(bw)}M" for bw in bandwidths])
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("Relative CI Width (No Covariates = 1.0)")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"plot_relative_ci_reduction_p{poly_order}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# forrest plot: point estimates vs unadjusted CI
def plot_forest_estimates(results_all):
    if results_all.empty:
        return
    df = results_all[
        (results_all["bandwidth"] == 125) & (results_all["poly_order"] == 1)
    ].copy()
    if len(df) == 0:
        return
    method_order = ["nocov", "ridge", "lasso", "rf", "lgbm", "xgb", "nnet", "sl"]
    methods = [m for m in method_order if m in df["method"].values]
    if "nocov" not in methods:
        return
    nocov = df[df["method"] == "nocov"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    # shaded band
    ax.axvspan(
        nocov["ci_lower"],
        nocov["ci_upper"],
        color="#000000",
        alpha=0.08,
        label="Unadjusted 95% CI",
    )
    # dashed line at nocov point estimate
    ax.axvline(
        nocov["beta1"],
        color="#000000",
        linestyle="--",
        linewidth=1.2,
        label=f"Unadjusted estimate ({nocov['beta1']:.1f})",
    )
    # solid line at zero
    ax.axvline(0, color="gray", linewidth=0.8)
    y_positions = list(range(len(methods)))
    for i, method in enumerate(methods):
        row = df[df["method"] == method].iloc[0]
        mc = METHOD_COLORS.get(method, "gray")
        ax.errorbar(
            row["beta1"],
            i,
            xerr=[
                [row["beta1"] - row["ci_lower"]],
                [row["ci_upper"] - row["beta1"]],
            ],
            fmt="o",
            markersize=8,
            color=mc,
            capsize=4,
            capthick=1.5,
            markerfacecolor=mc,
            markeredgecolor="black",
            linewidth=1.5,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_xlabel(r"Treatment effect ($\Delta$ users)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(figures_dir / "forest_estimates_p1.png", dpi=150, bbox_inches="tight")
    plt.close()


# standardised difference
def plot_standardized_diff(results_all):
    if results_all.empty:
        return
    df = results_all[
        (results_all["bandwidth"] == 125) & (results_all["poly_order"] == 1)
    ].copy()
    if len(df) == 0:
        return
    method_order = ["ridge", "lasso", "rf", "lgbm", "xgb", "nnet", "sl"]
    methods = [m for m in method_order if m in df["method"].values]
    nocov_rows = df[df["method"] == "nocov"]
    if len(nocov_rows) == 0 or not methods:
        return
    nocov = nocov_rows.iloc[0]
    beta_nocov = nocov["beta1"]
    se_nocov = nocov["se"]
    if np.isnan(se_nocov) or se_nocov <= 0:
        return
    std_diffs = []
    for method in methods:
        row = df[df["method"] == method].iloc[0]
        std_diffs.append((row["beta1"] - beta_nocov) / se_nocov)
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 6))
    # shaded band at +/- 2 (green, matching Griffin style)
    ax.axhspan(-2, 2, color="#009E73", alpha=0.15, label=r"$\pm 2$ SE band")
    ax.axhline(0, color="black", linewidth=0.8)
    x_pos = np.arange(len(methods))
    for i, (method, sd) in enumerate(zip(methods, std_diffs)):
        color = METHOD_COLORS.get(method, "gray") if abs(sd) <= 2 else "#D55E00"
        ax.bar(
            x_pos[i],
            sd,
            width=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        # annotate value
        va = "bottom" if sd >= 0 else "top"
        offset = 0.05 if sd >= 0 else -0.05
        ax.text(
            x_pos[i],
            sd + offset,
            f"{sd:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right"
    )
    ax.set_ylabel(
        r"$(\hat{\tau}_m - \hat{\tau}_{\mathrm{nocov}}) \;/\; "
        r"\mathrm{SE}_{\mathrm{nocov}}$"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(figures_dir / "standardized_diff_p1.png", dpi=150, bbox_inches="tight")
    plt.close()


# plot statistical power
def compute_power_analysis(results_all):
    if results_all.empty:
        return
    # build table
    col_labels = [
        "BW ($M)",
        "Poly",
        "Method",
        "N",
        "SE (nocov)",
        "MDE (80%)",
        "Paper Coef",
    ]
    cells = []
    power_rows = []
    for bw in sorted(results_all["bandwidth"].dropna().unique()):
        for poly in POLY_ORDERS:
            # get nocov SE as the baseline for MDE
            nocov_row = results_all[
                (results_all["bandwidth"] == bw)
                & (results_all["poly_order"] == poly)
                & (results_all["method"] == "nocov")
            ]
            if len(nocov_row) == 0:
                continue
            nocov_se = nocov_row["se"].values[0]
            n_obs = int(nocov_row["n_obs"].values[0])
            paper = PAPER_BENCHMARK_TABLE7.get((bw, poly), {})
            paper_beta = paper.get("beta1", np.nan)
            if np.isnan(nocov_se) or nocov_se <= 0:
                continue
            mde = 2.8 * nocov_se
            if not np.isnan(paper_beta):
                detectable = "Yes" if abs(paper_beta) > mde else "No"
            else:
                detectable = "N/A"
            cells.append(
                [
                    f"${bw}M",
                    str(poly),
                    "nocov",
                    f"{n_obs:,}",
                    f"{nocov_se:.2f}",
                    f"{mde:.2f}",
                    f"{paper_beta:.2f}" if not np.isnan(paper_beta) else "N/A",
                ]
            )
            power_rows.append(
                {
                    "bandwidth": bw,
                    "poly_order": poly,
                    "se": nocov_se,
                    "mde": mde,
                    "paper_beta": paper_beta,
                    "detectable": detectable,
                    "n_obs": n_obs,
                }
            )
    if not cells:
        return
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(output_dir / "power_analysis.csv", index=False)
    # render as table figure
    fig_h = 1.5 + 0.38 * max(len(cells), 1)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9d9d9")
    plt.tight_layout()
    plt.savefig(figures_dir / "plot_power_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# placebo cutoff and covariate smoothness tests
def robustness_predetermined_and_placebo(sample_by_bw):
    max_bw = max(sample_by_bw.keys())
    min_bw = min(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    if len(covars) == 0:
        return
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 100:
        return
    # Covariate smoothness at the cutoff
    near_cutoff = df_prep[df_prep["mktcap1_300"].abs() <= min_bw]
    left = near_cutoff[near_cutoff["mktcap1_300"] < 0]
    right = near_cutoff[near_cutoff["mktcap1_300"] >= 0]
    col_labels_a = [
        "Covariate",
        "Below Mean",
        "Above Mean",
        "Diff",
        "t-stat",
        "p-value",
    ]
    cells_a = []
    for cov in covars:
        left_vals = left[cov].dropna().values.astype(float)
        right_vals = right[cov].dropna().values.astype(float)
        if len(left_vals) < 5 or len(right_vals) < 5:
            continue
        mean_l = np.mean(left_vals)
        mean_r = np.mean(right_vals)
        diff = mean_r - mean_l
        se_diff = np.sqrt(
            np.var(left_vals, ddof=1) / len(left_vals)
            + np.var(right_vals, ddof=1) / len(right_vals)
        )
        if se_diff > 0:
            t_stat = diff / se_diff
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
        else:
            t_stat = np.nan
            p_val = np.nan
        label = COVARIATE_LABELS.get(cov, cov)
        cells_a.append(
            [
                label,
                f"{mean_l:.4f}",
                f"{mean_r:.4f}",
                f"{diff:+.4f}",
                f"{t_stat:.3f}" if not np.isnan(t_stat) else "N/A",
                f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
            ]
        )
    # render as table figure
    if cells_a:
        fig_h = 1.5 + 0.38 * max(len(cells_a), 1)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=cells_a,
            colLabels=col_labels_a,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.3)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#d9d9d9")
            # highlight significant p-values
            if c == 5 and r > 0:
                try:
                    pv = float(cell.get_text().get_text())
                    if pv < 0.05:
                        cell.set_facecolor("#f8d7da")
                except (ValueError, AttributeError):
                    pass
        plt.tight_layout()
        plt.savefig(
            figures_dir / "robustness_covariate_smoothness.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    # Placebo cutoff tests
    placebo_shifts = [-100, -50, 50, 100]
    placebo_results = []
    for placebo_shift in placebo_shifts:
        df_placebo = df_prep.copy()
        df_placebo["mktcap1_300"] = df_placebo["mktcap1_300"] - placebo_shift
        df_placebo["gt300"] = (df_placebo["mktcap1_300"] >= 0).astype(int)
        in_bw = df_placebo["mktcap1_300"].abs() <= max_bw
        df_sub = df_placebo[in_bw]
        if len(df_sub) < 50:
            continue
        try:
            rdd_data = DoubleMLRDDData(
                data=df_sub,
                y_col="dayuserchgw",
                d_cols="gt300",
                score_col="mktcap1_300",
                x_cols=covars,
            )
            est = RDFlex(
                obj_dml_data=rdd_data,
                ml_g=DummyRegressor(strategy="mean"),
                ml_m=None,
                fuzzy=False,
                cutoff=0,
                RDFLEX_N_Folds=RDFLEX_N_Folds,
                RDFLEX_N_REP=min(RDFLEX_N_REP, 5),
                h_fs=float(max_bw),
                fs_kernel="triangular",
                p=1,
            )
            est.fit()
            coef = float(est.coef[0])
            se = float(est.se[0])
            pval = float(est.pval[0])
            placebo_results.append(
                {
                    "cutoff": 300 + placebo_shift,
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                }
            )
        except Exception as exc:
            pass
    if not placebo_results:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot placebo cutoffs in blue
    for r in placebo_results:
        ax.errorbar(
            r["cutoff"],
            r["coef"],
            yerr=1.96 * r["se"],
            fmt="o",
            color="#1f77b4",
            capsize=4,
            markersize=8,
            linewidth=2,
        )
    # add paper's reference estimate at true cutoff
    paper_ref = PAPER_BENCHMARK_TABLE7.get((max(sample_by_bw.keys()), 1), {})
    if paper_ref:
        ax.errorbar(
            300,
            paper_ref["beta1"],
            yerr=1.96 * paper_ref["se"],
            fmt="D",
            color="#d62728",
            capsize=5,
            markersize=10,
            linewidth=2,
            label=f"Paper estimate: {paper_ref['beta1']:.2f}",
        )
        ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=300, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Cutoff Location ($M)")
    ax.set_ylabel("Treatment Effect Estimate")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "robustness_placebo_cutoffs.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


#  Plot density of running variable
def robustness_density_test(sample_by_bw):
    max_bw = max(sample_by_bw.keys())
    df_sample = sample_by_bw[max_bw]
    running = df_sample["mktcap1_300"].dropna().values
    n_bins = 50
    below = running[running <= 0]
    above = running[running > 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    bins_below = np.linspace(running.min(), 0, n_bins + 1)
    bins_above = np.linspace(0, running.max(), n_bins + 1)
    ax.hist(
        below,
        bins=bins_below,
        color="#1f77b4",
        alpha=0.7,
        label=f"Below cutoff (N={len(below):,})",
    )
    ax.hist(
        above,
        bins=bins_above,
        color="#d62728",
        alpha=0.7,
        label=f"Above cutoff (N={len(above):,})",
    )
    ax.axvline(x=0, color="black", linewidth=2, linestyle="--", label="Cutoff")
    # McCrary log-density discontinuity estimate
    narrow = 10.0  # $10M window on each side
    count_below = np.sum((running >= -narrow) & (running <= 0))
    count_above = np.sum((running > 0) & (running <= narrow))
    dens_below = count_below / (narrow * len(running))
    dens_above = count_above / (narrow * len(running))
    if dens_above > 0:
        density_ratio = dens_below / dens_above
    else:
        density_ratio = np.nan
    if dens_below > 0 and dens_above > 0 and count_below > 0 and count_above > 0:
        log_diff = np.log(dens_below) - np.log(dens_above)
        se_log = np.sqrt(1.0 / count_below + 1.0 / count_above)
        t_stat = log_diff / se_log
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        log_diff = np.nan
        t_stat = np.nan
        p_val = np.nan
    if not np.isnan(p_val):
        if p_val < 0.05:
            interp = "SIGNIFICANT — potential manipulation"
        else:
            interp = "NOT significant — no evidence of manipulation"
    else:
        interp = "N/A"
    ax.set_xlabel("Market Cap - $300M ($M)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "robustness_density_test.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    # save log density difference to CSV
    density_df = pd.DataFrame(
        [
            {
                "log_density_diff": log_diff,
                "t_stat": t_stat,
                "p_value": p_val,
                "density_ratio": density_ratio,
                "narrow_window": narrow,
                "n_below": len(below),
                "n_above": len(above),
                "interpretation": interp,
            }
        ]
    )
    density_df.to_csv(output_dir / "density_test_results.csv", index=False)


# Plot sensitivity to bandwidth choice
def robustness_bandwidth_sensitivity(sample_by_bw, results_all):
    if results_all.empty:
        return
    n_poly = len(POLY_ORDERS)
    fig, axes = plt.subplots(1, n_poly, figsize=(6 * n_poly, 6))
    if n_poly == 1:
        axes = [axes]
    for idx, p in enumerate(POLY_ORDERS):
        ax = axes[idx]
        sub = results_all[
            (results_all["method"] == "nocov") & (results_all["poly_order"] == p)
        ].sort_values("bandwidth")
        if len(sub) == 0:
            ax.set_title(f"Poly {p}: no data")
            continue
        bws = sub["bandwidth"].values.astype(float)
        coefs = sub["beta1"].values.astype(float)
        ses = sub["se"].values.astype(float)
        ci_lo = coefs - 1.96 * ses
        ci_hi = coefs + 1.96 * ses
        ax.plot(bws, coefs, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.fill_between(bws, ci_lo, ci_hi, alpha=0.2, color="#1f77b4")
        # paper benchmark
        paper_beta = PAPER_BENCHMARK_TABLE7.get((BANDWIDTHS[0], p), {}).get("beta1")
        if paper_beta is not None:
            ax.axhline(
                y=paper_beta,
                color="#d62728",
                linestyle="--",
                linewidth=1.5,
                label=f"Paper: {paper_beta:.2f}",
            )
            ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Bandwidth ($M)")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"Poly {p}")
    # title removed per figure optimization
    plt.tight_layout()
    fig.savefig(
        figures_dir / "robustness_bw_sensitivity.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# plot kernel sensitivity
def robustness_kernel_comparison(sample_by_bw, results_all):
    if results_all.empty:
        return
    # preferred specification: bw=125, poly=1
    bw = 125
    poly_order = 1
    if bw not in sample_by_bw:
        return
    df_sample = sample_by_bw[bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 100 or len(covars) == 0:
        return
    # tune once
    Z = df_prep[covars].values.astype(float)
    y_tune = df_prep["dayuserchgw"].values.astype(float)
    tuned = _tune_models(Z, y_tune, tag="kernel-robustness")
    configs_all = dict(_build_adjustment_configs(tuned, sample_n=len(df_prep)))
    # all methods that have a config (excludes rdrobust which runs in R)
    test_methods = [
        m for m in METHOD_COLORS.keys() if m in configs_all and m != "rdrobust"
    ]
    kernels = ["triangular", "epanechnikov"]
    kernel_results = []
    for kernel in kernels:
        for method_name in test_methods:
            ml_g = clone(configs_all[method_name])
            try:
                rdd_data = DoubleMLRDDData(
                    data=df_prep,
                    y_col="dayuserchgw",
                    d_cols="gt300",
                    score_col="mktcap1_300",
                    x_cols=covars,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=0,
                    RDFLEX_N_Folds=RDFLEX_N_Folds,
                    RDFLEX_N_REP=RDFLEX_N_REP,
                    h_fs=float(bw),
                    fs_kernel=kernel,
                    p=poly_order,
                )
                est.fit()
                rec = _record_rdflex(
                    est, bw, poly_order, method_name, len(df_prep), covars
                )
                rec["kernel"] = kernel
                kernel_results.append(rec)
                del est, rdd_data
                gc.collect()
            except Exception as exc:
                pass
    if not kernel_results:
        return
    kernel_df = pd.DataFrame(kernel_results)
    kernel_df.to_csv(output_dir / "kernel_robustness.csv", index=False)
    # plot grouped bar chart of relative CI width per kernel
    _plot_kernel_robustness(kernel_df)


# grouped bar chart of relative CI width per kernel
def _plot_kernel_robustness(kernel_df):
    methods_in_results = [
        m
        for m in METHOD_COLORS.keys()
        if m in kernel_df["method"].values and m != "nocov"
    ]
    if len(methods_in_results) == 0:
        return
    kernels = ["triangular", "epanechnikov"]
    # only plot kernels that actually have data
    kernels = [k for k in kernels if k in kernel_df["kernel"].unique()]
    n_kernels = len(kernels)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods_in_results))
    bar_width = 0.8 / max(n_kernels, 1)
    for i, kernel in enumerate(kernels):
        kdf = kernel_df[kernel_df["kernel"] == kernel]
        nocov_row = kdf[kdf["method"] == "nocov"]
        if len(nocov_row) == 0:
            continue
        nocov_ci = nocov_row["ci_width"].values[0]
        if np.isnan(nocov_ci) or nocov_ci <= 0:
            continue
        ratios = []
        for method in methods_in_results:
            m_row = kdf[kdf["method"] == method]
            if len(m_row) > 0 and not np.isnan(m_row["ci_width"].values[0]):
                ratios.append(m_row["ci_width"].values[0] / nocov_ci)
            else:
                ratios.append(np.nan)
        offset = (i - (n_kernels - 1) / 2) * bar_width
        # use METHOD_COLORS per method
        hatch = "//" if kernel == "epanechnikov" else None
        for j, (method, v) in enumerate(zip(methods_in_results, ratios)):
            mc = METHOD_COLORS.get(method, "gray")
            bar = ax.bar(
                x_pos[j] + offset,
                v if not np.isnan(v) else 0,
                width=bar_width,
                color=mc,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            if not np.isnan(v):
                change_pct = (v - 1) * 100
                ax.text(
                    x_pos[j] + offset,
                    v + 0.005,
                    f"{change_pct:+.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    legend_handles = [
        Patch(facecolor="#999999", edgecolor="black", label="Triangular"),
        Patch(facecolor="#999999", edgecolor="black", hatch="//", label="Epanechnikov"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods_in_results],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Relative CI Width (nocov = 1.0)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "robustness_kernel_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# Summary Statistics replication
def _build_summary_frame(df):
    d = df.drop_duplicates(subset=["ticker", "date"], keep="first")
    d = d.sort_values(["ticker", "date"])
    if "users_lag" not in d.columns and "users_close" in d.columns:
        d["users_lag"] = (
            pd.to_numeric(d["users_close"], errors="coerce")
            .groupby(d["ticker"])
            .shift(1)
        )
    out = pd.DataFrame(index=d.index)
    price_col = (
        "close_usd"
        if "close_usd" in d.columns
        else ("close" if "close" in d.columns else None)
    )
    mktcap_col = (
        "mktcap_usd"
        if "mktcap_usd" in d.columns
        else ("mktcap" if "mktcap" in d.columns else None)
    )
    ret_col = (
        "ret_usd" if "ret_usd" in d.columns else ("ret" if "ret" in d.columns else None)
    )
    if "users_close" in d.columns:
        out["users_close"] = pd.to_numeric(d["users_close"], errors="coerce")
    if "users_lag" in d.columns:
        out["users_lag"] = pd.to_numeric(d["users_lag"], errors="coerce")
    if "userchg" in d.columns:
        out["userchg"] = pd.to_numeric(d["userchg"], errors="coerce")
    if "users_close" in out.columns and "users_lag" in out.columns:
        users_lag_pos = out["users_lag"].where(out["users_lag"] > 0)
        ratio = out["users_close"] / users_lag_pos
        out["userratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    if price_col is not None:
        prc = pd.to_numeric(d[price_col], errors="coerce")
        out["prc"] = prc.where(prc > 0)
    if mktcap_col is not None:
        size = pd.to_numeric(d[mktcap_col], errors="coerce")
        if size.dropna().median() > 1e7:
            size = size / 1e6
        out["size($mil)"] = size.where(size > 0)
    if ret_col is not None:
        out["ret(%)"] = (pd.to_numeric(d[ret_col], errors="coerce") * 100).replace(
            [np.inf, -np.inf], np.nan
        )
    if price_col is not None and "open" in d.columns:
        lag_price = d.groupby("ticker")[price_col].shift(1)
        lag_price_pos = lag_price.where(lag_price > 0)
        open_num = pd.to_numeric(d["open"], errors="coerce")
        open_pos = open_num.where(open_num > 0)
        price_num = pd.to_numeric(d[price_col], errors="coerce")
        out["openret(%)"] = ((open_num - lag_price_pos) / lag_price_pos * 100).replace(
            [np.inf, -np.inf], np.nan
        )
        out["dayret(%)"] = ((price_num - open_pos) / open_pos * 100).replace(
            [np.inf, -np.inf], np.nan
        )
    out["date"] = d["date"].values
    out["ticker"] = d["ticker"].values
    return out


# compute descriptive statistics for the given variables
def _summary_stats(frame, variables):
    rows = []
    for v in variables:
        if v not in frame.columns:
            continue
        s = pd.to_numeric(frame[v], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "variable": v,
                "N": int(len(s)),
                "mean": float(s.mean()),
                "sd": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


# format a numeric table cell as count or rounded value
def _fmt_cell(x, is_count=False):
    if pd.isna(x):
        return ""
    if is_count:
        return f"{int(x):,}"
    ax = abs(x)
    if ax >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


# render one or more summary-statistics panels as a table figure
def _render_stats_table(title, subtitle, save_path, panels=None):
    col_labels = ["variable", "N", "mean", "sd", "min", "p25", "p50", "p75", "max"]
    if panels is None:
        panels = []
    total_rows = sum(len(p[1]) for p in panels) + len(panels)
    fig_h = 1.2 + 0.38 * max(total_rows, 1)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    cells = []
    row_labels = []
    for panel_name, pdf in panels:
        if panel_name:
            cells.append([panel_name] + [""] * (len(col_labels) - 1))
            row_labels.append("")
        for _, r in pdf.iterrows():
            cells.append(
                [
                    r["variable"],
                    _fmt_cell(r["N"], is_count=True),
                    _fmt_cell(r["mean"]),
                    _fmt_cell(r["sd"]),
                    _fmt_cell(r["min"]),
                    _fmt_cell(r["p25"]),
                    _fmt_cell(r["p50"]),
                    _fmt_cell(r["p75"]),
                    _fmt_cell(r["max"]),
                ]
            )
            row_labels.append("")
    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    tbl.scale(1.0, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9d9d9")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    row_offset = 1
    for panel_name, pdf in panels:
        if panel_name:
            for c in range(len(col_labels)):
                tbl[(row_offset, c)].set_facecolor("#f0f0f0")
                tbl[(row_offset, c)].set_text_props(weight="bold", ha="left")
            row_offset += 1
        row_offset += len(pdf)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# replicate summary statistics for all stock-days
def plot_table1_summary(df_master):
    frame = _build_summary_frame(df_master)
    stock_day_vars = [
        "users_close",
        "userchg",
        "userratio",
        "prc",
        "size($mil)",
        "ret(%)",
        "openret(%)",
        "dayret(%)",
    ]
    panel_a = _summary_stats(frame, stock_day_vars)
    daily = (
        frame.groupby("date")
        .agg(
            n_stocks=("ticker", "nunique"),
            users_close_sum=("users_close", "sum"),
            userchg_sum=("userchg", "sum"),
        )
        .reset_index()
    )
    daily["users_close_mil"] = daily["users_close_sum"] / 1e6
    daily["userchg_000"] = daily["userchg_sum"] / 1e3
    daily_renamed = daily.rename(
        columns={
            "n_stocks": "#stocks",
            "users_close_mil": "users_close(mil.)",
            "userchg_000": "userchg(000)",
        }
    )
    daily_vars = ["#stocks", "users_close(mil.)", "userchg(000)"]
    panel_b = _summary_stats(daily_renamed, daily_vars)
    _render_stats_table(
        title="Replication of Table I — Summary Statistics",
        subtitle=(
            "Panel A: stock-day observations. Panel B: daily aggregates "
            "(summed per day, averaged across days). Computed on our master panel."
        ),
        save_path=figures_dir / "plot_table1_summary.png",
        panels=[
            ("Panel A: Stock-Day Observations", panel_a),
            ("Panel B: Daily Observations", panel_b),
        ],
    )


# replicate summary statistics on Robinhood herding events
def plot_table2_herding_events(df_master):
    frame = _build_summary_frame(df_master)
    if "users_lag" not in frame.columns or "userratio" not in frame.columns:
        return
    mask = (
        frame["userratio"].notna()
        & (frame["userratio"] > 1)
        & (frame["users_lag"] >= 100)
    )
    candidates = frame[mask].copy()
    if len(candidates) == 0:
        return
    cutoff = candidates["userratio"].quantile(0.995)
    herding = candidates[candidates["userratio"] >= cutoff].copy()
    herd_vars = [
        "users_close",
        "userchg",
        "userratio",
        "prc",
        "size($mil)",
        "ret(%)",
        "openret(%)",
        "dayret(%)",
    ]
    panel = _summary_stats(herding, herd_vars)
    _render_stats_table(
        title="Replication of Table II — Robinhood Herding Events",
        subtitle=(
            f"Herding events: top 0.5% userratio with userratio > 1 and "
            f"users_close(t-1) >= 100. N = {len(herding):,} stock-days."
        ),
        save_path=figures_dir / "plot_table2_herding_events.png",
        panels=[("Herding event stock-days", panel)],
    )


# summary of RD estimates across methods
def plot_results_table(results_all):
    if results_all.empty:
        return
    # preferred specification: bw=125, p=1
    df = results_all[
        (results_all["bandwidth"] == 125) & (results_all["poly_order"] == 1)
    ].copy()
    if df.empty:
        return
    # order methods
    method_order = [m for m in METHOD_COLORS.keys() if m in df["method"].values]
    df = df.set_index("method").loc[method_order].reset_index()
    # compute CI width reduction vs nocov
    nocov_ci = df.loc[df["method"] == "nocov", "ci_width"].values
    nocov_ci_val = nocov_ci[0] if len(nocov_ci) > 0 else np.nan
    df["ci_reduction_pct"] = (df["ci_width"] - nocov_ci_val) / nocov_ci_val * 100
    # build table rows
    col_labels = [
        "Method",
        "Coefficient",
        "Std. Error",
        "p-value",
        "CI Lower",
        "CI Upper",
        "CI Width",
        "CI Reduction (%)",
        "Eff. N",
    ]
    cell_data = []
    for _, row in df.iterrows():
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        coef = f"{row['beta1']:.1f}" if not np.isnan(row["beta1"]) else "—"
        se = f"{row['se']:.1f}" if not np.isnan(row["se"]) else "—"
        pv = f"{row['pvalue']:.3f}" if not np.isnan(row["pvalue"]) else "—"
        ci_lo = f"{row['ci_lower']:.1f}" if not np.isnan(row["ci_lower"]) else "—"
        ci_up = f"{row['ci_upper']:.1f}" if not np.isnan(row["ci_upper"]) else "—"
        ci_w = f"{row['ci_width']:.1f}" if not np.isnan(row["ci_width"]) else "—"
        ci_red = (
            f"{row['ci_reduction_pct']:.1f}"
            if not np.isnan(row["ci_reduction_pct"]) and row["method"] != "nocov"
            else "—"
        )
        n_obs_val = row.get("n_obs", np.nan)
        if not np.isnan(n_obs_val):
            n_obs_str = f"{int(n_obs_val)}"
        else:
            n_obs_str = "—"
        cell_data.append(
            [method_label, coef, se, pv, ci_lo, ci_up, ci_w, ci_red, n_obs_str]
        )
    fig_h = 0.8 + 0.35 * len(cell_data)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    tbl.scale(1.0, 1.5)
    # style header
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    # method name column left-aligned
    for r in range(1, len(cell_data) + 1):
        tbl[(r, 0)].set_text_props(ha="left")
    # highlight best CI reduction
    best_idx = None
    best_val = 0.0
    for r_idx, row in enumerate(cell_data):
        try:
            val = float(row[7])
            if val < best_val:
                best_val = val
                best_idx = r_idx
        except (ValueError, TypeError):
            pass
    if best_idx is not None:
        for c in range(len(col_labels)):
            tbl[(best_idx + 1, c)].set_facecolor("#d4edda")
    plt.tight_layout(pad=0)
    save_path = figures_dir / "results_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.01)
    plt.close()


# run all plots for RDFlex results
def _run_plots(results, sample_by_bw, prefix, title_suffix):
    if results.empty:
        return
    for poly in POLY_ORDERS:
        plot_ci_width_comparison(
            results, poly_order=poly, prefix=prefix, title_suffix=title_suffix
        )
    # point estimate plots for every bandwidth and polynom
    for bw in BANDWIDTHS:
        for poly in POLY_ORDERS:
            plot_point_estimates(
                results,
                bandwidth=bw,
                poly_order=poly,
                prefix=prefix,
                title_suffix=title_suffix,
            )
    plot_covariate_adjustment_scatter(
        results, sample_by_bw, prefix=prefix, title_suffix=title_suffix
    )
    plot_rdd_paper_style(
        results, sample_by_bw, prefix=prefix, title_suffix=title_suffix
    )
    plot_results_table(results)
    plot_forest_estimates(results)
    plot_standardized_diff(results)


# seed robustness test
def run_seed_robustness(sample_by_bw, results_all):
    if results_all.empty:
        return
    # largest bandwidth
    bw = max(sample_by_bw.keys())
    poly_order = 1
    df_sample = sample_by_bw[bw]
    df_prep, covars = _prepare_rdflex_covariates(df_sample)
    required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
    covars, _, _ = _select_complete_rdflex_covariates(df_prep, covars, required_cols)
    df_prep = df_prep.dropna(subset=covars + required_cols)
    if len(df_prep) < 50 or len(covars) == 0:
        return
    # tune once and reuse across seeds
    Z = df_prep[covars].values.astype(float)
    y_tune = df_prep["dayuserchgw"].values.astype(float)
    tuned = _tune_models(Z, y_tune, tag="seed-robustness")
    configs_all = dict(_build_adjustment_configs(tuned, sample_n=len(df_prep)))
    test_methods = [
        m for m in METHOD_COLORS.keys() if m in configs_all and m != "rdrobust"
    ]
    seeds = [42, 123, 456, 789, 2024]
    seed_results = []
    # estimate every method
    for seed in seeds:
        np.random.seed(seed)
        for method_name in test_methods:
            ml_g = clone(configs_all[method_name])
            try:
                est = _run_single_rdflex(df_prep, covars, ml_g, bw, poly_order)
                rec = _record_rdflex(
                    est, bw, poly_order, method_name, len(df_prep), covars
                )
                rec["seed"] = seed
                seed_results.append(rec)
            except Exception:
                pass
    np.random.seed(None)
    if not seed_results:
        return
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(output_dir / "seed_robustness.csv", index=False)
    # one panel per method
    methods_in_results = [
        m for m in METHOD_COLORS.keys() if m in seed_df["method"].values
    ]
    n_methods = len(methods_in_results)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )
    for idx, method in enumerate(methods_in_results):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i, col_i]
        mdf = seed_df[seed_df["method"] == method].sort_values("seed")
        x_pos = np.arange(len(mdf))
        ax.errorbar(
            x_pos,
            mdf["beta1"].values,
            yerr=1.96 * mdf["se"].values,
            fmt="o",
            color=METHOD_COLORS.get(method, "gray"),
            capsize=5,
            capthick=1.5,
            markersize=7,
            linewidth=1.5,
        )
        mean_coef = mdf["beta1"].mean()
        ax.axhline(
            y=mean_coef,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_coef:.1f}",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(s) for s in mdf["seed"].values], fontsize=8)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
    # hide empty panels
    for idx in range(n_methods, n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes[row_i, col_i].set_visible(False)
    fig.supxlabel("Seed", fontsize=11)
    fig.supylabel("Treatment Effect", fontsize=11)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(
        figures_dir / "robustness_seed_stability.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# generate full results table
def plot_full_results_table():
    csv_path = output_dir / "results_combined.csv"
    if not csv_path.exists():
        csv_path = output_dir / "results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df["poly_order"] == 1].copy()
    if df.empty:
        return
    rows = []
    for bw in [50, 75, 100, 125]:
        sub = df[df["bandwidth"] == bw]
        if sub.empty:
            continue
        nocov = sub.loc[sub["method"] == "nocov", "ci_width"]
        base = nocov.values[0] if len(nocov) else np.nan
        for m in METHOD_COLORS.keys():
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            dci = (
                (r["ci_width"] - base) / base * 100
                if base and not np.isnan(base)
                else np.nan
            )
            rows.append(
                [
                    f"${bw}M",
                    METHOD_LABELS.get(m, m),
                    f"{r['beta1']:.1f}" if pd.notna(r["beta1"]) else "-",
                    f"{r['se']:.1f}" if pd.notna(r["se"]) else "-",
                    f"{r['ci_width']:.1f}" if pd.notna(r["ci_width"]) else "-",
                    f"{dci:.1f}" if pd.notna(dci) and m != "nocov" else "-",
                ]
            )
    if not rows:
        return
    col_labels = ["Bandwidth", "Method", "Coef", "SE", "CI Width", "CI Red. (%)"]
    fig, ax = plt.subplots(figsize=(12, 0.8 + 0.3 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    tbl.scale(1.0, 1.3)
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    plt.tight_layout(pad=0)
    plt.savefig(
        figures_dir / "results_table_full.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()


# run main
def main():
    t_start = time.time()
    # Mode selection: --plots-only skips all estimation, loads cached CSVs
    plots_only = "--plots-only" in sys.argv
    if plots_only:
        print("Reading Input ...")
        # load matched samples from saved CSVs
        sample_by_bw = {}
        for bw in BANDWIDTHS:
            csv_path = output_dir / f"matched_sample_bw{bw}.csv"
            if csv_path.exists():
                sample_by_bw[bw] = pd.read_csv(csv_path)
        if not sample_by_bw:
            return
        # load combined results
        combined_csv = output_dir / "results_combined.csv"
        results_csv = output_dir / "results.csv"
        if combined_csv.exists():
            results_rdflex = pd.read_csv(combined_csv)
        elif results_csv.exists():
            results_rdflex = pd.read_csv(results_csv)
        else:
            return
        # observation comparison
        plot_observation_comparison(sample_by_bw)
        # table1 and table2 summary
        master_csv = output_dir / "master_dataset.csv"
        if master_csv.exists():
            df_master_po = pd.read_csv(master_csv)
            plot_table1_summary(df_master_po)
            plot_table2_herding_events(df_master_po)
            del df_master_po
            gc.collect()
        # seed stability plot (from cached CSV)
        seed_csv = output_dir / "seed_robustness.csv"
        if seed_csv.exists():
            seed_df = pd.read_csv(seed_csv)
            methods_in_results = [
                m for m in METHOD_COLORS.keys() if m in seed_df["method"].values
            ]
            n_methods = len(methods_in_results)
            if n_methods > 0:
                n_cols = min(4, n_methods)
                n_rows = (n_methods + n_cols - 1) // n_cols
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
                )
                for idx, method in enumerate(methods_in_results):
                    row_i, col_i = divmod(idx, n_cols)
                    ax = axes[row_i, col_i]
                    mdf = seed_df[seed_df["method"] == method].sort_values("seed")
                    x_pos = np.arange(len(mdf))
                    ax.errorbar(
                        x_pos,
                        mdf["beta1"].values,
                        yerr=1.96 * mdf["se"].values,
                        fmt="o",
                        color=METHOD_COLORS.get(method, "gray"),
                        capsize=5,
                        capthick=1.5,
                        markersize=7,
                        linewidth=1.5,
                    )
                    mean_coef = mdf["beta1"].mean()
                    ax.axhline(
                        y=mean_coef,
                        color="gray",
                        linestyle="--",
                        linewidth=1,
                        label=f"Mean: {mean_coef:.1f}",
                    )
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([str(s) for s in mdf["seed"].values], fontsize=8)
                    ax.set_title(
                        METHOD_LABELS.get(method, method),
                        fontsize=10,
                    )
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color="black", linewidth=0.5)
                for idx in range(n_methods, n_rows * n_cols):
                    row_i, col_i = divmod(idx, n_cols)
                    axes[row_i, col_i].set_visible(False)
                fig.supxlabel("Seed", fontsize=11)
                fig.supylabel("Treatment Effect", fontsize=11)
                plt.tight_layout(rect=[0.03, 0.03, 1, 1])
                plt.savefig(
                    figures_dir / "robustness_seed_stability.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()
        # tuned hyperparameters table (from cached CSV)
        hp_csv = output_dir / "tuned_hyperparameters.csv"
        if hp_csv.exists():
            hp_df = pd.read_csv(hp_csv)
            _plot_hyperparameter_table(hp_df)
        # kernel robustness plot (from cached CSV)
        kernel_csv = output_dir / "kernel_robustness.csv"
        if kernel_csv.exists():
            kernel_df = pd.read_csv(kernel_csv)
            _plot_kernel_robustness(kernel_df)
        # forest estimate + standardised difference plots
        if not results_rdflex.empty:
            plot_forest_estimates(results_rdflex)
            plot_standardized_diff(results_rdflex)
    else:
        # build master dataset
        df_master = build_master_dataset()
        compute_rdd_variables(df_master)
        df_master.to_csv(output_dir / "master_dataset.csv", index=False)
        print("Build Master Dataset ...")
        # construct matched samples for all bandwidths
        sample_by_bw = {}
        for bw in BANDWIDTHS:
            matched = construct_matched_sample(df_master, bw)
            if len(matched) > 0:
                sample_by_bw[bw] = matched
        if not sample_by_bw:
            return
        # summary statistics
        plot_table1_summary(df_master)
        plot_table2_herding_events(df_master)
        # memory cleanup
        del df_master
        gc.collect()
        diagnose_sample_sizes(sample_by_bw, label="matched")
        # save matched samples and prepare covariates for estimation
        for bw, df_bw in sample_by_bw.items():
            df_prep, covars = _prepare_rdflex_covariates(df_bw)
            required_cols = ["dayuserchgw", "gt300", "mktcap1_300"]
            covars, _, _ = _select_complete_rdflex_covariates(
                df_prep, covars, required_cols
            )
            df_prep = df_prep.dropna(subset=covars + required_cols)
            save_cols = required_cols + covars
            df_prep[save_cols].to_csv(
                output_dir / f"matched_sample_bw{bw}.csv", index=False
            )
        # observation comparison plot
        plot_observation_comparison(sample_by_bw)
        # RDFlex estimation across all bandwidths, methods, and poly orders
        print("Estimating ...")
        results_rdflex = run_rdflex_path(sample_by_bw)
        results_rdflex.to_csv(output_dir / "results.csv", index=False)
        # merge linear rdrobust results
        rdrobust_csv = output_dir / "results_rdrobust.csv"
        if rdrobust_csv.exists():
            rdrobust_results = pd.read_csv(rdrobust_csv)
            # ensure column alignment
            for col in RDFLEX_RESULT_COLUMNS:
                if col not in rdrobust_results.columns:
                    rdrobust_results[col] = np.nan
            rdrobust_results = rdrobust_results[RDFLEX_RESULT_COLUMNS]
            results_rdflex = pd.concat(
                [results_rdflex, rdrobust_results], ignore_index=True
            )
            results_rdflex.to_csv(output_dir / "results_combined.csv", index=False)
        # generate seed plot and csv
        run_seed_robustness(sample_by_bw, results_rdflex)
        # end of full-mode estimation block
    print("Plotting ...")
    plot_full_results_table()
    _run_plots(results_rdflex, sample_by_bw, prefix="rdflex_", title_suffix=" [RDFlex]")
    gc.collect()
    # estimation with automatic bandwidth selection
    if plots_only:
        # load cached auto-bw results
        auto_bw_csv = output_dir / "results_auto_bw.csv"
        if auto_bw_csv.exists():
            results_rdflex_auto = pd.read_csv(auto_bw_csv)
        else:
            results_rdflex_auto = pd.DataFrame()
    else:
        results_rdflex_auto = run_rdflex_auto_bw(sample_by_bw)
        # merge rdrobust auto-bw results
        rdrobust_auto_csv = output_dir / "results_rdrobust_auto_bw.csv"
        if rdrobust_auto_csv.exists():
            rdr_auto = pd.read_csv(rdrobust_auto_csv)
            # ensure column alignment with RDFlex auto-bw results
            for col in RDFLEX_RESULT_COLUMNS + ["auto_bw"]:
                if col not in rdr_auto.columns:
                    rdr_auto[col] = np.nan
            rdr_auto = rdr_auto[RDFLEX_RESULT_COLUMNS + ["auto_bw"]]
            results_rdflex_auto = pd.concat(
                [results_rdflex_auto, rdr_auto], ignore_index=True
            )
    if not results_rdflex_auto.empty:
        if not plots_only:
            results_rdflex_auto.to_csv(output_dir / "results_auto_bw.csv", index=False)
        plot_rdflex_bw_comparison(results_rdflex, results_rdflex_auto)
    del results_rdflex_auto
    # memory cleanup
    gc.collect()
    # covariate diagnostics
    plot_covariate_importance(sample_by_bw)
    plot_covariate_balance(sample_by_bw)
    gc.collect()
    # covariate-running variable correlation diagnostic
    plot_covariate_running_var_correlation(sample_by_bw)
    gc.collect()
    # Lasso covariate selection diagnostic
    report_lasso_covariate_selection(sample_by_bw)
    gc.collect()
    # SE comparison plot
    plot_se_comparison(results_rdflex)
    plot_relative_se_reduction(results_rdflex)
    gc.collect()
    # super learner weights diagnostic
    plot_super_learner_weights(sample_by_bw)
    gc.collect()
    # statistical power analysis
    compute_power_analysis(results_rdflex)
    gc.collect()
    # Predetermined Covariates & Placebo Outcomes
    robustness_predetermined_and_placebo(sample_by_bw)
    gc.collect()
    # Density of Running Variable
    robustness_density_test(sample_by_bw)
    gc.collect()
    # Sensitivity to Bandwidth Choice
    robustness_bandwidth_sensitivity(sample_by_bw, results_rdflex)
    gc.collect()
    # Kernel Sensitivity
    robustness_kernel_comparison(sample_by_bw, results_rdflex)
    gc.collect()
    # Covariate Continuity
    cov_cont_csv = output_dir / "results_covariate_continuity.csv"
    if cov_cont_csv.exists():
        cov_cont = pd.read_csv(cov_cont_csv)
        col_labels = [
            "Covariate",
            "MSE-Optimal Bandwidth",
            "RD Estimator",
            "Robust p-value",
            "Robust 95% CI",
            "Eff. N",
        ]
        cells = []
        for _, row in cov_cont.iterrows():
            label = COVARIATE_LABELS.get(row["covariate"], row["covariate"])
            cells.append(
                [
                    label,
                    f"{row['mse_bandwidth']:.1f}",
                    f"{row['rd_estimator']:.4f}",
                    f"{row['robust_pvalue']:.3f}",
                    f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                    f"{int(row['eff_n'])}",
                ]
            )
        fig_h = 0.8 + 0.35 * max(len(cells), 1)
        fig, ax = plt.subplots(figsize=(16, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=cells,
            colLabels=col_labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(list(range(len(col_labels))))
        tbl.scale(1.0, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#d9d9d9")
            # highlight significant p-values in red (p < 0.05)
            if c == 3 and r > 0:
                try:
                    pv = float(cell.get_text().get_text())
                    if pv < 0.05:
                        cell.set_facecolor("#f8d7da")
                except (ValueError, AttributeError):
                    pass
        plt.tight_layout(pad=0)
        plt.savefig(
            figures_dir / "robustness_covariate_continuity_table.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        plt.close()
    gc.collect()
    print("Finished ...")


if __name__ == "__main__":
    main()