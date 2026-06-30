# Replication of Griffin & Shams (2020)

# Sharp and Fuzzy RDD.
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
#   original_replication/Price_Flow_Clean.dta   hourly price/flow panel
#   original_replication/EOM_Data.dta           daily end-of-month panel
#   results.csv / results_combined.csv          cached estimates (--plots-only)
#   results_optbw.csv                           cached optimal-bandwidth estimates
#   results_rdrobust.csv                        R rdrobust results (merged if present)
#   best_params_*.csv                           cached tuned hyperparameters
#   covariate_continuity_rdrobust.csv           rdrobust covariate-continuity test
#   kernel_robustness.csv                       cached kernel-robustness results
#   seed_robustness.csv                         cached seed-robustness results
# Outputs:
#   master_dataset.csv                          constructed analysis dataset
#   results.csv                                 assembled estimates
#   results_combined.csv                        estimates + merged rdrobust rows
#   results_optbw.csv                           optimal-bandwidth estimates
#   eom_results.csv                             Table VIII estimates
#   best_params_*.csv                           tuned hyperparameters per sample
#   rdrobust_sample_sharp_*.csv                 per-sample CSVs for R rdrobust
#   rdrobust_sample_fuzzy_*.csv                 per-sample CSVs for R rdrobust
#   power_analysis_{sharp,fuzzy}.csv            power-analysis tables
#   kernel_robustness.csv                       kernel-robustness results
#   seed_robustness.csv                         seed-robustness results
#   figures/*.png                               all figures and table images                                 all figure outputs

from __future__ import annotations
import os
import sys
import shutil
import warnings
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from doubleml import DoubleMLRDDData
from doubleml.rdd import RDFlex
from lightgbm import LGBMClassifier, LGBMRegressor
from linearmodels.iv import IV2SLS
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler as _SS
from scipy import stats as sp_stats
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPLICATION_DIR = SCRIPT_DIR.parent 
DATA_DIR = REPLICATION_DIR / "data"
FIG_DIR = REPLICATION_DIR / "figures"
RESULTS_DIR = REPLICATION_DIR / "results"
PRICE_PATH = DATA_DIR / "Price_Flow_Clean.dta"
EOM_PATH = DATA_DIR / "EOM_Data.dta"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Global Tuning Parameters
N_OPTUNA_TRIALS = 10  # Optuna trials per model
OPTUNA_CV = 5  # cross-validation folds for Optuna tuning
RDFLEX_N_Folds = 5  # cross-fitting folds for RDFlex estimation
RDFLEX_N_REP = 5  # repetitions for RDFlex

# paper reference estimates (basis points, t-stats, N)
PAPER_VALUES = {
    # Table V
    "Table_V|All": {"coef": 14.38, "tstat": 2.69, "n": 1602},
    "Table_V|Auth": {"coef": 34.22, "tstat": 2.87, "n": 464},
    "Table_V|NoAuth": {"coef": 3.82, "tstat": 0.92, "n": 1138},
    # Table VI Panel A
    "Table_VI_A|Auth": {"coef": 20.61, "tstat": 2.42, "n": 464},
    "Table_VI_A|NoAuth": {"coef": -3.40, "tstat": -0.74, "n": 1138},
    "Table_VI_A|Auth_NegRet": {"coef": 32.87, "tstat": 2.58, "n": 214},
    "Table_VI_A|Auth_PosRet": {"coef": 11.91, "tstat": 1.29, "n": 250},
    # Table VI Panel B
    "Table_VI_B|All": {"coef": 26.42, "tstat": 2.06, "n": 1602},
    "Table_VI_B|Auth": {"coef": 33.88, "tstat": 2.05, "n": 464},
    "Table_VI_B|Auth_NegRet": {"coef": 45.34, "tstat": 2.37, "n": 214},
    "Table_VI_B|Auth_PosRet": {"coef": 20.17, "tstat": 1.30, "n": 250},
}

# covariates used by RDFlex (all adjustment strategies share this set)
COVARIATES = [
    "lag_ret",
    "vol",
    "vol_x_lagret",
    "hour_of_day",
    "day_of_week",
    "lag_ret_eth",
    "lag_ret_xrp",
    # 5 additional covariates
    "lag_ret_neg",
    "netflow_bitt",
    "netflow_huob",
    "netflow_krak",
    "netflow_aggplbt",
]

COVARIATE_LABELS = {
    "lag_ret": "Lagged Return (CoinDesk)",
    "vol": "24h Volatility",
    "vol_x_lagret": "Vol x Lag Return",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "lag_ret_eth": "Lagged ETH Return",
    "lag_ret_xrp": "Lagged XRP Return",
    "lag_ret_neg": "Negative Lagged Return (dummy)",
    "netflow_bitt": "Net Tether Flow to Bittrex",
    "netflow_huob": "Net Tether Flow to Huobi",
    "netflow_krak": "Net Tether Flow to Kraken",
    "netflow_aggplbt": "Aggregate Flow Polo+Bittrex",
}

# exchanges
ROUND_DISC_FLOWS = [
    ("netflow_lsg", "1LSg"),
    ("netflow_bina", "Binance"),
    ("netflow_bitt", "Oth BTX"),
    ("netflow_hitb", "HitBTC"),
    ("netflow_huob", "Huobi"),
    ("netflow_krak", "Kraken"),
    ("netflow_okex", "OKEx"),
    ("netflow_polo", "Oth PLX"),
    ("netflow_aggplbt", "Aggregate Polo+Bittrex"),
]

# hyperparameter search spaces
RF_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
    "max_depth": {"type": "categorical", "choices": [3, 5, 7, 10, 15, None]},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    "max_features": {
        "type": "categorical",
        "choices": ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0],
    },
}

LGBM_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 50},
    "max_depth": {"type": "int", "low": 2, "high": 15},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "num_leaves": {"type": "int", "low": 8, "high": 128},
    "min_child_samples": {"type": "int", "low": 3, "high": 50},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.05},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0, "step": 0.05},
    "reg_alpha": {"type": "float", "low": 1e-5, "high": 50.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-5, "high": 50.0, "log": True},
}

XGB_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 50},
    "max_depth": {"type": "int", "low": 2, "high": 12},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "min_child_weight": {"type": "int", "low": 1, "high": 15},
    "gamma": {"type": "float", "low": 1e-5, "high": 5.0, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.05},
    "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0, "step": 0.05},
    "reg_alpha": {"type": "float", "low": 1e-5, "high": 50.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-5, "high": 50.0, "log": True},
}

# prepare labels for hyperparameter table
HYPERPARAM_DESCRIPTIONS = {
    # Random Forest (ml_g)
    "rf_g_n_estimators": "Number of trees in the forest",
    "rf_g_max_depth": "Maximum depth of each tree (None = unlimited)",
    "rf_g_min_samples_split": "Minimum samples to split an internal node",
    "rf_g_min_samples_leaf": "Minimum samples at each leaf node",
    "rf_g_max_features": "Fraction of features considered per split",
    # LightGBM (ml_g)
    "lgbm_g_n_estimators": "Number of boosting rounds",
    "lgbm_g_max_depth": "Maximum depth of each tree",
    "lgbm_g_learning_rate": "Step size shrinkage per boosting round",
    "lgbm_g_num_leaves": "Maximum number of leaves per tree",
    "lgbm_g_min_child_samples": "Minimum samples in a leaf",
    "lgbm_g_subsample": "Fraction of rows sampled per tree",
    "lgbm_g_colsample_bytree": "Fraction of features sampled per tree",
    "lgbm_g_reg_alpha": "L1 regularisation on leaf weights",
    "lgbm_g_reg_lambda": "L2 regularisation on leaf weights",
    # XGBoost (ml_g)
    "xgb_g_n_estimators": "Number of boosting rounds",
    "xgb_g_max_depth": "Maximum depth of each tree",
    "xgb_g_learning_rate": "Step size shrinkage per boosting round",
    "xgb_g_min_child_weight": "Minimum sum of instance weight in a leaf",
    "xgb_g_gamma": "Minimum loss reduction for a split",
    "xgb_g_subsample": "Fraction of rows sampled per tree",
    "xgb_g_colsample_bytree": "Fraction of features sampled per tree",
    "xgb_g_reg_alpha": "L1 regularisation on leaf weights",
    "xgb_g_reg_lambda": "L2 regularisation on leaf weights",
}
HP_METHOD_LABELS = {
    "rf_g": "Random Forest",
    "lgbm_g": "LightGBM",
    "xgb_g": "XGBoost",
}

# Okabe-Ito colorblind-safe palette (consistent with all replications)
OI_COLORS = {
    "no covariates": "#000000",
    "linear (rdrobust)": "#E69F00",
    "linear (Ridge)": "#56B4E9",
    "lasso+interactions": "#009E73",
    "ML (RandomForest)": "#F0E442",
    "ML (LightGBM)": "#0072B2",
    "ML (XGBoost)": "#D55E00",
    "ML (NNet)": "#CC79A7",
    "ML (SuperLearner)": "#999999",
}

# Clean method labels
OI_LABELS = {
    "no covariates": "No Covariates",
    "linear (rdrobust)": "rdrobust",
    "linear (Ridge)": "Ridge",
    "lasso+interactions": "Lasso",
    "ML (RandomForest)": "Random Forest",
    "ML (LightGBM)": "LightGBM",
    "ML (XGBoost)": "XGBoost",
    "ML (NNet)": "Neural Network",
    "ML (SuperLearner)": "Super Learner",
}


# data loading
def load_price_flow():
    # hourly price / flow panel
    df = pd.read_stata(PRICE_PATH, convert_categoricals=False)
    df = df.sort_values("htime").reset_index(drop=True)
    return df


def load_eom():
    # daily end-of-month panel for Table VIII
    df = pd.read_stata(EOM_PATH, convert_categoricals=False)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# RD variable construction (copied from the orginial replication Stata code)


def construct_rd_vars(df):
    # build running variable, cutoff dummy, instrument, outcome and covariates
    df = df.copy()
    # lag price used in the running variable
    df["lag_close"] = df["close_coindesk"].shift(1)
    # signed distance to nearest 500-dollar multiple, based on lag price
    nearest = (df["lag_close"] / 500).round() * 500
    df["price_dist"] = df["lag_close"] - nearest
    # below_cutoff: 0 if [0, 50), 1 if (-50, 0)
    df["below_cutoff"] = np.where(
        (df["price_dist"] >= 0) & (df["price_dist"] < 50),
        0,
        np.where((df["price_dist"] < 0) & (df["price_dist"] > -50), 1, np.nan),
    )
    # binned running variable used for qfit plots.
    df["round_prc_dist"] = np.floor(df["price_dist"] / 10) * 10 + 5
    # bandwidth indicator (|dist| <= 50)
    df["in_bandwidth"] = ((df["price_dist"].abs() <= 50)).astype(int)
    # instrument: below_cutoff AND after_auth
    in_bw = (df["price_dist"] >= -50) & (df["price_dist"] < 50)
    df["inst"] = np.where(
        in_bw,
        np.where(
            (df["price_dist"] < 0) & (df["price_dist"] > -50) & (df["after_auth"] == 1),
            1,
            0,
        ),
        np.nan,
    )
    # forward 3-hour return in basis points
    r = df["ret_coindesk"]
    df["fret"] = ((r.shift(-1) + r.shift(-2) + r.shift(-3)) / 3.0) * 10000.0

    # the paper rescales aggplbt flow by 100 before running regressions
    df["flow"] = df["netflow_aggplbt"]
    df["flow_scaled"] = df["netflow_aggplbt"] / 100.0
    df["netflow_lsg_scaled"] = df["netflow_lsg"] / 100.0
    df["netflow_plbt_scaled"] = df["netflow_plbt"] / 100.0
    df["netflow_oth_noplbt_scaled"] = df["netflow_oth_noplbt"] / 100.0

    # covariates for RDFlex
    df["lag_ret"] = df["ret_coindesk"].shift(1)
    df["vol_x_lagret"] = df["vol"] * df["lag_ret"]
    df["hour_of_day"] = pd.to_datetime(df["htime"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["htime"]).dt.dayofweek
    df["lag_ret_eth"] = df["ret_eth"].shift(1)
    df["lag_ret_xrp"] = df["ret_xrp"].shift(1)

    # sign of lag return (for the L.Ret<0 / L.Ret>0 split tables)
    df["lag_ret_neg"] = (df["ret_coindesk"] < 0).astype(int)

    return df


# Replication - Tables V, VI
def replicate_table_v(df):
    # first stage: flow = a + b * below_cutoff, robust SE
    out = {}
    use = df.dropna(subset=["flow", "below_cutoff"])
    for label, mask in [
        ("All", pd.Series(True, index=use.index)),
        ("Auth", use["after_auth"] == 1),
        ("NoAuth", use["after_auth"] == 0),
    ]:
        sub = use[mask]
        if len(sub) < 10:
            continue
        X = add_constant(sub["below_cutoff"].astype(float))
        m = OLS(sub["flow"].astype(float), X).fit(cov_type="HC1")
        out[label] = _fmt_ols(m, "below_cutoff", n=int(m.nobs))
    return out


def replicate_table_vi_a(df):
    # panel A
    out = {}
    use = df.dropna(subset=["fret", "below_cutoff", "ret_coindesk"])
    samples = {
        "Auth": use["after_auth"] == 1,
        "NoAuth": use["after_auth"] == 0,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for label, mask in samples.items():
        sub = use[mask]
        if len(sub) < 10:
            continue
        X = add_constant(sub["below_cutoff"].astype(float))
        m = OLS(sub["fret"].astype(float), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        out[label] = _fmt_ols(m, "below_cutoff", n=int(m.nobs))
    return out


def replicate_table_vi_b(df):
    # panel B
    out = {}
    use = df.dropna(subset=["fret", "flow_scaled", "inst", "ret_coindesk"])
    samples = {
        "All": pd.Series(True, index=use.index),
        "Auth": use["after_auth"] == 1,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for label, mask in samples.items():
        sub = use[mask]
        if len(sub) < 20:
            continue
        try:
            iv = IV2SLS(
                dependent=sub["fret"].astype(float),
                exog=pd.DataFrame({"const": 1.0}, index=sub.index),
                endog=sub[["flow_scaled"]].astype(float),
                instruments=sub[["inst"]].astype(float),
            ).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
            out[label] = {
                "coef": round(float(iv.params["flow_scaled"]), 2),
                "se": round(float(iv.std_errors["flow_scaled"]), 2),
                "tstat": round(float(iv.tstats["flow_scaled"]), 2),
                "pval": round(float(iv.pvalues["flow_scaled"]), 4),
                "n": int(iv.nobs),
            }
        except Exception as exc:
            out[label] = {"error": str(exc)}
    return out


def _fmt_ols(m, name, n):
    # format an OLS fit into a coef/se/tstat/pval/n dict
    return {
        "coef": round(float(m.params[name]), 2),
        "se": round(float(m.bse[name]), 2),
        "tstat": round(float(m.tvalues[name]), 2),
        "pval": round(float(m.pvalues[name]), 4),
        "n": n,
    }


def _suggest_from_space(trial, space):
    # translate a parameter-space spec into Optuna trial suggestions
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif kind == "float":
            if spec.get("log", False):
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=True,
                )
            else:
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", None),
                )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {kind}")
    return params


def _optuna_tune(base, space, X, y, scoring, n_iter=N_OPTUNA_TRIALS, cv=OPTUNA_CV):
    # run an Optuna search and return the best hyperparameters
    def objective(trial):
        params = _suggest_from_space(trial, space)
        model = clone(base)
        model.set_params(**params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    # show_progress_bar=True
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    return study.best_params


def tune_regressor(base, space, X, y, n_iter=N_OPTUNA_TRIALS, cv=OPTUNA_CV):
    # tune a regressor by maximising negative MSE
    return _optuna_tune(
        base=base,
        space=space,
        X=X,
        y=y,
        scoring="neg_mean_squared_error",
        n_iter=n_iter,
        cv=cv,
    )


def tune_classifier(base, space, X, y, n_iter=N_OPTUNA_TRIALS, cv=OPTUNA_CV):
    # tune a classifier by maximising negative log-loss
    return _optuna_tune(
        base=base,
        space=space,
        X=X,
        y=y,
        scoring="neg_log_loss",
        n_iter=n_iter,
        cv=cv,
    )


def tune_all_ml_models(X, y, d=None, n_iter=N_OPTUNA_TRIALS, cv=OPTUNA_CV, tag=""):
    # tune ml_g learners (and ml_m learners when d given for fuzzy)
    tuned = {}
    best_params = {}
    tuned["rf_g"] = tune_regressor(
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        RF_PARAM_SPACE,
        X,
        y,
        n_iter,
        cv,
    )
    best_params["rf_g"] = tuned["rf_g"]
    tuned["lgbm_g"] = tune_regressor(
        LGBMRegressor(
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        ),
        LGBM_PARAM_SPACE,
        X,
        y,
        n_iter,
        cv,
    )
    best_params["lgbm_g"] = tuned["lgbm_g"]
    tuned["xgb_g"] = tune_regressor(
        XGBRegressor(
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        ),
        XGB_PARAM_SPACE,
        X,
        y,
        n_iter,
        cv,
    )
    best_params["xgb_g"] = tuned["xgb_g"]

    if d is not None:
        tuned["rf_m"] = tune_classifier(
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
            ),
            RF_PARAM_SPACE,
            X,
            d,
            n_iter,
            cv,
        )
        best_params["rf_m"] = tuned["rf_m"]
        tuned["lgbm_m"] = tune_classifier(
            LGBMClassifier(
                verbose=-1,
                n_jobs=-1,
                random_state=42,
            ),
            LGBM_PARAM_SPACE,
            X,
            d,
            n_iter,
            cv,
        )
        best_params["lgbm_m"] = tuned["lgbm_m"]
        tuned["xgb_m"] = tune_classifier(
            XGBClassifier(
                verbosity=0,
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
            ),
            XGB_PARAM_SPACE,
            X,
            d,
            n_iter,
            cv,
        )
        best_params["xgb_m"] = tuned["xgb_m"]
    # where the best hyperparameters are cached for --plots-only mode
    bp_path = RESULTS_DIR / f"best_params_{tag.replace('/', '_')}.csv"
    # best_params as long-format CSV for --plots-only mode
    bp_rows = [
        {"method": k, "param": pk, "value": ("None" if pv is None else pv)}
        for k, v in best_params.items()
        for pk, pv in v.items()
    ]
    pd.DataFrame(bp_rows, columns=["method", "param", "value"]).to_csv(
        bp_path, index=False
    )
    return tuned, best_params


def _oi_color(adj):
    # Return Okabe-Ito color for a given adjustment label.
    return OI_COLORS.get(adj, "#333333")


def _oi_label(adj):
    # Return clean label for a given adjustment name.
    return OI_LABELS.get(adj, adj)


def _grey_squares(ax):
    # Apply grey-shaded-squares background style (consistent with all replications)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_axisbelow(True)


def plot_hyperparameter_table(best_params, title_tag):
    # Plot the hyperparameter table
    table_data = []
    for method in ["rf_g", "lgbm_g", "xgb_g"]:
        if method not in best_params:
            continue
        for param, val in best_params[method].items():
            key = f"{method}_{param}"
            # format value nicely
            try:
                fval = float(val)
                if fval < 0.01:
                    val_str = f"{fval:.2e}"
                elif fval == int(fval):
                    val_str = f"{int(fval)}"
                else:
                    val_str = f"{fval:.4f}"
            except (ValueError, TypeError):
                val_str = str(val)
            table_data.append(
                [
                    HP_METHOD_LABELS.get(method, method),
                    param,
                    val_str,
                    HYPERPARAM_DESCRIPTIONS.get(key, ""),
                ]
            )
    if not table_data:
        return
    n_rows_tbl = len(table_data) + 1
    row_height = 0.45
    fig_h = max(2.5, n_rows_tbl * row_height)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Parameter", "Best Value", "Description"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
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
    # save figure
    plt.savefig(
        FIG_DIR / "tuned_hyperparameters_table.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


# Define the ML models
class ScaledMLPRegressor(BaseEstimator, RegressorMixin):
    # MLP Regressor
    def __init__(
        self,
        n_hidden=20,
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
    ):
        # store hyperparameters (sklearn estimator convention)
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        # scale features then fit the MLP regressor
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.n_hidden,),
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(Xs, y)
        return self

    def predict(self, X):
        # scale features and predict, guarding against NaN/Inf
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        pred = self.model_.predict(Xs)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred


class ScaledMLPClassifier(ClassifierMixin, BaseEstimator):
    # MLPClassifier
    _estimator_type = "classifier"

    def __init__(
        self,
        n_hidden=20,
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
    ):
        # store hyperparameters
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def _more_tags(self):
        # y is required
        return {"requires_y": True}

    def fit(self, X, y, sample_weight=None):
        # scale features then fit the MLP classifier
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # set before fit as fallback
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(self.n_hidden,),
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(Xs, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        # scale features and predict class labels
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.model_.predict(Xs)

    def predict_proba(self, X):
        # predict probabilities, clipped away from 0/1 for the fuzzy Wald ratio
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        proba = self.model_.predict_proba(Xs)
        proba = np.clip(proba, 1e-4, 1 - 1e-4)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class LassoInteractionRegressor(BaseEstimator, RegressorMixin):
    # LassoCV + interaction features Regressor
    def __init__(self, cv=5, max_iter=10000, n_jobs=-1):
        # store hyperparameters
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        # scale, expand to degree-2 features, then fit LassoCV
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.poly_ = PolynomialFeatures(
            degree=2, interaction_only=False, include_bias=False
        )
        Xp = self.poly_.fit_transform(Xs)
        self.model_ = LassoCV(cv=self.cv, max_iter=self.max_iter, n_jobs=self.n_jobs)
        self.model_.fit(Xp, y)
        return self

    def predict(self, X):
        # transform features and predict, guarding against NaN/Inf
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        Xp = self.poly_.transform(Xs)
        pred = self.model_.predict(Xp)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred

    def get_feature_names(self, input_features):
        # expose the expanded polynomial feature names
        Xs_dummy = self.scaler_.transform(np.zeros((1, len(input_features))))
        self.poly_.fit(Xs_dummy)
        return self.poly_.get_feature_names_out(input_features)


class LassoInteractionClassifier(ClassifierMixin, BaseEstimator):
    # LassoCV + interaction features Classifier
    _estimator_type = "classifier"

    def __init__(self, cv=5, max_iter=10000):
        # store hyperparameters
        self.cv = cv
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        # fit L1 logistic regression
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.poly_ = PolynomialFeatures(
            degree=2, interaction_only=False, include_bias=False
        )
        Xp = self.poly_.fit_transform(Xs)
        self.model_ = LogisticRegressionCV(
            cv=self.cv,
            penalty="l1",
            solver="saga",
            max_iter=self.max_iter,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(Xp, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        # transform features and predict class labels
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        Xp = self.poly_.transform(Xs)
        return self.model_.predict(Xp)

    def predict_proba(self, X):
        # predict probabilities, clipped away from 0/1 for the fuzzy Wald ratio
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        Xp = self.poly_.transform(Xs)
        proba = self.model_.predict_proba(Xp)
        proba = np.clip(proba, 1e-4, 1 - 1e-4)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


# Stacked ensemble
class SuperLearnerRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_learners=None, random_state=42):
        # store the list of base learners and seed
        self.base_learners = base_learners
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        # stack cross-validated base predictions, then ridge-combine them
        from sklearn.model_selection import cross_val_predict

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.learners_ = [clone(l) for l in self.base_learners]
        # level-0: cross-validated predictions
        Z = np.column_stack(
            [cross_val_predict(clone(l), X, y, cv=5) for l in self.base_learners]
        )
        # level-1: ridge on stacked predictions
        self.meta_ = RidgeCV(cv=5)
        self.meta_.fit(Z, y)
        self.weights_ = np.clip(self.meta_.coef_, 0, None)
        w_sum = self.weights_.sum()
        if w_sum > 0:
            self.weights_ = self.weights_ / w_sum
        # refit base learners on full data
        for l in self.learners_:
            l.fit(X, y)
        return self

    def predict(self, X):
        # weighted combination of base-learner predictions
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([l.predict(X) for l in self.learners_])
        return preds @ self.weights_


class SuperLearnerClassifier(ClassifierMixin, BaseEstimator):
    # Stacked ensemble classifier for fuzzy ml_m
    _estimator_type = "classifier"

    def __init__(self, base_learners=None, random_state=42):
        # store the list of base learners and seed
        self.base_learners = base_learners
        self.random_state = random_state

    def _more_tags(self):
        # y is required
        return {"requires_y": True}

    def fit(self, X, y, sample_weight=None):
        # stack cross-validated base probabilities, then ridge-combine them
        from sklearn.model_selection import cross_val_predict

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.learners_ = [clone(l) for l in self.base_learners]
        # level-0: cross-validated probability predictions
        Z = np.column_stack(
            [
                cross_val_predict(clone(l), X, y, cv=5, method="predict_proba")[:, 1]
                for l in self.base_learners
            ]
        )
        # level-1: ridge on stacked probabilities
        self.meta_ = RidgeCV(cv=5)
        self.meta_.fit(Z, y.astype(float))
        self.weights_ = np.clip(self.meta_.coef_, 0, None)
        w_sum = self.weights_.sum()
        if w_sum > 0:
            self.weights_ = self.weights_ / w_sum
        # refit base learners on full data
        for l in self.learners_:
            l.fit(X, y)
        return self

    def predict(self, X):
        # threshold the stacked probability at 0.5 for class labels
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X):
        # weighted combination of base-learner probabilities, clipped from 0/1
        X = np.asarray(X, dtype=float)
        p1 = (
            np.column_stack([l.predict_proba(X)[:, 1] for l in self.learners_])
            @ self.weights_
        )
        # clip away from 0/1 to stabilise fuzzy RD Wald ratio
        p1 = np.clip(p1, 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p1, p1])


def get_adjustment_configs(tuned, fuzzy=False):
    # build the (label, ml_g, ml_m) learner triples for every adjustment method
    configs = [
        (
            "no covariates",
            DummyRegressor(strategy="mean"),
            DummyClassifier(strategy="most_frequent") if fuzzy else None,
        ),
        (
            "linear (Ridge)",
            RidgeCV(cv=5),
            (
                LogisticRegressionCV(cv=5, penalty="l2", max_iter=2000, n_jobs=-1)
                if fuzzy
                else None
            ),
        ),
        (
            "ML (RandomForest)",
            RandomForestRegressor(random_state=42, n_jobs=-1, **tuned["rf_g"]),
            (
                RandomForestClassifier(
                    random_state=42, n_jobs=-1, **tuned.get("rf_m", {})
                )
                if fuzzy
                else None
            ),
        ),
        (
            "ML (LightGBM)",
            LGBMRegressor(verbose=-1, n_jobs=-1, **tuned["lgbm_g"]),
            (
                LGBMClassifier(verbose=-1, n_jobs=-1, **tuned.get("lgbm_m", {}))
                if fuzzy
                else None
            ),
        ),
        (
            "ML (XGBoost)",
            XGBRegressor(verbosity=0, n_jobs=-1, **tuned["xgb_g"]),
            (
                XGBClassifier(
                    verbosity=0,
                    use_label_encoder=False,
                    n_jobs=-1,
                    **tuned.get("xgb_m", {}),
                )
                if fuzzy
                else None
            ),
        ),
        (
            "lasso+interactions",
            LassoInteractionRegressor(cv=5, max_iter=10000, n_jobs=-1),
            (LassoInteractionClassifier(cv=5, max_iter=10000) if fuzzy else None),
        ),
        (
            "ML (NNet)",
            ScaledMLPRegressor(n_hidden=20, alpha=0.001, max_iter=1000),
            (
                ScaledMLPClassifier(n_hidden=20, alpha=0.001, max_iter=1000)
                if fuzzy
                else None
            ),
        ),
        (
            "ML (SuperLearner)",
            SuperLearnerRegressor(
                base_learners=[
                    RidgeCV(cv=5),
                    LassoCV(cv=5, n_jobs=-1, max_iter=5000),
                    RandomForestRegressor(random_state=42, n_jobs=-1, **tuned["rf_g"]),
                    LGBMRegressor(verbose=-1, n_jobs=-1, **tuned["lgbm_g"]),
                    XGBRegressor(verbosity=0, n_jobs=-1, **tuned["xgb_g"]),
                    ScaledMLPRegressor(n_hidden=20, alpha=0.001, max_iter=1000),
                ]
            ),
            (
                SuperLearnerClassifier(
                    base_learners=[
                        LogisticRegressionCV(
                            cv=5, penalty="l2", max_iter=2000, n_jobs=-1
                        ),
                        LogisticRegressionCV(
                            cv=5, penalty="l1", solver="saga", max_iter=2000, n_jobs=-1
                        ),
                        RandomForestClassifier(
                            random_state=42, n_jobs=-1, **tuned.get("rf_m", {})
                        ),
                        LGBMClassifier(
                            verbose=-1, n_jobs=-1, **tuned.get("lgbm_m", {})
                        ),
                        XGBClassifier(verbosity=0, n_jobs=-1, **tuned.get("xgb_m", {})),
                        ScaledMLPClassifier(n_hidden=20, alpha=0.001, max_iter=1000),
                    ]
                )
                if fuzzy
                else None
            ),
        ),
    ]
    return configs


# run the estimation for sharp and fuzzy RD designs, across samples and adjustment methods
def run_rdflex_sharp(df):
    # sharp RDFlex at hardcoded h=50 across samples and adjustment methods
    out = []
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"] + COVARIATES).copy()
    samples = {
        "Auth": use["after_auth"] == 1,
        "NoAuth": use["after_auth"] == 0,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for sample_label, mask in samples.items():
        sub = use[mask]
        if len(sub) < 50:
            continue
        cols = ["fret", "below_cutoff", "price_dist"] + COVARIATES
        sub = sub[cols].copy()
        # one tuning pass per sample, shared across adjustments
        tuned, bp = tune_all_ml_models(
            sub[COVARIATES], sub["fret"], tag=f"sharp/{sample_label}"
        )
        plot_hyperparameter_table(bp, f"sharp_{sample_label}")
        for name, ml_g, _ in get_adjustment_configs(tuned, fuzzy=False):
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="below_cutoff",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=0,
                    # cross-fitting
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    # hardcode paper bandwidth ($50 buckets) instead of auto-selection
                    h_fs=50.0,
                    fs_specification="cutoff",
                    fs_kernel="triangular",
                    # final-stage bandwidth = paper bandwidth
                    h=50.0,
                )
                est.fit()
                out.append(_record_rdflex(est, "sharp", sample_label, name, len(sub)))
            except Exception as exc:
                out.append(
                    {
                        "method": "RDFlex",
                        "design": "sharp",
                        "sample": sample_label,
                        "adjustment": name,
                        "coef": np.nan,
                        "se": np.nan,
                        "tstat": np.nan,
                        "pval": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(sub)),
                        "error": str(exc)[:100],
                    }
                )
    return out


def run_rdflex_fuzzy(df):
    # dichotomize netflow_aggplbt at sample median and treat as fuzzy RD
    out = []
    use = df.dropna(subset=["fret", "flow", "price_dist"] + COVARIATES).copy()
    samples = {
        "All": pd.Series(True, index=use.index),
        "Auth": use["after_auth"] == 1,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for sample_label, mask in samples.items():
        sub = use[mask].copy()
        if len(sub) < 50:
            continue
        sub["high_flow"] = (sub["flow"] > sub["flow"].median()).astype(int)
        cols = ["fret", "high_flow", "price_dist"] + COVARIATES
        sub = sub[cols].copy()
        tuned, bp = tune_all_ml_models(
            sub[COVARIATES],
            sub["fret"],
            d=sub["high_flow"],
            tag=f"fuzzy/{sample_label}",
        )
        plot_hyperparameter_table(bp, f"fuzzy_{sample_label}")

        for name, ml_g, ml_m in get_adjustment_configs(tuned, fuzzy=True):
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="high_flow",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=ml_m,
                    fuzzy=True,
                    cutoff=0,
                    # cross-fitting
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    # hardcode paper bandwidth ($50 buckets) instead of auto-selection
                    h_fs=50.0,
                    fs_specification="cutoff",
                    fs_kernel="triangular",
                    # final-stage bandwidth = paper bandwidth
                    h=50.0,
                )
                est.fit()
                out.append(_record_rdflex(est, "fuzzy", sample_label, name, len(sub)))
            except Exception as exc:
                out.append(
                    {
                        "method": "RDFlex",
                        "design": "fuzzy",
                        "sample": sample_label,
                        "adjustment": name,
                        "coef": np.nan,
                        "se": np.nan,
                        "tstat": np.nan,
                        "pval": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(sub)),
                        "error": str(exc)[:100],
                    }
                )
    return out


def _record_rdflex(est, design, sample_label, adjustment, n):
    # turn a fitted RDFlex estimator into a results-row dict
    coef = float(est.coef[0])
    se = float(est.se[0])
    pval = float(est.pval[0])
    ci = est.confint()
    ci_low = float(ci.iloc[0, 0])
    ci_high = float(ci.iloc[0, 1])
    return {
        "method": "RDFlex",
        "design": design,
        "sample": sample_label,
        "adjustment": adjustment,
        "coef": round(coef, 2),
        "se": round(se, 2),
        "tstat": round(coef / se, 2) if se > 0 else np.nan,
        "pval": round(pval, 4),
        "ci_low": round(ci_low, 2),
        "ci_high": round(ci_high, 2),
        "n": int(n),
    }


# Optimal-bandwidth RDFlex runs
def run_rdflex_optbw_sharp(df):
    # Sharp RDFlex with automatic MSE-optimal bandwidth selection
    out = []
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"] + COVARIATES).copy()
    samples = {
        "Auth": use["after_auth"] == 1,
        "NoAuth": use["after_auth"] == 0,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for sample_label, mask in samples.items():
        sub = use[mask]
        if len(sub) < 50:
            continue
        cols = ["fret", "below_cutoff", "price_dist"] + COVARIATES
        sub = sub[cols].copy()
        tuned, bp = tune_all_ml_models(
            sub[COVARIATES], sub["fret"], tag=f"sharp-optbw/{sample_label}"
        )
        plot_hyperparameter_table(bp, f"sharp_optbw_{sample_label}")
        for name, ml_g, _ in get_adjustment_configs(tuned, fuzzy=False):
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="below_cutoff",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=0,
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    fs_specification="cutoff",
                    fs_kernel="triangular",
                    # leave h and h_fs at None since auto MSE-optimal bandwidth
                )
                est.fit()
                rec = _record_rdflex(est, "sharp", sample_label, name, len(sub))
                rec["bandwidth"] = "auto"
                try:
                    rec["h_selected"] = float(est.h)
                except Exception:
                    rec["h_selected"] = np.nan
                out.append(rec)
            except Exception as exc:
                out.append(
                    {
                        "method": "RDFlex",
                        "design": "sharp",
                        "sample": sample_label,
                        "adjustment": name,
                        "coef": np.nan,
                        "se": np.nan,
                        "tstat": np.nan,
                        "pval": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(sub)),
                        "bandwidth": "auto",
                        "error": str(exc)[:100],
                    }
                )
    return out


def run_rdflex_optbw_fuzzy(df):
    # Fuzzy RDFlex with automatic MSE-optimal bandwidth selection
    out = []
    use = df.dropna(subset=["fret", "flow", "price_dist"] + COVARIATES).copy()
    samples = {
        "All": pd.Series(True, index=use.index),
        "Auth": use["after_auth"] == 1,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    for sample_label, mask in samples.items():
        sub = use[mask].copy()
        if len(sub) < 50:
            continue
        sub["high_flow"] = (sub["flow"] > sub["flow"].median()).astype(int)
        cols = ["fret", "high_flow", "price_dist"] + COVARIATES
        sub = sub[cols].copy()
        tuned, bp = tune_all_ml_models(
            sub[COVARIATES],
            sub["fret"],
            d=sub["high_flow"],
            tag=f"fuzzy-optbw/{sample_label}",
        )
        plot_hyperparameter_table(bp, f"fuzzy_optbw_{sample_label}")
        for name, ml_g, ml_m in get_adjustment_configs(tuned, fuzzy=True):
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="high_flow",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=ml_m,
                    fuzzy=True,
                    cutoff=0,
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    fs_specification="cutoff",
                    fs_kernel="triangular",
                )
                est.fit()
                rec = _record_rdflex(est, "fuzzy", sample_label, name, len(sub))
                rec["bandwidth"] = "auto"
                try:
                    rec["h_selected"] = float(est.h)
                except Exception:
                    rec["h_selected"] = np.nan
                out.append(rec)
            except Exception as exc:
                out.append(
                    {
                        "method": "RDFlex",
                        "design": "fuzzy",
                        "sample": sample_label,
                        "adjustment": name,
                        "coef": np.nan,
                        "se": np.nan,
                        "tstat": np.nan,
                        "pval": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(sub)),
                        "bandwidth": "auto",
                        "error": str(exc)[:100],
                    }
                )
    return out


# Table VIII - end of month replication
def replicate_table_viii(eom):
    # Table VIII: end-of-month effect on BTC and value-weighted returns
    out = {}
    use = eom.dropna(subset=["btcret", "eom"]).copy()
    spec_masks = {
        "btcret_all": pd.Series(True, index=use.index),
        "btcret_no_issuance": use["issuance"] == 0,
        "btcret_with_issuance": use["issuance"] > 0,
    }
    for label, mask in spec_masks.items():
        sub = use[mask]
        if len(sub) < 5:
            continue
        X = add_constant(sub["eom"].astype(float))
        m = OLS(sub["btcret"].astype(float), X).fit(cov_type="HC1")
        out[label] = _fmt_ols(m, "eom", n=int(m.nobs))

    use2 = eom.dropna(subset=["vwret5", "eom"]).copy()
    spec2 = {
        "vwret5_all": pd.Series(True, index=use2.index),
        "vwret5_with_issuance": use2["issuance"] > 0,
    }
    for label, mask in spec2.items():
        sub = use2[mask]
        if len(sub) < 5:
            continue
        X = add_constant(sub["eom"].astype(float))
        m = OLS(sub["vwret5"].astype(float), X).fit(cov_type="HC1")
        out[label] = _fmt_ols(m, "eom", n=int(m.nobs))
    return out


# plots for paper replications
def _quadratic_fit_with_ci(x, y):
    # quadratic fit with 95 percent pointwise CI, returned on a dense grid
    if len(x) < 5 or len(np.unique(x)) < 3:
        return None
    X = np.column_stack([np.ones_like(x), x, x**2])
    try:
        m = OLS(y, X).fit(cov_type="HC1")
    except Exception:
        return None
    grid = np.linspace(x.min(), x.max(), 80)
    G = np.column_stack([np.ones_like(grid), grid, grid**2])
    params = np.asarray(m.params)
    pred = G @ params
    cov = np.asarray(m.cov_params())
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    se = np.sqrt(np.clip(var, 0, None))
    return grid, pred, pred - 1.96 * se, pred + 1.96 * se


def plot_fret_discontinuity(df):
    # RD scatterplots
    for auth, tag in [(1, "Auth"), (0, "NoAuth")]:
        sub = df.dropna(subset=["fret", "round_prc_dist"])
        sub = sub[(sub["after_auth"] == auth) & (sub["round_prc_dist"].abs() <= 100)]
        if len(sub) < 20:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x_vals = sub["round_prc_dist"].astype(float).values
        y_vals = sub["fret"].astype(float).values
        # small scattered dots: blue below, red above cutoff
        dot_colors = np.where(x_vals >= 0, "#d62728", "#1f77b4")
        ax.scatter(
            x_vals,
            y_vals,
            c=dot_colors,
            alpha=0.2,
            s=8,
            edgecolors="none",
            rasterized=True,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        # local linear fits on each side (matches the p=1 preferred spec)
        for side_mask, side_color in [(x_vals < 0, "navy"), (x_vals >= 0, "darkred")]:
            xm, ym = x_vals[side_mask], y_vals[side_mask]
            if len(xm) < 5:
                continue
            try:
                coeffs = np.polyfit(xm, ym, 1)
                xfit = np.linspace(xm.min(), xm.max(), 200)
                yfit = np.polyval(coeffs, xfit)
                ax.plot(xfit, yfit, color=side_color, linewidth=2.5)
            except np.linalg.LinAlgError:
                pass
        # large binned mean dots
        n_bins = 20
        bin_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = []
        for bi in range(n_bins):
            mask_bin = (x_vals >= bin_edges[bi]) & (x_vals < bin_edges[bi + 1])
            bin_means.append(
                np.nanmean(y_vals[mask_bin]) if mask_bin.sum() > 0 else np.nan
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
        ax.set_xlabel("Distance from Round Threshold")
        ax.set_ylabel("Forward 3h Return")
        _grey_squares(ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"Fret_Discontinuity_{tag}.png", dpi=150)
        plt.close(fig)


def plot_combined_round_discontinuity(df):
    # All exchanges on one 3×3 figure
    for auth, tag in [(1, "Auth"), (0, "NoAuth")]:
        n = len(ROUND_DISC_FLOWS)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.8 * nrows),
            sharex=True,
            sharey=True,
        )
        axes = np.array(axes).flatten()
        for idx, (col, label) in enumerate(ROUND_DISC_FLOWS):
            ax = axes[idx]
            if col not in df.columns:
                ax.axis("off")
                continue
            sub = df.dropna(subset=[col, "round_prc_dist", "after_auth"])
            sub = sub[
                (sub["after_auth"] == auth) & (sub["round_prc_dist"].abs() <= 100)
            ]
            if len(sub) < 10:
                ax.set_title(f"{label} (n<10)", fontsize=9)
                continue
            for side_df in [
                sub[sub["round_prc_dist"] < 0],
                sub[sub["round_prc_dist"] >= 0],
            ]:
                if len(side_df) < 5:
                    continue
                fit = _quadratic_fit_with_ci(
                    side_df["round_prc_dist"].astype(float),
                    side_df[col].astype(float),
                )
                if fit is None:
                    continue
                grid, pred, lo, hi = fit
                ax.fill_between(grid, lo, hi, color="lightgray", alpha=0.5)
                ax.plot(grid, pred, color="black", lw=1.5)
            # binned means
            bins = sub.groupby("round_prc_dist", as_index=False)[col].mean()
            ax.scatter(
                bins["round_prc_dist"],
                bins[col],
                s=50,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
            ax.axvline(0, color="black", lw=0.8, linestyle="--")
            ax.set_title(label, fontsize=10)
            _grey_squares(ax)
        # shared axis labels
        fig.supxlabel("Distance from Round Threshold", fontsize=11)
        fig.supylabel("Hourly Flow (BTC)", fontsize=11)
        for j in range(len(ROUND_DISC_FLOWS), len(axes)):
            axes[j].axis("off")
        fig.tight_layout(rect=[0.03, 0.04, 1, 1])
        fig.savefig(
            FIG_DIR / f"Round_Discontinuity_Combined_{tag}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_pooled_round_discontinuity(df):
    # pool all exchange flows into a single plot
    for auth, tag in [(1, "Auth"), (0, "NoAuth")]:
        all_x, all_y = [], []
        for col, label in ROUND_DISC_FLOWS:
            if col not in df.columns:
                continue
            sub = df.dropna(subset=[col, "round_prc_dist", "after_auth"])
            sub = sub[
                (sub["after_auth"] == auth) & (sub["round_prc_dist"].abs() <= 100)
            ]
            all_x.append(sub["round_prc_dist"].astype(float))
            all_y.append(sub[col].astype(float))
        if not all_x:
            continue
        x = pd.concat(all_x, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        for side_mask in [x < 0, x >= 0]:
            sx, sy = x[side_mask], y[side_mask]
            if len(sx) < 5:
                continue
            fit = _quadratic_fit_with_ci(sx.values, sy.values)
            if fit is None:
                continue
            grid, pred, lo, hi = fit
            ax.fill_between(grid, lo, hi, color="gray", alpha=0.3)
            ax.plot(grid, pred, color="black", lw=2)
        # binned means: black hollow circles
        combined = pd.DataFrame({"round_prc_dist": x, "flow": y})
        bins = combined.groupby("round_prc_dist", as_index=False)["flow"].mean()
        ax.scatter(
            bins["round_prc_dist"],
            bins["flow"],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            zorder=5,
        )
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel("Distance from Round Threshold")
        ax.set_ylabel("Hourly Average Flow (BTC)")
        _grey_squares(ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"Round_Discontinuity_Pooled_{tag}.png", dpi=150)
        plt.close(fig)


# plots for estimation results
def plot_coefficient_comparison(results, design, out_path):
    # horizontal error-bar plot grouped by sample and adjustment
    df = pd.DataFrame(results)
    df = df[df["design"] == design]
    df = df[df["method"] != "Paper (Griffin & Shams 2020)"]
    if df.empty:
        return
    # split successful vs errored rows
    ok = df[df["coef"].notna()].copy()
    err = df[df["coef"].isna()].copy()
    samples = df["sample"].unique().tolist()
    fig, axes = plt.subplots(
        len(samples), 1, figsize=(9, 1.9 * len(samples) + 1), sharex=True
    )
    if len(samples) == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        sub_ok = ok[ok["sample"] == s]
        sub_err = err[err["sample"] == s]
        # combined adjustment list
        all_adj = list(
            dict.fromkeys(
                list(sub_ok["adjustment"].values) + list(sub_err["adjustment"].values)
            )
        )
        ypos_map = {adj: i for i, adj in enumerate(all_adj)}
        paper_key = f"Table_VI_A|{s}" if design == "sharp" else f"Table_VI_B|{s}"
        paper = PAPER_VALUES.get(paper_key)
        if paper is not None:
            # paper point estimate + 95% CI
            p_coef = float(paper["coef"])
            p_tstat = float(paper["tstat"])
            p_se = abs(p_coef / p_tstat) if p_tstat != 0 else float("nan")
            p_lo = p_coef - 1.96 * p_se
            p_hi = p_coef + 1.96 * p_se
            ax.axvspan(
                p_lo,
                p_hi,
                color="red",
                alpha=0.12,
                label=f"Paper 95% CI = [{p_lo:.2f}, {p_hi:.2f}]",
            )
            ax.axvline(
                p_coef,
                color="red",
                ls="--",
                lw=1.2,
                label=f"Paper = {p_coef}",
            )
        # plot successful entries with Okabe-Ito colors
        if not sub_ok.empty:
            ypos = np.array([ypos_map[a] for a in sub_ok["adjustment"]])
            coefs = sub_ok["coef"].values.astype(float)
            los = sub_ok["ci_low"].astype(float).to_numpy()
            his = sub_ok["ci_high"].astype(float).to_numpy()
            los = np.where(np.isnan(los), coefs, los)
            his = np.where(np.isnan(his), coefs, his)
            for yi, ci, li, hi_val, adj in zip(
                ypos, coefs, los, his, sub_ok["adjustment"]
            ):
                ax.errorbar(
                    ci,
                    yi,
                    xerr=[[ci - li], [hi_val - ci]],
                    fmt="o",
                    color=_oi_color(adj),
                    capsize=3,
                    markersize=7,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
        # plot errored entries
        if not sub_err.empty:
            ypos_e = [ypos_map[a] for a in sub_err["adjustment"]]
            ax.scatter(
                [0] * len(ypos_e),
                ypos_e,
                marker="x",
                color="red",
                s=60,
                zorder=5,
                label="failed",
            )
            for yp in ypos_e:
                ax.annotate("  (failed)", (0, yp), fontsize=7, color="red", va="center")
        ax.set_yticks(list(range(len(all_adj))))
        ax.set_yticklabels([_oi_label(a) for a in all_adj], fontsize=9)
        ax.set_title(f"{design.title()} RD - Sample: {s}", fontsize=10)
        _grey_squares(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Estimated RD coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_covariate_importance_per_method(df):
    # Create one feature-importance bar plot per learner
    sub = df.dropna(subset=["fret", "below_cutoff"] + COVARIATES)
    sub = sub[sub["after_auth"] == 1]
    if len(sub) < 50:
        return
    X = sub[COVARIATES]
    y = sub["fret"]
    methods = {
        "Ridge": RidgeCV(cv=5),
        "Lasso": LassoCV(cv=5, n_jobs=-1, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(n_estimators=300, verbose=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, verbosity=0, random_state=42),
    }
    # standardise X for linear methods so coefficients are comparable
    X_scaled = pd.DataFrame(_SS().fit_transform(X), columns=X.columns, index=X.index)
    for name, model in methods.items():
        # use scaled features for linear models so coefficients reflect importance
        X_fit = X_scaled if name in ("Ridge", "Lasso") else X
        model.fit(X_fit, y)
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=COVARIATES)
        elif hasattr(model, "coef_"):
            imp = pd.Series(np.abs(model.coef_), index=COVARIATES)
        else:
            continue
        # normalize to relative importance
        total = imp.sum()
        if total > 0:
            imp = imp / total
        imp = imp.sort_values()
        # skip if all zero
        if imp.max() == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            [COVARIATE_LABELS[c] for c in imp.index],
            imp.values,
            color="steelblue",
        )
        ax.set_xlabel("Relative Feature Importance")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"covariate_importance_{name.lower()}.png", dpi=150)
        plt.close(fig)
    # plot NNet importance via permutation importance
    nnet = ScaledMLPRegressor(n_hidden=20, alpha=0.001, max_iter=1000)
    nnet.fit(X, y)
    perm = permutation_importance(nnet, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    imp_nn = pd.Series(perm.importances_mean, index=COVARIATES)
    total = imp_nn.sum()
    if total > 0:
        imp_nn = imp_nn / total
    imp_nn = imp_nn.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        [COVARIATE_LABELS[c] for c in imp_nn.index],
        imp_nn.values,
        color="steelblue",
    )
    ax.set_xlabel("Relative Feature Importance (Permutation)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "covariate_importance_nnet.png", dpi=150)
    plt.close(fig)
    # Lasso+interactions importance
    lasso_int = LassoInteractionRegressor(cv=5, max_iter=10000, n_jobs=-1)
    lasso_int.fit(X, y)
    feat_names = lasso_int.get_feature_names(np.array(COVARIATES))
    coefs = np.abs(lasso_int.model_.coef_)
    imp_li = pd.Series(coefs, index=feat_names)
    imp_li = imp_li[imp_li > 0].sort_values(ascending=True).tail(25)
    if len(imp_li) > 0:
        total = imp_li.sum()
        if total > 0:
            imp_li = imp_li / total
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_li.index.astype(str), imp_li.values, color="steelblue")
        ax.set_xlabel("Relative Coefficient Magnitude")
        fig.tight_layout()
        # save figure
        fig.savefig(FIG_DIR / "covariate_importance_lasso_interactions.png", dpi=150)
        plt.close(fig)


def plot_covariate_distribution_by_cutoff(df, out_path):
    # covariate distributions on either side of the cutoff
    sub = df.dropna(subset=["below_cutoff"] + COVARIATES)
    sub = sub[sub["after_auth"] == 1]
    if len(sub) < 50:
        return
    n = len(COVARIATES)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    for i, c in enumerate(COVARIATES):
        ax = axes[i]
        for val, color, label in [
            (0, "#d62728", "Above"),
            (1, "#1f77b4", "Below"),
        ]:
            vals = sub.loc[sub["below_cutoff"] == val, c].dropna()
            if len(vals) > 0:
                ax.hist(
                    vals, bins=25, alpha=0.5, color=color, label=label, density=True
                )
        ax.set_title(COVARIATE_LABELS[c], fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _grey_squares(ax)
    for j in range(len(COVARIATES), len(axes)):
        axes[j].axis("off")
    fig.supylabel("Density", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Table I: Summary Statistics

# coins tracked in Tbale I
TABLE_1_COINS = [
    "bcc",
    "bch",
    "bnb",
    "btc",
    "dash",
    "eos",
    "etc",
    "eth",
    "iota",
    "ltc",
    "neo",
    "omg",
    "xmr",
    "xrp",
    "zec",
]


def compute_table_1_panels(df):
    # Compute daily-return correlations (Panel B) and hourly-return autocorrelations (Panel C)
    ret_cols = [f"ret_{c}" for c in TABLE_1_COINS if f"ret_{c}" in df.columns]
    # Panel B: daily-return correlations
    daily = df.groupby("date")[ret_cols].sum(min_count=1)
    corr = daily.corr()
    # Panel C: autocorrelation coefficient + HAC t-stat at 1h/3h/5h.
    ac_rows = []
    for c in ret_cols:
        s = df[c].astype(float)
        row = {"coin": c.replace("ret_", "").upper()}
        for k in (1, 3, 5):
            rolled = s.rolling(k).sum() if k > 1 else s
            pair = pd.concat([rolled, rolled.shift(k)], axis=1).dropna()
            pair.columns = ["y", "x"]
            if len(pair) > k + 5:
                try:
                    res = OLS(pair["y"].values, add_constant(pair["x"].values)).fit(
                        cov_type="HAC", cov_kwds={"maxlags": k}
                    )
                    coef = float(res.params[1])
                    tstat = float(res.tvalues[1])
                except Exception:
                    coef, tstat = np.nan, np.nan
            else:
                coef, tstat = np.nan, np.nan
            row[f"{k}h_coef"] = coef
            row[f"{k}h_t"] = tstat
        ac_rows.append(row)
    ac_df = pd.DataFrame(ac_rows).set_index("coin")
    return corr, ac_df


def _render_panel_table(ax, cell_text, col_labels, row_labels, title, col_widths=None):
    # Render a single panel as a clean paper-style text table
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    n_rows = len(cell_text)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("white")
        if r == 0:
            # header row: bold with grey background
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
            cell.set_linewidth(0.8)
            cell.set_edgecolor("#999999")


def plot_table_1(df, out_path):
    # Replicate Table I
    corr, ac_df = compute_table_1_panels(df)
    if corr.empty or ac_df.empty:
        return
    # normalize coin labels to upper case
    corr.index = [i.replace("ret_", "").upper() for i in corr.index]
    corr.columns = [c.replace("ret_", "").upper() for c in corr.columns]
    # layout: one figure with two stacked paper-style table panels (B and C)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.25)
    # Panel B: Correlations
    axB = fig.add_subplot(gs[0, 0])
    # paper layout: rows = coins 2..N (BCH..ZEC), columns = coins 1..N-1 (BCC..XRP)
    coins_order = [c.upper() for c in TABLE_1_COINS if c.upper() in corr.index]
    row_coins = coins_order[1:]
    col_coins = coins_order[:-1]
    b_cells = []
    for i, rc in enumerate(row_coins):
        row = []
        for j, cc in enumerate(col_coins):
            # only include cells where column index <= row index
            if j <= i:
                v = corr.loc[rc, cc]
                row.append("" if pd.isna(v) else f"{v:.2f}")
            else:
                row.append("")
        b_cells.append(row)
    _render_panel_table(
        axB,
        cell_text=b_cells,
        col_labels=col_coins,
        row_labels=row_coins,
        title="B. Correlations",
    )
    # Panel C: Autocorrelations table
    axC = fig.add_subplot(gs[1, 0])
    c_col_labels = [
        "1-Hour\nCoefficient",
        "t-stats",
        "3-Hour\nCoefficient",
        "t-stats",
        "5-Hour\nCoefficient",
        "t-stats",
    ]
    c_cells = []
    c_rows = []
    for coin in [c.upper() for c in TABLE_1_COINS]:
        if coin not in ac_df.index:
            continue
        r = ac_df.loc[coin]
        c_rows.append(coin)
        c_cells.append(
            [
                "" if pd.isna(r["1h_coef"]) else f"{r['1h_coef']:.3f}",
                "" if pd.isna(r["1h_t"]) else f"{r['1h_t']:.3f}",
                "" if pd.isna(r["3h_coef"]) else f"{r['3h_coef']:.3f}",
                "" if pd.isna(r["3h_t"]) else f"{r['3h_t']:.3f}",
                "" if pd.isna(r["5h_coef"]) else f"{r['5h_coef']:.3f}",
                "" if pd.isna(r["5h_t"]) else f"{r['5h_t']:.3f}",
            ]
        )
    _render_panel_table(
        axC,
        cell_text=c_cells,
        col_labels=c_col_labels,
        row_labels=c_rows,
        title="C. Autocorrelations",
    )

    fig.suptitle(
        "Table I. Summary Statistics (Griffin & Shams 2020 replication)",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bandwidth_comparison(fixed_rows, optbw_rows, design, out_path):
    # auto bw vs hardcoded bandwdith plot
    df_fix = pd.DataFrame(fixed_rows)
    df_opt = pd.DataFrame(optbw_rows)
    df_fix = df_fix[df_fix["design"] == design]
    df_opt = df_opt[df_opt["design"] == design]
    # need at least one of the two
    if df_fix.empty and df_opt.empty:
        return
    # combine samples from both
    all_samples = set()
    if not df_fix.empty:
        all_samples.update(df_fix["sample"].unique())
    if not df_opt.empty:
        all_samples.update(df_opt["sample"].unique())
    for s in sorted(all_samples):
        sub_fix = (
            df_fix[df_fix["sample"] == s].copy() if not df_fix.empty else pd.DataFrame()
        )
        sub_opt = (
            df_opt[df_opt["sample"] == s].copy() if not df_opt.empty else pd.DataFrame()
        )
        # methods present in either run
        adj_order = [
            "no covariates",
            "linear (rdrobust)",
            "linear (Ridge)",
            "lasso+interactions",
            "ML (RandomForest)",
            "ML (LightGBM)",
            "ML (XGBoost)",
            "ML (NNet)",
            "ML (SuperLearner)",
        ]
        methods = [
            m
            for m in adj_order
            if m in sub_fix.get("adjustment", pd.Series()).values
            or m in sub_opt.get("adjustment", pd.Series()).values
        ]
        if not methods:
            continue
        # extract coefs and CI widths
        coef_fix, coef_auto, ci_fix, ci_auto = [], [], [], []
        for m in methods:
            r_f = (
                sub_fix[sub_fix["adjustment"] == m]
                if not sub_fix.empty
                else pd.DataFrame()
            )
            r_a = (
                sub_opt[sub_opt["adjustment"] == m]
                if not sub_opt.empty
                else pd.DataFrame()
            )
            coef_fix.append(
                float(r_f["coef"].iloc[0])
                if len(r_f) > 0 and pd.notna(r_f["coef"].iloc[0])
                else np.nan
            )
            coef_auto.append(
                float(r_a["coef"].iloc[0])
                if len(r_a) > 0 and pd.notna(r_a["coef"].iloc[0])
                else np.nan
            )
            ci_f = (
                (float(r_f["ci_high"].iloc[0]) - float(r_f["ci_low"].iloc[0]))
                if len(r_f) > 0 and pd.notna(r_f["ci_high"].iloc[0])
                else np.nan
            )
            ci_a = (
                (float(r_a["ci_high"].iloc[0]) - float(r_a["ci_low"].iloc[0]))
                if len(r_a) > 0 and pd.notna(r_a["ci_high"].iloc[0])
                else np.nan
            )
            ci_fix.append(ci_f)
            ci_auto.append(ci_a)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        x = np.arange(len(methods))
        w = 0.35
        clean_labels = [_oi_label(m) for m in methods]
        # panel 1: coefficient estimates
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
            coef_fix,
            w,
            label="Hardcoded (h=50)",
            color="#2ca02c",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Treatment Effect")
        ax1.set_title("Coefficient Estimates")
        ax1.legend(fontsize=8, frameon=False)
        ax1.axhline(0, color="gray", ls="--", alpha=0.5)
        _grey_squares(ax1)
        # panel 2: CI widths
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
            ci_fix,
            w,
            label="Hardcoded (h=50)",
            color="#2ca02c",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("CI Width")
        ax2.set_title("Confidence Interval Width")
        ax2.legend(fontsize=8, frameon=False)
        _grey_squares(ax2)
        plt.tight_layout()
        fig.savefig(
            out_path.parent / f"{out_path.stem}_{s}{out_path.suffix}",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


# Standard Error Comparison plots
def plot_se_comparison(results, design, out_path):
    # grouped bar chart: paper SE vs RDFlex method SE
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[df["design"] == design]
    rdflex = df[df["method"].isin(["RDFlex", "rdrobust"])].copy()
    rdflex = rdflex[rdflex["se"].notna() & (rdflex["se"] > 0)]
    if rdflex.empty:
        return
    samples = sorted(rdflex["sample"].unique())
    fig, ax = plt.subplots(figsize=(max(14, 4 * len(samples)), 7))
    x = np.arange(len(samples))
    # collect methods actually present
    methods_present = [m for m in adj_order if m in rdflex["adjustment"].values]
    n_groups = len(methods_present) + 1
    width = 0.8 / n_groups
    # paper SEs
    paper_ses = []
    for s in samples:
        paper_key = f"Table_VI_A|{s}" if design == "sharp" else f"Table_VI_B|{s}"
        paper = PAPER_VALUES.get(paper_key)
        if paper is not None:
            p_coef, p_tstat = float(paper["coef"]), float(paper["tstat"])
            paper_ses.append(abs(p_coef / p_tstat) if p_tstat != 0 else np.nan)
        else:
            paper_ses.append(np.nan)
    ax.bar(
        x - 0.4 + width / 2,
        paper_ses,
        width,
        label="Paper",
        color="white",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
    )
    # collect all SE values for y-axis capping
    all_se_vals = [v for v in paper_ses if not np.isnan(v)]
    # each method
    for j, method in enumerate(methods_present):
        ses = []
        for s in samples:
            msub = rdflex[(rdflex["sample"] == s) & (rdflex["adjustment"] == method)]
            ses.append(float(msub["se"].iloc[0]) if len(msub) > 0 else np.nan)
        all_se_vals.extend([v for v in ses if not np.isnan(v)])
        hatch = "///" if "opt-bw" in method else ""
        ax.bar(
            x - 0.4 + (j + 1.5) * width,
            ses,
            width,
            label=_oi_label(method),
            color=_oi_color(method),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
        )
    # cap y-axis if bars explode (just for better visuals)
    if all_se_vals:
        median_se = float(np.nanmedian(all_se_vals))
        max_se = float(np.nanmax(all_se_vals))
        if max_se > 5 * median_se and median_se > 0:
            y_cap = 3 * median_se
            ax.set_ylim(0, y_cap)
            # annotate truncated bars
            for j, method in enumerate(methods_present):
                for si, s in enumerate(samples):
                    msub = rdflex[
                        (rdflex["sample"] == s) & (rdflex["adjustment"] == method)
                    ]
                    if len(msub) > 0:
                        se_val = float(msub["se"].iloc[0])
                        if se_val > y_cap:
                            bx = x[si] - 0.4 + (j + 1.5) * width
                            ax.text(
                                bx,
                                y_cap * 0.95,
                                f"{se_val:.0f}",
                                ha="center",
                                va="top",
                                fontsize=6,
                                fontweight="bold",
                                color="red",
                                rotation=90,
                            )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Standard Error")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, fontsize=10)
    ax.legend(
        fontsize=7,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    _grey_squares(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# SE diagnostic plots
def plot_forest_estimates(results, design, sample, out_path):
    # One row per method, marker at point estimate with horizontal line for 95% CI
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[
        (df["design"] == design)
        & (df["sample"] == sample)
        & (df["method"].isin(["RDFlex", "rdrobust"]))
    ]
    df = df[df["coef"].notna()].copy()
    if df.empty:
        return
    # order rows
    rows = []
    for mk in adj_order:
        sub = df[df["adjustment"] == mk]
        if len(sub) > 0:
            rows.append(sub.iloc[0])
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    nocov = plot_df[plot_df["adjustment"] == "no covariates"]
    nocov_coef = float(nocov["coef"].iloc[0]) if len(nocov) > 0 else np.nan
    nocov_ci_lo = float(nocov["ci_low"].iloc[0]) if len(nocov) > 0 else np.nan
    nocov_ci_hi = float(nocov["ci_high"].iloc[0]) if len(nocov) > 0 else np.nan
    labels = [_oi_label(m) for m in plot_df["adjustment"]]
    coefs = plot_df["coef"].astype(float).values
    ci_lo = plot_df["ci_low"].astype(float).values
    ci_hi = plot_df["ci_high"].astype(float).values
    colors = [_oi_color(m) for m in plot_df["adjustment"]]
    y_pos = np.arange(len(labels))[::-1]
    # detect catastrophic CIs for axis clipping
    well_behaved_mask = np.abs(ci_hi - ci_lo) < 500
    if well_behaved_mask.any():
        wb_lo = np.nanmin(ci_lo[well_behaved_mask])
        wb_hi = np.nanmax(ci_hi[well_behaved_mask])
        x_margin = (wb_hi - wb_lo) * 0.15
        x_lo = wb_lo - x_margin
        x_hi = wb_hi + x_margin
    else:
        x_lo, x_hi = None, None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(labels))))
    # nocov CI shaded band
    if not np.isnan(nocov_ci_lo):
        ax.axvspan(
            nocov_ci_lo, nocov_ci_hi, color="#000000", alpha=0.07, label="No-cov 95% CI"
        )
    # reference lines
    ax.axvline(0, color="black", lw=1.0, zorder=1)
    if not np.isnan(nocov_coef):
        ax.axvline(
            nocov_coef,
            color="black",
            ls="--",
            lw=0.8,
            alpha=0.6,
            label=f"No-cov estimate ({nocov_coef:.1f})",
        )
    for i, yi in enumerate(y_pos):
        ci_width = ci_hi[i] - ci_lo[i]
        clipped = x_lo is not None and ci_width > 500
        lo_draw = max(ci_lo[i], x_lo) if clipped else ci_lo[i]
        hi_draw = min(ci_hi[i], x_hi) if clipped else ci_hi[i]
        ax.plot(
            [lo_draw, hi_draw], [yi, yi], color=colors[i], lw=2, solid_capstyle="butt"
        )
        # caps for non-clipped ends
        cap_len = 0.15
        if not clipped or ci_lo[i] >= x_lo:
            ax.plot(
                [ci_lo[i], ci_lo[i]],
                [yi - cap_len, yi + cap_len],
                color=colors[i],
                lw=2,
            )
        if not clipped or ci_hi[i] <= x_hi:
            ax.plot(
                [ci_hi[i], ci_hi[i]],
                [yi - cap_len, yi + cap_len],
                color=colors[i],
                lw=2,
            )
        # marker at point estimate
        coef_draw = coefs[i]
        if x_lo is not None and (coef_draw < x_lo or coef_draw > x_hi):
            coef_draw = np.clip(coefs[i], x_lo, x_hi)
        ax.plot(coef_draw, yi, "o", color=colors[i], markersize=7, zorder=5)
        # annotate clipped bars
        if clipped:
            ax.annotate(
                f"CI: [{ci_lo[i]:.0f}, {ci_hi[i]:.0f}]",
                xy=(hi_draw, yi),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=6,
                color="red",
                va="center",
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Forward 3h return (bp)")
    if x_lo is not None:
        ax.set_xlim(x_lo, x_hi)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _grey_squares(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_standardised_diff(results, design, sample, out_path):
    # Bar plot of (beta_m - beta_nocov) / se_nocov for each method
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[
        (df["design"] == design)
        & (df["sample"] == sample)
        & (df["method"].isin(["RDFlex", "rdrobust"]))
    ]
    df = df[df["coef"].notna()].copy()
    if df.empty:
        return
    nocov = df[df["adjustment"] == "no covariates"]
    if len(nocov) == 0:
        return
    nocov_coef = float(nocov["coef"].iloc[0])
    nocov_se = float(nocov["se"].iloc[0])
    if np.isnan(nocov_se) or nocov_se <= 0:
        return
    methods = []
    diffs = []
    colors = []
    for mk in adj_order:
        if mk == "no covariates":
            continue
        sub = df[df["adjustment"] == mk]
        if len(sub) == 0:
            continue
        beta_m = float(sub["coef"].iloc[0])
        if np.isnan(beta_m):
            continue
        methods.append(mk)
        diffs.append((beta_m - nocov_coef) / nocov_se)
        colors.append(_oi_color(mk))
    if not methods:
        return
    diffs_arr = np.array(diffs)
    labels = [_oi_label(m) for m in methods]
    # detect if any bars are way to large (only for visuals)
    well_behaved = np.abs(diffs_arr) < 20
    if well_behaved.all():
        y_cap = None
    else:
        wb_max = np.nanmax(np.abs(diffs_arr[well_behaved])) if well_behaved.any() else 5
        y_cap = max(5, wb_max * 1.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    display_diffs = diffs_arr.copy()
    clipped_mask = np.zeros(len(diffs_arr), dtype=bool)
    if y_cap is not None:
        for i in range(len(display_diffs)):
            if abs(display_diffs[i]) > y_cap:
                clipped_mask[i] = True
                display_diffs[i] = np.sign(display_diffs[i]) * y_cap * 0.95
    bar_colors = []
    for i, d in enumerate(diffs_arr):
        bar_colors.append(colors[i] if abs(d) <= 2.0 else "#d62728")
    bars = ax.bar(
        x,
        display_diffs,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    # shaded band
    ax.axhspan(-2, 2, color="green", alpha=0.08, label="$\\pm 2$ SE band")
    ax.axhline(0, color="black", lw=1.0)
    # annotate clipped bars
    for i in range(len(diffs_arr)):
        if clipped_mask[i]:
            ax.annotate(
                f"{diffs_arr[i]:.1f}",
                xy=(x[i], display_diffs[i]),
                xytext=(0, 5 * np.sign(display_diffs[i])),
                textcoords="offset points",
                ha="center",
                va="bottom" if display_diffs[i] > 0 else "top",
                fontsize=7,
                fontweight="bold",
                color="red",
            )
        else:
            ax.text(
                x[i],
                display_diffs[i] + 0.05 * np.sign(display_diffs[i]),
                f"{diffs_arr[i]:.2f}",
                ha="center",
                va="bottom" if display_diffs[i] >= 0 else "top",
                fontsize=7,
            )
    if y_cap is not None:
        ax.set_ylim(-y_cap, y_cap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(
        "$(\\hat{\\tau}_m - \\hat{\\tau}_{\\mathrm{nocov}}) \\,/\\, \\mathrm{SE}_{\\mathrm{nocov}}$"
    )
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    _grey_squares(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Compute Statistical power analysis and plot it
def compute_power_analysis(results, design):
    # Compute MDE at 80% power for each RDFlex adjustment method
    from scipy import stats as sp_stats

    df = pd.DataFrame(results)
    df = df[(df["design"] == design) & (df["method"].isin(["RDFlex", "rdrobust"]))]
    df = df[df["se"].notna() & (df["se"] > 0)]
    if df.empty:
        return pd.DataFrame()
    z_alpha = sp_stats.norm.ppf(0.975)  # 1.96
    z_beta = sp_stats.norm.ppf(0.80)  # 0.8416
    rows = []
    for _, row in df.iterrows():
        se = float(row["se"])
        mde = (z_alpha + z_beta) * se
        paper_key = (
            f"Table_VI_A|{row['sample']}"
            if design == "sharp"
            else f"Table_VI_B|{row['sample']}"
        )
        paper = PAPER_VALUES.get(paper_key)
        power_at_paper = None
        if paper is not None:
            tau = abs(float(paper["coef"]))
            if se > 0:
                power_at_paper = float(sp_stats.norm.cdf(tau / se - z_alpha))
        rows.append(
            {
                "sample": row["sample"],
                "adjustment": row["adjustment"],
                "n": row.get("n", "?"),
                "se": se,
                "mde_80pct": round(mde, 2),
                "paper_coef": abs(float(paper["coef"])) if paper else np.nan,
                "power_at_paper_effect": (
                    round(power_at_paper, 4) if power_at_paper is not None else None
                ),
                "detectable": (
                    "Yes"
                    if power_at_paper is not None and abs(float(paper["coef"])) > mde
                    else ("No" if paper is not None else "N/A")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_power_analysis(results, design, out_path):
    # plot power analysis table
    power_df = compute_power_analysis(results, design)
    if power_df.empty:
        return
    # filter out rdrobust opt-bw rows
    power_df = power_df[~power_df["adjustment"].str.contains("opt-bw", na=False)]
    if power_df.empty:
        return
    col_labels = [
        "Sample",
        "Adjustment",
        "N",
        "SE",
        "MDE (80%)",
        "Paper |Coef|",
    ]
    cells = []
    for _, row in power_df.iterrows():
        cells.append(
            [
                row["sample"],
                _oi_label(row["adjustment"]),
                f"{int(row['n']):,}" if not np.isnan(float(row["n"])) else "?",
                f"{row['se']:.2f}",
                f"{row['mde_80pct']:.2f}",
                (
                    f"{row['paper_coef']:.2f}"
                    if not np.isnan(row["paper_coef"])
                    else "N/A"
                ),
            ]
        )
    total_rows = len(cells)
    fig_h = 1.4 + 0.38 * max(total_rows, 1)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


# Main results (estimate, ci) plot
def plot_main_results_table(results, design, sample, out_path):
    # Generate main results summary table
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[(df["design"] == design) & (df["sample"] == sample)]
    df = df[df["method"].isin(["RDFlex", "rdrobust"])]
    if df.empty:
        return
    df = df[df["coef"].notna()].copy()
    df["ci_width"] = (df["ci_high"] - df["ci_low"]).astype(float)
    df["se"] = df["se"].astype(float)
    df["coef"] = df["coef"].astype(float)
    # compute MDE and delta CI
    nocov_row = df[df["adjustment"] == "no covariates"]
    nocov_ci = nocov_row["ci_width"].values[0] if len(nocov_row) > 0 else np.nan
    ordered = []
    for adj in adj_order:
        row = df[df["adjustment"] == adj]
        if len(row) > 0:
            ordered.append(row.iloc[0])
    if not ordered:
        return
    col_labels = ["Method", r"$\hat{\tau}$", "SE", "CI width", r"$\Delta$CI (%)", "MDE"]
    cells = []
    for row in ordered:
        ci_w = float(row["ci_width"])
        se_val = float(row["se"])
        coef_val = float(row["coef"])
        mde = 2.8 * se_val
        if np.isnan(nocov_ci) or row["adjustment"] == "no covariates":
            delta_str = "---"
        else:
            delta = (ci_w - nocov_ci) / nocov_ci * 100
            delta_str = f"{delta:+.1f}"
        cells.append(
            [
                _oi_label(row["adjustment"]),
                f"{coef_val:.1f}",
                f"{se_val:.1f}",
                f"{ci_w:.1f}",
                delta_str,
                f"{mde:.1f}",
            ]
        )
    total_rows = len(cells)
    fig_h = 1.4 + 0.38 * max(total_rows, 1)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    # color-code method column cells
    for r in range(1, total_rows + 1):
        method_name = cells[r - 1][0]
        # find original adj key for color
        for adj_key, label in OI_LABELS.items():
            if label == method_name:
                color = _oi_color(adj_key)
                tbl[(r, 0)].set_text_props(color=color, weight="bold")
                break
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# Sample comparison plots
def plot_sample_comparison(df, out_path):
    # sample comparison barplot
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"])
    samples = {
        "All": pd.Series(True, index=use.index),
        "Auth": use["after_auth"] == 1,
        "NoAuth": use["after_auth"] == 0,
        "Auth_NegRet": (use["after_auth"] == 1) & (use["ret_coindesk"] < 0),
        "Auth_PosRet": (use["after_auth"] == 1) & (use["ret_coindesk"] > 0),
    }
    paper_n = {
        "All": 1602,
        "Auth": 464,
        "NoAuth": 1138,
        "Auth_NegRet": 214,
        "Auth_PosRet": 250,
    }
    col_labels = ["Sample", "Paper N", "Replication N", "Difference", "Match"]
    cells = []
    for label, mask in samples.items():
        rn = int(mask.sum())
        pn = paper_n.get(label, 0)
        diff = rn - pn
        match = "Yes" if diff == 0 else "No"
        cells.append([label, f"{pn:,}", f"{rn:,}", f"{diff:+,}", match])
    # grouped bar chart
    labels = [c[0] for c in cells]
    paper_vals = [int(c[1].replace(",", "")) for c in cells]
    repl_vals = [int(c[2].replace(",", "")) for c in cells]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        repl_vals,
        width,
        label="Replication",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        paper_vals,
        width,
        label="Original Paper",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )
    # annotate bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Observations")
    ax.legend(frameon=False)
    _grey_squares(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ci_width_comparison(results, design, out_path):
    # CI width comparison
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[(df["design"] == design) & (df["method"].isin(["RDFlex", "rdrobust"]))]
    if df.empty:
        return
    df = df[df["coef"].notna()].copy()
    df["ci_width"] = (df["ci_high"] - df["ci_low"]).astype(float)
    samples = sorted(df["sample"].unique())
    n_samples = len(samples)
    cols = min(n_samples, 2)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    for idx, s in enumerate(samples):
        ax = axes[idx]
        sub = df[df["sample"] == s]
        methods = []
        ci_widths = []
        colors = []
        hatches = []
        for mk in adj_order:
            msub = sub[sub["adjustment"] == mk]
            if len(msub) > 0:
                methods.append(mk)
                ci_widths.append(msub["ci_width"].mean())
                colors.append(_oi_color(mk))
                hatches.append("///" if "opt-bw" in mk else "")
        x = np.arange(len(methods))
        bars = ax.bar(
            x, ci_widths, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
        )
        for bi, h in enumerate(hatches):
            if h:
                bars[bi].set_hatch(h)
        # annotate each bar with its CI width value
        for bi, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{ci_widths[bi]:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_oi_label(m) for m in methods], rotation=45, ha="right", fontsize=8
        )
        ax.set_ylabel("Average CI Width")
        ax.set_title(s)
        _grey_squares(ax)
    for j in range(n_samples, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Super Learner weights plot
def plot_super_learner_weights(df, out_path):
    # Super Learner stacking weights from a fresh fit on Auth sample
    sub = df.dropna(subset=["fret", "below_cutoff"] + COVARIATES)
    sub = sub[sub["after_auth"] == 1]
    if len(sub) < 50:
        return
    X = sub[COVARIATES].values
    y = sub["fret"].values
    base_names = ["Ridge", "Lasso", "RF", "LGBM", "XGB", "NNet"]
    base_learners = [
        RidgeCV(cv=5),
        LassoCV(cv=5, n_jobs=-1, max_iter=5000),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        LGBMRegressor(n_estimators=200, verbose=-1, random_state=42),
        XGBRegressor(n_estimators=200, verbosity=0, random_state=42),
        ScaledMLPRegressor(n_hidden=20, alpha=0.001, max_iter=1000),
    ]
    sl = SuperLearnerRegressor(base_learners=base_learners)
    sl.fit(X, y)
    # Okabe-Ito colors for base learners
    bl_colors = {
        "Ridge": "#56B4E9",
        "Lasso": "#009E73",
        "RF": "#F0E442",
        "LGBM": "#0072B2",
        "XGB": "#D55E00",
        "NNet": "#CC79A7",
    }
    colors = [bl_colors.get(n, "#999999") for n in base_names]
    # dual panel, weights + CV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Stacking weights
    ax1.set_title("Base Learner Weights", fontsize=11, fontweight="bold")
    bars1 = ax1.bar(
        base_names, sl.weights_, color=colors, edgecolor="black", linewidth=0.5
    )
    ax1.set_ylabel("Stacking Weight")
    ax1.set_xlabel("Base Learner")
    for i, w in enumerate(sl.weights_):
        ax1.text(i, w + 0.005, f"{w:.3f}", ha="center", fontsize=9)
    _grey_squares(ax1)
    # CV scores
    from sklearn.model_selection import cross_val_score

    cv_r2 = []
    for bl in base_learners:
        try:
            scores = cross_val_score(bl, X, y, cv=5, scoring="r2", n_jobs=1)
            cv_r2.append(max(scores.mean(), 0))
        except Exception:
            cv_r2.append(0.0)
    ax2.set_title("Base Learner Performance", fontsize=11, fontweight="bold")
    bars2 = ax2.bar(base_names, cv_r2, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("CV R²")
    ax2.set_xlabel("Base Learner")
    for i, r2 in enumerate(cv_r2):
        ax2.text(i, r2 + 0.001, f"{r2:.4f}", ha="center", fontsize=9)
    _grey_squares(ax2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Robustness/ Falsification tests
def robustness_predetermined_and_placebo(df):
    # Predetermined Covariates & Placebo Outcomes
    from scipy.stats import norm as _norm

    # Formal covariate continuity test
    rdrobust_cov_csv = RESULTS_DIR / "covariate_continuity_rdrobust.csv"
    if rdrobust_cov_csv.exists():
        cov_df = pd.read_csv(rdrobust_cov_csv)
        col_labels_a = [
            "Covariate",
            "MSE-Optimal BW",
            "RD Estimator",
            "Robust p-value",
            "Robust CI",
            "Eff. N",
        ]
        cells_a = []
        for _, row in cov_df.iterrows():
            label = COVARIATE_LABELS.get(row["covariate"], row["covariate"])
            cells_a.append(
                [
                    label,
                    f"{row['h_mse']:.1f}",
                    f"{row['rd_est']:.4f}",
                    f"{row['robust_pval']:.4f}",
                    f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
                    f"{int(row['n_eff']):,}",
                ]
            )
        title_a = "Formal Continuity-Based Analysis for Covariates (rdrobust, Auth)"
        subtitle_a = (
            "Each covariate tested as outcome with MSE-optimal bandwidth. "
            "Significant p-values (< 0.05) indicate a violation."
        )
    else:
        sub = df.dropna(subset=["below_cutoff", "price_dist"] + COVARIATES)
        sub = sub[(sub["price_dist"].abs() <= 50) & (sub["after_auth"] == 1)]
        col_labels_a = [
            "Covariate",
            "Below Mean",
            "Above Mean",
            "Diff",
            "t-stat",
            "p-value",
        ]
        cells_a = []
        for cov in COVARIATES:
            cov_data = sub.dropna(subset=[cov, "below_cutoff"])
            below = cov_data.loc[cov_data["below_cutoff"] == 1, cov].astype(float)
            above = cov_data.loc[cov_data["below_cutoff"] == 0, cov].astype(float)
            if len(below) < 10 or len(above) < 10:
                continue
            mean_b, mean_a = below.mean(), above.mean()
            diff = mean_b - mean_a
            se_d = np.sqrt(below.var() / len(below) + above.var() / len(above))
            if se_d > 0:
                t_s = diff / se_d
                p_v = 2 * (1 - _norm.cdf(abs(t_s)))
            else:
                t_s, p_v = np.nan, np.nan
            label = COVARIATE_LABELS.get(cov, cov)
            cells_a.append(
                [
                    label,
                    f"{mean_b:.4f}",
                    f"{mean_a:.4f}",
                    f"{diff:.4f}",
                    f"{t_s:.3f}" if not np.isnan(t_s) else "N/A",
                    f"{p_v:.4f}" if not np.isnan(p_v) else "N/A",
                ]
            )
        title_a = "Covariate Smoothness at the Cutoff (t-test, h=50, Auth)"
        subtitle_a = "Significant p-values (< 0.05) would indicate a violation of the RDD assumption."
    # render as table image
    total_rows = len(cells_a)
    fig_h = 1.4 + 0.38 * max(total_rows, 1)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells_a,
        colLabels=col_labels_a,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for c in range(len(col_labels_a)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    # no title — clean table style
    plt.savefig(
        FIG_DIR / "robustness_covariate_smoothness.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    # Placebo cutoff tests
    use = df.dropna(subset=["fret", "price_dist"])
    use = use[use["after_auth"] == 1]
    placebo_cutoffs = [-200, -100, 0, 100, 200]
    results_b = []
    for pc in placebo_cutoffs:
        sub_p = use.copy()
        sub_p["price_dist_shifted"] = sub_p["price_dist"] - pc
        sub_p["below_placebo"] = (sub_p["price_dist_shifted"] < 0).astype(int)
        sub_p = sub_p[sub_p["price_dist_shifted"].abs() <= 50]
        if len(sub_p) < 30:
            results_b.append((pc, np.nan, np.nan))
            continue
        X_p = add_constant(sub_p["below_placebo"].astype(float))
        try:
            m = OLS(sub_p["fret"].astype(float), X_p).fit(cov_type="HC1")
            results_b.append(
                (pc, float(m.params["below_placebo"]), float(m.bse["below_placebo"]))
            )
        except Exception:
            results_b.append((pc, np.nan, np.nan))
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, coef, se in results_b:
        if np.isnan(coef):
            continue
        color = "#d62728" if c == 0 else "#1f77b4"
        marker = "D" if c == 0 else "o"
        ax.errorbar(
            c,
            coef,
            yerr=1.96 * se,
            fmt=marker,
            color=color,
            capsize=4,
            markersize=8,
            linewidth=2,
        )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=0, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    # paper estimate annotation
    paper = PAPER_VALUES.get("Table_VI_A|Auth")
    if paper is not None:
        ax.annotate(
            f"Paper estimate: {paper['coef']}",
            xy=(0.98, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            color="#d62728",
        )
    ax.set_xlabel("Cutoff Location (dollars from round number)")
    ax.set_ylabel("Treatment Effect Estimate")
    _grey_squares(ax)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "robustness_placebo_cutoffs.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def robustness_density_test(df):
    # Density of Running Variable
    sub = df.dropna(subset=["price_dist"])
    rv = sub["price_dist"].astype(float).values
    # test in several window widths for robustness
    windows = [5, 10, 20, 50]
    test_results = []
    for w in windows:
        in_window = rv[(rv >= -w) & (rv <= w)]
        n_total = len(in_window)
        n_below = int(np.sum(in_window < 0))
        n_above = int(np.sum(in_window > 0))
        n_at_zero = int(np.sum(in_window == 0))
        # exclude observations exactly at the cutoff for the binomial test
        n_test = n_below + n_above
        if n_test < 10:
            test_results.append(
                (w, n_total, n_below, n_above, n_at_zero, np.nan, np.nan)
            )
            continue
        # two-sided binomial test
        p_val = float(sp_stats.binomtest(n_below, n_test, 0.5).pvalue)
        prop = n_below / n_test
        test_results.append((w, n_total, n_below, n_above, n_at_zero, prop, p_val))
    # histogram plot
    n_bins = 50
    below = rv[rv < 0]
    above = rv[rv >= 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    bins_below = np.linspace(max(rv.min(), -200), 0, n_bins + 1)
    bins_above = np.linspace(0, min(rv.max(), 200), n_bins + 1)
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
    ax.set_xlabel("Distance from Round Threshold (price_dist)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    _grey_squares(ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "robustness_density_test.png", dpi=150, bbox_inches="tight")
    plt.close()


def robustness_bandwidth_sensitivity(df):
    # Sensitivity to Bandwidth Choice
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"])
    use = use[use["after_auth"] == 1]
    if len(use) < 30:
        return
    bandwidths = list(range(20, 110, 10))
    results = []
    for h in bandwidths:
        sub = use[use["price_dist"].abs() <= h]
        if len(sub) < 10:
            results.append(
                {"bandwidth": h, "coef": np.nan, "se": np.nan, "n": len(sub)}
            )
            continue
        X_b = add_constant(sub["below_cutoff"].astype(float))
        try:
            m = OLS(sub["fret"].astype(float), X_b).fit(cov_type="HC1")
            results.append(
                {
                    "bandwidth": h,
                    "coef": float(m.params["below_cutoff"]),
                    "se": float(m.bse["below_cutoff"]),
                    "pval": float(m.pvalues["below_cutoff"]),
                    "n": int(m.nobs),
                }
            )
        except Exception:
            results.append(
                {"bandwidth": h, "coef": np.nan, "se": np.nan, "n": len(sub)}
            )
    res_df = pd.DataFrame(results)
    ok = res_df[res_df["coef"].notna()]
    fig, ax = plt.subplots(figsize=(10, 6))
    if not ok.empty:
        coefs = ok["coef"].values
        ses = ok["se"].values
        bws = ok["bandwidth"].values
        ax.plot(
            bws, coefs, "o-", color="#1f77b4", lw=2, markersize=6, label="Coefficient"
        )
        ax.fill_between(
            bws, coefs - 1.96 * ses, coefs + 1.96 * ses, alpha=0.2, color="#1f77b4"
        )
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        # paper benchmark
        paper = PAPER_VALUES.get("Table_VI_A|Auth")
        if paper is not None:
            ax.axhline(
                float(paper["coef"]),
                color="#d62728",
                ls="--",
                lw=1.5,
                label=f"Paper: {paper['coef']}",
            )
        ax.axvline(50, color="#d62728", ls=":", lw=1, alpha=0.5, label="Paper h=50")
        ax.legend(fontsize=8)
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Coefficient")
    _grey_squares(ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "robustness_bw_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()


# Kernel Sensitivity test
def robustness_kernel_comparison(df):
    # Compare RDFlex estimates using triangular vs epanechnikov kernels
    kernel_csv = RESULTS_DIR / "kernel_robustness.csv"
    # if cached CSV exists, just re-plot
    if kernel_csv.exists():
        kernel_df = pd.read_csv(kernel_csv)
        _plot_kernel_robustness(kernel_df)
        return
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"] + COVARIATES).copy()
    sub = use[use["after_auth"] == 1].copy()
    cols = ["fret", "below_cutoff", "price_dist"] + COVARIATES
    sub = sub[cols].copy()
    if len(sub) < 50:
        return
    # tune once, shared across both kernels
    tuned, _ = tune_all_ml_models(sub[COVARIATES], sub["fret"], tag="kernel-robustness")
    kernels = ["triangular", "epanechnikov"]
    kernel_results = []
    for kernel in kernels:
        for name, ml_g, _ in get_adjustment_configs(tuned, fuzzy=False):
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="below_cutoff",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=0,
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    h_fs=50.0,
                    fs_specification="cutoff",
                    fs_kernel=kernel,
                    h=50.0,
                )
                est.fit()
                rec = _record_rdflex(est, "sharp", "Auth", name, len(sub))
                rec["kernel"] = kernel
                ci = est.confint()
                rec["ci_width"] = float(ci.iloc[0, 1]) - float(ci.iloc[0, 0])
                kernel_results.append(rec)
            except Exception as exc:
                pass
    if not kernel_results:
        return
    kernel_df = pd.DataFrame(kernel_results)
    kernel_df.to_csv(kernel_csv, index=False)
    _plot_kernel_robustness(kernel_df)


def _plot_kernel_robustness(kernel_df):
    # relative CI width for triangular vs epanechnikov
    adj_order = list(OI_COLORS.keys())
    methods_in_results = [
        m
        for m in adj_order
        if m in kernel_df["adjustment"].values and m != "no covariates"
    ]
    if len(methods_in_results) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(methods_in_results))
    bar_width = 0.35
    for i, kernel in enumerate(["triangular", "epanechnikov"]):
        kdf = kernel_df[kernel_df["kernel"] == kernel]
        nocov_row = kdf[kdf["adjustment"] == "no covariates"]
        if len(nocov_row) == 0:
            continue
        nocov_ci = nocov_row["ci_width"].values[0]
        if np.isnan(nocov_ci) or nocov_ci <= 0:
            continue
        ratios = []
        colors = []
        for method in methods_in_results:
            m_row = kdf[kdf["adjustment"] == method]
            if len(m_row) > 0 and not np.isnan(m_row["ci_width"].values[0]):
                ratios.append(m_row["ci_width"].values[0] / nocov_ci)
            else:
                ratios.append(np.nan)
            colors.append(_oi_color(method))
        offset = (i - 0.5) * bar_width
        hatch = "" if kernel == "triangular" else "///"
        bars = ax.bar(
            x_pos + offset,
            ratios,
            width=bar_width,
            label=kernel.title(),
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
        )
        for bar, v in zip(bars, ratios):
            if not np.isnan(v):
                change = (v - 1) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{change:+.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [_oi_label(m) for m in methods_in_results],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Relative CI Width (No Covariates = 1.0)")
    # grey legend
    legend_handles = [
        Patch(facecolor="#999999", edgecolor="black", label="Triangular"),
        Patch(
            facecolor="#999999", edgecolor="black", hatch="///", label="Epanechnikov"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=9,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    _grey_squares(ax)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "robustness_kernel_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# assemble results and write CSVs
def assemble_results(ols_v, ols_vi_a, ols_vi_b, rdflex_sharp, rdflex_fuzzy):
    # combine paper, OLS and RDFlex rows into one results DataFrame
    rows = []
    # paper reference rows
    for key, val in PAPER_VALUES.items():
        panel, sample = key.split("|")
        if panel == "Table_V":
            design = "first_stage"
        elif panel == "Table_VI_A":
            design = "sharp"
        else:
            design = "fuzzy"
        # compute standard error from t-statistic and coefficient
        coef_p = float(val["coef"])
        tstat_p = float(val["tstat"])
        se_p = abs(coef_p / tstat_p) if tstat_p != 0 else float("nan")
        rows.append(
            {
                "method": "Paper (Griffin & Shams 2020)",
                "design": design,
                "sample": sample,
                "adjustment": "none (OLS)",
                "coef": coef_p,
                "tstat": tstat_p,
                "se": se_p,
                "ci_low": coef_p - 1.96 * se_p,
                "ci_high": coef_p + 1.96 * se_p,
                "n": val["n"],
            }
        )
    for label, v in ols_v.items():
        rows.append(
            {
                "method": "OLS replication",
                "design": "first_stage",
                "sample": label,
                "adjustment": "none (OLS)",
                **v,
            }
        )
    for label, v in ols_vi_a.items():
        rows.append(
            {
                "method": "OLS replication",
                "design": "sharp",
                "sample": label,
                "adjustment": "none (OLS)",
                **v,
            }
        )
    for label, v in ols_vi_b.items():
        if "error" in v:
            continue
        rows.append(
            {
                "method": "IV2SLS replication",
                "design": "fuzzy",
                "sample": label,
                "adjustment": "none (2SLS)",
                **v,
            }
        )
    rows.extend(rdflex_sharp)
    rows.extend(rdflex_fuzzy)
    return pd.DataFrame(rows)


def save_master_dataset(df):
    # write the constructed analysis columns to master_dataset.csv
    keep = (
        [
            "date",
            "htime",
            "after_auth",
            "close_coindesk",
            "ret_coindesk",
            "vol",
            "fret",
            "flow",
            "flow_scaled",
            "netflow_lsg",
            "netflow_plbt",
            "netflow_oth_noplbt",
            "price_dist",
            "round_prc_dist",
            "below_cutoff",
            "in_bandwidth",
            "inst",
        ]
        + COVARIATES
        + ["lag_ret_neg"]
    )
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(RESULTS_DIR / "master_dataset.csv", index=False)


# Relative CI width plot
def plot_relative_ci_width(results, design, out_path):
    # Grouped bar chart of relative CI width (method / no covariates) for each sample
    adj_order = list(OI_COLORS.keys())
    df = pd.DataFrame(results)
    df = df[(df["design"] == design) & (df["method"].isin(["RDFlex", "rdrobust"]))]
    if df.empty:
        return
    df = df[df["coef"].notna()].copy()
    df["ci_width"] = (df["ci_high"] - df["ci_low"]).astype(float)
    samples = sorted(df["sample"].unique())
    # methods excluding no_cov since it is the baseline
    ml_methods = [
        m for m in adj_order if m in df["adjustment"].values and m != "no covariates"
    ]
    if not ml_methods:
        return
    # compute relative CI width: method_ci_width / nocov_ci_width
    norm_rows = []
    for s in samples:
        nocov_row = df[(df["sample"] == s) & (df["adjustment"] == "no covariates")]
        if len(nocov_row) == 0:
            continue
        nocov_ci = nocov_row["ci_width"].values[0]
        if np.isnan(nocov_ci) or nocov_ci <= 0:
            continue
        for method in ml_methods:
            m_row = df[(df["sample"] == s) & (df["adjustment"] == method)]
            if len(m_row) == 0:
                continue
            m_ci = m_row["ci_width"].values[0]
            if np.isnan(m_ci):
                continue
            norm_rows.append(
                {
                    "sample": s,
                    "method": method,
                    "relative_ci": m_ci / nocov_ci,
                }
            )
    if not norm_rows:
        return
    norm_df = pd.DataFrame(norm_rows)
    n_methods = len(ml_methods)
    bar_width = 0.8 / max(n_methods, 1)
    x_base = np.arange(len(samples))
    fig, ax = plt.subplots(figsize=(max(12, n_methods * 1.5), 11))
    # cap y-axis if bars explode (just visuals)
    all_vals_flat = norm_df["relative_ci"].values
    median_val = float(np.nanmedian(all_vals_flat))
    y_cap = max(3.0, median_val * 5) if median_val < 5.0 else None
    for j, method in enumerate(ml_methods):
        vals = []
        for s in samples:
            row = norm_df[(norm_df["method"] == method) & (norm_df["sample"] == s)]
            vals.append(row["relative_ci"].values[0] if len(row) > 0 else np.nan)
        offset = (j - n_methods / 2 + 0.5) * bar_width
        # clip bar heights for display
        display_vals = []
        for v in vals:
            if y_cap is not None and not np.isnan(v) and v > y_cap:
                display_vals.append(y_cap * 0.95)
            else:
                display_vals.append(v)
        bars = ax.bar(
            x_base + offset,
            display_vals,
            width=bar_width,
            label=_oi_label(method),
            color=_oi_color(method),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        # annotate bars with change percentage
        for bi, (bar, v) in enumerate(zip(bars, vals)):
            if not np.isnan(v):
                change = (v - 1) * 100
                if y_cap is not None and v > y_cap:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{v:.1f}x\n({change:+.0f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                        color="red",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{change:+.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
    if y_cap is not None:
        ax.set_ylim(0, y_cap)
    ax.set_xticks(x_base)
    ax.set_xticklabels(samples)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Relative CI Width (No Covariates = 1.0)")
    ax.legend(
        fontsize=8,
        ncol=4,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
    )
    _grey_squares(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Seed Robustness test
def run_seed_robustness(df):
    # Uses Auth sample, sharp design, h=50
    use = df.dropna(subset=["fret", "below_cutoff", "price_dist"] + COVARIATES).copy()
    sub = use[use["after_auth"] == 1].copy()
    sub = sub[sub["price_dist"].abs() <= 50]
    cols = ["fret", "below_cutoff", "price_dist"] + COVARIATES
    sub = sub[cols].copy()
    if len(sub) < 50:
        return
    # tune models once
    tuned, _ = tune_all_ml_models(sub[COVARIATES], sub["fret"], tag="seed-robustness")
    # all adjustment methods
    configs = get_adjustment_configs(tuned, fuzzy=False)
    # exclude no covariates since it is deterministic and not affected by random seed
    test_configs = [
        (name, ml_g) for name, ml_g, _ in configs if name != "no covariates"
    ]
    seeds = [42, 123, 456, 789, 2024]
    seed_results = []
    for seed in seeds:
        np.random.seed(seed)
        for method_name, ml_g in test_configs:
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col="fret",
                    d_cols="below_cutoff",
                    score_col="price_dist",
                    x_cols=COVARIATES,
                )
                est = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=clone(ml_g),
                    ml_m=None,
                    fuzzy=False,
                    cutoff=0,
                    n_folds=RDFLEX_N_Folds,
                    n_rep=RDFLEX_N_REP,
                    h_fs=50.0,
                    fs_specification="cutoff",
                    fs_kernel="triangular",
                    h=50.0,
                )
                est.fit()
                rec = _record_rdflex(est, "sharp", "Auth", method_name, len(sub))
                rec["seed"] = seed
                seed_results.append(rec)
            except Exception as exc:
                pass
    # reset random state
    np.random.seed(None)
    if not seed_results:
        return
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(RESULTS_DIR / "seed_robustness.csv", index=False)
    _plot_seed_robustness(seed_df)


def _plot_seed_robustness(seed_df):
    # Plot coefficient estimates across seeds for each adjustment method
    methods_in_results = [
        m for m in OI_COLORS.keys() if m in seed_df["adjustment"].values
    ]
    # also include any methods not in OI_COLORS
    for m in seed_df["adjustment"].unique():
        if m not in methods_in_results:
            methods_in_results.append(m)
    n_methods = len(methods_in_results)
    if n_methods == 0:
        return
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    for idx, method in enumerate(methods_in_results):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i, col_i]
        mdf = seed_df[seed_df["adjustment"] == method].sort_values("seed")
        x_pos = np.arange(len(mdf))
        ax.errorbar(
            x_pos,
            mdf["coef"].values.astype(float),
            yerr=1.96 * mdf["se"].values.astype(float),
            fmt="o",
            color=_oi_color(method),
            capsize=5,
            capthick=1.5,
            markersize=7,
            linewidth=1.5,
        )
        # reference: mean across seeds
        mean_coef = mdf["coef"].astype(float).mean()
        ax.axhline(
            y=mean_coef,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_coef:.1f}",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(s) for s in mdf["seed"].values], fontsize=8)
        cv_pct = (
            mdf["coef"].astype(float).std() / abs(mean_coef) * 100
            if abs(mean_coef) > 1e-6
            else 0
        )
        ax.set_title(_oi_label(method), fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        _grey_squares(ax)
        ax.axhline(y=0, color="black", linewidth=0.5)
    # hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes[row_i, col_i].set_visible(False)
    fig.supxlabel("Seed", fontsize=11)
    fig.supylabel("Treatment Effect", fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(left=0.06, bottom=0.07)
    plt.savefig(
        FIG_DIR / "robustness_seed_stability.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# main
# render the full results CSV
def plot_full_results_table():
    csv_path = RESULTS_DIR / "results_combined.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df["method"].isin(["RDFlex", "rdrobust"])].copy()
    if df.empty:
        return
    df["ci_width"] = df["ci_high"] - df["ci_low"]
    adj_order = list(OI_COLORS.keys())
    samples = ["All", "Auth", "NoAuth", "Auth_NegRet", "Auth_PosRet"]
    rows = []
    for design in ["sharp", "fuzzy"]:
        for s in samples:
            sub = df[(df["design"] == design) & (df["sample"] == s)]
            if sub.empty:
                continue
            base_r = sub.loc[sub["adjustment"] == "no covariates", "ci_width"]
            base = base_r.values[0] if len(base_r) else np.nan
            for a in adj_order:
                r = sub[sub["adjustment"] == a]
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
                        design,
                        s,
                        OI_LABELS.get(a, a),
                        f"{r['coef']:.2f}" if pd.notna(r["coef"]) else "-",
                        f"{r['se']:.2f}" if pd.notna(r["se"]) else "-",
                        f"{r['ci_width']:.2f}" if pd.notna(r["ci_width"]) else "-",
                        f"{dci:.1f}" if pd.notna(dci) and a != "no covariates" else "-",
                    ]
                )
    if not rows:
        return
    col_labels = ["Design", "Sample", "Method", "Coef", "SE", "CI Width", "CI Red. (%)"]
    fig, ax = plt.subplots(figsize=(13, 0.8 + 0.28 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    tbl.scale(1.0, 1.2)
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    plt.tight_layout(pad=0)
    plt.savefig(
        FIG_DIR / "results_table_full.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()


def main():
    # full run or --plots-only mode
    plots_only = "--plots-only" in sys.argv
    print("Reading Input ...")
    price = load_price_flow()
    df = construct_rd_vars(price)
    eom = load_eom()
    ols_v = replicate_table_v(df)
    ols_vi_a = replicate_table_vi_a(df)
    ols_vi_b = replicate_table_vi_b(df)
    eom_results = replicate_table_viii(eom)
    if not plots_only:
        print("Estimating ...")
        rdflex_sharp = run_rdflex_sharp(df)
        rdflex_fuzzy = run_rdflex_fuzzy(df)
        optbw_sharp = run_rdflex_optbw_sharp(df)
        optbw_fuzzy = run_rdflex_optbw_fuzzy(df)
        print("Build Master Dataset ...")
        save_master_dataset(df)
        results = assemble_results(
            ols_v, ols_vi_a, ols_vi_b, rdflex_sharp, rdflex_fuzzy
        )
        results.to_csv(RESULTS_DIR / "results.csv", index=False)
        pd.DataFrame(eom_results).T.to_csv(RESULTS_DIR / "eom_results.csv")
        # save optimal-bandwidth results
        optbw_all = optbw_sharp + optbw_fuzzy
        if optbw_all:
            pd.DataFrame(optbw_all).to_csv(
                RESULTS_DIR / "results_optbw.csv", index=False
            )
        # save per-sample CSVs for R rdrobust script
        use_sharp = df.dropna(
            subset=["fret", "below_cutoff", "price_dist"] + COVARIATES
        ).copy()
        sharp_samples = {
            "Auth": use_sharp["after_auth"] == 1,
            "NoAuth": use_sharp["after_auth"] == 0,
            "Auth_NegRet": (use_sharp["after_auth"] == 1)
            & (use_sharp["ret_coindesk"] < 0),
            "Auth_PosRet": (use_sharp["after_auth"] == 1)
            & (use_sharp["ret_coindesk"] > 0),
        }
        for label, mask in sharp_samples.items():
            sub = use_sharp[mask][["fret", "below_cutoff", "price_dist"] + COVARIATES]
            sub.to_csv(RESULTS_DIR / f"rdrobust_sample_sharp_{label}.csv", index=False)
        use_fuzzy = df.dropna(subset=["fret", "flow", "price_dist"] + COVARIATES).copy()
        fuzzy_samples = {
            "All": pd.Series(True, index=use_fuzzy.index),
            "Auth": use_fuzzy["after_auth"] == 1,
            "Auth_NegRet": (use_fuzzy["after_auth"] == 1)
            & (use_fuzzy["ret_coindesk"] < 0),
            "Auth_PosRet": (use_fuzzy["after_auth"] == 1)
            & (use_fuzzy["ret_coindesk"] > 0),
        }
        for label, mask in fuzzy_samples.items():
            sub = use_fuzzy[mask].copy()
            sub["high_flow"] = (sub["flow"] > sub["flow"].median()).astype(int)
            sub = sub[["fret", "high_flow", "price_dist"] + COVARIATES]
            sub.to_csv(RESULTS_DIR / f"rdrobust_sample_fuzzy_{label}.csv", index=False)
    else:
        # plots-only: load cached results
        results_csv = RESULTS_DIR / "results_combined.csv"
        if not results_csv.exists():
            results_csv = RESULTS_DIR / "results.csv"
        results = pd.read_csv(results_csv)
        # load optimal-bandwidth results
        optbw_csv = RESULTS_DIR / "results_optbw.csv"
        if optbw_csv.exists():
            optbw_df = pd.read_csv(optbw_csv)
            optbw_sharp = optbw_df[optbw_df["design"] == "sharp"].to_dict("records")
            optbw_fuzzy = optbw_df[optbw_df["design"] == "fuzzy"].to_dict("records")
        else:
            optbw_sharp, optbw_fuzzy = [], []
        # reconstruct rdflex_sharp / rdflex_fuzzy from cached results
        rdflex_mask = results["method"].isin(["RDFlex", "rdrobust"])
        rdflex_sharp = results[rdflex_mask & (results["design"] == "sharp")].to_dict(
            "records"
        )
        rdflex_fuzzy = results[rdflex_mask & (results["design"] == "fuzzy")].to_dict(
            "records"
        )
    # regenerate hyperparameter table in --plots-only mode from cache
    if plots_only:
        for bp_file in sorted(RESULTS_DIR.glob("best_params_*.csv")):
            try:
                bp_df = pd.read_csv(bp_file)
                # rebuild nested {method: {param: value}} dict, "None" -> None
                bp_loaded = {}
                for _, _r in bp_df.iterrows():
                    _val = None if str(_r["value"]) == "None" else _r["value"]
                    bp_loaded.setdefault(_r["method"], {})[_r["param"]] = _val
                tag_name = bp_file.stem.replace("best_params_", "")
                plot_hyperparameter_table(bp_loaded, tag_name)
            except Exception:
                pass

    # merge rdrobust results if available
    # the R script rdrobust_griffin.R must be run separately before this step
    rdrobust_csv = RESULTS_DIR / "results_rdrobust.csv"
    if rdrobust_csv.exists():
        rdr = pd.read_csv(rdrobust_csv)
        # ensure column alignment with results DataFrame
        expected_cols = results.columns.tolist()
        for col in expected_cols:
            if col not in rdr.columns:
                rdr[col] = np.nan
        rdr = rdr[[c for c in expected_cols if c in rdr.columns]]
        # avoid duplicates if results_combined.csv already contains rdrobust rows
        if not plots_only:
            results = pd.concat([results, rdr], ignore_index=True)
            results.to_csv(RESULTS_DIR / "results_combined.csv", index=False)
        # inject rdrobust rows into bandwidth comparison lists
        rdr_fixed = rdr[rdr["adjustment"] == "linear (rdrobust)"]
        rdr_optbw = rdr[rdr["adjustment"] == "linear (rdrobust, opt-bw)"]
        if not plots_only:
            for _, row in rdr_fixed.iterrows():
                target = rdflex_sharp if row["design"] == "sharp" else rdflex_fuzzy
                target.append(row.to_dict())
            for _, row in rdr_optbw.iterrows():
                target = optbw_sharp if row["design"] == "sharp" else optbw_fuzzy
                target.append(row.to_dict())

    # everything runs in both modes (plots + diagnostics)
    print("Plotting ...")
    plot_full_results_table()
    plot_table_1(df, FIG_DIR / "Table1_SummaryStatistics.png")
    plot_fret_discontinuity(df)
    plot_combined_round_discontinuity(df)
    plot_pooled_round_discontinuity(df)
    plot_coefficient_comparison(
        results.to_dict("records"),
        "sharp",
        FIG_DIR / "rdflex_coefficient_comparison_sharp.png",
    )
    plot_coefficient_comparison(
        results.to_dict("records"),
        "fuzzy",
        FIG_DIR / "rdflex_coefficient_comparison_fuzzy.png",
    )
    plot_covariate_importance_per_method(df)
    plot_covariate_distribution_by_cutoff(df, FIG_DIR / "covariate_balance.png")
    # bandwidth comparison (Barber-style)
    plot_bandwidth_comparison(
        rdflex_sharp,
        optbw_sharp,
        "sharp",
        FIG_DIR / "bandwidth_comparison_sharp.png",
    )
    plot_bandwidth_comparison(
        rdflex_fuzzy,
        optbw_fuzzy,
        "fuzzy",
        FIG_DIR / "bandwidth_comparison_fuzzy.png",
    )
    plot_sample_comparison(df, FIG_DIR / "sample_comparison.png")
    plot_main_results_table(
        results.to_dict("records"),
        "sharp",
        "Auth",
        FIG_DIR / "main_results_table_sharp_auth.png",
    )
    plot_ci_width_comparison(
        results.to_dict("records"),
        "sharp",
        FIG_DIR / "ci_width_comparison_sharp.png",
    )
    plot_ci_width_comparison(
        results.to_dict("records"),
        "fuzzy",
        FIG_DIR / "ci_width_comparison_fuzzy.png",
    )
    # relative CI width plots
    plot_relative_ci_width(
        results.to_dict("records"),
        "sharp",
        FIG_DIR / "relative_ci_width_sharp.png",
    )
    plot_relative_ci_width(
        results.to_dict("records"),
        "fuzzy",
        FIG_DIR / "relative_ci_width_fuzzy.png",
    )
    plot_super_learner_weights(df, FIG_DIR / "super_learner_weights.png")
    plot_se_comparison(
        results.to_dict("records"),
        "sharp",
        FIG_DIR / "se_comparison_sharp.png",
    )
    plot_se_comparison(
        results.to_dict("records"),
        "fuzzy",
        FIG_DIR / "se_comparison_fuzzy.png",
    )
    # forest estimate plots and standardised difference plots
    for des, samp in [("sharp", "Auth"), ("fuzzy", "Auth")]:
        tag = f"{des}_{samp.lower()}"
        plot_forest_estimates(
            results.to_dict("records"),
            des,
            samp,
            FIG_DIR / f"forest_estimates_{tag}.png",
        )
        plot_standardised_diff(
            results.to_dict("records"),
            des,
            samp,
            FIG_DIR / f"standardised_diff_{tag}.png",
        )
    plot_power_analysis(
        results.to_dict("records"),
        "sharp",
        FIG_DIR / "power_analysis_sharp.png",
    )
    plot_power_analysis(
        results.to_dict("records"),
        "fuzzy",
        FIG_DIR / "power_analysis_fuzzy.png",
    )
    for des in ("sharp", "fuzzy"):
        pwr = compute_power_analysis(results.to_dict("records"), des)
        if not pwr.empty:
            pwr.to_csv(RESULTS_DIR / f"power_analysis_{des}.csv", index=False)
    robustness_predetermined_and_placebo(df)
    robustness_density_test(df)
    robustness_bandwidth_sensitivity(df)
    robustness_kernel_comparison(df)
    # seed robustness plot
    if not plots_only:
        run_seed_robustness(df)
    else:
        # in plots-only mode, load cached seed robustness CSV if it exists
        seed_csv = RESULTS_DIR / "seed_robustness.csv"
        if seed_csv.exists():
            seed_df = pd.read_csv(seed_csv)
            _plot_seed_robustness(seed_df)
    print("Finished ...")


if __name__ == "__main__":
    main()