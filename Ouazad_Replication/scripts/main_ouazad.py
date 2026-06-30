# Replication of Ouazad & Kahn (2023)
#
# Sharp RDD at the conforming loan limit.
#
# Adjustment methods:
#   1. No Covariates (DummyRegressor)
#   2. Ridge
#   3. Lasso (with 2nd-order interaction features per Olma 2024 p.27)
#   4. Random Forest (Optuna-tuned)
#   5. LightGBM (Optuna-tuned)
#   6. XGBoost (Optuna-tuned)
#   7. Neural Net (MLPRegressor, Optuna-tuned)
#   8. Super Learner (stacking ensemble)

# Inputs:
#   est_sample.csv                              (estimation sample from prepare_data.R)
#   results_rdrobust.csv                        (linear rdrobust results from R, optional)
#   results_covariate_continuity.csv            (covariate continuity from R, optional)
#   rdd_rdflex_checkpoint.csv                   (RDFlex checkpoint resume, optional)
#   tuned_hyperparameters_checkpoint.csv        (hyperparameter checkpoint resume, optional)
#   covariate_importance.csv                    (re-read for plotting)
#   super_learner_weights.csv                   (re-read for plotting)
#   rdd_rdflex_results.csv                      (plots-only mode)
#   rdd_paper_results.csv                       (plots-only mode)
#   seed_robustness.csv                         (plots-only mode)
#   kernel_robustness.csv                       (plots-only mode)
#   tuned_hyperparameters.csv                   (plots-only mode)
# Output:
#   rdrobust_samples/sample_<depvar>_t<t>.csv   (samples for rdrobust_ouazad.R)
#   rdd_paper_results.csv
#   rdd_rdflex_checkpoint.csv                   (deleted on success)
#   tuned_hyperparameters_checkpoint.csv        (deleted on success)
#   rdd_rdflex_results.csv
#   covariate_importance.csv
#   super_learner_weights.csv
#   tuned_hyperparameters.csv
#   seed_robustness.csv
#   kernel_robustness.csv
#   results.csv                                 (results table for all methods)
#   figures/*.png                             (all plots and table images)

import gc
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from scipy.stats import norm
from statsmodels.api import OLS
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from doubleml import DoubleMLRDDData
from doubleml.rdd import RDFlex
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# GPU acceleration for LightGBM and XGBoost (fallback to CPU) if available
try:
    _test_xgb = XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
    _test_xgb.fit([[0]], [0])
    XGB_GPU = {"device": "cuda"}
except Exception:
    XGB_GPU = {}
try:
    _test_lgb = LGBMRegressor(device="gpu", n_estimators=1, verbose=-1)
    _test_lgb.fit([[0]], [0])
    LGB_GPU = {"device": "gpu"}
except Exception:
    LGB_GPU = {}
try:
    del _test_xgb
except NameError:
    pass
try:
    del _test_lgb
except NameError:
    pass

# paths
script_dir = Path(__file__).parent.absolute()
repo_dir = script_dir.parent 
data_dir = repo_dir / "data"
figures_dir = repo_dir / "figures"
results_dir = repo_dir / "results"
for _d in (data_dir, figures_dir, results_dir):
    _d.mkdir(exist_ok=True)

# Global Tuning Parameters
N_OPTUNA_TRIALS = 10  # Optuna trials per model
OPTUNA_CV = 5  # cross-validation folds for Optuna tuning
RDFLEX_N_FOLDS = 5  # cross-fitting folds for RDFlex estimation
RDFLEX_N_REP = 5  # repetitions for RDFlex

# paper values
PAPER_BENCHMARKS = {
    # approved — Table 5 (columns 1-4 = bw 1%-4%)
    ("approved", 0.01): {
        1: (0.0383, 0.0149),
        2: (0.0526, 0.0275),
        3: (0.0648, 0.0270),
        4: (0.0208, 0.0430),
    },
    ("approved", 0.02): {
        1: (0.0348, 0.0080),
        2: (0.0501, 0.0162),
        3: (0.0609, 0.0188),
        4: (0.0264, 0.0351),
    },
    ("approved", 0.03): {
        1: (0.0300, 0.0061),
        2: (0.0425, 0.0117),
        3: (0.0583, 0.0160),
        4: (0.0284, 0.0297),
    },
    ("approved", 0.05): {
        1: (0.0268, 0.0067),
        2: (0.0360, 0.0100),
        3: (0.0570, 0.0146),
        4: (0.0291, 0.0259),
    },
    # originated — Table 6
    ("originated", 0.01): {
        1: (0.0457, 0.0261),
        2: (0.0246, 0.0434),
        3: (0.0501, 0.0320),
        4: (-0.0083, 0.0508),
    },
    ("originated", 0.02): {
        1: (0.0427, 0.0177),
        2: (0.0408, 0.0244),
        3: (0.0607, 0.0262),
        4: (0.0060, 0.0432),
    },
    ("originated", 0.03): {
        1: (0.0356, 0.0125),
        2: (0.0393, 0.0171),
        3: (0.0608, 0.0218),
        4: (0.0140, 0.0355),
    },
    ("originated", 0.05): {
        1: (0.0304, 0.0106),
        2: (0.0349, 0.0143),
        3: (0.0604, 0.0194),
        4: (0.0177, 0.0290),
    },
    # securitized — Table 7
    ("securitized", 0.01): {
        1: (0.0466, 0.0314),
        2: (0.0342, 0.0234),
        3: (0.0872, 0.0372),
        4: (0.1663, 0.0440),
    },
    ("securitized", 0.02): {
        1: (0.0567, 0.0309),
        2: (0.0290, 0.0259),
        3: (0.1042, 0.0262),
        4: (0.1773, 0.0413),
    },
    ("securitized", 0.03): {
        1: (0.0496, 0.0299),
        2: (0.0202, 0.0268),
        3: (0.1002, 0.0249),
        4: (0.1688, 0.0418),
    },
    ("securitized", 0.05): {
        1: (0.0398, 0.0288),
        2: (0.0118, 0.0269),
        3: (0.0933, 0.0246),
        4: (0.1583, 0.0432),
    },
}
# paper Table observation counts
PAPER_N = {
    "approved": 2_572_574,
    "originated": 2_572_574,
    "securitized": 2_049_035,
}

# constants
CUTOFF = 0.0
BANDWIDTHS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
RDFLEX_BWS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
DEPVARS = ["approved", "originated", "securitized"]
KERNEL_BW = 0.01

# covariates used by RDFlex (all adjustment strategies share this set)
RDFLEX_COVARIATES = [
    "applicant_income_log",
    "log_loan_amount",
    "agency_1",
    "agency_2",
    "agency_3",
    "agency_5",
    "agency_7",
    "agency_9",
    "loan_purpose",
    "occupancy",
    "year",
]
# excluding log_loan_amount and year but keep for diagnostic plots
_ESTIMATION_EXCLUDED = ("log_loan_amount", "year")
RDFLEX_ESTIMATION_COVARIATES = [
    c for c in RDFLEX_COVARIATES if c not in _ESTIMATION_EXCLUDED
]

# visual parameters
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

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
    "nnet": "Neural Net",
    "sl": "Super Learner",
}
COVARIATE_LABELS = {
    "applicant_income_log": "Log Applicant Income",
    "log_loan_amount": "Log Loan Amount",
    "agency_1": "Agency: OCC (1)",
    "agency_2": "Agency: FRS (2)",
    "agency_3": "Agency: FDIC (3)",
    "agency_5": "Agency: NCUA (5)",
    "agency_7": "Agency: HUD (7)",
    "agency_9": "Agency: CFPB (9)",
    "loan_purpose": "Loan Purpose",
    "occupancy": "Occupancy Type",
    "year": "Year",
}
DEPVAR_TABLE = {
    "approved": "Table V",
    "originated": "Table VI",
    "securitized": "Table VII",
}

# hyperparameter search spaces
RF_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
    "max_depth": {"type": "categorical", "choices": [3, 5, 7, 10, None]},
    "min_samples_split": {"type": "int", "low": 2, "high": 12},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 6},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8]},
}
LGBM_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "min_child_samples": {"type": "int", "low": 5, "high": 30},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
}
XGB_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
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
        "choices": [(50,), (100,), (50, 25), (100, 50)],
    },
    "alpha": {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
    "learning_rate_init": {"type": "float", "low": 1e-4, "high": 0.01, "log": True},
}

# hyperparameter labels
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
    "nnet_learning_rate_init": "Initial learning rate for SGD/Adam",
}


# Load the estimation sample CSV
def load_data():
    csv_path = data_dir / "est_sample.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            "Run 'Rscript prepare_data.R' first to convert the RDS."
        )
    print("Reading Input ...")
    df = pd.read_csv(csv_path, low_memory=False)
    # clean column names
    df.columns = df.columns.str.replace(".", "_", regex=False)
    # derived variables
    if "applicant_income" in df.columns:
        df["applicant_income_log"] = np.log(df["applicant_income"].clip(lower=1))
    else:
        df["applicant_income_log"] = np.nan
    # log loan amount covariate
    if "loan_amount" in df.columns:
        df["log_loan_amount"] = np.log(df["loan_amount"].clip(lower=1))
    elif "log_loan_amount" not in df.columns:
        df["log_loan_amount"] = np.nan
    # agency dummies
    if "agency" in df.columns:
        for a in [1, 2, 3, 5, 7, 9]:
            df[f"agency_{a}"] = (df["agency"] == a).astype(float)
    # ensure boolean outcomes
    for col in ["approved", "originated", "securitized"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["treated"] = df["treated"].astype(float)
    df["below_limit"] = df["below_limit"].astype(float)
    return df


# Build estimation sample
def build_estimation_sample(df, depvar, bw=0.20):
    # keep only columns needed
    required_cols = [
        "time",
        "diff_log_loan_amount",
        "action_type",
        "year",
        "name_event",
        "treated",
        "below_limit",
        "ZCTA5CE10",
        depvar,
    ]
    # also include covariates if available
    for c in RDFLEX_COVARIATES:
        if c in df.columns and c not in required_cols:
            required_cols.append(c)
    required_cols = [c for c in required_cols if c in df.columns]
    # filter time window
    est = df.loc[df["time"].between(-4, 4), required_cols]
    # recompute below_limit
    est["below_limit"] = (est["diff_log_loan_amount"] <= 0).astype(float)
    # filter by bandwidth
    est = est.loc[est["diff_log_loan_amount"].abs() <= bw]
    # filter by depvar
    if depvar == "securitized":
        est = est.loc[est["action_type"] == 1]
    elif depvar in ("approved", "originated"):
        est = est.loc[est["action_type"].isin([1, 2, 3])]
    # year filter from paper
    est = est.loc[est["year"] >= 2001]
    # fill control events
    est["name_event"] = est["name_event"].fillna("control")
    est.loc[est["name_event"] == "", "name_event"] = "control"
    return est.reset_index(drop=True)


# Save estimation sample CSVs
def export_rdrobust_samples(df):
    rdrobust_dir = results_dir / "rdrobust_samples"
    rdrobust_dir.mkdir(exist_ok=True)
    for depvar in DEPVARS:
        # use bw=0.20 to replicate paper
        est = build_estimation_sample(df, depvar, bw=0.20)
        for t in [1, 2, 3, 4]:
            sub = est[(est["treated"] == 1) & (est["time"] == t)].copy()
            covars = [c for c in RDFLEX_COVARIATES if c in sub.columns]
            save_cols = [depvar, "diff_log_loan_amount", "below_limit"] + covars
            save_cols = [c for c in save_cols if c in sub.columns]
            sub_clean = sub.dropna(subset=save_cols)
            csv_path = rdrobust_dir / f"sample_{depvar}_t{t}.csv"
            sub_clean[save_cols].to_csv(csv_path, index=False)


# Load results_rdrobust.csv
def import_rdrobust_results():
    rdrobust_csv = results_dir / "results_rdrobust.csv"
    if not rdrobust_csv.exists():
        return None
    rdr = pd.read_csv(rdrobust_csv)
    # align columns to match RDFlex result format
    rdflex_cols = [
        "depvar",
        "bandwidth",
        "time",
        "method",
        "coef",
        "se",
        "t_stat",
        "pvalue",
        "ci_lower",
        "ci_upper",
        "ci_width",
        "n_obs",
    ]
    for col in rdflex_cols:
        if col not in rdr.columns:
            rdr[col] = np.nan
    rdr = rdr[rdflex_cols]
    return rdr


# Gaussian kernel weights.
def gaussian_kernel(x, bw):
    return norm.pdf(x / bw)


# Replicate paper RDD results
def replicate_paper_rdd(df):
    import pyfixest as pf

    all_results = []
    for depvar in DEPVARS:
        est = build_estimation_sample(df, depvar, bw=0.20)
        # create interacted FE variables
        est["year_below"] = (
            est["year"].astype(str) + "_" + est["below_limit"].astype(int).astype(str)
        )
        est["zip_below"] = (
            est["ZCTA5CE10"].astype(str)
            + "_"
            + est["below_limit"].astype(int).astype(str)
        )
        est["event_below"] = (
            est["name_event"].astype(str)
            + "_"
            + est["below_limit"].astype(int).astype(str)
        )
        # time dummy columns
        time_range = range(-4, 5)
        for t in time_range:
            if t == -1:
                continue  # reference period
            tname = f"time_m{abs(t)}" if t < 0 else f"time_{t}"
            est[tname] = (est["time"] == t).astype(float)
        for bw_h in BANDWIDTHS:
            # kernel weights
            est["kweight"] = gaussian_kernel(est["diff_log_loan_amount"], bw_h)
            # depvar ~ treated:time_dummies + below_limit:treated:time_dummies
            time_terms_jumbo = []
            time_terms_conf = []
            for t in time_range:
                if t == -1:
                    continue
                tname = f"time_m{abs(t)}" if t < 0 else f"time_{t}"
                time_terms_jumbo.append(f"{tname}:treated")
                time_terms_conf.append(f"below_limit:{tname}:treated")
            rhs = " + ".join(time_terms_jumbo + time_terms_conf)
            formula = f"{depvar} ~ {rhs} | year_below + zip_below + event_below"
            try:
                reg = pf.feols(
                    formula,
                    data=est,
                    weights="kweight",
                    vcov={"CRV1": "ZCTA5CE10 + year"},
                )
                coefs = reg.coef()
                ses = reg.se()
                n_obs = int(getattr(reg, "_N", len(est)))
                # extract below_limit x treated x time coefficients
                for t in [1, 2, 3, 4]:
                    tname = f"time_{t}"
                    coef_key = f"below_limit:{tname}:treated"
                    # check both orderings
                    if coef_key not in coefs.index:
                        coef_key = f"treated:{tname}:below_limit"
                    if coef_key not in coefs.index:
                        # try searching for partial match
                        matches = [
                            k
                            for k in coefs.index
                            if "below_limit" in k and tname in k and "treated" in k
                        ]
                        coef_key = matches[0] if matches else None
                    if coef_key and coef_key in coefs.index:
                        c = coefs[coef_key]
                        s = ses[coef_key]
                        all_results.append(
                            {
                                "depvar": depvar,
                                "bandwidth": bw_h,
                                "time": t,
                                "coef": c,
                                "se": s,
                                "t_stat": c / s if s > 0 else np.nan,
                                "n_obs": n_obs,
                                "method": "paper_rdd",
                            }
                        )
            except Exception:
                pass
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "rdd_paper_results.csv", index=False)
    return results_df


# Draw Optuna params from a search-space
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


# Tune a model
def _optuna_tune(base, space, Z, y, n_trials=N_OPTUNA_TRIALS, cv=OPTUNA_CV):
    def objective(trial):
        params = _suggest_from_space(trial, space)
        model = clone(base)
        model.set_params(**params)
        scores = cross_val_score(
            model, Z, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    # memory cleanup
    del study
    return best


# Tune the acutale models, return best params
def _tune_rdflex_models(Z, y, tag=""):
    tuned = {}
    tuned["rf"] = _optuna_tune(
        RandomForestRegressor(random_state=42, n_jobs=1),
        RF_PARAM_SPACE,
        Z,
        y,
    )
    gc.collect()
    tuned["lgbm"] = _optuna_tune(
        LGBMRegressor(verbose=-1, n_jobs=1, random_state=42, **LGB_GPU),
        LGBM_PARAM_SPACE,
        Z,
        y,
    )
    gc.collect()
    tuned["xgb"] = _optuna_tune(
        XGBRegressor(verbosity=0, n_jobs=1, random_state=42, **XGB_GPU),
        XGB_PARAM_SPACE,
        Z,
        y,
    )
    gc.collect()
    tuned["nnet"] = _optuna_tune(
        MLPRegressor(max_iter=2000, early_stopping=True, random_state=42),
        NNET_PARAM_SPACE,
        Z,
        y,
    )
    gc.collect()
    return tuned


# adjustment configs from tuned hyperparameters
def _rdflex_configs(tuned):
    # n_jobs=1 against memory overload
    base_learners = [
        ("ridge", RidgeCV(cv=5)),
        ("lasso", LassoPolyRegressor(degree=2, cv=5, max_iter=5000)),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=1, **tuned["rf"])),
        ("lgbm", LGBMRegressor(verbose=-1, n_jobs=1, **tuned["lgbm"], **LGB_GPU)),
        (
            "xgb",
            XGBRegressor(
                verbosity=0, n_jobs=1, random_state=42, **tuned["xgb"], **XGB_GPU
            ),
        ),
        (
            "nnet",
            MLPRegressor(
                max_iter=2000, early_stopping=True, random_state=42, **tuned["nnet"]
            ),
        ),
    ]
    return [
        ("nocov", DummyRegressor(strategy="mean")),
        ("ridge", RidgeCV(cv=5)),
        ("lasso", LassoPolyRegressor(degree=2, cv=5, max_iter=5000)),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=1, **tuned["rf"])),
        ("lgbm", LGBMRegressor(verbose=-1, n_jobs=1, **tuned["lgbm"], **LGB_GPU)),
        (
            "xgb",
            XGBRegressor(
                verbosity=0, n_jobs=1, random_state=42, **tuned["xgb"], **XGB_GPU
            ),
        ),
        (
            "nnet",
            MLPRegressor(
                max_iter=2000, early_stopping=True, random_state=42, **tuned["nnet"]
            ),
        ),
        ("sl", SuperLearnerRegressor(base_learners=base_learners)),
    ]


# Record a single RDFlex result as a dict
def _record_rdflex(est, depvar, bandwidth, time_period, adjustment, n):
    coef = float(est.coef[0])
    se = float(est.se[0])
    pval = float(est.pval[0])
    ci = est.confint()
    return {
        "depvar": depvar,
        "bandwidth": bandwidth,
        "time": time_period,
        "method": adjustment,
        "coef": coef,
        "se": se,
        "t_stat": coef / se if se > 0 else np.nan,
        "pvalue": pval,
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
        "ci_width": float(ci.iloc[0, 1] - ci.iloc[0, 0]),
        "n_obs": int(n),
    }


# save best hyperparameter config
def save_tuned_hyperparameters(tuned, depvar, time_period):
    rows = []
    for method, params in tuned.items():
        for param_name, param_value in params.items():
            key = f"{method}_{param_name}"
            rows.append(
                {
                    "depvar": depvar,
                    "time": time_period,
                    "method": method,
                    "parameter": param_name,
                    "value": param_value,
                    "description": HYPERPARAM_DESCRIPTIONS.get(key, ""),
                }
            )
    return rows


# Export best hyperparameter config
def export_all_hyperparameters(all_hp_rows):
    if not all_hp_rows:
        return
    hp_df = pd.DataFrame(all_hp_rows)
    hp_csv = results_dir / "tuned_hyperparameters.csv"
    hp_df.to_csv(hp_csv, index=False)
    # also generate a summary table image for the thesis
    _plot_hyperparameter_table(hp_df)


# Estimation functions
class LassoPolyRegressor(BaseEstimator, RegressorMixin):
    # Lasso Regressor with interactions
    def __init__(self, degree=2, cv=5, max_iter=5000, random_state=42):
        self.degree = degree
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.poly_ = PolynomialFeatures(
            degree=self.degree, interaction_only=False, include_bias=False
        )
        X_poly = self.poly_.fit_transform(X)
        self.lasso_ = LassoCV(
            cv=self.cv,
            n_jobs=1,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.lasso_.fit(X_poly, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_poly = self.poly_.transform(X)
        return self.lasso_.predict(X_poly)


# Super Learner Regressor
class SuperLearnerRegressor(BaseEstimator, RegressorMixin):
    # stacking ensemble
    def __init__(self, base_learners=None):
        self.base_learners = base_learners or []

    def _fit_learner(self, learner, X, y, sample_weight=None):
        # fit a base learner, forwarding sample_weight
        try:
            learner.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            learner.fit(X, y)

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.fitted_ = []
        holdout_preds = np.zeros((len(y), len(self.base_learners)))
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for j, (name, learner) in enumerate(self.base_learners):
            fitted_learner = clone(learner)
            # cross-validated holdout predictions for stacking
            for train_idx, val_idx in kf.split(X):
                fold_model = clone(learner)
                sw_fold = (
                    sample_weight[train_idx] if sample_weight is not None else None
                )
                self._fit_learner(fold_model, X[train_idx], y[train_idx], sw_fold)
                holdout_preds[val_idx, j] = fold_model.predict(X[val_idx])
            # refit on full data
            self._fit_learner(fitted_learner, X, y, sample_weight)
            self.fitted_.append((name, fitted_learner))
        # meta-learner
        self.meta_ = RidgeCV(cv=5)
        self.meta_.fit(holdout_preds, y)
        return self

    def predict(self, X):
        base_preds = np.column_stack([m.predict(X) for _, m in self.fitted_])
        return self.meta_.predict(base_preds)

    def get_weights(self):
        # return dict of base learner name
        coefs = self.meta_.coef_
        return {name: float(coefs[j]) for j, (name, _) in enumerate(self.fitted_)}


# run estimation
def run_rdflex_estimation(df):
    # saves partial results after each block for fallback
    rows = []
    rdflex_bws = RDFLEX_BWS
    importance_data = []
    sl_weights_data = []
    all_hp_rows = []
    # load any previously saved partial results
    checkpoint_csv = results_dir / "rdd_rdflex_checkpoint.csv"
    completed_blocks = set()
    if checkpoint_csv.exists():
        prev = pd.read_csv(checkpoint_csv)
        rows = prev.to_dict("records")
        # identify blocks that are fully done
        for (dv, tp), grp in prev.groupby(["depvar", "time"]):
            completed_blocks.add((dv, int(tp)))
    # also load any previously saved hyperparameters
    hp_checkpoint = results_dir / "tuned_hyperparameters_checkpoint.csv"
    if hp_checkpoint.exists():
        hp_prev = pd.read_csv(hp_checkpoint)
        all_hp_rows = hp_prev.to_dict("records")
    for depvar in DEPVARS:
        est = build_estimation_sample(df, depvar, bw=0.20)
        for t in [1, 2, 3, 4]:
            # skip already-completed blocks
            if (depvar, t) in completed_blocks:
                continue
            sub = est[(est["treated"] == 1) & (est["time"] == t)].copy()
            if len(sub) < 100:
                continue
            covars = [c for c in RDFLEX_ESTIMATION_COVARIATES if c in sub.columns]
            sub = sub.dropna(
                subset=covars + [depvar, "below_limit", "diff_log_loan_amount"]
            )
            if len(sub) < 100 or len(covars) == 0:
                continue
            # tune once per block
            Z = sub[covars].values.astype(float)
            y = sub[depvar].values.astype(float)
            tuned = _tune_rdflex_models(Z, y, tag=f"{depvar} t+{t}")
            # save tuned hyperparameters
            all_hp_rows.extend(save_tuned_hyperparameters(tuned, depvar, t))
            bw_list = rdflex_bws + ["auto"]
            for bw_h in bw_list:
                for name, ml_g in _rdflex_configs(tuned):
                    try:
                        rdd_data = DoubleMLRDDData(
                            data=sub,
                            y_col=depvar,
                            d_cols="below_limit",
                            score_col="diff_log_loan_amount",
                            x_cols=covars,
                        )
                        # RDFlex parameters
                        rdflex_kwargs = dict(
                            obj_dml_data=rdd_data,
                            ml_g=ml_g,
                            ml_m=None,
                            fuzzy=False,
                            cutoff=CUTOFF,
                            n_folds=RDFLEX_N_FOLDS,
                            n_rep=RDFLEX_N_REP,
                            fs_kernel="triangular",
                        )
                        if bw_h == "auto":
                            rdflex_kwargs["fs_specification"] = (
                                "cutoff"  # MSE-optimal bandwidth
                            )
                        else:
                            rdflex_kwargs["h_fs"] = float(bw_h)
                            rdflex_kwargs["fs_specification"] = (
                                "cutoff"  # MSE-optimal bandwidth
                            )
                        rdest = RDFlex(**rdflex_kwargs)
                        rdest.fit()
                        result = _record_rdflex(
                            rdest,
                            depvar,
                            bw_h if bw_h != "auto" else -1.0,
                            t,
                            name,
                            len(sub),
                        )
                        if bw_h == "auto":
                            result["bandwidth"] = "auto"
                            # try to extract the auto bandwidth
                            try:
                                result["auto_bw"] = float(rdest.bandwidth)
                            except Exception:
                                result["auto_bw"] = np.nan
                        rows.append(result)
                        # collect covariate importance for tree-based models
                        if name in ("rf", "lgbm", "xgb") and hasattr(
                            ml_g, "feature_importances_"
                        ):
                            for ci, cname in enumerate(covars):
                                importance_data.append(
                                    {
                                        "depvar": depvar,
                                        "time": t,
                                        "method": name,
                                        "bandwidth": bw_h,
                                        "covariate": cname,
                                        "importance": ml_g.feature_importances_[ci],
                                    }
                                )
                        # collect LASSO/Ridge coefficient magnitudes as importance
                        if name == "lasso" and hasattr(ml_g, "lasso_"):
                            coefs = np.abs(ml_g.lasso_.coef_)
                            # only keep the first len(covars) raw feature coefficients
                            raw_coefs = (
                                coefs[: len(covars)]
                                if len(coefs) >= len(covars)
                                else coefs
                            )
                            total = raw_coefs.sum() if raw_coefs.sum() > 0 else 1.0
                            for ci, cname in enumerate(covars):
                                importance_data.append(
                                    {
                                        "depvar": depvar,
                                        "time": t,
                                        "method": name,
                                        "bandwidth": bw_h,
                                        "covariate": cname,
                                        "importance": (
                                            raw_coefs[ci] / total
                                            if ci < len(raw_coefs)
                                            else 0.0
                                        ),
                                    }
                                )
                        if name == "ridge" and hasattr(ml_g, "coef_"):
                            coefs = np.abs(ml_g.coef_)
                            raw_coefs = (
                                coefs[: len(covars)]
                                if len(coefs) >= len(covars)
                                else coefs
                            )
                            total = raw_coefs.sum() if raw_coefs.sum() > 0 else 1.0
                            for ci, cname in enumerate(covars):
                                importance_data.append(
                                    {
                                        "depvar": depvar,
                                        "time": t,
                                        "method": name,
                                        "bandwidth": bw_h,
                                        "covariate": cname,
                                        "importance": (
                                            raw_coefs[ci] / total
                                            if ci < len(raw_coefs)
                                            else 0.0
                                        ),
                                    }
                                )
                        if name == "nnet" and hasattr(ml_g, "coefs_"):
                            # sum abs of first layer weights per feature
                            w = np.abs(ml_g.coefs_[0])
                            feat_imp = w.sum(axis=1)
                            raw_imp = feat_imp[: len(covars)]
                            total = raw_imp.sum() if raw_imp.sum() > 0 else 1.0
                            for ci, cname in enumerate(covars):
                                importance_data.append(
                                    {
                                        "depvar": depvar,
                                        "time": t,
                                        "method": name,
                                        "bandwidth": bw_h,
                                        "covariate": cname,
                                        "importance": (
                                            raw_imp[ci] / total
                                            if ci < len(raw_imp)
                                            else 0.0
                                        ),
                                    }
                                )
                        # collect super learner weights
                        if name == "sl" and hasattr(ml_g, "get_weights"):
                            try:
                                wts = ml_g.get_weights()
                                for wname, wval in wts.items():
                                    sl_weights_data.append(
                                        {
                                            "depvar": depvar,
                                            "time": t,
                                            "bandwidth": bw_h,
                                            "base_learner": wname,
                                            "weight": wval,
                                        }
                                    )
                            except Exception:
                                pass
                        # memory cleanup
                        del rdest
                        del rdd_data
                    except Exception:
                        rows.append(
                            {
                                "depvar": depvar,
                                "bandwidth": bw_h if bw_h != "auto" else "auto",
                                "time": t,
                                "method": name,
                                "coef": np.nan,
                                "se": np.nan,
                                "t_stat": np.nan,
                                "pvalue": np.nan,
                                "ci_lower": np.nan,
                                "ci_upper": np.nan,
                                "ci_width": np.nan,
                                "n_obs": len(sub),
                            }
                        )
                # memory cleanup after each bandwidth
                gc.collect()
            # checkpoint: save after each (depvar, time) block
            pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)
            pd.DataFrame(all_hp_rows).to_csv(hp_checkpoint, index=False)
            # memory cleanup for tuned models and sub-DataFrame
            del tuned, sub, Z, y
            gc.collect()
        # memory cleanup for estimation sample after all time periods for this depvar
        del est
        gc.collect()
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / "rdd_rdflex_results.csv", index=False)
    # remove checkpoint files
    if checkpoint_csv.exists():
        checkpoint_csv.unlink()
    if hp_checkpoint.exists():
        hp_checkpoint.unlink()
    if importance_data:
        pd.DataFrame(importance_data).to_csv(
            results_dir / "covariate_importance.csv", index=False
        )
    if sl_weights_data:
        pd.DataFrame(sl_weights_data).to_csv(
            results_dir / "super_learner_weights.csv", index=False
        )
    # export tuned hyperparameters
    export_all_hyperparameters(all_hp_rows)
    return results_df


# compute power analysis table
def compute_power_analysis(df, rdflex_df):
    rdflex_norm = _normalize_method_col(rdflex_df)
    col_labels = [
        "Depvar",
        "Time",
        "N (bw=5%)",
        "Avg SE (nocov)",
        "MDE (80%)",
        "Paper Coef",
    ]
    cells = []
    for depvar in DEPVARS:
        # get sample size at bw=0.05
        est = build_estimation_sample(df, depvar, bw=0.05)
        paper_bench = PAPER_BENCHMARKS.get((depvar, 0.05), {})
        for t in [1, 2, 3, 4]:
            sub = est[(est["treated"] == 1) & (est["time"] == t)]
            n_obs = len(sub)
            # get nocov SE from RDFlex results
            nocov_sub = rdflex_norm[
                (rdflex_norm["depvar"] == depvar)
                & (rdflex_norm["time"] == t)
                & (rdflex_norm["method"] == "nocov")
                & (rdflex_norm["bandwidth"] == 0.05)
            ]
            if len(nocov_sub) > 0:
                avg_se = nocov_sub["se"].values[0]
            else:
                avg_se = np.nan
            # MDE at 80% power, alpha=0.05
            mde = 2.8 * avg_se if not np.isnan(avg_se) else np.nan
            # paper coefficient
            bench = paper_bench.get(t)
            paper_coef = bench[0] if bench else np.nan
            cells.append(
                [
                    depvar.title(),
                    f"t+{t}",
                    f"{n_obs:,}",
                    f"{avg_se:.4f}" if not np.isnan(avg_se) else "N/A",
                    f"{mde:.4f}" if not np.isnan(mde) else "N/A",
                    f"{paper_coef:.4f}" if not np.isnan(paper_coef) else "N/A",
                ]
            )
    save_path = figures_dir / "power_analysis.png"
    _render_paper_table(
        title="",
        subtitle="",
        save_path=save_path,
        col_labels=col_labels,
        cells=cells,
        highlight_rules=[],
    )


#  predetermined covariate smoothness and placebo cutoffs
def robustness_predetermined_and_placebo(df):
    # covariate smoothness at the cutoff
    est = build_estimation_sample(df, "approved", bw=0.05)
    sub = est[(est["treated"] == 1) & (est["time"] == 1)].copy()
    covars = [c for c in RDFLEX_COVARIATES if c in sub.columns]
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
        # skip constant covariates
        if sub[cov].nunique() <= 1:
            continue
        cov_data = sub.dropna(subset=[cov, "below_limit"])
        below = cov_data.loc[cov_data["below_limit"] == 1, cov].astype(float)
        above = cov_data.loc[cov_data["below_limit"] == 0, cov].astype(float)
        if len(below) < 10 or len(above) < 10:
            continue
        mean_below = below.mean()
        mean_above = above.mean()
        diff = mean_below - mean_above
        se_diff = np.sqrt(below.var() / len(below) + above.var() / len(above))
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
                f"{mean_below:.4f}",
                f"{mean_above:.4f}",
                f"{diff:.4f}",
                f"{t_stat:.3f}" if not np.isnan(t_stat) else "N/A",
                f"{p_val:.4f}" if not np.isnan(p_val) else "N/A",
            ]
        )
    save_path_a = figures_dir / "robustness_covariate_smoothness.png"
    # highlight p-values < 0.05
    _render_paper_table(
        title="",
        subtitle="",
        save_path=save_path_a,
        col_labels=col_labels_a,
        cells=cells_a,
        highlight_rules=[
            {
                "col_idx": 5,
                "mode": "numeric_lt",
                "threshold": 0.05,
                "color": "#f8d7da",
            }
        ],
    )
    # placebo cutoff test
    depvar = "approved"
    bw = 0.05
    est_full = build_estimation_sample(df, depvar, bw=0.20)
    sub_p = est_full[(est_full["treated"] == 1) & (est_full["time"] == 1)].copy()
    sub_p = sub_p.dropna(subset=[depvar, "diff_log_loan_amount"])
    placebo_cutoffs = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    results = []
    for pc in placebo_cutoffs:
        in_bw = sub_p[
            (sub_p["diff_log_loan_amount"] >= pc - bw)
            & (sub_p["diff_log_loan_amount"] <= pc + bw)
        ].copy()
        if len(in_bw) < 50:
            results.append((pc, np.nan, np.nan))
            continue
        in_bw["fake_treat"] = (in_bw["diff_log_loan_amount"] <= pc).astype(float)
        dist = (in_bw["diff_log_loan_amount"] - pc).abs()
        in_bw["weight"] = np.maximum(1 - dist / bw, 0)
        treated = in_bw[in_bw["fake_treat"] == 1]
        control = in_bw[in_bw["fake_treat"] == 0]
        if len(treated) < 10 or len(control) < 10:
            results.append((pc, np.nan, np.nan))
            continue
        y1 = np.average(treated[depvar].values, weights=treated["weight"].values)
        y0 = np.average(control[depvar].values, weights=control["weight"].values)
        coef = y1 - y0
        n1 = len(treated)
        n0 = len(control)
        var1 = np.average(
            (treated[depvar].values - y1) ** 2, weights=treated["weight"].values
        )
        var0 = np.average(
            (control[depvar].values - y0) ** 2, weights=control["weight"].values
        )
        se = np.sqrt(var1 / n1 + var0 / n0)
        results.append((pc, coef, se))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (c, coef, se) in enumerate(results):
        if np.isnan(coef):
            continue
        color = "#d62728" if c == 0.0 else "#1f77b4"
        marker = "D" if c == 0.0 else "o"
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
    ax.set_xlabel("Cutoff Location (%)")
    ax.set_ylabel("Treatment Effect Estimate")
    plt.tight_layout()
    save_path_b = figures_dir / "robustness_placebo_cutoffs.png"
    plt.savefig(save_path_b, dpi=150, bbox_inches="tight")
    plt.close()


# density of the running variable
def robustness_density_test(df):
    est = build_estimation_sample(df, "approved", bw=0.20)
    sub = est[(est["treated"] == 1) & (est["time"] == 1)].copy()
    rv = sub["diff_log_loan_amount"].dropna().values
    n_bins = 50
    below = rv[rv <= 0]
    above = rv[rv > 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    bins_below = np.linspace(rv.min(), 0, n_bins + 1)
    bins_above = np.linspace(0, rv.max(), n_bins + 1)
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
    # log-density discontinuity estimate
    narrow = 0.01
    count_below = np.sum((rv >= -narrow) & (rv <= 0))
    count_above = np.sum((rv > 0) & (rv <= narrow))
    dens_below = count_below / (narrow * len(rv))
    dens_above = count_above / (narrow * len(rv))
    if dens_above > 0:
        density_ratio = dens_below / dens_above
    else:
        density_ratio = np.nan
    if dens_below > 0 and dens_above > 0:
        log_diff = np.log(dens_below) - np.log(dens_above)
        # SE of log density difference
        se_log = np.sqrt(1.0 / count_below + 1.0 / count_above)
        t_stat = log_diff / se_log
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        log_diff = np.nan
        t_stat = np.nan
        p_val = np.nan
    ax.set_xlabel("Log Distance to Conforming Limit")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_path = figures_dir / "robustness_density_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# covariate continuity table from rdrobust output
def robustness_formal_covariate_continuity():
    cov_cont_csv = results_dir / "results_covariate_continuity.csv"
    if not cov_cont_csv.exists():
        return
    cov_cont = pd.read_csv(cov_cont_csv)
    # render as table
    col_labels = [
        "Covariate",
        "MSE-Opt BW",
        "RD Est.",
        "Robust p",
        "95% CI",
        "Eff. N",
    ]
    cells = []
    for _, row in cov_cont.iterrows():
        label = COVARIATE_LABELS.get(row["covariate"], row["covariate"])
        cells.append(
            [
                label,
                f"{row['mse_bandwidth']:.4f}",
                f"{row['rd_estimator']:.4f}",
                f"{row['robust_pvalue']:.3f}",
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                f"{int(row['eff_n'])}",
            ]
        )
    # highlight p-values < 0.05
    _render_paper_table(
        title="",
        subtitle="",
        save_path=figures_dir / "robustness_covariate_continuity_table.png",
        col_labels=col_labels,
        cells=cells,
        highlight_rules=[
            {
                "col_idx": 3,
                "mode": "numeric_lt",
                "threshold": 0.05,
                "color": "#f8d7da",
            }
        ],
    )


# seed stability test
def run_seed_robustness(df, rdflex_results):
    if rdflex_results is None or rdflex_results.empty:
        return
    # preferred specification
    depvar = "approved"
    bw = 0.05
    t_target = 1
    est = build_estimation_sample(df, depvar, bw=0.20)
    sub = est[(est["treated"] == 1) & (est["time"] == t_target)].copy()
    del est  # free immediately — only sub is needed
    gc.collect()
    covars = [c for c in RDFLEX_ESTIMATION_COVARIATES if c in sub.columns]
    sub = sub.dropna(subset=covars + [depvar, "below_limit", "diff_log_loan_amount"])
    if len(sub) < 100 or len(covars) == 0:
        return
    # tune models once
    Z = sub[covars].values.astype(float)
    y = sub[depvar].values.astype(float)
    tuned = _tune_rdflex_models(Z, y, tag="seed-robustness")
    # all methods excluding rdrobust and sl
    configs_all = dict(_rdflex_configs(tuned))
    test_methods = [
        m
        for m in METHOD_COLORS.keys()
        if m in configs_all and m not in ("rdrobust", "sl")
    ]
    seeds = [42, 123, 456, 789, 2024]
    seed_results = []
    for seed in seeds:
        np.random.seed(seed)
        for method_name in test_methods:
            ml_g = clone(configs_all[method_name])
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col=depvar,
                    d_cols="below_limit",
                    score_col="diff_log_loan_amount",
                    x_cols=covars,
                )
                rdest = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=CUTOFF,
                    n_folds=RDFLEX_N_FOLDS,
                    n_rep=RDFLEX_N_REP,
                    fs_kernel="triangular",
                    h_fs=float(bw),
                    fs_specification="cutoff",
                )
                rdest.fit()
                rec = _record_rdflex(rdest, depvar, bw, t_target, method_name, len(sub))
                rec["seed"] = seed
                seed_results.append(rec)
                del rdest
                del rdd_data
                gc.collect()
            except Exception:
                pass
    # reset random state
    np.random.seed(None)
    if not seed_results:
        return
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv(results_dir / "seed_robustness.csv", index=False)
    # plot seed sensitivity results
    _plot_seed_robustness(seed_df)


# kernel robustness test
def robustness_kernel_comparison(df, rdflex_results):
    if rdflex_results is None or rdflex_results.empty:
        return
    # preferred specification: approved, t+1, bw=0.05
    depvar = "approved"
    bw = 0.05
    t_target = 1
    est = build_estimation_sample(df, depvar, bw=0.20)
    sub = est[(est["treated"] == 1) & (est["time"] == t_target)].copy()
    # memory cleanup
    del est
    gc.collect()
    covars = [c for c in RDFLEX_ESTIMATION_COVARIATES if c in sub.columns]
    sub = sub.dropna(subset=covars + [depvar, "below_limit", "diff_log_loan_amount"])
    if len(sub) < 100 or len(covars) == 0:
        return
    # tune once
    Z = sub[covars].values.astype(float)
    y = sub[depvar].values.astype(float)
    tuned = _tune_rdflex_models(Z, y, tag="kernel-robustness")
    kernels = ["triangular", "epanechnikov"]
    test_methods = [m for m in METHOD_COLORS.keys() if m not in ("rdrobust", "sl")]
    kernel_results = []
    for kernel in kernels:
        for name, ml_g in _rdflex_configs(tuned):
            if name in ("rdrobust", "sl"):
                continue
            try:
                rdd_data = DoubleMLRDDData(
                    data=sub,
                    y_col=depvar,
                    d_cols="below_limit",
                    score_col="diff_log_loan_amount",
                    x_cols=covars,
                )
                rdest = RDFlex(
                    obj_dml_data=rdd_data,
                    ml_g=ml_g,
                    ml_m=None,
                    fuzzy=False,
                    cutoff=CUTOFF,
                    n_folds=RDFLEX_N_FOLDS,
                    n_rep=RDFLEX_N_REP,
                    fs_kernel=kernel,
                    h_fs=float(bw),
                    fs_specification="cutoff",
                )
                rdest.fit()
                rec = _record_rdflex(rdest, depvar, bw, t_target, name, len(sub))
                rec["kernel"] = kernel
                kernel_results.append(rec)
                del rdest
                del rdd_data
                gc.collect()
            except Exception:
                pass
    if not kernel_results:
        return
    kernel_df = pd.DataFrame(kernel_results)
    kernel_df.to_csv(results_dir / "kernel_robustness.csv", index=False)
    # plot kernel robustness results
    _plot_kernel_robustness(kernel_df)


# plot functions
def _normalize_method_col(df):
    out = df.copy()
    if "method" in out.columns:
        out["method"] = out["method"].str.replace("rdflex_", "", regex=False)
    # convert bandwidth to numeric
    if "bandwidth" in out.columns:
        out["bandwidth"] = out["bandwidth"].apply(
            lambda x: float(x) if str(x) != "auto" else x
        )
    return out


# Paper replication plot
def plot_paper_replication(results_df):
    for depvar in DEPVARS:
        sub = results_df[
            (results_df["depvar"] == depvar) & (results_df["method"] == "paper_rdd")
        ]
        if len(sub) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, bw_h in enumerate([0.02, 0.03, 0.05, 0.10]):
            ax = axes[idx // 2][idx % 2]
            bw_sub = sub[sub["bandwidth"] == bw_h]
            if len(bw_sub) == 0:
                ax.set_title(f"bw={bw_h:.0%} — no data")
                continue
            times = bw_sub["time"].values
            coefs = bw_sub["coef"].values
            ses = bw_sub["se"].values
            ax.errorbar(
                times,
                coefs,
                yerr=1.96 * ses,
                fmt="o-",
                color="#1f77b4",
                label="Our estimate",
                capsize=4,
            )
            # paper benchmarks
            bench = PAPER_BENCHMARKS.get((depvar, bw_h), {})
            if bench:
                bt = list(bench.keys())
                bc = [bench[t][0] for t in bt]
                bs = [bench[t][1] for t in bt]
                ax.errorbar(
                    bt,
                    bc,
                    yerr=[1.96 * s for s in bs],
                    fmt="s--",
                    color="#d62728",
                    label="Paper",
                    capsize=4,
                    alpha=0.7,
                )
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"Bandwidth = {bw_h:.0%}")
            ax.legend(fontsize=8)
        fig.supxlabel("Years Since Hurricane")
        fig.supylabel("Treatment Effect")
        plt.tight_layout()
        plt.savefig(figures_dir / f"rdd_replication_{depvar}.png", dpi=150)
        plt.close()


# plot coefficient comparison across methods
def plot_coefficient_comparison(rdflex_df):
    rdflex_n = _normalize_method_col(rdflex_df)
    for depvar in DEPVARS:
        table_name = DEPVAR_TABLE.get(depvar, depvar)
        for bw_show in [0.05]:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for idx, t in enumerate([1, 2, 3, 4]):
                ax = axes[idx // 2][idx % 2]
                method_labels_list = []
                coefs = []
                ci_lo = []
                ci_hi = []
                colors = []
                bench = PAPER_BENCHMARKS.get((depvar, bw_show), {}).get(t)
                if bench:
                    p_coef, p_se = bench
                    ax.axvline(
                        p_coef,
                        color="red",
                        ls="--",
                        lw=2,
                        label=f"Paper {table_name}",
                        zorder=1,
                    )
                    ax.axvspan(
                        p_coef - 1.96 * p_se,
                        p_coef + 1.96 * p_se,
                        color="red",
                        alpha=0.1,
                        zorder=0,
                    )
                # ML methods
                if rdflex_n is not None and len(rdflex_n) > 0:
                    for mk in METHOD_COLORS:
                        rdf_sub = rdflex_n[
                            (rdflex_n["depvar"] == depvar)
                            & (rdflex_n["bandwidth"] == bw_show)
                            & (rdflex_n["time"] == t)
                            & (rdflex_n["method"] == mk)
                        ]
                        if len(rdf_sub) > 0 and not np.isnan(rdf_sub["coef"].values[0]):
                            c = rdf_sub["coef"].values[0]
                            s = (
                                rdf_sub["se"].values[0]
                                if not np.isnan(rdf_sub["se"].values[0])
                                else 0
                            )
                            method_labels_list.append(METHOD_LABELS[mk])
                            coefs.append(c)
                            ci_lo.append(c - 1.96 * s)
                            ci_hi.append(c + 1.96 * s)
                            colors.append(METHOD_COLORS[mk])
                if len(coefs) == 0:
                    ax.set_title(f"t+{t} - no data")
                    continue
                y_pos = np.arange(len(method_labels_list))
                xerr_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
                xerr_hi = [hi - c for c, hi in zip(coefs, ci_hi)]
                ax.errorbar(
                    coefs,
                    y_pos,
                    xerr=[xerr_lo, xerr_hi],
                    fmt="o",
                    color="steelblue",
                    ecolor="steelblue",
                    capsize=4,
                    markersize=6,
                    zorder=5,
                )
                ax.set_yticks(y_pos)
                ax.set_yticklabels(method_labels_list, fontsize=9)
                ax.axvline(0, color="gray", ls="--", alpha=0.5)
                ax.set_xlabel("Treatment Effect")
                ax.set_title(f"t + {t}")
                if idx == 0:
                    ax.legend(fontsize=8, loc="best")
            plt.tight_layout()
            save_path = figures_dir / f"coef_comparison_{depvar}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()


# CI width comparison across methods
def plot_ci_width_comparison(rdflex_df):
    rdflex_n = _normalize_method_col(rdflex_df)
    for depvar in DEPVARS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, t in enumerate([1, 2, 3, 4]):
            ax = axes[idx // 2][idx % 2]
            sub = rdflex_n[(rdflex_n["depvar"] == depvar) & (rdflex_n["time"] == t)]
            if len(sub) == 0:
                ax.set_title(f"t+{t} - no data")
                continue
            methods = []
            ci_widths = []
            colors = []
            for mk in METHOD_COLORS:
                msub = sub[sub["method"] == mk]
                if len(msub) > 0:
                    methods.append(METHOD_LABELS[mk])
                    ci_widths.append(msub["ci_width"].mean())
                    colors.append(METHOD_COLORS[mk])
            x = np.arange(len(methods))
            ax.bar(x, ci_widths, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"t + {t}")
        fig.supylabel("95% Confidence Interval Width")
        plt.tight_layout()
        save_path = figures_dir / f"ci_width_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# Forest plot of estimates and CIs across methods
def plot_forest_estimates(rdflex_df):
    if rdflex_df is None or rdflex_df.empty:
        return
    rdf = _normalize_method_col(rdflex_df)
    bw, t = 0.05, 1
    for depvar in DEPVARS:
        sub = rdf[
            (rdf["depvar"] == depvar) & (rdf["bandwidth"] == bw) & (rdf["time"] == t)
        ].copy()
        sub = sub.dropna(subset=["coef", "se"])
        if sub.empty:
            continue
        # order by METHOD_COLORS key order
        method_order = list(METHOD_COLORS.keys())
        rows = []
        for m in method_order:
            r = sub[sub["method"] == m]
            if len(r) == 0:
                continue
            rows.append(r.iloc[0])
        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        # nocov baseline
        nocov = plot_df[plot_df["method"] == "nocov"]
        if nocov.empty:
            continue
        nocov_coef = nocov["coef"].values[0]
        nocov_lo = nocov["ci_lower"].values[0]
        nocov_hi = nocov["ci_upper"].values[0]
        labels = [METHOD_LABELS.get(m, m) for m in plot_df["method"]]
        coefs = plot_df["coef"].values.astype(float)
        ci_lo = plot_df["ci_lower"].values.astype(float)
        ci_hi = plot_df["ci_upper"].values.astype(float)
        colors = [METHOD_COLORS.get(m, "gray") for m in plot_df["method"]]
        y_pos = np.arange(len(labels))
        xerr_lo = coefs - ci_lo
        xerr_hi = ci_hi - coefs
        fig, ax = plt.subplots(figsize=(10, 6))
        # nocov CI shaded band
        ax.axvspan(nocov_lo, nocov_hi, color="gray", alpha=0.15, label="Unadj. 95% CI")
        # nocov point estimate dashed line
        ax.axvline(nocov_coef, color="black", ls="--", lw=1.2, label="Unadj. estimate")
        # zero line
        ax.axvline(0, color="black", ls="-", lw=0.8)
        # method whiskers
        for i in range(len(y_pos)):
            ax.errorbar(
                coefs[i],
                y_pos[i],
                xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                capsize=4,
                markersize=7,
                zorder=5,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Treatment Effect (probability points)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        save_path = figures_dir / f"forest_estimates_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# standardized difference plot: (coef - nocov_coef) / nocov_se
def plot_standardized_diff(rdflex_df):
    if rdflex_df is None or rdflex_df.empty:
        return
    rdf = _normalize_method_col(rdflex_df)
    bw, t = 0.05, 1
    for depvar in DEPVARS:
        sub = rdf[
            (rdf["depvar"] == depvar) & (rdf["bandwidth"] == bw) & (rdf["time"] == t)
        ].copy()
        sub = sub.dropna(subset=["coef", "se"])
        if sub.empty:
            continue
        # nocov baseline
        nocov = sub[sub["method"] == "nocov"]
        if nocov.empty:
            continue
        nocov_coef = nocov["coef"].values[0]
        nocov_se = nocov["se"].values[0]
        if nocov_se == 0 or np.isnan(nocov_se):
            continue
        # order by METHOD_COLORS
        method_order = [m for m in METHOD_COLORS.keys() if m != "nocov"]
        labels = []
        diffs = []
        colors = []
        for m in method_order:
            r = sub[sub["method"] == m]
            if len(r) == 0:
                continue
            d = (r["coef"].values[0] - nocov_coef) / nocov_se
            labels.append(METHOD_LABELS.get(m, m))
            diffs.append(d)
            colors.append(METHOD_COLORS.get(m, "gray"))
        if not labels:
            continue
        diffs = np.array(diffs)
        fig, ax = plt.subplots(figsize=(10, 6))
        # SE shaded band (green like Griffin)
        ax.axhspan(-2, 2, color="green", alpha=0.08, label="$\\pm 2$ SE band")
        bar_colors = []
        for d, c in zip(diffs, colors):
            bar_colors.append(c if abs(d) <= 2 else "#d62728")
        x = np.arange(len(labels))
        ax.bar(x, diffs, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("(τ̂ₘ − τ̂_nocov) / SE_nocov")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        save_path = figures_dir / f"standardized_diff_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# auto bandwidth vs hardcoded bandwidth comparison
def plot_bw_auto_vs_hardcoded(rdflex_df):
    rdflex_n = _normalize_method_col(rdflex_df)
    for depvar in DEPVARS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        # separate auto and hardcoded
        auto = rdflex_n[
            (rdflex_n["depvar"] == depvar) & (rdflex_n["bandwidth"] == "auto")
        ]
        hardcoded = rdflex_n[
            (rdflex_n["depvar"] == depvar)
            & (rdflex_n["bandwidth"] != "auto")
            & (rdflex_n["bandwidth"] == 0.05)
        ]
        methods = []
        coef_auto = []
        coef_hard = []
        ci_auto = []
        ci_hard = []
        for mk in METHOD_COLORS:
            a = auto[auto["method"] == mk]
            h = hardcoded[hardcoded["method"] == mk]
            if len(a) > 0 or len(h) > 0:
                methods.append(METHOD_LABELS[mk])
                # average across time periods
                coef_auto.append(a["coef"].mean() if len(a) > 0 else np.nan)
                coef_hard.append(h["coef"].mean() if len(h) > 0 else np.nan)
                ci_auto.append(a["ci_width"].mean() if len(a) > 0 else np.nan)
                ci_hard.append(h["ci_width"].mean() if len(h) > 0 else np.nan)
        x = np.arange(len(methods))
        w = 0.35
        # coefficients
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
            label="Hardcoded (5%)",
            color="#2ca02c",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Treatment Effect")
        ax1.set_title("Coefficient Estimates")
        ax1.legend()
        ax1.axhline(0, color="gray", ls="--", alpha=0.5)
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
            ci_hard,
            w,
            label="Hardcoded (5%)",
            color="#2ca02c",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("CI Width")
        ax2.set_title("Confidence Interval Width")
        ax2.legend()
        plt.tight_layout()
        save_path = figures_dir / f"bw_auto_vs_hardcoded_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# feature importance plots per method
def plot_covariate_importance_per_method():
    imp_path = results_dir / "covariate_importance.csv"
    if not imp_path.exists():
        return
    imp = pd.read_csv(imp_path)
    for depvar in DEPVARS:
        dsub = imp[imp["depvar"] == depvar]
        if len(dsub) == 0:
            continue
        # order methods by METHOD_COLORS key order
        method_order = list(METHOD_COLORS.keys())
        available_methods = [m for m in method_order if m in dsub["method"].unique()]
        if not available_methods:
            available_methods = list(dsub["method"].unique())
        n_methods = len(available_methods)
        n_cols = min(n_methods, 4)
        n_rows = (n_methods + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_methods == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        for idx, method in enumerate(available_methods):
            ri, ci = divmod(idx, n_cols)
            ax = axes[ri, ci]
            msub = dsub[dsub["method"] == method]
            avg = msub.groupby("covariate")["importance"].mean().sort_values()
            labels = [COVARIATE_LABELS.get(cv, cv) for cv in avg.index]
            ax.barh(
                labels, avg.values, color=METHOD_COLORS.get(method, "gray"), alpha=0.8
            )
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.set_xlabel("Importance")
        # hide unused subplots
        for idx in range(n_methods, n_rows * n_cols):
            ri, ci = divmod(idx, n_cols)
            axes[ri, ci].set_visible(False)
        plt.tight_layout()
        save_path = figures_dir / f"covariate_importance_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# distribution of covariates
def plot_covariate_balance(df):
    est = build_estimation_sample(df, "approved", bw=0.05)
    covars = [c for c in RDFLEX_COVARIATES if c in est.columns]
    if len(covars) == 0:
        return
    below = est[est["below_limit"] == 1]
    above = est[est["below_limit"] == 0]
    n_covars = len(covars)
    n_cols = min(3, n_covars)
    n_rows = (n_covars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_covars == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    for i, cov in enumerate(covars):
        ax = axes[i // n_cols][i % n_cols]
        label = COVARIATE_LABELS.get(cov, cov)
        b_vals = below[cov].dropna()
        a_vals = above[cov].dropna()
        if len(b_vals) > 0 and len(a_vals) > 0:
            bins = np.histogram_bin_edges(pd.concat([b_vals, a_vals]), bins=30)
            ax.hist(
                b_vals,
                bins=bins,
                alpha=0.6,
                color="#1f77b4",
                label="Below",
                density=True,
            )
            ax.hist(
                a_vals,
                bins=bins,
                alpha=0.6,
                color="#d62728",
                label="Above",
                density=True,
            )
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=10)
    # hide empty subplots
    for i in range(n_covars, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.supylabel("Density")
    plt.tight_layout()
    save_path = figures_dir / f"covariate_balance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# sample size comparison
def plot_sample_n_comparison(df):
    depvars = DEPVARS
    our_n = []
    paper_n_list = []
    for depvar in depvars:
        est = build_estimation_sample(df, depvar, bw=0.20)
        our_n.append(len(est))
        paper_n_list.append(PAPER_N.get(depvar, 0))
    x = np.arange(len(depvars))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_w / 2, our_n, bar_w, label="Replication", color="#1f77b4")
    ax.bar(x + bar_w / 2, paper_n_list, bar_w, label="Original", color="#ff7f0e")
    for i, (ours, paper) in enumerate(zip(our_n, paper_n_list)):
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
    ax.set_xticklabels([d.title() for d in depvars])
    ax.set_xlabel("Dependent Variable")
    ax.set_ylabel("Number of Observations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "plot_sample_n_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# super learner weights plot
def plot_super_learner_weights():
    wt_path = results_dir / "super_learner_weights.csv"
    if not wt_path.exists():
        return
    wt = pd.read_csv(wt_path)
    for depvar in DEPVARS:
        dsub = wt[wt["depvar"] == depvar]
        if len(dsub) == 0:
            continue
        # average weights across time periods and bandwidths
        avg = dsub.groupby("base_learner")["weight"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [METHOD_COLORS.get(bl, "gray") for bl in avg.index]
        labels = [METHOD_LABELS.get(bl, bl) for bl in avg.index]
        ax.barh(labels, avg.values, color=colors, alpha=0.8)
        ax.set_xlabel("Average Stacking Weight")
        ax.set_title(
            f"Super Learner Weights — {depvar.title()}", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        save_path = figures_dir / f"super_learner_weights_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# Quadratic fit with 95% pointwise CI
def _linear_fit_with_ci(x, y):
    if len(x) < 5 or len(np.unique(x)) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    try:
        m = OLS(y, X).fit(cov_type="HC1")
    except Exception:
        return None
    grid = np.linspace(x.min(), x.max(), 80)
    G = np.column_stack([np.ones_like(grid), grid])
    params = np.asarray(m.params)
    pred = G @ params
    cov = np.asarray(m.cov_params())
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    se = np.sqrt(np.clip(var, 0, None))
    return grid, pred, pred - 1.96 * se, pred + 1.96 * se


# covariate-adjusted scatter plots with local linear fit and CI (matches the p=1 preferred spec)
def plot_covariate_adjustment_scatter(rdflex_df, df):
    for depvar in DEPVARS:
        est = build_estimation_sample(df, depvar, bw=0.05)
        if len(est) == 0:
            continue
        # pool treated observations across time periods
        sub = est[est["treated"] == 1].copy()
        if len(sub) < 50:
            continue
        x = sub["diff_log_loan_amount"].astype(float)
        y = sub[depvar].astype(float)
        fig, ax = plt.subplots(figsize=(9, 5))
        # linear fit with CI on each side of the cutoff
        left = sub[sub["diff_log_loan_amount"] < 0]
        right = sub[sub["diff_log_loan_amount"] >= 0]
        for side_df in [left, right]:
            if len(side_df) < 5:
                continue
            fit = _linear_fit_with_ci(
                side_df["diff_log_loan_amount"].astype(float),
                side_df[depvar].astype(float),
            )
            if fit is None:
                continue
            grid, pred, lo, hi = fit
            ax.fill_between(grid, lo, hi, color="gray", alpha=0.3)
            ax.plot(grid, pred, color="black", lw=2)
        # individual observations
        ax.scatter(
            x,
            y,
            s=4,
            alpha=0.05,
            color="steelblue",
            edgecolors="none",
            rasterized=True,
        )
        # binned means
        n_bins = 40
        sub["rv_bin"] = pd.cut(sub["diff_log_loan_amount"], bins=n_bins)
        binned = (
            sub.groupby("rv_bin", observed=True)
            .agg(
                rv_mean=("diff_log_loan_amount", "mean"),
                y_mean=(depvar, "mean"),
            )
            .dropna()
        )
        ax.scatter(
            binned["rv_mean"],
            binned["y_mean"],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.axvline(0, color="red", lw=1.2)
        ax.set_xlabel("Log Distance to Conforming Limit")
        ax.set_ylabel(f"P({depvar.title()})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = figures_dir / f"rdd_scatter_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# Format coefficient/SE
def _fmt_coef(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


# plot paper table
def _render_paper_table(
    title, subtitle, save_path, col_labels, cells, group_rows=None, highlight_rules=None
):
    if group_rows is None:
        group_rows = []
    if highlight_rules is None:
        highlight_rules = []
    total_rows = len(cells)
    fig_h = 0.8 + 0.38 * max(total_rows, 1)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        cellLoc="right",
        colLoc="center",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    # style header row
    for c in range(len(col_labels)):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#d9d9d9")
    # style group header rows
    for gr in group_rows:
        row_idx = gr + 1
        for c in range(len(col_labels)):
            tbl[(row_idx, c)].set_facecolor("#f0f0f0")
            tbl[(row_idx, c)].set_text_props(weight="bold", ha="left")
    # left-align first column for data rows
    for r in range(1, total_rows + 1):
        if (r - 1) not in group_rows:
            tbl[(r, 0)].set_text_props(ha="left")
    # apply conditional highlighting rules
    for rule in highlight_rules:
        ci = rule["col_idx"]
        mode = rule["mode"]
        for r in range(1, total_rows + 1):
            if (r - 1) in group_rows:
                continue
            cell = tbl[(r, ci)]
            txt = cell.get_text().get_text().strip()
            if mode == "numeric_lt":
                try:
                    val = float(txt)
                    if val < rule["threshold"]:
                        cell.set_facecolor(rule["color"])
                except (ValueError, AttributeError):
                    pass
            elif mode == "exact_match":
                if txt == rule["match_value"]:
                    cell.set_facecolor(rule["color"])
                elif "alt_color" in rule and txt and txt != "N/A":
                    cell.set_facecolor(rule["alt_color"])
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, loc="left", pad=4)
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# generate main result table
def plot_main_results_table(rdflex_df):
    if rdflex_df is None or rdflex_df.empty:
        return
    rdf = _normalize_method_col(rdflex_df)
    # filter to preferred specification
    sub = rdf[
        (rdf["depvar"] == "approved") & (rdf["bandwidth"] == 0.05) & (rdf["time"] == 1)
    ].copy()
    if sub.empty:
        return
    # order methods by METHOD_COLORS key order
    method_order = list(METHOD_COLORS.keys())
    available = set(sub["method"].unique())
    methods = [m for m in method_order if m in available]
    if not methods:
        return
    # get nocov CI width as baseline for delta-CI calculation
    nocov_row = sub[sub["method"] == "nocov"]
    nocov_ci = nocov_row["ci_width"].values[0] if len(nocov_row) > 0 else np.nan
    col_labels = [
        "Method",
        "τ̂",
        "SE",
        "CI Width",
        "ΔCI (%)",
    ]
    cells = []
    for method in methods:
        m_row = sub[sub["method"] == method]
        if len(m_row) == 0:
            continue
        coef = m_row["coef"].values[0]
        se = m_row["se"].values[0]
        ci_w = m_row["ci_width"].values[0]
        # super learner or other methods may have NaN values
        if pd.isna(coef) or pd.isna(se):
            cells.append(
                [
                    METHOD_LABELS.get(method, method),
                    "failed",
                    "failed",
                    "failed",
                    "failed",
                ]
            )
            continue
        # compute delta-CI relative to nocov
        if (not np.isnan(nocov_ci)) and nocov_ci > 0 and (not np.isnan(ci_w)):
            delta_ci = ((ci_w - nocov_ci) / nocov_ci) * 100
            delta_str = f"{delta_ci:+.1f}%"
        else:
            delta_str = "N/A"
        cells.append(
            [
                METHOD_LABELS.get(method, method),
                f"{coef:.4f}",
                f"{se:.4f}",
                f"{ci_w:.4f}" if not np.isnan(ci_w) else "N/A",
                delta_str,
            ]
        )
    save_path = figures_dir / "main_results_table.png"
    _render_paper_table(
        title="",
        subtitle="",
        save_path=save_path,
        col_labels=col_labels,
        cells=cells,
        highlight_rules=[],
    )


# paper benchmark vs our replication table
def replicate_paper_tables(paper_results):
    col_labels = ["BW", "t", "Paper Coef", "Paper SE", "Our Coef", "Our SE", "Diff"]
    for depvar in DEPVARS:
        table_name = DEPVAR_TABLE.get(depvar, depvar)
        paper_sub = paper_results[paper_results["depvar"] == depvar].copy()
        cells = []
        group_rows = []
        for t in [1, 2, 3, 4]:
            group_rows.append(len(cells))
            cells.append([f"t + {t}"] + [""] * (len(col_labels) - 1))
            for bw_h in [0.01, 0.02, 0.03, 0.05]:
                bench = PAPER_BENCHMARKS.get((depvar, bw_h), {}).get(t)
                our = paper_sub[
                    (paper_sub["bandwidth"] == bw_h) & (paper_sub["time"] == t)
                ]
                paper_coef = bench[0] if bench else np.nan
                paper_se = bench[1] if bench else np.nan
                our_coef = our["coef"].values[0] if len(our) > 0 else np.nan
                our_se = our["se"].values[0] if len(our) > 0 else np.nan
                diff = (
                    our_coef - paper_coef
                    if not np.isnan(paper_coef) and not np.isnan(our_coef)
                    else np.nan
                )
                cells.append(
                    [
                        f"{bw_h:.0%}",
                        str(t),
                        _fmt_coef(paper_coef),
                        _fmt_coef(paper_se),
                        _fmt_coef(our_coef),
                        _fmt_coef(our_se),
                        _fmt_coef(diff),
                    ]
                )
        _render_paper_table(
            title=f"Replication of {table_name} — {depvar.title()}",
            subtitle="",
            save_path=figures_dir
            / f"plot_{table_name.replace(' ', '').lower()}_replication.png",
            col_labels=col_labels,
            cells=cells,
            group_rows=group_rows,
        )


# correlation of each covariate with the running variable
def plot_covariate_running_var_correlation(df):
    for depvar in DEPVARS:
        est = build_estimation_sample(df, depvar, bw=0.05)
        # keep only treated==1, time==1 for a clean sample
        sub = est[(est["treated"] == 1) & (est["time"] == 1)].copy()
        covars = [c for c in RDFLEX_COVARIATES if c in sub.columns]
        sub = sub.dropna(subset=covars + ["diff_log_loan_amount"])
        if len(sub) < 50:
            continue
        # compute Pearson correlation
        correlations = {}
        for cov in covars:
            corr = sub[cov].astype(float).corr(sub["diff_log_loan_amount"])
            correlations[cov] = corr
        # sort by absolute correlation
        sorted_covs = sorted(correlations.keys(), key=lambda c: abs(correlations[c]))
        labels = [COVARIATE_LABELS.get(c, c) for c in sorted_covs]
        values = [correlations[c] for c in sorted_covs]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728" if abs(v) > 0.3 else "#1f77b4" for v in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson Correlation with Running Variable")
        # add value labels
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        save_path = figures_dir / f"covariate_rv_correlation_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# covariates selection
def plot_lasso_covariate_selection(df):
    for depvar in DEPVARS:
        est = build_estimation_sample(df, depvar, bw=0.05)
        sub = est[(est["treated"] == 1) & (est["time"] == 1)].copy()
        covars = [c for c in RDFLEX_COVARIATES if c in sub.columns]
        sub = sub.dropna(subset=covars + [depvar])
        if len(sub) < 50:
            continue
        X = sub[covars].values.astype(float)
        y = sub[depvar].values.astype(float)
        # fit LassoCV to determine which covariates are selected
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(X, y)
        coefs = lasso.coef_
        selected = np.abs(coefs) > 0
        # sort by absolute coefficient
        sort_idx = np.argsort(np.abs(coefs))
        sorted_labels = [COVARIATE_LABELS.get(covars[i], covars[i]) for i in sort_idx]
        sorted_coefs = coefs[sort_idx]
        sorted_selected = selected[sort_idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["black" if s else "#999999" for s in sorted_selected]
        ax.barh(sorted_labels, sorted_coefs, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Relative Feature Importance")
        n_selected = int(np.sum(selected))
        # add value labels
        for i, v in enumerate(sorted_coefs):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        save_path = figures_dir / f"lasso_covariate_selection_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# SE comparision plot
def plot_se_comparison(rdflex_df):
    rdflex_df = _normalize_method_col(rdflex_df)
    for depvar in DEPVARS:
        paper_bench = PAPER_BENCHMARKS.get((depvar, 0.05))
        if paper_bench is None:
            continue
        # get RDFlex results at bw=0.05
        sub = rdflex_df[
            (rdflex_df["depvar"] == depvar) & (rdflex_df["bandwidth"] == 0.05)
        ].copy()
        if len(sub) == 0:
            # try closest available bw
            avail_bws = rdflex_df.loc[
                rdflex_df["depvar"] == depvar, "bandwidth"
            ].unique()
            numeric_bws = [
                b for b in avail_bws if isinstance(b, (int, float)) and b > 0
            ]
            if not numeric_bws:
                continue
            closest = min(numeric_bws, key=lambda b: abs(b - 0.05))
            sub = rdflex_df[
                (rdflex_df["depvar"] == depvar) & (rdflex_df["bandwidth"] == closest)
            ].copy()
        times = [1, 2, 3, 4]
        # method order follows METHOD_COLORS key order
        method_order = list(METHOD_COLORS.keys())
        available = set(sub["method"].unique())
        methods = [m for m in method_order if m in available]
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(times))
        n_groups = len(methods) + 1
        width = 0.8 / n_groups
        # paper SEs
        paper_ses = [paper_bench.get(t, (np.nan, np.nan))[1] for t in times]
        ax.bar(
            x - 0.4 + width / 2,
            paper_ses,
            width,
            label="Paper",
            color="white",
            edgecolor="black",
            linewidth=1.0,
            hatch="//",
        )
        # our methods SEs
        for j, method in enumerate(methods):
            method_sub = sub[sub["method"] == method]
            ses = []
            for t in times:
                t_sub = method_sub[method_sub["time"] == t]
                ses.append(t_sub["se"].values[0] if len(t_sub) > 0 else np.nan)
            color = METHOD_COLORS.get(method, "#333333")
            label = METHOD_LABELS.get(method, method)
            ax.bar(
                x - 0.4 + (j + 1.5) * width,
                ses,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Standard Error")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{t}" for t in times])
        ax.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        save_path = figures_dir / f"se_comparison_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# Bandwidth sensitivity plot of nocov coefficient across bandwidths
def robustness_bandwidth_sensitivity(rdflex_df):
    rdflex_df = _normalize_method_col(rdflex_df)
    for depvar in DEPVARS:
        sub = rdflex_df[
            (rdflex_df["depvar"] == depvar) & (rdflex_df["method"] == "nocov")
        ].copy()
        # filter to numeric bandwidths only
        sub = sub[sub["bandwidth"] != "auto"].copy()
        sub["bandwidth"] = sub["bandwidth"].astype(float)
        sub["coef"] = sub["coef"].astype(float)
        sub["se"] = sub["se"].astype(float)
        if len(sub) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, t in enumerate([1, 2, 3, 4]):
            ax = axes[idx]
            t_sub = sub[sub["time"] == t].sort_values("bandwidth")
            if len(t_sub) == 0:
                ax.set_title(f"t+{t}: no data")
                continue
            bws = t_sub["bandwidth"].values.astype(float)
            coefs = t_sub["coef"].values.astype(float)
            ses = t_sub["se"].values.astype(float)
            ci_lo = coefs - 1.96 * ses
            ci_hi = coefs + 1.96 * ses
            ax.plot(bws, coefs, "o-", color="#1f77b4", linewidth=2, markersize=6)
            ax.fill_between(bws, ci_lo, ci_hi, alpha=0.2, color="#1f77b4")
            # paper benchmark
            bench = PAPER_BENCHMARKS.get((depvar, 0.05), {}).get(t)
            if bench:
                ax.axhline(
                    y=bench[0],
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.text(
                    0.97,
                    0.97,
                    f"Paper: {bench[0]:.4f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#d62728",
                    ha="right",
                    va="top",
                )
            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"t+{t}")
        fig.supxlabel("Bandwidth (%)")
        fig.supylabel("Coefficient")
        plt.tight_layout()
        save_path = figures_dir / f"robustness_bw_sensitivity_{depvar}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# Relative CI width reduction plots
def plot_relative_ci_reduction(rdflex_df):
    if rdflex_df is None or rdflex_df.empty:
        return
    rdflex_bws = RDFLEX_BWS
    ml_methods = [
        m
        for m in METHOD_COLORS.keys()
        if m != "nocov" and m in rdflex_df["method"].values
    ]
    if not ml_methods:
        return
    # averaged across time periods
    for depvar in DEPVARS:
        dep_df = rdflex_df[rdflex_df["depvar"] == depvar].copy()
        if len(dep_df) == 0:
            continue
        norm_rows = []
        for bw in rdflex_bws:
            for t in [1, 2, 3, 4]:
                nocov_row = dep_df[
                    (dep_df["bandwidth"] == bw)
                    & (dep_df["method"] == "nocov")
                    & (dep_df["time"] == t)
                ]
                if len(nocov_row) == 0:
                    continue
                nocov_ci = nocov_row["ci_width"].values[0]
                if np.isnan(nocov_ci) or nocov_ci <= 0:
                    continue
                for method in ml_methods:
                    m_row = dep_df[
                        (dep_df["bandwidth"] == bw)
                        & (dep_df["method"] == method)
                        & (dep_df["time"] == t)
                    ]
                    if len(m_row) == 0:
                        continue
                    m_ci = m_row["ci_width"].values[0]
                    if np.isnan(m_ci):
                        continue
                    norm_rows.append(
                        {
                            "bandwidth": bw,
                            "time": t,
                            "method": method,
                            "relative_ci": m_ci / nocov_ci,
                            "change_pct": (m_ci / nocov_ci - 1) * 100,
                        }
                    )
        if not norm_rows:
            continue
        norm_df = pd.DataFrame(norm_rows)
        # average across time periods
        avg_df = (
            norm_df.groupby(["bandwidth", "method"])
            .agg(
                relative_ci=("relative_ci", "mean"),
                change_pct=("change_pct", "mean"),
            )
            .reset_index()
        )
        n_methods = len(ml_methods)
        bar_width = 0.8 / max(n_methods, 1)
        x_base = np.arange(len(rdflex_bws))
        fig, ax = plt.subplots(figsize=(max(12, n_methods * 1.5), 7))
        for j, method in enumerate(ml_methods):
            vals = []
            for bw in rdflex_bws:
                row = avg_df[(avg_df["method"] == method) & (avg_df["bandwidth"] == bw)]
                vals.append(row["relative_ci"].values[0] if len(row) > 0 else np.nan)
            offset = (j - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_base + offset,
                vals,
                width=bar_width,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "gray"),
                alpha=0.9,
            )
            # annotate bars with change %
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    change = (v - 1) * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{change:+.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{bw:.0%}" for bw in rdflex_bws])
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("Relative CI Width (No Covariates = 1.0)")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # 15% headroom for labels
        ax.legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"relative_ci_reduction_{depvar}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    # relative ci reduction summary plot
    all_norm = []
    for depvar in DEPVARS:
        dep_df = rdflex_df[rdflex_df["depvar"] == depvar]
        for t in [1, 2, 3, 4]:
            nocov_row = dep_df[
                (dep_df["bandwidth"] == 0.05)
                & (dep_df["method"] == "nocov")
                & (dep_df["time"] == t)
            ]
            if len(nocov_row) == 0:
                continue
            nocov_ci = nocov_row["ci_width"].values[0]
            if np.isnan(nocov_ci) or nocov_ci <= 0:
                continue
            for method in ml_methods:
                m_row = dep_df[
                    (dep_df["bandwidth"] == 0.05)
                    & (dep_df["method"] == method)
                    & (dep_df["time"] == t)
                ]
                if len(m_row) == 0:
                    continue
                m_ci = m_row["ci_width"].values[0]
                if np.isnan(m_ci):
                    continue
                all_norm.append(
                    {
                        "depvar": depvar,
                        "time": t,
                        "method": method,
                        "relative_ci": m_ci / nocov_ci,
                    }
                )
    if all_norm:
        summary_df = pd.DataFrame(all_norm)
        avg_summary = summary_df.groupby("method")["relative_ci"].mean().reset_index()
        avg_summary = avg_summary.sort_values("relative_ci")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [METHOD_COLORS.get(m, "gray") for m in avg_summary["method"]]
        bars = ax.barh(
            [METHOD_LABELS.get(m, m) for m in avg_summary["method"]],
            avg_summary["relative_ci"],
            color=colors,
            alpha=0.9,
        )
        for bar, v in zip(bars, avg_summary["relative_ci"]):
            change = (v - 1) * 100
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{change:+.1f}%",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel("Relative CI Width (No Covariates = 1.0)")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(
            figures_dir / "relative_ci_reduction_summary.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# plot seed robustness figure
def _plot_seed_robustness(seed_df):
    methods_in_results = [
        m for m in METHOD_COLORS.keys() if m in seed_df["method"].values
    ]
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
        mdf = seed_df[seed_df["method"] == method].sort_values("seed")
        x_pos = np.arange(len(mdf))
        ax.errorbar(
            x_pos,
            mdf["coef"].values,
            yerr=1.96 * mdf["se"].values,
            fmt="o",
            color=METHOD_COLORS.get(method, "gray"),
            capsize=5,
            capthick=1.5,
            markersize=7,
            linewidth=1.5,
        )
        # reference: mean across seeds
        mean_coef = mdf["coef"].mean()
        ax.axhline(
            y=mean_coef,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_coef:.4f}",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(s) for s in mdf["seed"].values], fontsize=8)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
    # hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes[row_i, col_i].set_visible(False)
    fig.supxlabel("Seed")
    fig.supylabel("Treatment Effect")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "robustness_seed_stability.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# plot kernel robustness figure
def _plot_kernel_robustness(kernel_df):
    kernels = kernel_df["kernel"].unique()
    methods_in_results = [
        m
        for m in METHOD_COLORS.keys()
        if m in kernel_df["method"].values and m != "nocov"
    ]
    if len(methods_in_results) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods_in_results))
    bar_width = 0.35
    for i, kernel in enumerate(["triangular", "epanechnikov"]):
        kdf = kernel_df[kernel_df["kernel"] == kernel]
        nocov_row = kdf[kdf["method"] == "nocov"]
        if len(nocov_row) == 0:
            continue
        nocov_ci = nocov_row["ci_width"].values[0]
        if np.isnan(nocov_ci) or nocov_ci <= 0:
            continue
        ratios = []
        colors = []
        for method in methods_in_results:
            m_row = kdf[kdf["method"] == method]
            if len(m_row) > 0 and not np.isnan(m_row["ci_width"].values[0]):
                ratios.append(m_row["ci_width"].values[0] / nocov_ci)
            else:
                ratios.append(np.nan)
            colors.append(METHOD_COLORS.get(method, "gray"))
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
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods_in_results],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Relative CI Width (nocov = 1.0)")
    # legend
    from matplotlib.patches import Patch

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
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "robustness_kernel_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


# plot the best hyperparameter config
def _plot_hyperparameter_table(hp_df):
    pref = hp_df[(hp_df["depvar"] == "approved") & (hp_df["time"] == 1)]
    if len(pref) == 0:
        return
    methods = ["rf", "lgbm", "xgb", "nnet"]
    table_data = []
    for method in methods:
        mdf = pref[pref["method"] == method]
        for _, row in mdf.iterrows():
            val = row["value"]
            # format value nicely
            if isinstance(val, float) and not np.isnan(val):
                if val < 0.01:
                    val_str = f"{val:.2e}"
                elif val == int(val):
                    val_str = f"{int(val)}"
                else:
                    val_str = f"{val:.4f}"
            elif isinstance(val, float) and np.isnan(val):
                if row["parameter"] == "max_depth":
                    val_str = "None"
                else:
                    val_str = "—"
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
    n_rows_table = len(table_data) + 1  # +1 for header
    row_height = 0.3
    fig_height = n_rows_table * row_height
    fig = plt.figure(figsize=(14, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Method", "Parameter", "Best Value", "Description"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width([0, 1, 2, 3])
    table.scale(1, 1.2)
    # style header
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # alternate row colours
    for i in range(1, len(table_data) + 1):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)
    fig.savefig(
        figures_dir / "tuned_hyperparameters_table.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


# plot the main results table with all methods and time periods
def plot_full_results_table():
    csv_path = results_dir / "rdd_rdflex_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[pd.to_numeric(df["bandwidth"], errors="coerce") == 0.05].copy()
    if df.empty:
        return
    method_order = [
        "nocov",
        "rdrobust",
        "ridge",
        "lasso",
        "rf",
        "lgbm",
        "xgb",
        "nnet",
        "sl",
    ]
    rows = []
    for dep in DEPVARS:
        for t in [1, 2, 3, 4]:
            sub = df[
                (df["depvar"] == dep)
                & (df["time"].astype(str).isin([str(t), f"t+{t}"]))
            ]
            if sub.empty:
                continue
            nocov = sub.loc[sub["method"] == "nocov", "ci_width"]
            base = nocov.values[0] if len(nocov) else np.nan
            for m in method_order:
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
                        dep,
                        f"t+{t}",
                        m,
                        f"{r['coef']:.4f}" if pd.notna(r["coef"]) else "-",
                        f"{r['se']:.4f}" if pd.notna(r["se"]) else "-",
                        f"{r['ci_width']:.4f}" if pd.notna(r["ci_width"]) else "-",
                        f"{dci:.1f}" if pd.notna(dci) and m != "nocov" else "-",
                    ]
                )
    if not rows:
        return
    col_labels = [
        "Outcome",
        "Horizon",
        "Method",
        "Coef",
        "SE",
        "CI Width",
        "CI Red. (%)",
    ]
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
        figures_dir / "results_table_full.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()


# run main
def main():
    t_start = time.time()
    # mode selection: --plots-only skips all estimation, loads cached CSVs
    plots_only = "--plots-only" in sys.argv
    if plots_only:
        # load data
        df = load_data()
        # load cached RDFlex results
        rdflex_csv = results_dir / "rdd_rdflex_results.csv"
        if rdflex_csv.exists():
            rdflex_results = pd.read_csv(rdflex_csv)
            # convert numeric values back to float so filtering works
            if "bandwidth" in rdflex_results.columns:
                rdflex_results["bandwidth"] = rdflex_results["bandwidth"].apply(
                    lambda x: float(x) if str(x) != "auto" else x
                )
        else:
            return
        # merge rdrobust results if available
        rdrobust_results = import_rdrobust_results()
        if rdrobust_results is not None:
            rdflex_results = pd.concat(
                [rdflex_results, rdrobust_results], ignore_index=True
            )
        # load cached paper results
        paper_csv = results_dir / "rdd_paper_results.csv"
        if paper_csv.exists():
            paper_results = pd.read_csv(paper_csv)
        else:
            paper_results = pd.DataFrame()
        # plots that only need raw data
        plot_covariate_balance(df)
        plot_covariate_adjustment_scatter(None, df)
        plot_sample_n_comparison(df)
        # paper replication plots if paper results available
        if not paper_results.empty:
            plot_paper_replication(paper_results)
            replicate_paper_tables(paper_results)
    else:
        # build data, estimate, save, then plot
        df = load_data()
        print("Build Master Dataset ...")
        export_rdrobust_samples(df)
        plot_covariate_balance(df)
        plot_sample_n_comparison(df)
        print("Estimating ...")
        paper_results = replicate_paper_rdd(df)
        plot_paper_replication(paper_results)
        replicate_paper_tables(paper_results)
        gc.collect()
        plot_covariate_adjustment_scatter(None, df)
        # clear memory
        plt.close("all")
        gc.collect()
        rdflex_results = run_rdflex_estimation(df)
        # merge rdrobust results if available
        rdrobust_results = import_rdrobust_results()
        if rdrobust_results is not None:
            rdflex_results = pd.concat(
                [rdflex_results, rdrobust_results], ignore_index=True
            )
    # plots and analysis
    print("Plotting ...")
    plot_full_results_table()
    plot_main_results_table(rdflex_results)
    plot_coefficient_comparison(rdflex_results)
    plot_ci_width_comparison(rdflex_results)
    plot_forest_estimates(rdflex_results)
    plot_standardized_diff(rdflex_results)
    plot_bw_auto_vs_hardcoded(rdflex_results)
    plot_covariate_importance_per_method()
    plot_super_learner_weights()
    plot_relative_ci_reduction(rdflex_results)
    plot_covariate_running_var_correlation(df)
    plot_lasso_covariate_selection(df)
    plot_se_comparison(rdflex_results)
    compute_power_analysis(df, rdflex_results)
    robustness_predetermined_and_placebo(df)
    robustness_density_test(df)
    robustness_bandwidth_sensitivity(rdflex_results)
    robustness_formal_covariate_continuity()
    # clear memory
    plt.close("all")
    gc.collect()
    seed_csv = results_dir / "seed_robustness.csv"
    if plots_only and seed_csv.exists():
        # re-render from cached CSV
        seed_df = pd.read_csv(seed_csv)
        _plot_seed_robustness(seed_df)
    elif not plots_only:
        run_seed_robustness(df, rdflex_results)
    plt.close("all")
    gc.collect()
    kernel_csv = results_dir / "kernel_robustness.csv"
    if plots_only and kernel_csv.exists():
        kernel_df = pd.read_csv(kernel_csv)
        _plot_kernel_robustness(kernel_df)
    elif not plots_only:
        robustness_kernel_comparison(df, rdflex_results)
    hp_csv = results_dir / "tuned_hyperparameters.csv"
    if hp_csv.exists():
        hp_df = pd.read_csv(hp_csv)
        _plot_hyperparameter_table(hp_df)
    # save consolidated results.csv
    if not plots_only:
        all_rows = []
        if len(paper_results) > 0:
            pr = paper_results.copy()
            pr["path"] = "paper_rdd"
            all_rows.append(pr)
        if len(rdflex_results) > 0:
            rr = rdflex_results.copy()
            rr["path"] = "rdflex"
            all_rows.append(rr)
        if all_rows:
            results = pd.concat(all_rows, ignore_index=True)
            results.to_csv(results_dir / "results.csv", index=False)
    elapsed = time.time() - t_start
    print("Finished ...")


if __name__ == "__main__":
    main()