# Linear covariate adjustment using rdrobust for comparison with the ML-based RDFlex models in main_griffin.py.
#
# This script:
#   1. Reads the per-sample CSVs from results
#   2. Runs rdrobust with the same covariates and bandwidth (h=50)
#   3. Saves results to results_rdrobust.csv

# install rdrobust if not available
if (!requireNamespace("rdrobust", quietly = TRUE)) {
    install.packages("rdrobust", repos = "https://cloud.r-project.org")
}
library(rdrobust)

# constants
BANDWIDTH <- 50
COVARIATES <- c(
    "lag_ret", "vol", "vol_x_lagret",
    "hour_of_day", "day_of_week",
    "lag_ret_eth", "lag_ret_xrp",
    "lag_ret_neg",
    "netflow_bitt", "netflow_huob", "netflow_krak", "netflow_aggplbt"
)

# working directory
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
}
# scripts/ lives one level below the replication root
replication_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
results_dir <- file.path(replication_dir, "results")

# sample definitions
SHARP_SAMPLES <- c("Auth", "NoAuth", "Auth_NegRet", "Auth_PosRet")
FUZZY_SAMPLES <- c("All", "Auth", "Auth_NegRet", "Auth_PosRet")

# collect results
results_list <- list()
row_idx <- 0

# Sharp RD
for (sample_label in SHARP_SAMPLES) {
    csv_path <- file.path(results_dir, sprintf("rdrobust_sample_sharp_%s.csv", sample_label))
    if (!file.exists(csv_path)) {
        cat(sprintf("[SKIP] %s — file not found: %s\n", sample_label, csv_path))
        cat("Run main_griffin.py first to generate sample CSVs.\n")
        next
    }
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    y <- df$fret
    x <- df$price_dist
    # define available covariates
    avail_covs <- intersect(COVARIATES, colnames(df))
    covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])
    # drop rows with NA
    complete <- complete.cases(y, x, covs_matrix)
    y <- y[complete]
    x <- x[complete]
    covs_matrix <- covs_matrix[complete, , drop = FALSE]
    tryCatch({
        # setup same parameters as in main_griffin.py
        fit <- rdrobust(
            y      = y,
            x      = x,
            c      = 0,
            p      = 1,
            h      = BANDWIDTH,
            covs   = covs_matrix,
            kernel = "triangular"
        )
        # extract robust estimates
        coef_val   <- fit$coef[1]
        se_val     <- fit$se[3]
        pval_val   <- fit$pv[3]
        ci_lower   <- fit$ci[3, 1]
        ci_upper   <- fit$ci[3, 2]
        t_stat_val <- coef_val / se_val
        n_obs      <- fit$N_h[1] + fit$N_h[2]
        row_idx <- row_idx + 1
        results_list[[row_idx]] <- data.frame(
            method     = "rdrobust",
            design     = "sharp",
            sample     = sample_label,
            adjustment = "linear (rdrobust)",
            coef       = round(coef_val, 2),
            se         = round(se_val, 2),
            tstat      = round(t_stat_val, 2),
            pval       = round(pval_val, 4),
            ci_low     = round(ci_lower, 2),
            ci_high    = round(ci_upper, 2),
            n          = n_obs,
            stringsAsFactors = FALSE
        )
    }, error = function(e) {
        cat(sprintf("    FAILED — %s\n", conditionMessage(e)))
        row_idx <<- row_idx + 1
        results_list[[row_idx]] <<- data.frame(
            method     = "rdrobust",
            design     = "sharp",
            sample     = sample_label,
            adjustment = "linear (rdrobust)",
            coef       = NA,
            se         = NA,
            tstat      = NA,
            pval       = NA,
            ci_low     = NA,
            ci_high    = NA,
            n          = nrow(df),
            stringsAsFactors = FALSE
        )
    })
}

# Fuzzy RD
for (sample_label in FUZZY_SAMPLES) {
    csv_path <- file.path(results_dir, sprintf("rdrobust_sample_fuzzy_%s.csv", sample_label))
    if (!file.exists(csv_path)) {
        cat(sprintf("[SKIP] %s — file not found: %s\n", sample_label, csv_path))
        next
    }
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    cat(sprintf("\n[rdrobust] Fuzzy / %s  n=%d\n", sample_label, nrow(df)))
    y <- df$fret
    x <- df$price_dist
    d <- df$high_flow
    # covariates
    avail_covs <- intersect(COVARIATES, colnames(df))
    covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])
    # drop rows with NA
    complete <- complete.cases(y, x, d, covs_matrix)
    y <- y[complete]
    x <- x[complete]
    d <- d[complete]
    covs_matrix <- covs_matrix[complete, , drop = FALSE]
    tryCatch({
        # setup same parameters as in main_griffin.py
        fit <- rdrobust(
            y      = y,
            x      = x,
            c      = 0,
            p      = 1,
            h      = BANDWIDTH,
            covs   = covs_matrix,
            fuzzy  = d,
            kernel = "triangular"
        )
        coef_val   <- fit$coef[1]
        se_val     <- fit$se[3]
        pval_val   <- fit$pv[3]
        ci_lower   <- fit$ci[3, 1]
        ci_upper   <- fit$ci[3, 2]
        t_stat_val <- coef_val / se_val
        n_obs      <- fit$N_h[1] + fit$N_h[2]
        row_idx <- row_idx + 1
        results_list[[row_idx]] <- data.frame(
            method     = "rdrobust",
            design     = "fuzzy",
            sample     = sample_label,
            adjustment = "linear (rdrobust)",
            coef       = round(coef_val, 2),
            se         = round(se_val, 2),
            tstat      = round(t_stat_val, 2),
            pval       = round(pval_val, 4),
            ci_low     = round(ci_lower, 2),
            ci_high    = round(ci_upper, 2),
            n          = n_obs,
            stringsAsFactors = FALSE
        )
    }, error = function(e) {
        cat(sprintf("    FAILED — %s\n", conditionMessage(e)))
        row_idx <<- row_idx + 1
        results_list[[row_idx]] <<- data.frame(
            method     = "rdrobust",
            design     = "fuzzy",
            sample     = sample_label,
            adjustment = "linear (rdrobust)",
            coef       = NA,
            se         = NA,
            tstat      = NA,
            pval       = NA,
            ci_low     = NA,
            ci_high    = NA,
            n          = nrow(df),
            stringsAsFactors = FALSE
        )
    })
}


# Sharp RD with automatic bandwidth
for (sample_label in SHARP_SAMPLES) {
    csv_path <- file.path(results_dir, sprintf("rdrobust_sample_sharp_%s.csv", sample_label))
    if (!file.exists(csv_path)) {
        cat(sprintf("[SKIP] %s — file not found\n", sample_label))
        next
    }
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    y <- df$fret
    x <- df$price_dist
    avail_covs <- intersect(COVARIATES, colnames(df))
    covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])
    complete <- complete.cases(y, x, covs_matrix)
    y <- y[complete]
    x <- x[complete]
    covs_matrix <- covs_matrix[complete, , drop = FALSE]
    tryCatch({
        # sharp RD with optimal bandwidth same as in main_griffin.py
        fit <- rdrobust(
            y      = y,
            x      = x,
            c      = 0,
            p      = 1,
            covs   = covs_matrix,
            kernel = "triangular"
        )
        coef_val   <- fit$coef[1]
        se_val     <- fit$se[3]
        pval_val   <- fit$pv[3]
        ci_lower   <- fit$ci[3, 1]
        ci_upper   <- fit$ci[3, 2]
        t_stat_val <- coef_val / se_val
        n_obs      <- fit$N_h[1] + fit$N_h[2]
        h_mse      <- fit$bws[1, 1]
        row_idx <- row_idx + 1
        results_list[[row_idx]] <- data.frame(
            method     = "rdrobust",
            design     = "sharp",
            sample     = sample_label,
            adjustment = "linear (rdrobust, opt-bw)",
            coef       = round(coef_val, 2),
            se         = round(se_val, 2),
            tstat      = round(t_stat_val, 2),
            pval       = round(pval_val, 4),
            ci_low     = round(ci_lower, 2),
            ci_high    = round(ci_upper, 2),
            n          = n_obs,
            stringsAsFactors = FALSE
        )
    }, error = function(e) {
        cat(sprintf("[FAILED] %s — %s\n", sample_label, conditionMessage(e)))
        row_idx <<- row_idx + 1
        results_list[[row_idx]] <<- data.frame(
            method     = "rdrobust",
            design     = "sharp",
            sample     = sample_label,
            adjustment = "linear (rdrobust, opt-bw)",
            coef       = NA,
            se         = NA,
            tstat      = NA,
            pval       = NA,
            ci_low     = NA,
            ci_high    = NA,
            n          = nrow(df),
            stringsAsFactors = FALSE
        )
    })
}

# Fuzzy RD with automatic bandwidth
for (sample_label in FUZZY_SAMPLES) {
    csv_path <- file.path(results_dir, sprintf("rdrobust_sample_fuzzy_%s.csv", sample_label))
    if (!file.exists(csv_path)) {
        cat(sprintf("[SKIP] %s — file not found\n", sample_label))
        next
    }
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    y <- df$fret
    x <- df$price_dist
    d <- df$high_flow
    avail_covs <- intersect(COVARIATES, colnames(df))
    covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])
    complete <- complete.cases(y, x, d, covs_matrix)
    y <- y[complete]
    x <- x[complete]
    d <- d[complete]
    covs_matrix <- covs_matrix[complete, , drop = FALSE]
    tryCatch({
        # fuzzy RD with optimal bandwidth same as in main_griffin.py
        fit <- rdrobust(
            y      = y,
            x      = x,
            c      = 0,
            p      = 1,
            covs   = covs_matrix,
            fuzzy  = d,
            kernel = "triangular"
        )
        coef_val   <- fit$coef[1]
        se_val     <- fit$se[3]
        pval_val   <- fit$pv[3]
        ci_lower   <- fit$ci[3, 1]
        ci_upper   <- fit$ci[3, 2]
        t_stat_val <- coef_val / se_val
        n_obs      <- fit$N_h[1] + fit$N_h[2]
        h_mse      <- fit$bws[1, 1]
        row_idx <- row_idx + 1
        results_list[[row_idx]] <- data.frame(
            method     = "rdrobust",
            design     = "fuzzy",
            sample     = sample_label,
            adjustment = "linear (rdrobust, opt-bw)",
            coef       = round(coef_val, 2),
            se         = round(se_val, 2),
            tstat      = round(t_stat_val, 2),
            pval       = round(pval_val, 4),
            ci_low     = round(ci_lower, 2),
            ci_high    = round(ci_upper, 2),
            n          = n_obs,
            stringsAsFactors = FALSE
        )
    }, error = function(e) {
        cat(sprintf("[FAILED] %s — %s\n", sample_label, conditionMessage(e)))
        row_idx <<- row_idx + 1
        results_list[[row_idx]] <<- data.frame(
            method     = "rdrobust",
            design     = "fuzzy",
            sample     = sample_label,
            adjustment = "linear (rdrobust, opt-bw)",
            coef       = NA,
            se         = NA,
            tstat      = NA,
            pval       = NA,
            ci_low     = NA,
            ci_high    = NA,
            n          = nrow(df),
            stringsAsFactors = FALSE
        )
    })
}

# save results to CSV
if (length(results_list) > 0) {
    results_df <- do.call(rbind, results_list)
    rownames(results_df) <- NULL
    out_path <- file.path(results_dir, "results_rdrobust.csv")
    write.csv(results_df, out_path, row.names = FALSE)
    # print summary
    cat("\nSummary:\n")
    print(results_df[, c("design", "sample", "coef", "se", "ci_low", "ci_high", "n")])
} else {
    cat("\nERROR: No results produced. Make sure sample CSV files exist.\n")
    cat("Run main_griffin.py first.\n")
}

# covariate continuity test
cov_csv <- file.path(results_dir, "rdrobust_sample_sharp_Auth.csv")
if (file.exists(cov_csv)) {
    df_cov <- read.csv(cov_csv, stringsAsFactors = FALSE)
    x_cov <- df_cov$price_dist
    cov_results <- list()
    cov_idx <- 0
    for (cov_name in COVARIATES) {
        if (!(cov_name %in% colnames(df_cov))) {
            cat(sprintf("  [SKIP] %s — not in data\n", cov_name))
            next
        }
        y_cov <- df_cov[[cov_name]]
        ok <- complete.cases(y_cov, x_cov)
        y_c <- y_cov[ok]
        x_c <- x_cov[ok]
        if (length(y_c) < 50) {
            cat(sprintf("[SKIP] %s — too few obs (%d)\n", cov_name, length(y_c)))
            next
        }
        tryCatch({
            # continuity test using optimal bandwidth
            fit <- rdrobust(y = y_c, x = x_c, c = 0, p = 1, kernel = "triangular")
            coef_val   <- fit$coef[1]
            se_val     <- fit$se[3]
            pval_val   <- fit$pv[3]
            ci_lower   <- fit$ci[3, 1]
            ci_upper   <- fit$ci[3, 2]
            h_val      <- fit$bws[1, 1]
            n_eff      <- fit$N_h[1] + fit$N_h[2]
            cov_idx <- cov_idx + 1
            cov_results[[cov_idx]] <- data.frame(
                covariate  = cov_name,
                h_mse      = round(h_val, 3),
                rd_est     = round(coef_val, 4),
                robust_pval = round(pval_val, 4),
                ci_low     = round(ci_lower, 4),
                ci_high    = round(ci_upper, 4),
                n_eff      = n_eff,
                stringsAsFactors = FALSE
            )
            sig <- if (pval_val < 0.05) " *" else ""
        }, error = function(e) {
            cat(sprintf("[FAILED] %s — %s\n", cov_name, conditionMessage(e)))
        })
    }

    if (length(cov_results) > 0) {
        cov_df <- do.call(rbind, cov_results)
        rownames(cov_df) <- NULL
        cov_out <- file.path(results_dir, "covariate_continuity_rdrobust.csv")
        write.csv(cov_df, cov_out, row.names = FALSE)
        cat(sprintf("\n[SAVED] %d rows to %s\n", nrow(cov_df), cov_out))
    }
} else {
    cat("\nERROR: No results produced. Auth sample CSV not found.\n")
    cat("Run main_griffin.py first.\n")
}
