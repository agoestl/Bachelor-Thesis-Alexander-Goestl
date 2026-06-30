# Linear covariate adjustment using rdrobust for comparison with the ML-based RDFlex models in main_ouazad.py.
#
# This script:
#   1. Reads the per-sample CSVs saved by main_ouazad.py
#   2. Runs rdrobust with the same covariates, bandwidths, and kernel
#   3. Saves results to results_rdrobust.csv in the same column format as RDFlex

# install rdrobust if not available
if (!requireNamespace("rdrobust", quietly = TRUE)) {
    install.packages("rdrobust", repos = "https://cloud.r-project.org")
}
library(rdrobust)

# constants
DEPVARS    <- c("approved", "originated", "securitized")
# Full bandwidth grid
BANDWIDTHS <- c(0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20)
CUTOFF     <- 0.0
COVARIATES <- c(
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
    "year"
)

# paths — script lives in Ouazad_Replication/scripts/
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
}
repo_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
results_dir <- file.path(repo_dir, "results")
setwd(script_dir)

sample_dir <- file.path(results_dir, "rdrobust_samples")
if (!dir.exists(sample_dir)) {
    stop(paste0(
        "Sample directory not found: ", sample_dir, "\n",
        "Run main_ouazad.py first to generate sample CSVs."
    ))
}

# collect results
results_list <- list()
row_idx <- 0

# main loop
for (depvar in DEPVARS) {
    for (t in 1:4) {
        csv_path <- file.path(sample_dir, sprintf("sample_%s_t%d.csv", depvar, t))
        if (!file.exists(csv_path)) {
            cat(sprintf("[SKIP] %s t+%d — file not found: %s\n", depvar, t, csv_path))
            cat("Run main_ouazad.py first to generate sample CSVs.\n")
            next
        }
        df <- read.csv(csv_path, stringsAsFactors = FALSE)
        # outcome and running variable
        y <- df[[depvar]]
        x <- df$diff_log_loan_amount
        # define covariates matrix
        avail_covs <- intersect(COVARIATES, colnames(df))
        covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])

        # drop rows with any NA in outcome, running var, or covariates
        complete <- complete.cases(y, x, covs_matrix)
        y <- y[complete]
        x <- x[complete]
        covs_matrix <- covs_matrix[complete, , drop = FALSE]

        # run rdrobust for each bandwidth
        for (bw in BANDWIDTHS) {
            tryCatch({
                # setup same parameters as in main_ouazad.py
                fit <- rdrobust(
                    y      = y,
                    x      = x,
                    c      = CUTOFF,
                    p      = 1,
                    h      = bw,
                    covs   = covs_matrix,
                    kernel = "triangular"
                )
                # extract robust estimates
                coef_val   <- fit$coef[1]
                se_val     <- fit$se[3]
                pval_val   <- fit$pv[3]
                ci_lower   <- fit$ci[3, 1]
                ci_upper   <- fit$ci[3, 2]
                ci_width   <- ci_upper - ci_lower
                t_stat_val <- coef_val / se_val
                n_obs      <- fit$N_h[1] + fit$N_h[2]
                row_idx <- row_idx + 1
                results_list[[row_idx]] <- data.frame(
                    depvar    = depvar,
                    bandwidth = bw,
                    time      = t,
                    method    = "rdrobust",
                    coef      = coef_val,
                    se        = se_val,
                    t_stat    = t_stat_val,
                    pvalue    = pval_val,
                    ci_lower  = ci_lower,
                    ci_upper  = ci_upper,
                    ci_width  = ci_width,
                    n_obs     = n_obs,
                    stringsAsFactors = FALSE
                )
            }, error = function(e) {
                cat(sprintf("FAILED — %s\n", conditionMessage(e)))
                row_idx <<- row_idx + 1
                results_list[[row_idx]] <<- data.frame(
                    depvar    = depvar,
                    bandwidth = bw,
                    time      = t,
                    method    = "rdrobust",
                    coef      = NA,
                    se        = NA,
                    t_stat    = NA,
                    pvalue    = NA,
                    ci_lower  = NA,
                    ci_upper  = NA,
                    ci_width  = NA,
                    n_obs     = length(y),
                    stringsAsFactors = FALSE
                )
            })
        }
        # automatic bandwidth selection
        tryCatch({
            fit_auto <- rdrobust(
                y      = y,
                x      = x,
                c      = CUTOFF,
                p      = 1,
                covs   = covs_matrix,
                kernel = "triangular"
            )
            coef_val   <- fit_auto$coef[1]
            se_val     <- fit_auto$se[3]
            pval_val   <- fit_auto$pv[3]
            ci_lower   <- fit_auto$ci[3, 1]
            ci_upper   <- fit_auto$ci[3, 2]
            ci_width   <- ci_upper - ci_lower
            t_stat_val <- coef_val / se_val
            n_obs      <- fit_auto$N_h[1] + fit_auto$N_h[2]
            auto_bw    <- fit_auto$bws[1, 1]
            row_idx <- row_idx + 1
            results_list[[row_idx]] <- data.frame(
                depvar    = depvar,
                bandwidth = "auto",
                time      = t,
                method    = "rdrobust",
                coef      = coef_val,
                se        = se_val,
                t_stat    = t_stat_val,
                pvalue    = pval_val,
                ci_lower  = ci_lower,
                ci_upper  = ci_upper,
                ci_width  = ci_width,
                n_obs     = n_obs,
                stringsAsFactors = FALSE
            )
        }, error = function(e) {
            cat(sprintf("FAILED — %s\n", conditionMessage(e)))
            row_idx <<- row_idx + 1
            results_list[[row_idx]] <<- data.frame(
                depvar    = depvar,
                bandwidth = "auto",
                time      = t,
                method    = "rdrobust",
                coef      = NA,
                se        = NA,
                t_stat    = NA,
                pvalue    = NA,
                ci_lower  = NA,
                ci_upper  = NA,
                ci_width  = NA,
                n_obs     = length(y),
                stringsAsFactors = FALSE
            )
        })
    }
}

# covariate continuity test
cov_csv <- file.path(sample_dir, "sample_approved_t1.csv")
if (!file.exists(cov_csv)) {
    cat(sprintf("[SKIP] Covariate continuity test — file not found: %s\n", cov_csv))
    cat("Run main_ouazad.py first to generate sample CSVs.\n")
} else {
    df_cov <- read.csv(cov_csv, stringsAsFactors = FALSE)
    x_cov  <- df_cov$diff_log_loan_amount  # running variable
    cov_results <- list()
    cov_idx <- 0
    for (cov_name in COVARIATES) {
        if (!(cov_name %in% colnames(df_cov))) {
            cat(sprintf("[SKIP] %s not in data\n", cov_name))
            next
        }
        y_cov <- df_cov[[cov_name]]
        ok <- complete.cases(y_cov, x_cov)
        y_sub <- y_cov[ok]
        x_sub <- x_cov[ok]
        if (length(y_sub) < 50) {
            cat(sprintf("[SKIP] %s: too few observations (%d)\n", cov_name, length(y_sub)))
            next
        }
        tryCatch({
            # let rdrobust choose MSE-optimal bandwidth for this covariate
            fit_cov <- rdrobust(
                y      = y_sub,
                x      = x_sub,
                c      = CUTOFF,
                p      = 1,
                kernel = "triangular"
            )
            rd_est    <- fit_cov$coef[1]
            rob_pval  <- fit_cov$pv[3]
            rob_ci_lo <- fit_cov$ci[3, 1]
            rob_ci_hi <- fit_cov$ci[3, 2]
            bw_h      <- fit_cov$bws[1, 1]
            eff_n     <- fit_cov$N_h[1] + fit_cov$N_h[2]
            sig <- ifelse(rob_pval < 0.05, " ***", "")
            cov_idx <- cov_idx + 1
            cov_results[[cov_idx]] <- data.frame(
                covariate     = cov_name,
                rd_estimator  = rd_est,
                robust_pvalue = rob_pval,
                ci_lower      = rob_ci_lo,
                ci_upper      = rob_ci_hi,
                mse_bandwidth = bw_h,
                eff_n         = eff_n,
                stringsAsFactors = FALSE
            )
        }, error = function(e) {
            cat(sprintf("FAILED — %s\n", conditionMessage(e)))
        })
    }
    if (length(cov_results) > 0) {
        cov_df <- do.call(rbind, cov_results)
        rownames(cov_df) <- NULL
        cov_out <- file.path(results_dir, "results_covariate_continuity.csv")
        write.csv(cov_df, cov_out, row.names = FALSE)
        cat(sprintf("\nSaved covariate continuity results to %s\n", cov_out))
        cat("\nCovariate Continuity Summary:\n")
        print(cov_df)
    }
}
# combine and save
if (length(results_list) > 0) {
    results_df <- do.call(rbind, results_list)
    rownames(results_df) <- NULL
    out_path <- file.path(results_dir, "results_rdrobust.csv")
    write.csv(results_df, out_path, row.names = FALSE)
    # print summary table
    cat("\nSummary:\n")
    print(results_df[, c("depvar", "time", "bandwidth", "coef", "se", "ci_lower", "ci_upper", "n_obs")])
} else {
    cat("\nERROR: No results produced. Make sure rdrobust_samples/ CSVs exist.\n")
    cat("Run main_ouazad.py first.\n")
}