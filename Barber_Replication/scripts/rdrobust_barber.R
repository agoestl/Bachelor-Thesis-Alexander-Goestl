# Linear covariate adjustment using rdrobust for comparison with the ML-based RDFlex models in main_barber.py.
#
#   1. Reads the per-sample CSVs saved by main_barber.py
#   2. Runs rdrobust with the same covariates and bandwidth
#   3. Saves results to results_rdrobust.csv in the same format as results.csv

# install rdrobust if not available
if (!requireNamespace("rdrobust", quietly = TRUE)) {
    install.packages("rdrobust", repos = "https://cloud.r-project.org")
}
library(rdrobust)

# constants
BANDWIDTHS   <- c(50, 75, 100, 125)
POLY_ORDERS  <- c(1, 2, 3)
COVARIATES   <- c("abs_ret", "log_volume", "log_users",
                   "mkt_ret", "asvi", "log_users_lag")

# working directory
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
}
results_dir <- file.path(dirname(script_dir), "results")

# collect results
results_list <- list()
row_idx <- 0

for (bw in BANDWIDTHS) {
    # read matched sample saved by main_barber.py
    csv_path <- file.path(results_dir, sprintf("matched_sample_bw%d.csv", bw))
    if (!file.exists(csv_path)) {
        cat(sprintf("[SKIP] bw=$%dM — file not found: %s\n", bw, csv_path))
        cat("Run main_barber.py first to generate matched sample CSVs.\n")
        next
    }
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    cat(sprintf("\n  [rdrobust] bw=$%dM  n=%d\n", bw, nrow(df)))
    # outcome, treatment, running variable
    y  <- df$dayuserchgw
    x  <- df$mktcap1_300
    d  <- df$gt300
    # covariates only include those present in the sample CSV
    avail_covs <- intersect(COVARIATES, colnames(df))
    covs_matrix <- as.matrix(df[, avail_covs, drop = FALSE])
    # drop rows with any NA in outcome, running var, or covariates
    complete <- complete.cases(y, x, covs_matrix)
    y   <- y[complete]
    x   <- x[complete]
    covs_matrix <- covs_matrix[complete, , drop = FALSE]
    for (p in POLY_ORDERS) {
        tryCatch({
            # setup parameter just as in main_barber.py
            fit <- rdrobust(
                y    = y,
                x    = x,
                c    = 0,
                p    = p,
                h    = bw,
                covs = covs_matrix,
                kernel = "triangular"
            )
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
                bandwidth      = bw,
                poly_order     = p,
                method         = "rdrobust",
                beta1          = coef_val,
                se             = se_val,
                t_stat         = t_stat_val,
                pvalue         = pval_val,
                ci_lower       = ci_lower,
                ci_upper       = ci_upper,
                ci_width       = ci_width,
                n_obs          = n_obs,
                covariates_used = paste(avail_covs, collapse = ", "),
                stringsAsFactors = FALSE
            )
        }, error = function(e) {
            cat(sprintf("FAILED — %s\n", conditionMessage(e)))
            row_idx <<- row_idx + 1
            results_list[[row_idx]] <<- data.frame(
                bandwidth       = bw,
                poly_order      = p,
                method          = "rdrobust",
                beta1           = NA,
                se              = NA,
                t_stat          = NA,
                pvalue          = NA,
                ci_lower        = NA,
                ci_upper        = NA,
                ci_width        = NA,
                n_obs           = length(y),
                covariates_used = paste(avail_covs, collapse = ", "),
                stringsAsFactors = FALSE
            )
        })
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
    print(results_df[, c("bandwidth", "poly_order", "beta1", "se", "ci_width")])
} else {
    cat("\nERROR: No results produced. Make sure matched_sample_bw*.csv files exist.\n")
    cat("Run main_barber.py first.\n")
}

# use the largest bandwidth sample
max_bw_auto <- max(BANDWIDTHS)
auto_csv <- file.path(results_dir, sprintf("matched_sample_bw%d.csv", max_bw_auto))

auto_results <- list()
auto_idx <- 0

# Run rdrobust with the automatically selected MSE-optimal bandwidth
if (!file.exists(auto_csv)) {
    cat(sprintf("[SKIP] auto-bw — file not found: %s\n", auto_csv))
} else {
    df_auto <- read.csv(auto_csv, stringsAsFactors = FALSE)
    y_auto <- df_auto$dayuserchgw
    x_auto <- df_auto$mktcap1_300
    avail_covs_auto <- intersect(COVARIATES, colnames(df_auto))
    covs_auto <- as.matrix(df_auto[, avail_covs_auto, drop = FALSE])
    complete_auto <- complete.cases(y_auto, x_auto, covs_auto)
    y_auto <- y_auto[complete_auto]
    x_auto <- x_auto[complete_auto]
    covs_auto <- covs_auto[complete_auto, , drop = FALSE]
    if (length(y_auto) >= 50) {
        cat("Running rdrobust with MSE-optimal bandwidth (p=1) ...\n")
        tryCatch({
            # rdrobust picks the MSE-optimal bandwidth
            fit_auto <- rdrobust(
                y      = y_auto,
                x      = x_auto,
                c      = 0,
                p      = 1,
                covs   = covs_auto,
                kernel = "triangular"
            )
            coef_auto   <- fit_auto$coef[1]
            se_auto     <- fit_auto$se[3]
            pval_auto   <- fit_auto$pv[3]
            ci_lo_auto  <- fit_auto$ci[3, 1]
            ci_hi_auto  <- fit_auto$ci[3, 2]
            ci_w_auto   <- ci_hi_auto - ci_lo_auto
            t_auto      <- coef_auto / se_auto
            n_auto      <- fit_auto$N_h[1] + fit_auto$N_h[2]
            h_mse       <- fit_auto$bws[1, 1]
            auto_idx <- auto_idx + 1
            auto_results[[auto_idx]] <- data.frame(
                bandwidth      = h_mse,
                poly_order     = 1,
                method         = "rdrobust",
                beta1          = coef_auto,
                se             = se_auto,
                t_stat         = t_auto,
                pvalue         = pval_auto,
                ci_lower       = ci_lo_auto,
                ci_upper       = ci_hi_auto,
                ci_width       = ci_w_auto,
                n_obs          = n_auto,
                covariates_used = paste(avail_covs_auto, collapse = ", "),
                auto_bw        = h_mse,
                stringsAsFactors = FALSE
            )
        }, error = function(e) {
            cat(sprintf("FAILED — %s\n", conditionMessage(e)))
        })
    } else {
        cat("Too few observations, skipping auto-bw\n")
    }
}

# save auto-bw results
if (length(auto_results) > 0) {
    auto_df <- do.call(rbind, auto_results)
    rownames(auto_df) <- NULL
    auto_out <- file.path(results_dir, "results_rdrobust_auto_bw.csv")
    write.csv(auto_df, auto_out, row.names = FALSE)
    cat(sprintf("\nSaved auto-bw results to %s\n", auto_out))
} else {
    cat("\nNo auto-bw results produced.\n")
}

# use the largest bandwidth sample for maximum power
max_bw <- max(BANDWIDTHS)
cov_csv <- file.path(results_dir, sprintf("matched_sample_bw%d.csv", max_bw))

# Covariate continuity test
if (!file.exists(cov_csv)) {
    cat(sprintf("[SKIP] Covariate continuity test — file not found: %s\n", cov_csv))
} else {
    df_cov <- read.csv(cov_csv, stringsAsFactors = FALSE)
    x_cov  <- df_cov$mktcap1_300  # running variable
    cov_results <- list()
    cov_idx <- 0
    for (cov_name in COVARIATES) {
        if (!(cov_name %in% colnames(df_cov))) {
            cat(sprintf("  [SKIP] %s not in data\n", cov_name))
            next
        }
        y_cov <- df_cov[[cov_name]]
        ok <- complete.cases(y_cov, x_cov)
        y_sub <- y_cov[ok]
        x_sub <- x_cov[ok]
        tryCatch({
            # automatically choose MSE-optimal bandwidth
            fit_cov <- rdrobust(
                y    = y_sub,
                x    = x_sub,
                c    = 0,
                p    = 1,
                kernel = "triangular"
            )
            rd_est    <- fit_cov$coef[1]
            rob_pval  <- fit_cov$pv[3]
            rob_ci_lo <- fit_cov$ci[3, 1]
            rob_ci_hi <- fit_cov$ci[3, 2]
            bw_h      <- fit_cov$bws[1, 1]
            eff_n     <- fit_cov$N_h[1] + fit_cov$N_h[2]

            sig <- ifelse(rob_pval < 0.05, " ***", "")
            cat(sprintf("RD=%.4f  p=%.3f  CI=[%.3f, %.3f]  h=%.1f  N_eff=%d%s\n",
                        rd_est, rob_pval, rob_ci_lo, rob_ci_hi, bw_h, eff_n, sig))

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
        print(cov_df)
    }
}
