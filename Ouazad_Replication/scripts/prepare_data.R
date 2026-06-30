# Converts the large RDS estimation sample to a CSV with only needed columns

# install tidyverse if not available
if (!requireNamespace("tidyverse", quietly = TRUE)) {
    install.packages("tidyverse", repos = "https://cloud.r-project.org")
}
library(tidyverse)

# paths
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
}
repo_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
data_dir <- file.path(repo_dir, "data")
figures_dir <- file.path(repo_dir, "figures")
results_dir <- file.path(repo_dir, "results")

dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

in_rds <- file.path(data_dir, "est_sample_for_revision.rds")
out_csv <- file.path(data_dir, "est_sample.csv")

if (!file.exists(in_rds)) {
  stop(paste0("Missing input file: ", in_rds))
}

cat("Reading RDS...\n")
est_sample <- readRDS(in_rds)

# keep only columns needed for replication
keep_cols <- c(
  "year", "time", "treated", "below_limit", "above_limit",
  "diff_log_loan_amount", "log_loan_amount",
  "ZCTA5CE10", "name_event", "year_event",
  "action.type", "loan.purpose", "occupancy",
  "approved", "originated", "securitized",
  "applicant.income", "agency",
  "effective_loanlimit", "county.fips"
)

# add time dummies
time_cols <- grep("^time_", names(est_sample), value = TRUE)
year_cols <- grep("^year_", names(est_sample), value = TRUE)
keep_cols <- unique(c(keep_cols, time_cols, year_cols))

# filter to available columns
keep_cols <- intersect(keep_cols, names(est_sample))

cat(sprintf("Subsetting %d rows x %d cols -> %d cols\n",
            nrow(est_sample), ncol(est_sample), length(keep_cols)))

est_sub <- est_sample[, keep_cols]

# apply filter to time window of interest
est_sub <- est_sub %>%
  filter(time >= -10 & time <= 4)

# write CSV
cat("Writing CSV...\n")
write_csv(est_sub, out_csv)            