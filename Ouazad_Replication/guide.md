**Guide, on how to use the scripts and data for the Ouazad \& Kahn (2023) Replication.**

## Execution Order

### Step 1: Prepare data

```bash
Rscript scripts/prepare_data.R
```

Converts the RDS estimation sample to `data/est_sample.csv` with only the columns needed for replication.

### Step 2: Run main estimation

```bash
python scripts/main_ouazad.py
```

### Step 3: Run rdrobust estimation

```bash
Rscript scripts/rdrobust_ouazad.R
```

### Regenerating figures only

To regenerate all figures from cached result CSVs without re-running estimation:

```bash
python scripts/main_ouazad.py --plots-only
```