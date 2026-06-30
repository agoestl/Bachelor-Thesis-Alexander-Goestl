**Guide, on how to use the scripts and data for the Griffin \& Shams (2020) Replication.**

## Execution Order

### Step 1: Run main estimation

```bash
python scripts/main_griffin.py
```

### Step 2: Run rdrobust estimation

```bash
Rscript scripts/rdrobust_griffin.R
```

### Regenerating figures only

To regenerate all figures from cached result CSVs without re-running estimation:

```bash
python scripts/main_griffin.py --plots-only
```