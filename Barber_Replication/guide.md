**Guide, on how to use the scripts and data for the Barber, Huang, Odean (2022) Replication.**

## Execution Order

### Step 1: Run main estimation

```bash
python scripts/main_barber.py
```

### Step 2: Run rdrobust estimation

```bash
Rscript scripts/rdrobust_barber.R
```

### Regenerating figures only

To regenerate all figures from cached result CSVs without re-running estimation:

```bash
python scripts/main_barber.py --plots-only
```