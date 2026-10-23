# Recovered April pilot schedules

These five CSVs are preserved byte-for-byte from commit
`c8dda7ddb351669bb8e2f08c3cc95117e9d5bb29`, formerly the tip of
`susan/main-pull-local-changes`. They contain 11 named UTT pilot runs with 60
requested trials across `no_query_trusted_distinguisher`,
`50_50_trusted_distinguisher`, and `yes_query_trusted_distinguisher`.

The maintained batch launcher accepts these schedules. For example:

```bash
python run_experiment_batch.py --csv configs/historical/0420_experiment_2.csv --output-dir results/utt-pilot
```

This command makes paid model calls. These files are partial historical launch
configurations, not evidence that the runs completed and not the paper's full
UTT reproduction manifest. Routes resolve through the current model registry;
inspect deviations from the recorded environment before comparing results.

Other historical records and cluster-specific scripts remain recoverable as
described in the [branch consolidation record](../../docs/history/branch-consolidation-2026-10-22.md).
