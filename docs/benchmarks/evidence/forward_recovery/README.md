# Evidence scope

`final/` records the final vector-store implementation. Earlier logs in this directory are diagnostic experiments, not final benchmark or validation results.

- `ablate_forward.py` uses the adjacent `frozen_pointer_forward.py` to compare runtime N with exact N at matched launch configuration. Run from the repository root with its source importable.
- `store_probe.py` and `check_store_target.py` were run against the TMA-store source at commit `5f1d93b`; `tma_store_5f1d93b.py` preserves that source for inspection. To reproduce that historical probe, use an isolated checkout of that commit. The probe generates a temporary pointer-store variant and overrides dispatch only within its process; benchmark source inventories alone do not identify that experimental override.
- `initcheck.log` contains uninitialized-read reports and was interrupted for investigation. Its original command metadata remained `running` because the parent driver interruption ended the recorder; see `initcheck-interruption.json`. This is not a successful check and no exit code is inferred.
- The final implementation uses ordinary vector output stores, retains TMA reads, and has its own complete validation records.
