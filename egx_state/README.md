# EGX scanner state

The daily GitHub Action runs `egx_seasonality_scanner.py` inside this folder and commits these files back,
so each run remembers the previous one:

- `cycle_state.json` – the current cycle (stocks, entry prices, stops)
- `cycle_history.json` – every finished cycle
- `ticker_names_cache.json` – company names (refreshed weekly)
- `last_run.txt` – the date of the last successful run (a second start the same day does nothing)

**Before the first run**, copy `cycle_state.json` and `cycle_history.json` from PythonAnywhere into this
folder. Without them the first run starts a brand-new cycle.
