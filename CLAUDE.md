# Rules for every session in this repo

## Never keep work only on the cloud machine
The cloud machine is temporary: its disk (scratchpad, caches, /tmp) is wiped when the session restarts.
- Backtest results (summary tables, trade lists, charts) are committed to `results/` and pushed as soon as each run finishes.
- Downloaded market data (EODHD / Databento / Alpaca) is copied to permanent storage as soon as it is downloaded:
  the Supabase project "Spread Staging" (tables or a storage bucket), never only the local cache.
- Before starting a new download, check that permanent storage for data already saved.
- Code the user pastes into PythonAnywhere is delivered on a private copy page (artifact) with their keys;
  the public repo never holds keys.
