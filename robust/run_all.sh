#!/usr/bin/env bash
# Full robust pipeline. Safe to rerun; finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*"; }

N=$(wc -l < data/fires.csv); N=$((N - 1))
until [ "$(ls data/fires/*.npz 2>/dev/null | wc -l | tr -d ' ')" -ge "$N" ]; do
  if ! pgrep -f "python -m robust.fetch" >/dev/null; then
    log "fetch workers stopped early; retrying missing fires"
    python -m robust.fetch
  fi
  sleep 30
done
log "all $N fires downloaded"

[ -f results/metrics_baselines.csv ] || python -m robust.baselines > results/log_baselines.txt 2>&1 &
BASE=$!

for v in bands_idx bands; do
  if [ ! -f "results/metrics_unet_${v}.csv" ] || [ "$(python -c "import pandas as pd; print(pd.read_csv('results/metrics_unet_${v}.csv').fire.nunique())")" -lt "$N" ]; then
    log "training unet_${v}"
    python -m robust.unet --variant "$v" > "results/log_unet_${v}.txt" 2>&1 || log "unet_${v} FAILED"
  fi
done

wait $BASE || true
log "running report"
python -m robust.report > results/log_report.txt 2>&1 || log "report FAILED"
log "ALL DONE"
