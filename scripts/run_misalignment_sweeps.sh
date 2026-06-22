#!/usr/bin/env bash
# Item 1 (LLM misalignment sweep): re-sweep the rotated contrastive pairs at
# every tilt theta x ensemble seed. Each cell is the 40-pair overlap-filtered
# subsample (rotcontrastive_pairs_*.txt) swept on the rotated direction cache
# for that (theta, seed). ~40 pairs x 7 theta x 4 seeds ~= 1.1k sweeps.
#
# Prereq: scripts/rotated_contrastive_directions.py has written the 28 rotated
# caches + the pairs file (CPU, cheap). Run THIS from repo root on a GPU pod.
# Resumable via --skip_existing.
#
# After this finishes:
#   python scripts/fit_misalignment_thrfixed.py --target gemma --layer 2
#   python -m scripts.plotting.plot_misalignment_llm --target gemma --layer 2
set -euo pipefail

TARGET=gemma
LAYER=2
DIR=results/directions
PAIRS="$DIR/rotcontrastive_pairs_${TARGET}_L${LAYER}_ov0p1.txt"

THETAS="00 15 30 45 60 75 90"
SEEDS="0 1 2 3"

if [[ ! -f "$PAIRS" ]]; then
  echo "missing pairs file $PAIRS; run rotated_contrastive_directions.py first" >&2
  exit 1
fi

for th in $THETAS; do
  for s in $SEEDS; do
    cache="$DIR/dirs_${TARGET}_L${LAYER}_rotcontrastive_th${th}_s${s}.pkl"
    if [[ ! -f "$cache" ]]; then
      echo "### skip th${th} s${s}: missing $cache" >&2
      continue
    fi
    echo "### theta=${th} seed=${s}"
    python scripts/sweep_2d.py --target "$TARGET" --layer "$LAYER" \
        --directions_pkl "$cache" \
        --pairs_file "$PAIRS" \
        --skip_existing
  done
done

echo "ALL MISALIGNMENT SWEEPS DONE"
