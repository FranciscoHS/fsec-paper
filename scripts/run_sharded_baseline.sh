#!/usr/bin/env bash
# Run one shard of the baseline-strengthening GPU sweeps on a pod that mounts
# the shared /workspace volume. Usage: run_sharded_baseline.sh i/N
#
# Item 2 (random-diff, 528 pairs) is sharded over PAIRS (1 model load/shard).
# Item 1 (misalignment, 28 caches x 40 pairs) is sharded over CACHES, so each
# pod loads the model once per cache it owns (not once per pair) — 28/N loads.
# All sweeps are --skip_existing and write atomically to the shared results
# dir, so shards never collide and the run is fully resumable.
set -uo pipefail
SHARD="${1:?usage: run_sharded_baseline.sh i/N}"
I="${SHARD%/*}"; N="${SHARD#*/}"
cd /workspace/fsec-anonymous
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 MPLCONFIGDIR=/workspace/.mplconfig
DIR=results/directions
PAIRS="$DIR/rotcontrastive_pairs_gemma_L2_ov0p1.txt"
SW="python -u scripts/sweep_2d.py --target gemma --layer 2 --skip_existing"

echo "### shard $SHARD starting $(date -u +%H:%M:%S)"

# --- Item 2: random-diff null, sharded over the 528 pairs ---
$SW --directions_pkl "$DIR/dirs_gemma_L2_randomdiff_fineweb.pkl" --shard "$SHARD"

# --- Item 1: misalignment, sharded over the 28 rotated caches ---
CACHES=()
for th in 00 15 30 45 60 75 90; do
  for s in 0 1 2 3; do
    CACHES+=("$DIR/dirs_gemma_L2_rotcontrastive_th${th}_s${s}.pkl")
  done
done
for ((k=I; k<${#CACHES[@]}; k+=N)); do
  echo "### cache $k: $(basename "${CACHES[$k]}")"
  $SW --directions_pkl "${CACHES[$k]}" --pairs_file "$PAIRS"
done

echo "### SHARD $SHARD DONE $(date -u +%H:%M:%S)"
