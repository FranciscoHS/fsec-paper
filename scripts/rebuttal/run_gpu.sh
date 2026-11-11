#!/usr/bin/env bash
# Rebuttal GPU runs (E4 four-quadrant sweeps, E5 MLP-neuron baseline) on a
# single multi-GPU RunPod pod. Resumable: every sweep uses --skip_existing and
# writes atomically.
#
# Usage (from the repo root on the pod):
#   bash scripts/rebuttal/run_gpu.sh setup     # pip deps
#   bash scripts/rebuttal/run_gpu.sh warmup    # GPU 0: model + anchor caches,
#                                              #   MLP-neuron dirs, (+,+) repro check
#   bash scripts/rebuttal/run_gpu.sh shards    # one background shard per GPU
#   bash scripts/rebuttal/run_gpu.sh status    # progress counts
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/workspace/hf_cache FSEC_CACHE_DIR=/workspace \
       MPLCONFIGDIR=/workspace/.mplconfig TOKENIZERS_PARALLELISM=false
mkdir -p logs/rebuttal

DIR=results/directions
# venv inheriting the pod's system torch/CUDA (system pip is PEP 668-locked)
VENV=${VENV:-.venv}
PY=$VENV/bin/python
SW="$PY -u scripts/sweep_2d.py --target gemma --layer 2 --skip_existing"
NGPU=$(nvidia-smi -L | wc -l)

case "${1:-}" in
setup)
  test -d $VENV || python -m venv --system-site-packages $VENV
  $PY -m pip install -q transformers accelerate datasets huggingface_hub scipy matplotlib
  $PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
  { test -n "${HF_TOKEN:-}" || test -s $HF_HOME/token; } && echo "HF token available" || echo "WARNING: no HF token (Gemma is gated)"
  ;;
warmup)
  export CUDA_VISIBLE_DEVICES=0
  # (+,+) reproducibility check: also downloads the model and builds the
  # FineWeb anchor cache before any parallel shard touches it.
  $SW --directions_pkl $DIR/dirs_gemma_L2_signed.pkl \
      --pairs_file results/rebuttal/pairs_quadrants_check.txt \
      2>&1 | tee logs/rebuttal/warmup_check.log
  $PY -u scripts/rebuttal/mlp_neuron_directions.py \
      2>&1 | tee logs/rebuttal/warmup_mlpneuron.log
  ;;
shards)
  test -f $DIR/dirs_gemma_L2_mlpneuron.pkl || { echo "run warmup first"; exit 1; }
  for ((i=0; i<NGPU; i++)); do
    ( export CUDA_VISIBLE_DEVICES=$i
      $SW --directions_pkl $DIR/dirs_gemma_L2_signed.pkl \
          --pairs_file results/rebuttal/pairs_quadrants.txt --shard "$i/$NGPU"
      $SW --directions_pkl $DIR/dirs_gemma_L2_mlpneuron.pkl --shard "$i/$NGPU"
      echo "### SHARD $i/$NGPU DONE $(date -u +%H:%M:%S)"
    ) > logs/rebuttal/shard_$i.log 2>&1 &
  done
  echo "launched $NGPU shards; logs in logs/rebuttal/"
  ;;
status)
  echo "quadrants: $(ls results/sweeps_2d/*_dirsigned.pkl 2>/dev/null | wc -l) / $((954 + 8))"
  echo "mlpneuron: $(ls results/sweeps_2d/*_dirmlpneuron.pkl 2>/dev/null | wc -l) / 528"
  grep -h "DONE" logs/rebuttal/shard_*.log 2>/dev/null
  ;;
*) echo "usage: $0 {setup|warmup|shards|status}"; exit 1 ;;
esac
