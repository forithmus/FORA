#!/usr/bin/env bash
# Per-node launcher of the CT writer trainer (SFT or DPO). Run once on every node of the allocation, e.g. from
# `srun --ntasks-per-node=1 bash reportgen/ct/run_node.sh <config.json> [extra trainer flags]`.
#
# Required environment:
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE (4)
#   DENSE_NPZ      frozen classifier logits of all studies (.npz)        -> node-local mmap cache
#   FINDING_NPZ    finding store (.npz)                                   -> node-local mmap cache
#   LOCAL_CACHE    node-local scratch directory for both caches
#   DINO_CACHE_ROOT root of the staged node-local block-23 caches
# Optional (DPO): PAIRS_SQLITE (global pair store) -> node-local copy holding only this node's train reports.
# The node's block-23 DINO cache must already be staged (see data-preprocessing/).
set -euo pipefail
CONFIG=${1:?config json}; shift
STAGE=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
: "${NNODES:?}" "${NODE_RANK:?}" "${DINO_CACHE_ROOT:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${DENSE_NPZ:?}" "${FINDING_NPZ:?}" "${LOCAL_CACHE:?}"
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

dense_root="$LOCAL_CACHE/dense_$(basename "$DENSE_NPZ" .npz)"
finding_root="$LOCAL_CACHE/finding_$(basename "$FINDING_NPZ" .npz)"
mkdir -p "$LOCAL_CACHE"
(
  flock 9
  python3 -m fora_report.writer.dense_cache --source "$DENSE_NPZ" --output "$dense_root"
  python3 -m fora_report.writer.dense_cache --kind loc --source "$FINDING_NPZ" --output "$finding_root"
) 9>"$LOCAL_CACHE/.dense_cache.lock"

extra=()
if [[ -n "${PAIRS_SQLITE:-}" ]]; then
  : "${MIL_CONFIG:?MIL_CONFIG (classifier config) is needed for the node-local pair filter}" "${INDEX:?}"
  pairs_local="$LOCAL_CACHE/dpo_pairs_node_local.sqlite"
  ( flock 9; python3 -m fora_report.ct.dpo_pairs_local --source "$PAIRS_SQLITE" --output "$pairs_local" \
      --mil-config "$MIL_CONFIG" --index "$INDEX" --cache-root "${DINO_CACHE_ROOT:?}" \
  ) 9>"$pairs_local.lock"
  extra+=(--dpo_pairs "$pairs_local")
fi

exec python3 -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$GPUS_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m fora_report.ct.train --config "$CONFIG" \
  --dino_cache_root "$DINO_CACHE_ROOT" --precomputed_mil_predictions "$dense_root" --finding_store "$finding_root" \
  "${extra[@]}" "$@"
