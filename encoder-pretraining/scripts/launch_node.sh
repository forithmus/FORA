#!/usr/bin/env bash
# One task per node: starts 4 training ranks of fora_pretrain.train.
# usage: launch_node.sh NNODES NODE_RANK MASTER_ADDR MASTER_PORT OUTPUT_DIR ARGS_FILE [extra trainer args...]
# Environment: FORA_H5 (H5 folder), FORA_SPLITS (split CSV), optional FORA_STAGING (node-local dir), FORA_COUNTS (CT
# reconstruction-count cache). PYTHONPATH must contain the repository root and the dinov3 checkout.
set -euo pipefail
nnodes=${1:?}; node_rank=${2:?}; master=${3:?}; port=${4:?}; out=${5:?}; argsfile=${6:?}; shift 6
here=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$out"
# shellcheck disable=SC2046
exec python3 -m torch.distributed.run --nnodes="$nnodes" --nproc-per-node=4 --node-rank="$node_rank" \
  --master-addr="$master" --master-port="$port" -m fora_pretrain.train --output-dir "$out" \
  --h5-folder "${FORA_H5:?}" --splits-csv "${FORA_SPLITS:?}" --staging-dir "${FORA_STAGING:-}" \
  --assignment-cache "${FORA_COUNTS:-}" $(grep -v '^#' "$argsfile") "$@"
