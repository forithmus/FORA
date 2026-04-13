#!/bin/bash
#SBATCH --job-name=fora_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=450G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORA_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config.yaml"}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

echo "==================================================================================="
echo "FORA SFT Report Generation Training"
echo "==================================================================================="
echo "CONFIG:           $CONFIG"
echo "FORA_ROOT:        $FORA_ROOT"
echo "SLURM_JOB_ID:     $SLURM_JOB_ID"
echo "SLURM_NNODES:     $SLURM_NNODES"
echo "GPUS_PER_NODE:    $GPUS_PER_NODE"
echo "Running on host:  $(hostname)"
echo "Start time:       $(date)"
echo "==================================================================================="

# Environment — activate your virtualenv or conda environment here
# source "${FORA_ROOT}/venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}:${FORA_ROOT}/rad_rate:${FORA_ROOT}/vision_encoder:${FORA_ROOT}/scripts:$PYTHONPATH"

cd "${SCRIPT_DIR}"
mkdir -p logs

# Distributed setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP: $head_node_ip"

export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Launch
srun -N $SLURM_NNODES -n $SLURM_NNODES bash -c "
accelerate launch \
    --multi_gpu \
    --num_machines=$SLURM_NNODES \
    --num_processes=$((SLURM_NNODES * $GPUS_PER_NODE)) \
    --mixed_precision bf16 \
    --machine_rank=\$SLURM_PROCID \
    --main_process_ip=$head_node_ip \
    --main_process_port=29500 \
    train.py --config $CONFIG
"

echo "==================================================================================="
echo "Job finished with exit code $?"
echo "End time: $(date)"
echo "==================================================================================="
