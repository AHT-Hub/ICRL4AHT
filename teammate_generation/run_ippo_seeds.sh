#!/bin/bash
# Script to run multiple IPPO training tasks in series with different seeds
# Usage: ./scripts/run_ippo_seeds.sh --layout <layout> --num_chunks <num> --seeds <start>-<end> --output_dir <dir> --gpu <gpu_id>
# Example: ./scripts/run_ippo_seeds.sh --layout grounded_coord_simple --num_chunks 15 --seeds 0-4 --output_dir mate_rl_models/ippo_grounded_coord_simple --gpu 0

set -e  # Exit on error

# Default values
LAYOUT="grounded_coord_simple"
NUM_CHUNKS=15
SEEDS="0-9"  # Default: seeds 0 through 9
OUTPUT_DIR_BASE=""  # Base output dir, seed will be appended
GPU=0
CONDA_ENV="aht"  # Default conda environment
EXTRA_ARGS=""
MY_HOME="/data1/jingyuheng"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layout)
            LAYOUT="$2"
            shift 2
            ;;
        --num_chunks)
            NUM_CHUNKS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR_BASE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --extra)
            # Capture all remaining arguments as extra args
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --layout      Layout name (default: grounded_coord_simple)"
            echo "  --num_chunks  Number of chunks (default: 15)"
            echo "  --seeds       Seed range, e.g., '0-4' or comma-separated '0,1,2,5' (default: 0-2)"
            echo "  --output_dir  Base output directory, _seed<N> will be appended (default: mate_rl_models/ippo_<layout>)"
            echo "  --gpu         GPU ID (default: 0)"
            echo "  --conda_env   Conda environment name (default: aht)"
            echo "  --extra       Extra arguments to pass to the training script (must be last)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --layout grounded_coord_simple --num_chunks 15 --seeds 0-4 --gpu 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default output_dir_base if not specified
if [ -z "$OUTPUT_DIR_BASE" ]; then
    OUTPUT_DIR_BASE="mate_rl_models/ippo_${LAYOUT}"
fi

# Parse seeds (supports both range "0-4" and comma-separated "0,1,2,5")
parse_seeds() {
    local seed_spec="$1"
    local seeds=()

    if [[ "$seed_spec" == *-* ]]; then
        # Range format: "0-4"
        local start=$(echo "$seed_spec" | cut -d'-' -f1)
        local end=$(echo "$seed_spec" | cut -d'-' -f2)
        for ((i=start; i<=end; i++)); do
            seeds+=($i)
        done
    elif [[ "$seed_spec" == *,* ]]; then
        # Comma-separated format: "0,1,2,5"
        IFS=',' read -ra seeds <<< "$seed_spec"
    else
        # Single seed
        seeds=($seed_spec)
    fi

    echo "${seeds[@]}"
}

SEED_LIST=($(parse_seeds "$SEEDS"))

echo "========================================"
echo "IPPO Training - Multiple Seeds"
echo "========================================"
echo "Layout:      $LAYOUT"
echo "Num chunks:  $NUM_CHUNKS"
echo "Seeds:       ${SEED_LIST[*]}"
echo "Output base: $OUTPUT_DIR_BASE"
echo "GPU:         $GPU"
echo "Conda env:   $CONDA_ENV"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args:  $EXTRA_ARGS"
fi
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source conda to enable conda activate in script
# Try common conda locations
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$MY_HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$MY_HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh, trying conda activate directly..."
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Run training for each seed
for SEED in "${SEED_LIST[@]}"; do
    OUTPUT_DIR="${OUTPUT_DIR_BASE}_seed${SEED}"
    echo "========================================"
    echo "Starting training with seed=$SEED"
    echo "Output dir: $OUTPUT_DIR"
    echo "Time: $(date)"
    echo "========================================"

    unset LD_LIBRARY_PATH
    python "$SCRIPT_DIR/train_ippo_overcooked_v2.py" \
        --layout "$LAYOUT" \
        --num_chunks "$NUM_CHUNKS" \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" \
        --gpu "$GPU" \
        $EXTRA_ARGS

    echo ""
    echo "Completed training with seed=$SEED"
    echo ""
done

echo "========================================"
echo "All training runs completed!"
echo "Time: $(date)"
echo "========================================"
