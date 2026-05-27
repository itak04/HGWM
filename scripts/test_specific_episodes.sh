#!/bin/bash
# Convenience script for running HGWM on a fixed list of episodes.
# Usage: bash scripts/test_specific_episodes.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VENV_NAME="hgwm"
CONFIG="HGWM"

# Episode ids to evaluate
EPISODES=(994 964)

echo "Running HGWM on episodes: ${EPISODES[@]}"

source ${CONDA_PATH}
conda activate ${VENV_NAME}
cd ${ROOT_DIR}

for episode in "${EPISODES[@]}"; do
    echo "===================================="
    echo "Episode ${episode}"
    echo "===================================="

    LOG_DIR="logs/test_episode_${episode}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p ${LOG_DIR}

    python scripts/main.py \
        --config ${CONFIG} \
        --name "test_episode_${episode}" \
        -ne 1 \
        -ms 500 \
        -lf 1 \
        --dataset hm3d_v0.1 \
        2>&1 | tee "${LOG_DIR}/test_output.log"

    echo "Episode ${episode} done. Logs in ${LOG_DIR}"
done

echo "All requested episodes finished."
