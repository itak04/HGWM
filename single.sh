#!/bin/bash
ROOT_DIR=/home/ps/dqf/GoalNav/WMNavigation
CONDA_PATH=/home/ps/anaconda3/etc/profile.d/conda.sh
VENV_NAME="wmnav"
PORT=2000
NAME="wmnav-qwen2_5vl-7B-hm3dv1"

# Start the aggregator
cd $ROOT_DIR
source $CONDA_PATH
conda activate $VENV_NAME

# Store the aggregator process ID
python scripts/aggregator.py --port $PORT --name $NAME --sleep 10 --config WMNav --project WMNav &
AGGREGATOR_PID=$!

# Wait for aggregator to start
sleep 5

# Run the evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/main.py --config WMNav -ms 20 -ne 20 --name $NAME --instances 10 --parallel -lf 1 --port $PORT --dataset hm3d_v0.1

# After evaluation completes, send termination signal and kill the aggregator
curl -X POST http://localhost:$PORT/terminate
wait $AGGREGATOR_PID || true