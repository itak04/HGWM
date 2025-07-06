#!/bin/bash

# Configuration Variables (原始设计的简化版本)
ROOT_DIR=/home/ps/dqf/GoalNav/WMNavigation
CONDA_PATH=/home/ps/anaconda3/etc/profile.d/conda.sh
NUM_GPU=5
INSTANCES=50
NUM_EPISODES_PER_INSTANCE=40
MAX_STEPS_PER_EPISODE=40
TASK="ObjectNav"
DATASET="hm3d_v0.1"
CFG="WMNav"
NAME="wmnav-qwen2_5vl-7B-hm3dv1"
PROJECT_NAME="WMNav"
VENV_NAME="wmnav" # Name of the conda environment
GPU_LIST=(3 4 5 6 7) # List of GPU IDs to use
SLEEP_INTERVAL=200
LOG_FREQ=1
PORT=2000  # 聚合器端口
CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

# 检查端口是否已被占用
if lsof -i:${PORT} > /dev/null 2>&1; then
  echo "端口 ${PORT} 已被占用，请使用其他端口或关闭占用该端口的进程"
  exit 1
fi

# Start Aggregator Session
echo "启动聚合器在端口 ${PORT}..."
tmux new-session -d -s "$AGGREGATOR_SESSION"
tmux send-keys -t $AGGREGATOR_SESSION "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && python scripts/aggregator.py --name ${TASK}_${NAME} --project ${PROJECT_NAME} --sleep ${SLEEP_INTERVAL} --config ${CFG} --port ${PORT}" C-m
SESSION_NAMES+=("$AGGREGATOR_SESSION")

# 等待聚合器启动
echo "等待聚合器在端口 ${PORT} 启动..."
for i in {1..30}; do
  if curl -s http://localhost:${PORT}/status >/dev/null 2>&1; then
    echo "聚合器已就绪!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "警告: 聚合器启动超时，但继续执行..."
    echo "请检查端口 ${PORT} 是否可用以及聚合器是否正常运行"
  fi
  sleep 1
done

# Cleanup Function
cleanup() {
  echo -e "\n捕获到中断信号。清理tmux会话..."

  for session in "${SESSION_NAMES[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "已终止会话: $session"
    fi
  done
}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

# Start Tmux Sessions for Each Instance
echo "启动 ${INSTANCES} 个实例..."
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  GPU_ID=${GPU_LIST[$((instance_id % ${#GPU_LIST[@]}))]}
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME"
  tmux send-keys -t $SESSION_NAME "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id" C-m
  SESSION_NAMES+=("$SESSION_NAME")
done

echo "所有实例已启动，开始监控..."

# Monitor Tmux Sessions
while true; do
  sleep $SLEEP_INTERVAL

  ALL_DONE=true

  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "$SESSION_NAME 已完成"
    else
      ALL_DONE=false
    fi
  done

  if $ALL_DONE; then
    echo "所有实例完成"
    
    # 获取最终结果
    echo "$(date): 获取最终结果..."
    if curl -s http://localhost:${PORT}/status > /tmp/final_results.json 2>/dev/null; then
      echo "=== 最终评估结果 ==="
      cat /tmp/final_results.json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Episodes Completed: {data.get(\"episodes_completed\", 0)}')
    print(f'Success Rate: {data.get(\"success_rate\", 0):.3f}')
    print(f'SPL: {data.get(\"spl\", 0):.3f}')
    print(f'Total Spend: \${data.get(\"total_spend\", 0)}')
except Exception as e:
    print(f'Could not parse aggregator results: {e}')
"
      echo "=========================="
    else
      echo "$(date): 无法获取聚合器状态"
    fi
    
    # 发送终止信号（修复：使用正确的变量名）
    echo "$(date): 发送终止信号给聚合器..."
    if curl -X POST http://localhost:${PORT}/terminate 2>/dev/null; then
      echo "$(date): 终止信号发送成功"
    else
      echo "$(date): 无法发送终止信号到聚合器"
    fi

    sleep 10
    if tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
      tmux kill-session -t "$AGGREGATOR_SESSION"
      echo "已终止聚合器会话: $AGGREGATOR_SESSION"
    fi
    break  # 退出监控循环
  fi
done

echo "脚本执行完成"
