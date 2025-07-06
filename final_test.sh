#!/bin/bash

# Quick test configuration for debugging - FINAL VERSION
ROOT_DIR=/home/ps/dqf/GoalNav/WMNavigation
CONDA_PATH=/home/ps/anaconda3/etc/profile.d/conda.sh
NUM_GPU=1
INSTANCES=2  # Reduced for testing
NUM_EPISODES_PER_INSTANCE=2  # Much smaller
MAX_STEPS_PER_EPISODE=3  # Even smaller for faster testing
TASK="ObjectNav"
DATASET="hm3d_v0.1"
CFG="WMNav"
NAME="wmnav-final-test"
PROJECT_NAME="WMNav"
VENV_NAME="wmnav"
GPU_LIST=(0)
SLEEP_INTERVAL=20  # Shorter interval
LOG_FREQ=1
PORT=20003  # Use the simple aggregator port

# 只为 wandb 设置代理
export WANDB_HTTP_PROXY=socks5://127.0.0.1:7897
export WANDB_HTTPS_PROXY=socks5://127.0.0.1:7897
unset HTTP_PROXY
unset HTTPS_PROXY
unset NO_PROXY

CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --dataset ${DATASET}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="simple_aggregator"

# 检查端口是否已被占用
if ! curl -s http://localhost:${PORT}/status >/dev/null 2>&1; then
  echo "错误: 简化版aggregator未运行在端口 ${PORT}"
  echo "请先运行: python simple_aggregator.py --name test-simple --port ${PORT}"
  exit 1
fi

echo "✅ 简化版aggregator运行正常"

# Cleanup Function
cleanup() {
  echo -e "\n捕获到中断信号。正在清理..."
  
  # 首先尝试优雅关闭聚合器
  echo "发送终止信号给aggregator..."
  if curl -X POST http://localhost:${PORT}/terminate 2>/dev/null; then
    echo "终止信号发送成功"
    sleep 2
  else
    echo "无法发送终止信号"
  fi

  # 清理所有worker会话
  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "已终止会话: $SESSION_NAME"
    fi
  done
}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

# Start Tmux Sessions for Each Instance
echo "启动worker会话..."
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  GPU_ID=${GPU_LIST[$((instance_id % ${#GPU_LIST[@]}))]}
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"

  tmux new-session -d -s "$SESSION_NAME"
  tmux send-keys -t $SESSION_NAME "source ${CONDA_PATH} && conda activate ${VENV_NAME} && cd ${ROOT_DIR} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id" C-m
  SESSION_NAMES+=("$SESSION_NAME")
  echo "启动了worker会话: $SESSION_NAME (instance $instance_id)"
done

# Enhanced Monitor Loop with Fixed Detection Logic
echo "开始监控实例..."
echo "预期总运行时间: 大约 $((NUM_EPISODES_PER_INSTANCE * MAX_STEPS_PER_EPISODE * 25 / 60)) 分钟"

MONITOR_COUNT=0
CONSECUTIVE_ALL_DONE=0  # 连续检测到完成的次数

while true; do
  sleep $SLEEP_INTERVAL
  MONITOR_COUNT=$((MONITOR_COUNT + 1))

  ALL_DONE=true
  ACTIVE_INSTANCES=0

  # 检查聚合器健康状态并显示状态
  if curl -s http://localhost:${PORT}/status >/dev/null 2>&1; then
    AGGREGATOR_STATUS=$(curl -s http://localhost:${PORT}/status | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Episodes: {data.get(\"episodes_completed\", 0)}/{data.get(\"total_episodes\", \"?\")}, Instances: {data.get(\"instances_connected\", 0)}, Success Rate: {data.get(\"success_rate\", 0):.2f}')
except Exception as e:
    print('解析状态失败')
" 2>/dev/null)
    echo "Aggregator状态: $AGGREGATOR_STATUS"
  else
    echo "警告: aggregator不响应，可能已崩溃"
  fi

  # 检查每个worker实例的状态
  for instance_id in $(seq 0 $((INSTANCES - 1))); do
    SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      # 检查Python进程是否仍在运行
      last_line=$(tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -1)
      if echo "$last_line" | grep -q "(wmnav) ps@"; then
        # 会话显示shell提示符，意味着Python进程已完成
        if [ $MONITOR_COUNT -eq 1 ] || [ $((MONITOR_COUNT % 5)) -eq 0 ]; then
          echo "$SESSION_NAME 已完成 (Python进程结束)"
        fi
      else
        # Python进程仍在运行
        ALL_DONE=false
        ACTIVE_INSTANCES=$((ACTIVE_INSTANCES + 1))
      fi
    else
      if [ $MONITOR_COUNT -eq 1 ] || [ $((MONITOR_COUNT % 5)) -eq 0 ]; then
        echo "$SESSION_NAME 已完成 (会话不存在)"
      fi
    fi
  done

  echo "$(date '+%H:%M:%S'): 活跃实例: $ACTIVE_INSTANCES/$INSTANCES (检查次数: $MONITOR_COUNT)"

  if $ALL_DONE; then
    CONSECUTIVE_ALL_DONE=$((CONSECUTIVE_ALL_DONE + 1))
    echo "所有实例完成检测次数: $CONSECUTIVE_ALL_DONE"
    
    # 需要连续2次检测到完成才确认完成（避免假阳性）
    if [ $CONSECUTIVE_ALL_DONE -ge 2 ]; then
      echo "✅ 确认所有实例完成！"
      
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
    print(f'Instances Connected: {data.get(\"instances_connected\", 0)}')
except Exception as e:
    print(f'解析结果失败: {e}')
" 2>/dev/null || echo "无法解析结果"
        echo "=========================="
      else
        echo "$(date): 无法获取聚合器状态"
      fi
      
      # 清理worker会话
      for instance_id in $(seq 0 $((INSTANCES - 1))); do
        SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
          tmux kill-session -t "$SESSION_NAME"
          echo "已清理worker会话: $SESSION_NAME"
        fi
      done
      
      break
    fi
  else
    CONSECUTIVE_ALL_DONE=0  # 重置计数器
  fi
  
  # 防止无限循环：如果监控超过一定次数，强制退出
  if [ $MONITOR_COUNT -gt 30 ]; then
    echo "警告: 监控次数超过限制，强制退出"
    cleanup
    break
  fi
done

echo "✅ 脚本执行完成"
