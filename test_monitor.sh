#!/bin/bash

# 测试修复后的监控逻辑
TASK="ObjectNav"
NAME="wmnav-quick-test"
INSTANCES=2
PORT=20002

echo "测试修复后的监控逻辑..."

ALL_DONE=true
ACTIVE_INSTANCES=0

echo "检查聚合器健康状态..."
if ! curl -s http://localhost:${PORT}/status >/dev/null 2>&1; then
  echo "警告: 聚合器不响应，可能已崩溃"
else
  echo "聚合器响应正常"
  curl -s http://localhost:${PORT}/status | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Episodes Completed: {data.get(\"episodes_completed\", 0)}')
    print(f'Instances Connected: {data.get(\"instances_connected\", 0)}')
    print(f'Total Episodes: {data.get(\"total_episodes\", 0)}')
except Exception as e:
    print(f'解析聚合器状态失败: {e}')
"
fi

echo -e "\n检查worker状态..."
for instance_id in $(seq 0 $((INSTANCES - 1))); do
  SESSION_NAME="${TASK}_${NAME}_${instance_id}/${INSTANCES}"
  echo "检查会话: $SESSION_NAME"
  
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    # Check if Python process is still running in this session
    if tmux capture-pane -t "$SESSION_NAME" -p | tail -1 | grep -q "(wmnav) ps@"; then
      # Session shows shell prompt, meaning Python process finished
      echo "$SESSION_NAME 已完成 (Python进程结束)"
    else
      # Python process still running
      echo "$SESSION_NAME 仍在运行"
      ALL_DONE=false
      ACTIVE_INSTANCES=$((ACTIVE_INSTANCES + 1))
    fi
  else
    echo "$SESSION_NAME 已完成 (会话不存在)"
  fi
done

echo -e "\n监控结果:"
echo "ALL_DONE: $ALL_DONE"
echo "ACTIVE_INSTANCES: $ACTIVE_INSTANCES/$INSTANCES"

if $ALL_DONE; then
  echo -e "\n✅ 所有实例已完成！"
  echo "获取最终结果..."
  
  if curl -s http://localhost:${PORT}/status > /tmp/final_results.json 2>/dev/null; then
    echo "=== 最终评估结果 ==="
    cat /tmp/final_results.json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Episodes Completed: {data.get(\"episodes_completed\", 0)}')
    print(f'Success Rate: {data.get(\"success_rate\", 0):.3f}' if 'success_rate' in data else 'Success Rate: N/A')
    print(f'SPL: {data.get(\"spl\", 0):.3f}' if 'spl' in data else 'SPL: N/A')
    print(f'Total Spend: \${data.get(\"total_spend\", 0)}')
except Exception as e:
    print(f'解析结果失败: {e}')
" 2>/dev/null || echo "无法解析结果"
    echo "=========================="
  else
    echo "无法获取聚合器状态"
  fi
  
  echo -e "\n现在可以安全地关闭监控脚本"
else
  echo -e "\n❌ 仍有 $ACTIVE_INSTANCES 个实例在运行"
fi
