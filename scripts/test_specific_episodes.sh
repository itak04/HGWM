#!/bin/bash

# 测试特定剧集的便捷脚本
# 使用方法: bash scripts/test_specific_episodes.sh

ROOT_DIR=/home/ps/dqf/GoalNav/WMNavigation
CONDA_PATH=/home/ps/anaconda3/etc/profile.d/conda.sh
VENV_NAME="wmnav"
CONFIG="CoTGNav"

# 要测试的剧集ID
EPISODES=(994 964)

echo "🎯 开始测试特定剧集..."
echo "测试剧集: ${EPISODES[@]}"

# 激活conda环境
source ${CONDA_PATH}
conda activate ${VENV_NAME}
cd ${ROOT_DIR}

# 为每个剧集创建单独的测试
for episode in "${EPISODES[@]}"; do
    echo "========================================"
    echo "🧪 测试剧集 ${episode}"
    echo "========================================"
    
    # 创建专门的日志目录
    LOG_DIR="logs/test_episode_${episode}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p ${LOG_DIR}
    
    # 运行测试
    python scripts/main.py \
        --config ${CONFIG} \
        --name "test_episode_${episode}" \
        -ne 1 \
        -ms 500 \
        -lf 1 \
        --dataset hm3d_v0.1 \
        2>&1 | tee "${LOG_DIR}/test_output.log"
    
    echo "✅ 剧集 ${episode} 测试完成"
    echo "📝 日志保存在: ${LOG_DIR}"
    echo ""
done

echo "🎉 所有测试完成！"
echo "💡 检查日志文件中的 'GOAL_SUBGRAPH_RESPONSE' 部分获取目标子图信息"
