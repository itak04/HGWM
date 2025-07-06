#!/bin/bash

# 测试单个worker连接到聚合器
# 这个脚本模拟一个worker实例

ROOT_DIR=/home/ps/dqf/GoalNav/WMNavigation
CONDA_PATH=/home/ps/anaconda3/etc/profile.d/conda.sh
VENV_NAME="wmnav"
CFG="WMNav"
NAME="test-worker"
PORT=20001
DATASET="hm3d_v0.1"

echo "测试单个worker连接到聚合器..."
echo "聚合器端口: ${PORT}"

# 检查聚合器是否运行
if ! curl -s http://localhost:${PORT}/status >/dev/null 2>&1; then
    echo "错误: 聚合器未在端口 ${PORT} 运行"
    echo "请先启动聚合器：python scripts/aggregator.py --name test --port ${PORT}"
    exit 1
fi

echo "✓ 聚合器正在运行"

# 启动单个worker实例进行测试
echo "启动测试worker..."

CMD="python scripts/main.py --config ${CFG} -ms 5 -ne 1 --name ${NAME} --instances 1 --parallel -lf 1 --port ${PORT} --dataset ${DATASET} --instance 0"

echo "执行命令: ${CMD}"

cd ${ROOT_DIR}
source ${CONDA_PATH}
conda activate ${VENV_NAME}

# 设置环境变量
export HTTPS_PROXY=socks5://127.0.0.1:7897
export HTTP_PROXY=socks5://127.0.0.1:7897

# 运行测试
${CMD}
