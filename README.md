# WMNavigation - Vision-Language Model Navigation System

该项目实现了基于视觉语言模型的室内物体导航系统，使用 Qwen2.5-VL-7B-Instruct 模型通过 SiliconFlow API 进行物体导航任务。

## 核心功能

- **物体导航 (ObjectNav)**: 在 HM3D 环境中导航到指定目标物体
- **VLM 集成**: 使用 SiliconFlow API 调用 Qwen2.5-VL-7B-Instruct 模型
- **并行评估**: 支持多实例并行运行以提高评估效率
- **完整指标**: 计算 Success Rate (SR) 和 SPL (Success Rate Weighted by Inverse Path Length)

## 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate wmnav

# 设置API密钥
export SILICONFLOW_API_KEY="your_api_key_here"
```

### 2. 测试VLM连接
```bash
python3 test_vlm_image.py
```

### 3. 运行完整评估
```bash
./parallel.sh
```

### 4. 查看结果
```bash
# 查看快速摘要
python3 show_results.py

# 查看详细分析
python3 analyze_navigation_metrics.py

# 查看完整报告
cat EVALUATION_REPORT.md
```

## 核心文件说明

### 评估脚本
- `parallel.sh`: 主要评估脚本，启动多个并行实例
- `analyze_navigation_metrics.py`: 分析脚本，计算 SR 和 SPL 指标
- `show_results.py`: 快速结果摘要显示

### 测试脚本
- `test_vlm_image.py`: VLM 图像传输功能测试

### 配置和源码
- `config/WMNav.yaml`: 系统配置文件
- `src/api.py`: SiliconFlow API 集成
- `src/WMNav_agent.py`: 导航智能体实现
- `src/WMNav_env.py`: 环境包装器

### 结果文件
- `navigation_evaluation_results.json`: 完整的评估结果数据
- `EVALUATION_REPORT.md`: 详细的评估报告
- `logs/`: 所有实例的执行日志

## 评估指标

### Success Rate (SR)
成功完成导航任务的回合百分比：
```
SR = (成功回合数 / 总回合数) × 100%
```

### SPL (Success Rate Weighted by Inverse Path Length)
考虑路径效率的成功率：
```
SPL = (1/N) × Σ(Success_i × shortest_path_i / max(actual_path_i, shortest_path_i))
```

## 最新评估结果

基于 434 个评估回合的结果：

- **Success Rate**: 52.8%
- **SPL**: 0.365
- **平均步数**: 12.6 (所有回合), 8.9 (成功回合)

### 按目标对象分析
| 目标对象 | 成功率 | SPL   | 回合数 |
|---------|--------|-------|--------|
| Sofa    | 61.3%  | 0.441 | 80     |
| Bed     | 56.8%  | 0.425 | 88     |
| Toilet  | 54.1%  | 0.391 | 74     |
| TV      | 52.1%  | 0.309 | 73     |
| Chair   | 48.2%  | 0.310 | 83     |
| Plant   | 33.3%  | 0.238 | 36     |

## 技术特性

### API 集成
- 支持 SOCKS5 代理环境
- 自动处理代理设置以确保 API 调用正常
- 错误处理和重试机制

### 并行处理
- 多 GPU 支持
- tmux 会话管理
- 自动结果聚合

### 鲁棒性
- 日志备份分析（当聚合器失败时）
- 完整的错误处理
- 可恢复的评估流程

## 开发说明

### 添加新模型
在 `src/api.py` 中添加新的 VLM 类，遵循现有的接口：
```python
class NewVLM:
    def call_chat(self, image, text_prompt):
        # 实现调用逻辑
        pass
```

### 修改评估参数
编辑 `parallel.sh` 中的配置变量：
```bash
INSTANCES=10              # 并行实例数
NUM_EPISODES_PER_INSTANCE=20  # 每实例回合数
MAX_STEPS_PER_EPISODE=20  # 最大步数
```

### 自定义分析
`analyze_navigation_metrics.py` 提供了完整的分析框架，可以轻松添加新的指标计算。

## 故障排除

### API 连接问题
1. 检查 `SILICONFLOW_API_KEY` 是否正确设置
2. 运行 `test_vlm_image.py` 测试连接
3. 检查代理设置是否影响 API 调用

### 评估中断
1. 检查 tmux 会话: `tmux list-sessions`
2. 查看日志文件确认已完成的回合
3. 运行 `analyze_navigation_metrics.py` 分析已有数据

### 性能优化
1. 调整 `SLEEP_INTERVAL` 减少监控频率
2. 增加 `INSTANCES` 提高并行度
3. 优化 VLM 提示词以提高响应速度

---

*项目基于 HM3D 数据集和 Habitat 模拟器*  
*使用 Qwen2.5-VL-7B-Instruct 通过 SiliconFlow API*
