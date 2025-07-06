#!/usr/bin/env python3

import requests
import json
import time

# 测试向aggregator发送数据
def test_aggregator_communication():
    port = 20003
    base_url = f"http://localhost:{port}"
    
    print("测试aggregator通信...")
    
    # 1. 检查aggregator是否运行
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ Aggregator响应正常: {response.json()}")
        else:
            print(f"❌ Aggregator响应错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到aggregator: {e}")
        return False
    
    # 2. 测试发送数据
    test_data = {
        'instance': 0,  # 使用数字而不是字符串
        'episode_ndx': 999,  # 使用唯一的episode number
        'total_episodes': 4,
        'spend': 10.5,
        'task': 'ObjectNav',
        'task_data': {
            'goal_reached': True,
            'spl': 0.85,
            'distance_to_goal': 2.3
        },
        'spl': 0.85,
        'goal_reached': True
    }
    
    try:
        print(f"发送测试数据: {json.dumps(test_data, indent=2)}")
        response = requests.post(f"{base_url}/log", json=test_data, timeout=5)
        if response.status_code == 200:
            print(f"✅ 数据发送成功: {response.json()}")
        else:
            print(f"❌ 数据发送失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ 发送数据时出错: {e}")
        return False
    
    # 3. 再次检查状态
    time.sleep(1)
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ 发送后的状态: {json.dumps(status, indent=2)}")
            if status.get('episodes_completed', 0) > 0:
                print(f"✅ Episodes计数已更新: {status['episodes_completed']}")
                return True
            else:
                print("❌ Episodes计数未更新")
                return False
        else:
            print(f"❌ 无法获取更新后的状态: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取状态时出错: {e}")
        return False

if __name__ == "__main__":
    success = test_aggregator_communication()
    print(f"\n测试结果: {'成功' if success else '失败'}")
