#!/usr/bin/env python3
"""
聚合器测试脚本
用于测试聚合器是否正常工作
"""

import requests
import subprocess
import time
import json

def check_port_usage(port):
    """检查端口是否被占用"""
    try:
        result = subprocess.run(['lsof', '-i', f':{port}'], 
                              capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(f"✓ 端口 {port} 被占用:")
            print(result.stdout)
            return True
        else:
            print(f"✗ 端口 {port} 未被占用")
            return False
    except Exception as e:
        print(f"✗ 检查端口失败: {e}")
        return False

def check_tmux_sessions():
    """检查tmux会话"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], 
                              capture_output=True, text=True, timeout=10)
        aggregator_sessions = [line for line in result.stdout.split('\n') 
                             if 'aggregator' in line]
        if aggregator_sessions:
            print("✓ 聚合器tmux会话:")
            for session in aggregator_sessions:
                print(f"  {session}")
            return True
        else:
            print("✗ 没有找到聚合器tmux会话")
            return False
    except Exception as e:
        print(f"✗ 检查tmux会话失败: {e}")
        return False

def test_aggregator_connection(port):
    """测试聚合器连接"""
    print(f"\n=== 测试聚合器连接 (端口 {port}) ===")
    
    try:
        # 测试状态端点
        response = requests.get(f'http://localhost:{port}/status', timeout=5)
        print(f"✓ 聚合器状态响应: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  响应内容: {json.dumps(data, indent=2)}")
                return True
            except:
                print(f"  响应内容: {response.text}")
                return True
        else:
            print(f"  错误响应: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到聚合器 (Connection Error)")
        return False
    except requests.exceptions.Timeout:
        print("✗ 聚合器响应超时")
        return False
    except Exception as e:
        print(f"✗ 聚合器连接失败: {e}")
        return False

def test_aggregator_log_endpoint(port):
    """测试聚合器日志端点"""
    print(f"\n=== 测试聚合器日志端点 ===")
    
    # 构造测试数据
    test_data = {
        "instance": 999,
        "episode_ndx": 999,
        "total_episodes": 1000,
        "spend": 0.01,
        "task": "ObjectNav",
        "task_data": {
            "goal_reached": True,
            "spl": 0.5,
            "test": True
        }
    }
    
    try:
        response = requests.post(
            f'http://localhost:{port}/log',
            json=test_data,
            timeout=5
        )
        print(f"✓ 日志端点响应: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  响应内容: {json.dumps(data, indent=2)}")
                return True
            except:
                print(f"  响应内容: {response.text}")
                return True
        else:
            print(f"  错误响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 日志端点测试失败: {e}")
        return False

def main():
    port = 20001
    
    print("=" * 50)
    print("聚合器测试脚本")
    print("=" * 50)
    
    # 1. 检查端口
    port_in_use = check_port_usage(port)
    
    # 2. 检查tmux会话
    tmux_exists = check_tmux_sessions()
    
    # 3. 测试连接
    if port_in_use:
        connection_ok = test_aggregator_connection(port)
        
        if connection_ok:
            # 4. 测试日志端点
            log_ok = test_aggregator_log_endpoint(port)
            
            if log_ok:
                print("\n🎉 聚合器工作正常!")
            else:
                print("\n⚠️  聚合器连接正常，但日志端点有问题")
        else:
            print("\n❌ 聚合器无法连接")
    else:
        print("\n❌ 聚合器未启动或端口未被占用")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
