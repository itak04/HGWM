#!/usr/bin/env python3

import threading
import time
import argparse
import json
import os

from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Threading components
lock = threading.Lock()
terminate_event = threading.Event()

# Aggregated metrics
episode_data = []  # Stores episode data
episodes_completed = set()  # Tracks completed episodes
cumulative_metrics = {'episodes_completed': 0}  # Tracks cumulative metrics for all episodes
total_episodes = [1]  # Total episodes to be completed
spend_per_instance = {}  # Stores spend per instance
task_log = {}  # Logs data for each task

@app.route('/status', methods=['GET'])
def health_check():
    """Health check endpoint"""
    with lock:
        # Calculate additional metrics
        success_rate = 0
        spl = 0
        total_spend = sum(spend_per_instance.values())
        
        if episode_data:
            successful_episodes = sum(1 for ep in episode_data if ep.get('goal_reached', False))
            success_rate = successful_episodes / len(episode_data) if episode_data else 0
            spl = sum(ep.get('spl', 0) for ep in episode_data) / len(episode_data) if episode_data else 0
        
        return jsonify({
            'status': 'healthy',
            'episodes_completed': cumulative_metrics.get('episodes_completed', 0),
            'instances_connected': len(spend_per_instance),
            'total_episodes': total_episodes[0],
            'success_rate': success_rate,
            'spl': spl,
            'total_spend': total_spend
        }), 200

@app.route('/terminate', methods=['POST'])
def terminate():
    """Endpoint to receive termination signal and shutdown the server."""
    with lock:
        print("Received termination signal.")
    terminate_event.set()
    return jsonify({'status': 'terminating'}), 200

@app.route('/log', methods=['POST'])
def log_metrics():
    """Endpoint to log metrics for each instance."""
    data = request.json
    required_keys = ['instance', 'episode_ndx', 'total_episodes', 'spend', 'task', 'task_data']
    
    print(f"收到数据: {json.dumps(data, indent=2)}")
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"缺少必需字段: {missing_keys}")
        return jsonify({
            'status': 'error',
            'message': f'Missing key(s): {", ".join(missing_keys)}'
        }), 400

    # Log the data within a lock to ensure thread safety
    with lock:
        instance = data['instance']
        spend_per_instance[instance] = data['spend']
        total_episodes[0] = data['total_episodes']

        # Log task-specific data
        task = data['task']
        task_log.setdefault(task, []).append(data['task_data'])

        # Log unique episode data
        if data['episode_ndx'] not in episodes_completed:
            episodes_completed.add(data['episode_ndx'])
            episode_data.append(data)
            cumulative_metrics['episodes_completed'] += 1
            print(f"✅ 新episode记录: {data['episode_ndx']}, 总计: {cumulative_metrics['episodes_completed']}")

            # Update cumulative metrics with the new episode data
            for key, value in data.items():
                if key not in required_keys and isinstance(value, (int, float)):
                    cumulative_metrics[key] = cumulative_metrics.get(key, 0) + value
        else:
            print(f"⚠️  Episode {data['episode_ndx']} 已存在，跳过")

    return jsonify({'status': 'success'}), 200

def periodic_status():
    """定期打印状态信息"""
    while not terminate_event.is_set():
        time.sleep(10)
        with lock:
            print(f"状态更新 - Episodes: {cumulative_metrics.get('episodes_completed', 0)}/{total_episodes[0]}, Instances: {len(spend_per_instance)}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Simple Aggregator for Testing')
    parser.add_argument('--name', type=str, required=True, help='Name for the run')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the Flask server')
    args = parser.parse_args()

    print(f"启动简化版聚合器: {args.name} on port {args.port}")

    # Start status monitoring in a separate thread
    status_thread = threading.Thread(target=periodic_status, daemon=True)
    status_thread.start()

    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except KeyboardInterrupt:
        print("Aggregator received KeyboardInterrupt. Shutting down.")
    finally:
        terminate_event.set()
        print("Simple aggregator has shut down.")
