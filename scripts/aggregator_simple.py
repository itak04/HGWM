#!/usr/bin/env python3
"""
Simplified version of aggregator.py with minimal dependencies and robust operation.
This version removes WandB integration and complex threading to focus on reliable
data aggregation and clean shutdown.
"""

import argparse
import json
import threading
import time
import traceback
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Threading components
lock = threading.Lock()
shutdown_flag = False

# Aggregated metrics - same structure as original but simplified
episode_data = []  # Stores episode data
episodes_completed = set()  # Tracks completed episodes
cumulative_metrics = {'episodes_completed': 0}  # Tracks cumulative metrics for all episodes
total_episodes = [1]  # Total episodes to be completed
spend_per_instance = {}  # Stores spend per instance
task_log = {}  # Logs data for each task

def compute_final_metrics():
    """Compute final aggregated metrics for all completed episodes."""
    if not episode_data:
        return {}
    
    metrics = {
        'episodes_completed': len(episode_data),
        'total_spend': sum(spend_per_instance.values()),
        'instances_connected': len(spend_per_instance),
        'total_episodes': total_episodes[0]
    }
    
    # Aggregate goal-based metrics if available
    all_goals = []
    for episode in episode_data:
        if 'task_data' in episode and 'goal_data' in episode['task_data']:
            all_goals.extend(episode['task_data']['goal_data'])
    
    if all_goals:
        metrics['goals_completed'] = len(all_goals)
        metrics['success_rate'] = sum(goal['goal_reached'] for goal in all_goals) / len(all_goals)
        metrics['spl'] = sum(goal['spl'] for goal in all_goals) / len(all_goals)
    
    return metrics

@app.route('/status', methods=['GET'])
def health_check():
    """Health check endpoint with final metrics"""
    with lock:
        basic_status = {
            'status': 'healthy',
            'episodes_completed': cumulative_metrics.get('episodes_completed', 0),
            'instances_connected': len(spend_per_instance),
            'total_episodes': total_episodes[0]
        }
        
        # Add final metrics if episodes are completed
        if episode_data:
            final_metrics = compute_final_metrics()
            basic_status.update(final_metrics)
        
        return jsonify(basic_status), 200

@app.route('/terminate', methods=['POST'])
def terminate():
    """Endpoint to receive termination signal and shutdown the server."""
    global shutdown_flag
    with lock:
        print("Received termination signal.")
        shutdown_flag = True
    
    # Compute and log final results
    if episode_data:
        final_metrics = compute_final_metrics()
        print("\n=== FINAL AGGREGATION RESULTS ===")
        print(f"Episodes Completed: {final_metrics.get('episodes_completed', 0)}")
        print(f"Total Episodes Expected: {final_metrics.get('total_episodes', 0)}")
        print(f"Instances Connected: {final_metrics.get('instances_connected', 0)}")
        print(f"Total Spend: ${final_metrics.get('total_spend', 0):.2f}")
        if 'success_rate' in final_metrics:
            print(f"Success Rate: {final_metrics['success_rate']:.3f}")
            print(f"SPL: {final_metrics['spl']:.3f}")
            print(f"Goals Completed: {final_metrics['goals_completed']}")
        print("==================================\n")
    
    return jsonify({'status': 'terminating'}), 200

@app.route('/log', methods=['POST'])
def log_metrics():
    """Endpoint to log metrics for each instance."""
    data = request.json
    required_keys = ['instance', 'episode_ndx', 'total_episodes', 'spend', 'task', 'task_data']
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"Error: Missing key(s): {', '.join(missing_keys)}")
        return jsonify({
            'status': 'error',
            'message': f'Missing key(s): {", ".join(missing_keys)}'
        }), 400

    # Log the data within a lock to ensure thread safety
    with lock:
        instance = data['instance']
        episode_ndx = data['episode_ndx']
        
        print(f"Received data from instance {instance}, episode {episode_ndx}")
        
        # Update instance spend and total episodes
        spend_per_instance[instance] = data['spend']
        total_episodes[0] = data['total_episodes']

        # Log task-specific data
        task = data['task']
        task_log.setdefault(task, []).append(data['task_data'])

        # Log unique episode data
        if episode_ndx not in episodes_completed:
            episodes_completed.add(episode_ndx)
            episode_data.append(data)
            cumulative_metrics['episodes_completed'] += 1
            
            print(f"Episode {episode_ndx} completed. Total: {cumulative_metrics['episodes_completed']}/{total_episodes[0]}")

            # Update cumulative metrics with the new episode data
            for key, value in data.items():
                if key not in required_keys and isinstance(value, (int, float)):
                    cumulative_metrics[key] = cumulative_metrics.get(key, 0) + value
        else:
            print(f"Episode {episode_ndx} already recorded (duplicate)")

    return jsonify({'status': 'success'}), 200

def shutdown_server():
    """Gracefully shutdown the Flask server"""
    import os
    import signal
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Simplified Aggregator for Parallel Workers')
    parser.add_argument('--name', type=str, required=True, help='Name for the run group')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the Flask server')
    parser.add_argument('--project', type=str, default='NewPaper', help='Project Name (ignored)')
    parser.add_argument('--config', type=str, default='ObjectNav', help='Config file (ignored)')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep interval (ignored)')
    args = parser.parse_args()

    print(f"Starting simplified aggregator '{args.name}' on port {args.port}")
    print("Note: WandB logging is disabled in this version")

    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except KeyboardInterrupt:
        print("Aggregator received KeyboardInterrupt. Shutting down.")
    finally:
        # Final summary
        if episode_data:
            final_metrics = compute_final_metrics()
            print("\n=== FINAL SHUTDOWN SUMMARY ===")
            print(f"Episodes Completed: {final_metrics.get('episodes_completed', 0)}")
            print(f"Total Episodes Expected: {final_metrics.get('total_episodes', 0)}")
            print(f"Instances Connected: {final_metrics.get('instances_connected', 0)}")
            print(f"Total Spend: ${final_metrics.get('total_spend', 0):.2f}")
            if 'success_rate' in final_metrics:
                print(f"Success Rate: {final_metrics['success_rate']:.3f}")
                print(f"SPL: {final_metrics['spl']:.3f}")
            print("===============================")
        
        print("Simplified aggregator has shut down.")
