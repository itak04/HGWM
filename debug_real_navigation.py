#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import habitat_sim
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simWrapper import SimWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)

def debug_real_navigation():
    """Debug the real navigation setup to understand why paths fail."""
    
    print("=== Real Navigation Debug ===")
    
    # Try to load the actual episode data
    episode_file = 'data/objectnav_hm3d_v1/val/objectnav_hm3d_v1_val.json'
    if not os.path.exists(episode_file):
        print(f"❌ Episode file not found: {episode_file}")
        return
        
    # Load episodes
    with open(episode_file, 'r') as f:
        episode_data = json.load(f)
    
    print(f"📊 Loaded {len(episode_data['episodes'])} episodes")
    
    # Find episode 1400 (from the logs)
    target_episode = None
    for i, episode in enumerate(episode_data['episodes']):
        if i == 1400:  # Episode index 1400
            target_episode = episode
            break
    
    if not target_episode:
        print("❌ Could not find episode 1400")
        return
        
    print(f"🎯 Found episode 1400:")
    print(f"   Scene: {target_episode['scene_id']}")
    print(f"   Object: {target_episode['object_category']}")
    print(f"   Start position: {target_episode['start_position']}")
    print(f"   Start rotation: {target_episode['start_rotation']}")
    print(f"   Geodesic distance: {target_episode['info']['geodesic_distance']}")
    
    # Extract scene info
    scene_parts = target_episode['scene_id'].split('/')
    scene_id = scene_parts[1]
    scene_path = os.path.join(os.environ.get("DATASET_ROOT", "/home/ps/dqf/GoalNav/WMNavigation/data"), f'{target_episode["scene_id"]}')
    
    print(f"🏠 Scene ID: {scene_id}")
    print(f"🏠 Scene path: {scene_path}")
    
    # Check if scene exists
    if not os.path.exists(scene_path):
        print(f"❌ Scene path not found: {scene_path}")
        return
    
    # Set up sim config
    sim_cfg = {
        'scene_id': scene_id,
        'scene_path': scene_path,
        'scene_config': '/home/ps/dqf/GoalNav/WMNavigation/data/hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json',
        'use_goal_image_agent': False,
        'allow_slide': True,
        'agent_radius': 0.25,
        'agent_height': 1.5,
        'sensor_cfg': {
            'pitch': np.deg2rad(-30),
            'fov': np.deg2rad(90),
            'height': 1.5,
            'img_height': 224,
            'img_width': 224
        },
        'goal_image_agent_fov': np.deg2rad(90)
    }
    
    try:
        # Initialize SimWrapper
        print("🔧 Initializing SimWrapper...")
        sim_wrapper = SimWrapper(sim_cfg)
        print("✅ SimWrapper initialized successfully")
        
        # Check pathfinder
        pathfinder = sim_wrapper.sim.pathfinder
        print(f"🗺️  Pathfinder loaded: {pathfinder.is_loaded}")
        
        if not pathfinder.is_loaded:
            print("❌ Pathfinder not loaded!")
            return
            
        print(f"🌐 Navigable area: {pathfinder.navigable_area:.2f} sq meters")
        
        # Check start position
        start_pos = np.array(target_episode['start_position'])
        print(f"🎯 Start position: {start_pos}")
        
        # Check if start position is navigable
        is_navigable = pathfinder.is_navigable(start_pos)
        print(f"✅ Start position navigable: {is_navigable}")
        
        if not is_navigable:
            print("❌ Start position is not navigable! This could be the problem.")
            # Try to snap to navigable
            snapped = pathfinder.snap_point(start_pos)
            print(f"🔧 Snapped position: {snapped}")
            snap_navigable = pathfinder.is_navigable(snapped)
            print(f"✅ Snapped position navigable: {snap_navigable}")
        
        # Now we need to check the goal positions
        # Load the goals file
        goals_file = '/home/ps/dqf/GoalNav/WMNavigation/data/hm3d_v0.1/goals.json'
        if not os.path.exists(goals_file):
            print(f"❌ Goals file not found: {goals_file}")
            return
            
        with open(goals_file, 'r') as f:
            goals_data = json.load(f)
            
        # Get goals for this scene and object
        if scene_id not in goals_data:
            print(f"❌ Scene {scene_id} not found in goals")
            return
            
        scene_goals = goals_data[scene_id]
        object_key = f'{scene_parts[2]}_{target_episode["object_category"]}'
        if target_episode["object_category"] == "tv_monitor":
            object_key = f'{scene_parts[2]}_tv screen'
            
        print(f"🔍 Looking for object key: {object_key}")
        
        if object_key not in scene_goals:
            print(f"❌ Object key {object_key} not found in scene goals")
            print(f"Available keys: {list(scene_goals.keys())}")
            return
            
        all_objects = scene_goals[object_key]
        print(f"🎯 Found {len(all_objects)} object instances")
        
        # Extract view positions
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
                
        print(f"🎯 Total view positions: {len(view_positions)}")
        
        # Check if goal positions are navigable
        navigable_goals = 0
        for i, goal_pos in enumerate(view_positions):
            goal_array = np.array(goal_pos, dtype=np.float32)
            is_nav = pathfinder.is_navigable(goal_array)
            if is_nav:
                navigable_goals += 1
            else:
                print(f"❌ Goal position {i} not navigable: {goal_pos}")
                
        print(f"✅ Navigable goals: {navigable_goals}/{len(view_positions)}")
        
        if navigable_goals == 0:
            print("❌ No goals are navigable! This is the problem.")
            return
            
        # Test pathfinding with the actual positions
        print("\n=== Testing Real Pathfinding ===")
        
        # Set up MultiGoalShortestPath like in the real code
        path_calculator = habitat_sim.MultiGoalShortestPath()
        path_calculator.requested_start = start_pos
        path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        
        print(f"🎯 Start: {start_pos}")
        print(f"🎯 Goals shape: {np.array(view_positions, dtype=np.float32).shape}")
        
        # Test pathfinding
        found = pathfinder.find_path(path_calculator)
        print(f"🛤️  Path found: {found}")
        
        if found:
            print(f"📏 Distance: {path_calculator.geodesic_distance:.2f}")
        else:
            print("❌ NO PATH FOUND - This reproduces the error!")
            
            # Debug why
            print("\n=== Debugging Path Failure ===")
            
            # Check individual paths to each goal
            for i, goal_pos in enumerate(view_positions[:5]):  # Check first 5
                single_path = habitat_sim.ShortestPath()
                single_path.requested_start = start_pos
                single_path.requested_end = np.array(goal_pos, dtype=np.float32)
                
                single_found = pathfinder.find_path(single_path)
                print(f"Goal {i}: {single_found} (distance: {single_path.geodesic_distance:.2f if single_found else 'N/A'})")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_real_navigation()
