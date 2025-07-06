#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import habitat_sim
import json
import gzip

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simWrapper import SimWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)

def debug_wmnav_environment():
    """Debug the WMNavEnv setup that's causing NO PATH FOUND."""
    
    print("=== WMNavEnv Debug ===")
    
    # Simulate the exact same setup as WMNavEnv
    dataset = 'hm3d_v0.1'
    split = 'val'
    
    # Get paths like in WMNavEnv
    if dataset == 'hm3d_v0.1':
        scene_config_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
        objnav_path = 'objectnav_hm3d_v1'
    
    dataset_root = os.environ.get("DATASET_ROOT", "/home/ps/dqf/GoalNav/WMNavigation/data")
    scene_config = os.path.join(dataset_root, scene_config_path)
    
    print(f"📁 Dataset root: {dataset_root}")
    print(f"🗂️  Scene config: {scene_config}")
    print(f"📊 Objnav path: {objnav_path}")
    
    # Check if paths exist
    if not os.path.exists(scene_config):
        print(f"❌ Scene config not found: {scene_config}")
        return
    
    objnav_dir = os.path.join(dataset_root, objnav_path, f'{split}/content')
    if not os.path.exists(objnav_dir):
        print(f"❌ Objnav directory not found: {objnav_dir}")
        return
        
    print(f"✅ Objnav directory exists: {objnav_dir}")
    
    # Load episodes like WMNavEnv does
    all_episodes = []
    goals = {}
    
    print("📊 Loading episodes...")
    for f in sorted(os.listdir(objnav_dir)):
        if f.endswith('.json.gz'):
            file_path = os.path.join(objnav_dir, f)
            print(f"   Loading {f}")
            try:
                with gzip.open(file_path, 'rt') as gz:
                    js = json.load(gz)
                    hsh = f.split('.')[0]
                    goals[hsh] = js['goals_by_category']
                    all_episodes += js['episodes']
            except Exception as e:
                print(f"   ❌ Error loading {f}: {e}")
    
    print(f"📊 Total episodes loaded: {len(all_episodes)}")
    
    if len(all_episodes) == 0:
        print("❌ No episodes loaded!")
        return
    
    # Find episode 1400 (from logs)
    if len(all_episodes) <= 1400:
        print(f"❌ Episode 1400 doesn't exist (only {len(all_episodes)} episodes)")
        return
        
    episode = all_episodes[1400]
    print(f"\n🎯 Episode 1400:")
    print(f"   Scene ID: {episode['scene_id']}")
    print(f"   Object: {episode['object_category']}")
    print(f"   Start position: {episode['start_position']}")
    print(f"   Start rotation: {episode['start_rotation']}")
    print(f"   Geodesic distance: {episode['info']['geodesic_distance']}")
    
    # Parse scene info like WMNavEnv does
    f = episode['scene_id'].split('/')[1:]
    scene_id = f[1][2:5]
    scene_path = os.path.join(dataset_root, 'hm3d_v0.1', f'{split}/{f[1]}/{f[2]}')
    
    print(f"\n🏠 Parsed scene info:")
    print(f"   Scene ID: {scene_id}")
    print(f"   Scene path: {scene_path}")
    
    # Check if scene exists
    if not os.path.exists(scene_path):
        print(f"❌ Scene path not found: {scene_path}")
        return
    else:
        print(f"✅ Scene path exists")
    
    # Get goals like WMNavEnv does
    goals_key = f[1][6:]
    if goals_key not in goals:
        print(f"❌ Goals key '{goals_key}' not found in goals")
        print(f"Available goals keys: {list(goals.keys())}")
        return
        
    scene_goals = goals[goals_key]
    object_key = f'{f[-1]}_{episode["object_category"]}'
    
    print(f"🎯 Looking for object key: {object_key}")
    
    if object_key not in scene_goals:
        print(f"❌ Object key '{object_key}' not found in scene goals")
        print(f"Available object keys: {list(scene_goals.keys())}")
        return
        
    all_objects = scene_goals[object_key]
    print(f"🎯 Found {len(all_objects)} object instances")
    
    # Extract view positions like WMNavEnv does
    view_positions = []
    for obj in all_objects:
        for vp in obj['view_points']:
            view_positions.append(vp['agent_state']['position'])
            
    print(f"🎯 Total view positions: {len(view_positions)}")
    
    if len(view_positions) == 0:
        print("❌ No view positions found!")
        return
    
    # Set up simulator like WMNavEnv does
    sim_cfg = {
        'scene_id': scene_id,
        'scene_path': scene_path,
        'scene_config': scene_config,
        'use_goal_image_agent': False,
        'allow_slide': True,
        'agent_radius': 0.18,
        'agent_height': 0.88,
        'sensor_cfg': {
            'pitch': -0.25,
            'fov': 79,
            'height': 0.88,
            'img_height': 480,
            'img_width': 640
        }
    }
    
    try:
        print("\n🔧 Initializing SimWrapper...")
        sim_wrapper = SimWrapper(sim_cfg)
        print("✅ SimWrapper initialized successfully")
        
        # Check pathfinder
        pathfinder = sim_wrapper.sim.pathfinder
        print(f"🗺️  Pathfinder loaded: {pathfinder.is_loaded}")
        
        if not pathfinder.is_loaded:
            print("❌ Pathfinder not loaded!")
            return
            
        print(f"🌐 Navigable area: {pathfinder.navigable_area:.2f} sq meters")
        
        # Test like WMNavEnv does
        print("\n🧪 Testing path calculation like WMNavEnv...")
        
        # Set agent position
        start_pos = np.array(episode['start_position'])
        print(f"🎯 Start position: {start_pos}")
        
        # Check if start position is navigable
        is_navigable = pathfinder.is_navigable(start_pos)
        print(f"✅ Start position navigable: {is_navigable}")
        
        if not is_navigable:
            print("❌ Start position is not navigable!")
            # Try to snap to navigable
            snapped = pathfinder.snap_point(start_pos)
            print(f"🔧 Snapped position: {snapped}")
            snap_navigable = pathfinder.is_navigable(snapped)
            print(f"✅ Snapped position navigable: {snap_navigable}")
            if snap_navigable:
                start_pos = snapped
                print("🔧 Using snapped position")
        
        # Test pathfinding like in _calculate_metrics
        path_calculator = habitat_sim.MultiGoalShortestPath()
        path_calculator.requested_start = start_pos
        view_positions_array = np.array(view_positions, dtype=np.float32)
        path_calculator.requested_ends = view_positions_array
        
        print(f"🎯 Requested start: {path_calculator.requested_start}")
        print(f"🎯 Requested ends shape: {view_positions_array.shape}")
        print(f"🎯 First few goals: {view_positions_array[:3]}")
        print(f"🎯 Path calculator ends type: {type(path_calculator.requested_ends)}")
        
        # This is the exact call that fails in the logs
        found = pathfinder.find_path(path_calculator)
        print(f"🛤️  Path found: {found}")
        
        if found:
            print(f"📏 Distance: {path_calculator.geodesic_distance:.2f}")
            print("✅ Navigation should work!")
        else:
            print("❌ NO PATH FOUND - This reproduces the error!")
            
            # Debug individual goals
            print("\n🔍 Testing individual goals...")
            navigable_goals = 0
            for i, goal_pos in enumerate(view_positions[:5]):
                goal_array = np.array(goal_pos, dtype=np.float32)
                is_nav = pathfinder.is_navigable(goal_array)
                print(f"   Goal {i}: navigable={is_nav}, pos={goal_pos}")
                if is_nav:
                    navigable_goals += 1
                    
                    # Test single path
                    single_path = habitat_sim.ShortestPath()
                    single_path.requested_start = start_pos
                    single_path.requested_end = goal_array
                    
                    single_found = pathfinder.find_path(single_path)
                    if single_found:
                        print(f"      ✅ Single path found: {single_path.geodesic_distance:.2f}")
                    else:
                        print(f"      ❌ Single path failed")
                        
            print(f"📊 Total navigable goals: {navigable_goals}/{len(view_positions)}")
            
            if navigable_goals == 0:
                print("❌ No goals are navigable - this is the problem!")
            else:
                print("🤔 Some goals are navigable but MultiGoalShortestPath fails")
                
                # Check navigation islands
                print("\n🏝️  Checking navigation islands...")
                print(f"Total islands: {pathfinder.num_islands}")
                
                start_island = pathfinder.get_island(start_pos)
                print(f"Start position island: {start_island}")
                
                for i, goal_pos in enumerate(view_positions[:5]):
                    goal_array = np.array(goal_pos, dtype=np.float32)
                    if pathfinder.is_navigable(goal_array):
                        goal_island = pathfinder.get_island(goal_array)
                        print(f"Goal {i} island: {goal_island} (same as start: {goal_island == start_island})")
                        
                        if goal_island == start_island:
                            print(f"   🤔 Goal {i} is on same island but path still fails!")
                            # Try with more detailed debugging
                            test_path = habitat_sim.ShortestPath()
                            test_path.requested_start = start_pos
                            test_path.requested_end = goal_array
                            
                            print(f"   Start: {test_path.requested_start}")
                            print(f"   End: {test_path.requested_end}")
                            
                            found = pathfinder.find_path(test_path)
                            print(f"   Found: {found}")
                            if found:
                                print(f"   Distance: {test_path.geodesic_distance}")
                            else:
                                print("   ❌ Path failed even though both points are navigable and on same island!")
                                
                # Check if the start position needs to be adjusted
                print(f"\n🔧 Checking if start position needs adjustment...")
                distance_threshold = 0.1  # 10cm
                nearby_navigable = []
                
                for dx in [-0.1, 0, 0.1]:
                    for dz in [-0.1, 0, 0.1]:
                        test_pos = start_pos + np.array([dx, 0, dz])
                        if pathfinder.is_navigable(test_pos):
                            nearby_navigable.append(test_pos)
                            
                print(f"Nearby navigable positions: {len(nearby_navigable)}")
                
                if len(nearby_navigable) > 1:
                    # Try with a slightly adjusted start position
                    adjusted_start = nearby_navigable[1]  # Use second position
                    print(f"🔧 Testing with adjusted start: {adjusted_start}")
                    
                    for i, goal_pos in enumerate(view_positions[:3]):
                        goal_array = np.array(goal_pos, dtype=np.float32)
                        if pathfinder.is_navigable(goal_array):
                            test_path = habitat_sim.ShortestPath()
                            test_path.requested_start = adjusted_start
                            test_path.requested_end = goal_array
                            
                            found = pathfinder.find_path(test_path)
                            print(f"   Goal {i} with adjusted start: {found}")
                            if found:
                                print(f"   ✅ Success! Distance: {test_path.geodesic_distance}")
                                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_wmnav_environment()
