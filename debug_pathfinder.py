#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import habitat_sim
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simWrapper import SimWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)

def check_pathfinder():
    """Debug the pathfinder setup to understand why NO PATH FOUND occurs."""
    
    # Load config (simplified version)
    sim_cfg = {
        'scene_id': '00801-HaxA7YrQdEC',  # A sample scene
        'scene_path': '/home/ps/dqf/GoalNav/WMNavigation/data/hm3d_v0.1/val/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb',
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
    
    print("=== Pathfinder Debug ===")
    
    # Check if scene file exists
    if not os.path.exists(sim_cfg['scene_path']):
        print(f"❌ Scene file not found: {sim_cfg['scene_path']}")
        return
    else:
        print(f"✅ Scene file exists: {sim_cfg['scene_path']}")
    
    if not os.path.exists(sim_cfg['scene_config']):
        print(f"❌ Scene config not found: {sim_cfg['scene_config']}")
        return
    else:
        print(f"✅ Scene config exists: {sim_cfg['scene_config']}")
    
    try:
        # Initialize SimWrapper
        print("🔧 Initializing SimWrapper...")
        sim_wrapper = SimWrapper(sim_cfg)
        print("✅ SimWrapper initialized successfully")
        
        # Check pathfinder status
        pathfinder = sim_wrapper.sim.pathfinder
        print(f"🗺️  Pathfinder loaded: {pathfinder.is_loaded}")
        
        if pathfinder.is_loaded:
            print(f"🌐 Navigable area: {pathfinder.navigable_area:.2f} sq meters")
            print(f"🏝️  Island count: {pathfinder.num_islands}")
            
            # Get navigable bounds
            bounds = pathfinder.get_bounds()
            print(f"📏 Bounds: {bounds}")
            
            # Test pathfinding with simple positions
            print("\n=== Testing Pathfinding ===")
            
            # Get random navigable points
            try:
                start_pos = pathfinder.get_random_navigable_point()
                end_pos = pathfinder.get_random_navigable_point()
                print(f"🎯 Random start: {start_pos}")
                print(f"🎯 Random end: {end_pos}")
                
                # Test basic pathfinding
                path = habitat_sim.ShortestPath()
                path.requested_start = start_pos
                path.requested_end = end_pos
                
                found = pathfinder.find_path(path)
                print(f"🛤️  Path found: {found}")
                if found:
                    print(f"📏 Distance: {path.geodesic_distance:.2f}")
                
                # Test MultiGoalShortestPath (like in the real code)
                print("\n=== Testing MultiGoalShortestPath ===")
                multi_path = habitat_sim.MultiGoalShortestPath()
                multi_path.requested_start = start_pos
                multi_path.requested_ends = np.array([end_pos], dtype=np.float32)
                
                found_multi = pathfinder.find_path(multi_path)
                print(f"🛤️  Multi-path found: {found_multi}")
                if found_multi:
                    print(f"📏 Multi-distance: {multi_path.geodesic_distance:.2f}")
                    
            except Exception as e:
                print(f"❌ Error during pathfinding test: {e}")
                
        else:
            print("❌ Pathfinder not loaded - this is the problem!")
            
            # Check for navmesh files
            scene_dir = os.path.dirname(sim_cfg['scene_path'])
            navmesh_files = list(Path(scene_dir).glob("*.navmesh"))
            print(f"🔍 Looking for navmesh files in: {scene_dir}")
            print(f"🗺️  Found navmesh files: {navmesh_files}")
            
            if navmesh_files:
                print("🔧 Trying to load navmesh manually...")
                for navmesh_file in navmesh_files:
                    try:
                        pathfinder.load_nav_mesh(str(navmesh_file))
                        print(f"✅ Loaded navmesh: {navmesh_file}")
                        print(f"🗺️  Pathfinder loaded: {pathfinder.is_loaded}")
                        break
                    except Exception as e:
                        print(f"❌ Failed to load {navmesh_file}: {e}")
            else:
                print("❌ No navmesh files found!")
                
    except Exception as e:
        print(f"❌ Error initializing SimWrapper: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pathfinder()
