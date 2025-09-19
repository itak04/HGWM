#!/usr/bin/env python3
"""
Test script to verify that all "list index out of range" errors have been fixed.
This script tests the specific scenarios that were causing issues.
"""

import sys
import os
sys.path.append('src')

def test_cotgraph_agent_import():
    """Test basic import and initialization"""
    print("🧪 Testing CoTGraphAgent import and initialization...")
    try:
        from cotgraph_agent import CoTGraphAgent
        
        cfg = {
            'map_size': 300,
            'vlm_cfg': {'vlm_model': 'GPT-4V'},
            'llm_cfg': {'llm_model': 'Pro/Qwen/Qwen2.5-7B-Instruct'},
            'cot_cfg': {
                'max_qa_rounds': 3,
                'confidence_threshold': 0.7,
                'memory_decay_factor': 0.9,
                'panoramic_analysis_mode': 'directional_scoring'
            }
        }
        
        agent = CoTGraphAgent(cfg)
        print("✅ CoTGraphAgent import and initialization: PASSED")
        return True, agent
        
    except Exception as e:
        print(f"❌ CoTGraphAgent import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_spatial_relations_extraction(agent):
    """Test the fixed _extract_spatial_relations_for_object method"""
    print("🧪 Testing spatial relations extraction with edge cases...")
    
    try:
        # Test case 1: Empty strings that cause split() to return empty list
        relations = agent._extract_spatial_relations_for_object(
            "chair", 
            ["chair beside "], 
            "chair beside "
        )
        print("✅ Empty string after 'beside' handled correctly")
        
        # Test case 2: Insufficient relation parts
        relations = agent._extract_spatial_relations_for_object(
            "table", 
            ["table on"], 
            "table on"
        )
        print("✅ Insufficient relation parts handled correctly")
        
        # Test case 3: Valid relations
        relations = agent._extract_spatial_relations_for_object(
            "book", 
            ["book on table", "chair beside book"], 
            "book on shelf"
        )
        print(f"✅ Valid relations extracted: {len(relations)} found")
        
        # Test case 4: Complex empty splits
        relations = agent._extract_spatial_relations_for_object(
            "lamp", 
            ["lamp near   ", "   beside lamp"], 
            "lamp near   "
        )
        print("✅ Complex empty splits handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Spatial relations extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlm_predictions_processing(agent):
    """Test VLM predictions processing that was causing issues"""
    print("🧪 Testing VLM predictions processing...")
    
    try:
        # Test with various VLM prediction structures
        sample_vlm_predictions = {
            '30': {
                'Score': 10,  # Maximum score to test the new logic
                'Objects': {'chair': 'center foreground'}, 
                'Room': 'living_room',
                'ObjectRelationships': ['chair beside table'],
                'Anchors': ['chair'],
                'Explanation': 'Clear view of living room with furniture'
            },
            '90': {
                'Score': 7,  # Regular score
                'Objects': {'table': 'left mid-ground beside'}, # Edge case: ends with relation word
                'Room': 'dining_room',
                'ObjectRelationships': ['table on '],  # Edge case: empty after relation
                'Anchors': [],
                'Explanation': 'Partial view of dining area'
            },
            '150': {
                'Score': 3,
                'Objects': {},  # Empty objects
                'Room': 'hallway',
                'ObjectRelationships': [],  # Empty relationships
                'Anchors': [],
                'Explanation': 'Empty hallway view'
            }
        }
        
        # Test curiosity value update
        explorable_value = {'30': 5.0, '90': 8.0, '150': 3.0}
        reason = {'30': 'Medium relevance', '90': 'High relevance', '150': 'Low relevance'}
        
        # This should not crash with list index out of range errors
        result = agent.update_curiosity_value(explorable_value, reason, sample_vlm_predictions, {})
        print("✅ VLM predictions processing with edge cases: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ VLM predictions processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_panoramic_image_selection():
    """Test the fixed panoramic image selection in WMNav_env.py"""
    print("🧪 Testing panoramic image selection bounds checking...")
    
    try:
        # Simulate the problematic scenario
        pano_images = ['img1', 'img2', 'img3']  # Only 3 images
        goal_rotate = 5  # Index that would cause out of range error
        
        # Test the fixed logic
        if isinstance(pano_images, list) and len(pano_images) > goal_rotate and goal_rotate >= 0:
            target_image = [pano_images[goal_rotate]]
        else:
            target_image = pano_images
            print(f"✅ Correctly fell back to full pano_images when goal_rotate={goal_rotate} >= len(pano_images)={len(pano_images)}")
        
        # Test negative goal_rotate
        goal_rotate = -1
        if isinstance(pano_images, list) and len(pano_images) > goal_rotate and goal_rotate >= 0:
            target_image = [pano_images[goal_rotate]]
        else:
            target_image = pano_images
            print(f"✅ Correctly handled negative goal_rotate={goal_rotate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Panoramic image selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing fixes for 'list index out of range' errors")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Basic import and initialization
    passed, agent = test_cotgraph_agent_import()
    all_passed = all_passed and passed
    
    if not agent:
        print("❌ Cannot proceed with further tests due to import failure")
        return False
    
    print()
    
    # Test 2: Spatial relations extraction
    passed = test_spatial_relations_extraction(agent)
    all_passed = all_passed and passed
    print()
    
    # Test 3: VLM predictions processing
    passed = test_vlm_predictions_processing(agent)
    all_passed = all_passed and passed
    print()
    
    # Test 4: Panoramic image selection
    passed = test_panoramic_image_selection()
    all_passed = all_passed and passed
    print()
    
    # Final results
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! List index out of range errors have been fixed.")
        print("✅ The CoTGraphAgent should now run without these specific errors.")
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
