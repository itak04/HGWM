#!/usr/bin/env python3
"""
测试目标子图构建的修复是否有效
"""

import sys
import os
sys.path.append('/home/ps/dqf/GoalNav/WMNavigation/src')

def test_goal_subgraph_parsing():
    print("🧪 测试目标子图LLM解析修复...")
    
    # Mock LLM响应数据 - 类似于用户报告的响应
    mock_llm_response = """{
    'target_room_hierarchy': {
        'primary_room': 'bedroom',
        'secondary_rooms': ['guest_room', 'master_bedroom'],
        'adjacent_zones': ['hallway', 'bathroom']
    },
    'hallway_navigation': {
        'room_connections': {
            'hallway_to_bedroom': {'from': 'hallway', 'to': 'bedroom', 'visual_cues': ['doorway', 'opening']},
            'bathroom_to_bedroom': {'from': 'bathroom', 'to': 'bedroom', 'visual_cues': ['adjacent_door', 'shared_wall']}
        },
        'wayfinding_objects': ['staircase', 'bathroom_door', 'main_hallway']
    },
    'spatial_anchor_objects': {
        'nightstand': {'name': 'nightstand', 'correlation': 0.95, 'spatial_relation': 'beside_bed', 'room_indicator': True},
        'dresser': {'name': 'dresser', 'correlation': 0.90, 'spatial_relation': 'opposite_wall', 'room_indicator': True},
        'bed_frame': {'name': 'bed_frame', 'correlation': 0.98, 'spatial_relation': 'center_room', 'room_indicator': True}
    },
    'exploration_strategy': {
        'frontier_priorities': ['bedroom_doorways', 'master_bedroom_access', 'guest_room_entrances']
    },
    'overlap_weights': {
        'room_hierarchy_match': 0.35,
        'anchor_objects_detected': 0.30,
        'hallway_connectivity': 0.20,
        'spatial_arrangement': 0.15
    },
    'confidence': 0.92
}"""
    
    # Create a minimal test agent
    class TestAgent:
        def __init__(self):
            self.goal_subgraph_cache = {}
            self.last_cached_goal = None
            self.unified_memory = {
                'goal_subgraphs': {}
            }
        
        def _get_fallback_goal_subgraph(self, goal):
            return {
                'target_room_hierarchy': {
                    'primary_room': 'unknown',
                    'secondary_rooms': [],
                    'adjacent_zones': []
                },
                'spatial_anchor_objects': [goal],
                'exploration_strategy': {
                    'frontier_priorities': ['forward', 'left', 'right']
                },
                'hallway_navigation': {
                    'room_connections': {},
                    'wayfinding_objects': []
                }
            }
            
        # 复制修复后的_eval_response方法
        def _eval_response(self, response):
            import re
            import json
            import ast
            
            if not response or not isinstance(response, str):
                return {}
            
            # Clean response
            result = response.strip()
            result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", result)
            result = re.sub(r'"([^"]*)":', r"'\1':", result)
            result = re.sub(r':\s*"([^"]*)"', r": '\1'", result)
            
            print(f"🔧 Cleaned response for parsing: {result[:200]}...")
            
            # Try different extraction strategies
            brace_patterns = [
                (lambda s: s[s.index('{') + 1:s.rindex('}')], "outer_double_braces"),
                (lambda s: s[s.rindex('{'):s.rindex('}') + 1], "single_braces_last"),
                (lambda s: s[s.index('{'):s.rindex('}')+1], "complete_braces"),
                (lambda s: s[s.index('{', s.index('{') + 1):s.rindex('}', 0, s.rindex('}'))], "inner_braces"),
                (lambda s: s[s.index('{') + 2:s.rindex('}') - 1] if s.count('{') >= 3 else s[s.index('{'):s.rindex('}')+1], "triple_braces"),
            ]
            
            for extract_func, pattern_name in brace_patterns:
                try:
                    extracted = extract_func(result)
                    if not extracted:
                        continue
                    
                    print(f"🎯 Trying {pattern_name}: {extracted[:100]}...")
                    
                    # Try ast.literal_eval first
                    try:
                        eval_resp = ast.literal_eval(extracted)
                        if isinstance(eval_resp, dict):
                            print(f"✅ Successfully parsed with {pattern_name} using ast.literal_eval")
                            
                            # Check what type of data we got
                            expected_directions = ['30', '90', '150', '210', '270', '330']
                            direction_keys = [k for k in eval_resp.keys() if k in expected_directions]
                            
                            # Check for goal subgraph indicators
                            subgraph_indicators = ['target_room_hierarchy', 'spatial_anchor_objects', 'hallway_navigation', 'exploration_strategy']
                            subgraph_keys = [k for k in eval_resp.keys() if k in subgraph_indicators]
                            
                            if len(direction_keys) >= 3:  # Directional VLM data
                                print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                                return eval_resp
                            elif len(subgraph_keys) >= 2:  # Goal subgraph data
                                print(f"✅ Confirmed goal subgraph data with keys: {subgraph_keys}")
                                return eval_resp
                            elif len(eval_resp) > 0:  # Any valid dictionary
                                print(f"✅ Valid dictionary data with {len(eval_resp)} keys: {list(eval_resp.keys())[:5]}")
                                return eval_resp
                            else:
                                print(f"⚠️ Empty dictionary, continuing search...")
                    except (ValueError, SyntaxError) as e:
                        print(f"   ast.literal_eval failed for {pattern_name}: {e}")
                    
                    # Try JSON parsing as backup
                    try:
                        json_str = extracted.replace("'", '"')
                        eval_resp = json.loads(json_str)
                        if isinstance(eval_resp, dict):
                            print(f"✅ Successfully parsed with {pattern_name} using json.loads")
                            
                            expected_directions = ['30', '90', '150', '210', '270', '330']
                            direction_keys = [k for k in eval_resp.keys() if k in expected_directions]
                            
                            subgraph_indicators = ['target_room_hierarchy', 'spatial_anchor_objects', 'hallway_navigation', 'exploration_strategy']
                            subgraph_keys = [k for k in eval_resp.keys() if k in subgraph_indicators]
                            
                            if len(direction_keys) >= 3:  # Directional VLM data
                                print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                                return eval_resp
                            elif len(subgraph_keys) >= 2:  # Goal subgraph data
                                print(f"✅ Confirmed goal subgraph data with keys: {subgraph_keys}")
                                return eval_resp
                            elif len(eval_resp) > 0:  # Any valid dictionary
                                print(f"✅ Valid dictionary data with {len(eval_resp)} keys: {list(eval_resp.keys())[:5]}")
                                return eval_resp
                            else:
                                print(f"⚠️ Empty dictionary, continuing search...")
                    except json.JSONDecodeError as e:
                        print(f"   json.loads failed for {pattern_name}: {e}")
                        
                except (ValueError, IndexError) as e:
                    print(f"   Pattern {pattern_name} extraction failed: {e}")
                    continue
            
            return {}
        
        # 复制修复后的_construct_goal_subgraph_via_llm方法
        def test_goal_subgraph_construction(self, goal, mock_response):
            print(f"🏠 Testing goal subgraph construction for: {goal}")
            
            # Simulate LLM response parsing
            subgraph = self._eval_response(mock_response)
            
            # Ensure we have a valid dictionary
            if not isinstance(subgraph, dict):
                print(f"⚠️ LLM response parsing failed for goal: {goal}, got {type(subgraph)}")
                # Try to get a fallback goal subgraph
                subgraph = self._get_fallback_goal_subgraph(goal)
                if not isinstance(subgraph, dict):
                    print(f"❌ Fallback subgraph also invalid, returning empty dict")
                    return {}
            
            # Cache the result
            self.goal_subgraph_cache[goal] = subgraph
            self.last_cached_goal = goal
            
            # Update unified memory with hierarchical data - ensure it's a dict
            try:
                if isinstance(subgraph, dict):
                    self.unified_memory['goal_subgraphs'][goal] = {
                        'data': subgraph,
                        'timestamp': 1234567890,
                        'access_count': self.unified_memory['goal_subgraphs'].get(goal, {}).get('access_count', 0) + 1,
                        'type': 'hierarchical_room_aware'
                    }
                    # Also store directly for backward compatibility
                    self.unified_memory['goal_subgraphs'][f"{goal}_direct"] = subgraph
                    print(f"✅ Updated unified memory successfully")
                else:
                    print(f"⚠️ Cannot update unified memory: subgraph is not a dict ({type(subgraph)})")
            except Exception as e:
                print(f"⚠️ Error updating unified memory: {e}")
            
            print(f"✅ Goal subgraph constructed and cached for: {goal}")
            
            # Debug: Print key parts with safe access
            try:
                target_room_hierarchy = subgraph.get('target_room_hierarchy', {}) if isinstance(subgraph, dict) else {}
                spatial_anchors = subgraph.get('spatial_anchor_objects', {}) if isinstance(subgraph, dict) else {}
                primary_room = target_room_hierarchy.get('primary_room', 'unknown') if isinstance(target_room_hierarchy, dict) else 'unknown'
                secondary_rooms = target_room_hierarchy.get('secondary_rooms', []) if isinstance(target_room_hierarchy, dict) else []
                
                print(f"📋 Primary room: {primary_room}")
                print(f"🏠 Secondary rooms: {secondary_rooms}")
                print(f"🔗 Spatial anchor objects: {len(spatial_anchors) if isinstance(spatial_anchors, dict) else 0}")
                
                # Show anchor objects for debugging  
                if isinstance(spatial_anchors, dict) and spatial_anchors:
                    anchor_names = []
                    for key, value in spatial_anchors.items():
                        if isinstance(value, dict):
                            anchor_names.append(value.get('name', key))
                        else:
                            anchor_names.append(str(key))
                    print(f"⚓ Anchor objects: {anchor_names}")
            except Exception as e:
                print(f"⚠️ Error in debug output: {e}")
            
            return subgraph
    
    # Run the test
    test_agent = TestAgent()
    
    print("\n🎯 测试目标子图解析...")
    result = test_agent.test_goal_subgraph_construction('bed', mock_llm_response)
    
    print(f"\n📊 测试结果:")
    print(f"   解析结果类型: {type(result)}")
    print(f"   是否为字典: {isinstance(result, dict)}")
    
    if isinstance(result, dict):
        print(f"   键数量: {len(result)}")
        print(f"   主要键: {list(result.keys())}")
        
        # 验证关键字段
        required_fields = ['target_room_hierarchy', 'spatial_anchor_objects', 'hallway_navigation']
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print(f"   ✅ 所有必需字段都存在")
            
            # 验证字段类型
            target_room = result.get('target_room_hierarchy', {})
            if isinstance(target_room, dict):
                primary_room = target_room.get('primary_room')
                print(f"   ✅ 主要房间: {primary_room}")
            else:
                print(f"   ❌ target_room_hierarchy不是字典: {type(target_room)}")
                
            spatial_anchors = result.get('spatial_anchor_objects', {})
            if isinstance(spatial_anchors, dict):
                print(f"   ✅ 空间锚点对象: {len(spatial_anchors)}个")
            else:
                print(f"   ❌ spatial_anchor_objects不是字典: {type(spatial_anchors)}")
                
            print(f"🎉 目标子图解析测试成功！")
            return True
        else:
            print(f"   ❌ 缺失必需字段: {missing_fields}")
            return False
    else:
        print(f"   ❌ 解析结果不是字典")
        return False

if __name__ == "__main__":
    success = test_goal_subgraph_parsing()
    print(f"\n{'='*60}")
    print(f"最终结果: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
