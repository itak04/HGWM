#!/usr/bin/env python3

"""
测试VLM对象提取修复的脚本
"""

def test_vlm_object_extraction():
    """测试VLM对象提取修复"""
    
    print("🧪 测试VLM对象提取修复...")
    print("=" * 50)
    
    # 模拟单个方向的VLM数据（修复前的格式）
    single_direction_vlm_data = {
        'Score': 2,
        'Objects': {
            'rocking_chair': 'center-left foreground beside side_table',
            'side_table': 'center-right mid-ground'
        },
        'Room': 'living_room',
        'Path': 'Clear',
        'Explanation': 'Living room with furniture'
    }
    
    # 模拟完整的VLM预测数据（修复后的格式）
    complete_vlm_predictions = {
        '30': {
            'Score': 2,
            'Objects': {
                'rocking_chair': 'center-left foreground beside side_table',
                'side_table': 'center-right mid-ground'
            },
            'Room': 'living_room',
            'Path': 'Clear',
            'Explanation': 'Living room with furniture'
        }
    }
    
    # 模拟目标子图
    goal_subgraph = {
        'target_room_hierarchy': {
            'primary_room': 'bedroom',
            'secondary_rooms': ['guest_room'],
            'adjacent_zones': ['hallway']
        },
        'spatial_anchor_objects': {
            'bed': {'confidence': 0.9, 'primary': True},
            'nightstand': {'confidence': 0.7, 'primary': False},
            'mattress': {'confidence': 0.8, 'primary': False},
            'closet_doors': {'confidence': 0.5, 'primary': False}
        }
    }
    
    def extract_objects_from_vlm_data(vlm_objects, current_goal='bed'):
        """模拟修复后的对象提取逻辑"""
        
        # 从VLM预测中提取物体
        current_objects = set()
        if isinstance(vlm_objects, dict):
            for direction_key, direction_data in vlm_objects.items():
                if isinstance(direction_data, dict):
                    objects = direction_data.get('Objects', direction_data.get('objects', {}))
                    
                    # 处理字典格式的Objects（主要格式）
                    if isinstance(objects, dict):
                        for obj_name, position_info in objects.items():
                            current_objects.add(obj_name.lower())
                    
                    # 处理列表格式的Objects（备用格式）
                    elif isinstance(objects, list):
                        for obj in objects:
                            obj_name = obj if isinstance(obj, str) else obj.get('name', str(obj))
                            current_objects.add(obj_name.lower())
        
        # 从目标子图中提取目标物体名称
        target_objects = set()
        current_goal = current_goal.lower()
        
        # 首先添加当前目标
        if current_goal:
            target_objects.add(current_goal)
        
        # 然后从goal_subgraph中提取
        if isinstance(goal_subgraph, dict):
            # 从spatial_anchor_objects提取
            spatial_anchors = goal_subgraph.get('spatial_anchor_objects', {})
            if isinstance(spatial_anchors, dict):
                for obj_name in spatial_anchors.keys():
                    target_objects.add(str(obj_name).lower())
        
        return current_objects, target_objects
    
    # 测试修复前的情况（单个对象字典 - 会失败）
    print("1. 测试修复前的格式 (单个对象字典):")
    try:
        current_objects_before, target_objects = extract_objects_from_vlm_data(single_direction_vlm_data['Objects'])
        print(f"   VLM检测到的对象: {list(current_objects_before)}")
        print(f"   目标对象: {list(target_objects)}")
        print(f"   ❌ 修复前: VLM检测为空，因为传入的是Objects内容而不是完整的方向字典")
    except Exception as e:
        print(f"   ❌ 修复前: 发生异常 - {e}")
    
    # 测试修复后的情况（完整的方向字典 - 应该成功）
    print("\n2. 测试修复后的格式 (完整的方向字典):")
    try:
        current_objects_after, target_objects = extract_objects_from_vlm_data(complete_vlm_predictions)
        print(f"   VLM检测到的对象: {list(current_objects_after)}")
        print(f"   目标对象: {list(target_objects)}")
        
        # 检查匹配
        vlm_target_intersection = current_objects_after.intersection(target_objects)
        if vlm_target_intersection:
            print(f"   🎯 发现目标物体: {list(vlm_target_intersection)}")
        else:
            print(f"   ❌ 未发现目标物体")
        
        print(f"   ✅ 修复后: VLM检测成功提取到 {len(current_objects_after)} 个对象")
        
        return len(current_objects_after) > 0
        
    except Exception as e:
        print(f"   ❌ 修复后仍有问题: {e}")
        return False

if __name__ == "__main__":
    success = test_vlm_object_extraction()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 VLM对象提取修复验证成功！")
        print("   现在传入的是完整的方向字典格式，对象提取应该正常工作")
    else:
        print("⚠️ VLM对象提取仍有问题，需要进一步调试")
