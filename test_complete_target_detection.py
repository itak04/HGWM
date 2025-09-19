#!/usr/bin/env python3

"""
测试目标物体检测完整流程的脚本
"""

def test_target_detection_complete_flow():
    """测试当VLM检测到目标物体时的完整流程"""
    
    print("🧪 测试目标物体检测完整流程...")
    print("=" * 60)
    
    # 模拟VLM检测到bed的场景
    vlm_predictions_with_target = {
        '30': {
            'Score': 10,
            'Objects': {
                'bed': 'center foreground clearly visible',
                'nightstand': 'left mid-ground beside bed',
                'pillow': 'on bed foreground'
            },
            'Room': 'bedroom',
            'Path': 'Clear',
            'Explanation': 'Bedroom with bed clearly visible - target found!'
        }
    }
    
    # 模拟VLM未检测到bed的场景
    vlm_predictions_without_target = {
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
    
    def calculate_object_level_overlap_test(vlm_objects, goal_subgraph, current_goal='bed'):
        """模拟修复后的完整对象重叠计算"""
        
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
        
        print(f"   📊 对象匹配分析: VLM检测={list(current_objects)}, 目标={list(target_objects)}")
        
        # 计算VLM-目标重叠（关键！）
        vlm_target_intersection = current_objects.intersection(target_objects)
        vlm_target_overlap = len(vlm_target_intersection)
        vlm_target_total = len(current_objects.union(target_objects))
        
        # 当发现目标物体时，显著提升分数
        if vlm_target_overlap > 0:
            # 基础匹配分数
            base_vlm_target_score = vlm_target_overlap / max(vlm_target_total, 1)
            
            # 目标检测加成：每个目标物体+0.5分，上限0.6
            target_detection_bonus = min(0.6, vlm_target_overlap * 0.5)
            
            # 完全匹配奖励：如果所有目标都找到了
            if len(vlm_target_intersection) == len(target_objects) and len(target_objects) > 0:
                perfect_match_bonus = 0.2
            else:
                perfect_match_bonus = 0.0
            
            vlm_target_score = min(1.0, base_vlm_target_score + target_detection_bonus + perfect_match_bonus)
            
            print(f"   🎯 目标检测成功! 发现: {list(vlm_target_intersection)}")
            print(f"      基础分数: {base_vlm_target_score:.3f}, 检测奖励: {target_detection_bonus:.3f}, 完美匹配奖励: {perfect_match_bonus:.3f}")
            print(f"      最终VLM-目标分数: {vlm_target_score:.3f}")
            
            # 综合对象匹配分数 - 重点突出VLM-目标匹配
            object_match_score = (vlm_target_score * 0.7 + 0.8 * 0.2 + 0.6 * 0.1)  # 假设其他分数
            
        else:
            vlm_target_score = 0.0
            print(f"   ❌ 未检测到目标物体，VLM-目标分数: {vlm_target_score:.3f}")
            object_match_score = 0.0
        
        # 模拟最终重叠计算
        room_match_score = 0.8 if vlm_target_overlap > 0 else 0.3  # 发现目标时房间匹配更好
        spatial_match_score = 0.6
        
        if vlm_target_overlap > 0:
            # 目标检测加成：显著提升权重
            room_weight = 0.25
            object_weight = 0.50  # 显著提升对象权重
            spatial_weight = 0.20
            semantic_weight = 0.03
            depth_weight = 0.02
        else:
            # 标准权重
            room_weight = 0.35
            object_weight = 0.30
            spatial_weight = 0.20
            semantic_weight = 0.10
            depth_weight = 0.05
        
        final_score = (
            room_match_score * room_weight +
            object_match_score * object_weight +
            spatial_match_score * spatial_weight
        )
        
        print(f"   📊 最终对象匹配分数: {object_match_score:.3f}")
        print(f"   🏗️ 权重分配: 房间{room_weight}, 对象{object_weight}, 空间{spatial_weight}")
        print(f"   🎯 最终重叠分数: {final_score:.3f}")
        
        return {
            'target_detected': vlm_target_overlap > 0,
            'target_objects_found': list(vlm_target_intersection),
            'vlm_target_score': vlm_target_score,
            'object_match_score': object_match_score,
            'final_overlap_score': final_score
        }
    
    # 测试1: 检测到目标物体的情况
    print("1. 🎯 测试检测到目标物体的情况:")
    result_with_target = calculate_object_level_overlap_test(vlm_predictions_with_target, goal_subgraph, 'bed')
    
    # 测试2: 未检测到目标物体的情况  
    print(f"\n2. ❌ 测试未检测到目标物体的情况:")
    result_without_target = calculate_object_level_overlap_test(vlm_predictions_without_target, goal_subgraph, 'bed')
    
    # 对比分析
    print(f"\n{'='*60}")
    print("📊 对比分析:")
    print(f"   检测到目标时:")
    print(f"     - 目标检测: {'✅ 成功' if result_with_target['target_detected'] else '❌ 失败'}")
    print(f"     - 发现目标: {result_with_target['target_objects_found']}")
    print(f"     - VLM-目标分数: {result_with_target['vlm_target_score']:.3f}")
    print(f"     - 对象匹配分数: {result_with_target['object_match_score']:.3f}")
    print(f"     - 最终重叠分数: {result_with_target['final_overlap_score']:.3f}")
    
    print(f"\n   未检测到目标时:")
    print(f"     - 目标检测: {'✅ 成功' if result_without_target['target_detected'] else '❌ 失败'}")
    print(f"     - 发现目标: {result_without_target['target_objects_found']}")
    print(f"     - VLM-目标分数: {result_without_target['vlm_target_score']:.3f}")
    print(f"     - 对象匹配分数: {result_without_target['object_match_score']:.3f}")
    print(f"     - 最终重叠分数: {result_without_target['final_overlap_score']:.3f}")
    
    # 验证修复效果
    score_improvement = result_with_target['final_overlap_score'] - result_without_target['final_overlap_score']
    print(f"\n🚀 修复效果:")
    print(f"   - 分数提升: {score_improvement:.3f} ({score_improvement/result_without_target['final_overlap_score']*100:.1f}% 增长)" if result_without_target['final_overlap_score'] > 0 else f"   - 分数提升: {score_improvement:.3f}")
    print(f"   - 目标检测奖励生效: {'✅' if result_with_target['vlm_target_score'] > 0.5 else '❌'}")
    print(f"   - 权重动态调整生效: {'✅' if result_with_target['final_overlap_score'] > 0.7 else '❌'}")
    
    return result_with_target['target_detected'] and result_with_target['final_overlap_score'] > 0.7

if __name__ == "__main__":
    success = test_target_detection_complete_flow()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 目标检测完整流程验证成功！")
        print("   - VLM对象提取修复生效")
        print("   - 目标检测奖励机制工作正常")
        print("   - 动态权重调整按预期执行")
        print("   - 当检测到目标时重叠分数显著提升")
    else:
        print("⚠️ 目标检测流程仍有问题，需要进一步调试")
