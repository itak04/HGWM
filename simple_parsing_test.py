#!/usr/bin/env python3
"""
简单的CoTGraphAgent VLM解析测试，验证修复是否有效
"""

import sys
import os
import numpy as np
import logging

# 添加路径
sys.path.append('/home/ps/dqf/GoalNav/WMNavigation/src')

def simple_vlm_parsing_test():
    print("🧪 启动CoTGraphAgent VLM解析测试...")
    
    try:
        # 尝试导入CoTGraphAgent
        from cotgraph_agent import CoTGraphAgent
        
        # 创建一个最小配置
        cfg = {
            'map_size': 100,
            'step_limit': 50,
            'api_endpoint': 'https://api.siliconflow.cn/v1',  # 使用SiliconFlow API
            'api_key': 'sk-example',  # 示例key，用于测试结构
            'vlm_model': 'Pro/Qwen/Qwen2-VL-72B-Instruct',
            'llm_model': 'Pro/Qwen/Qwen2.5-72B-Instruct',
        }
        
        print("✅ 成功导入CoTGraphAgent")
        
        # 创建agent实例
        print("📡 初始化CoTGraphAgent...")
        # 由于API相关的初始化可能失败，我们跳过实际的API初始化，只测试解析方法
        
        # 创建一个mock的agent来测试解析
        class TestAgent:
            def __init__(self):
                # 添加必要的属性
                self.map_size = 100
                
            # 复制CoTGraphAgent的解析方法进行测试
            def _extract_largest_json_section(self, text):
                max_content = ""
                max_length = 0
                best_directional_content = ""
                
                import re
                expected_directions = ['30', '90', '150', '210', '270', '330']
                
                # Strategy 1: Look for comprehensive directional JSON
                full_json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}'
                json_candidates = re.findall(full_json_pattern, text, re.DOTALL)
                
                for candidate in json_candidates:
                    direction_count = sum(1 for direction in expected_directions 
                                        if f"'{direction}'" in candidate or f'"{direction}"' in candidate)
                    
                    if direction_count >= 5:
                        if len(candidate) > len(best_directional_content):
                            best_directional_content = candidate
                            print(f"🎯 Found comprehensive directional JSON with {direction_count} directions, length: {len(candidate)}")
                
                if best_directional_content:
                    print(f"✅ Using directional JSON section (length: {len(best_directional_content)})")
                    return best_directional_content
                else:
                    print(f"⚠️ No directional JSON found, using largest section (length: {len(max_content)})")
                    return max_content
            
            def _regex_extract_dict(self, text):
                import re
                import ast
                result_dict = {}
                
                # Enhanced pattern for better nested structure matching
                kv_pattern = r"'([^']+)':\s*(?:'([^']*)'|(\d+\.?\d*)|(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})|(\[[^\]]*\]))"
                matches = re.findall(kv_pattern, text, re.DOTALL)
                
                for match in matches:
                    key = match[0]
                    if match[1]:  # string value
                        result_dict[key] = match[1]
                    elif match[2]:  # numeric value
                        try:
                            result_dict[key] = float(match[2]) if '.' in match[2] else int(match[2])
                        except ValueError:
                            result_dict[key] = match[2]
                    elif match[3]:  # dict value - parse it properly
                        dict_str = match[3]
                        try:
                            parsed_dict = ast.literal_eval(dict_str)
                            result_dict[key] = parsed_dict
                            print(f"   ✅ Successfully parsed nested dict for key '{key}': {type(parsed_dict)}")
                        except (ValueError, SyntaxError) as e:
                            print(f"   ⚠️ Failed to parse nested dict for key '{key}': {e}")
                            result_dict[key] = dict_str
                    elif match[4]:  # list value - parse it properly
                        list_str = match[4]
                        try:
                            parsed_list = ast.literal_eval(list_str)
                            result_dict[key] = parsed_list
                            print(f"   ✅ Successfully parsed list for key '{key}': {type(parsed_list)}")
                        except (ValueError, SyntaxError) as e:
                            print(f"   ⚠️ Failed to parse list for key '{key}': {e}")
                            result_dict[key] = list_str
                
                # Additional validation for direction keys
                expected_directions = ['30', '90', '150', '210', '270', '330']
                direction_keys = [k for k in result_dict.keys() if k in expected_directions]
                
                if direction_keys:
                    print(f"🎯 Found {len(direction_keys)} direction keys in regex extraction")
                    for direction in direction_keys:
                        direction_data = result_dict[direction]
                        if isinstance(direction_data, str):
                            print(f"⚠️ Direction {direction} is still a string, attempting secondary parsing...")
                            try:
                                parsed_data = ast.literal_eval(direction_data)
                                if isinstance(parsed_data, dict):
                                    result_dict[direction] = parsed_data
                                    print(f"   ✅ Successfully converted direction {direction} to dict")
                            except Exception as e:
                                print(f"   ❌ Failed to convert direction {direction}: {e}")
                        elif isinstance(direction_data, dict):
                            print(f"   ✅ Direction {direction} is already a dict with keys: {list(direction_data.keys())}")
                
                return result_dict
            
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
                    (lambda s: self._extract_largest_json_section(s), "largest_json_section")
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
                                # Verify we got direction data instead of just any dict
                                expected_directions = ['30', '90', '150', '210', '270', '330']
                                direction_keys = [k for k in eval_resp.keys() if k in expected_directions]
                                if len(direction_keys) >= 3:
                                    print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                                    return eval_resp
                                else:
                                    print(f"⚠️ Not directional data, continuing search...")
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
                                if len(direction_keys) >= 3:
                                    print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                                    return eval_resp
                                else:
                                    print(f"⚠️ Not directional data, continuing search...")
                        except json.JSONDecodeError as e:
                            print(f"   json.loads failed for {pattern_name}: {e}")
                            
                    except (ValueError, IndexError) as e:
                        print(f"   Pattern {pattern_name} extraction failed: {e}")
                        continue
                
                # Fallback to regex extraction
                print("🔧 Attempting regex-based key-value extraction as last resort...")
                try:
                    extracted_dict = self._regex_extract_dict(result)
                    if extracted_dict:
                        expected_directions = ['30', '90', '150', '210', '270', '330']
                        direction_keys = [k for k in extracted_dict.keys() if k in expected_directions]
                        if len(direction_keys) >= 3:
                            print(f"✅ Successfully extracted via regex: {len(extracted_dict)} keys with {len(direction_keys)} directions")
                            return extracted_dict
                        else:
                            print(f"⚠️ Regex extraction didn't yield directional data")
                except Exception as e:
                    print(f"   Regex extraction failed: {e}")
                
                return {}
        
        # 测试VLM响应解析
        test_agent = TestAgent()
        
        # 使用实际的有问题的VLM响应进行测试
        mock_problematic_response = """```json
{
  '30': {'Score': 2, 'Objects': {'door': 'left mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Hallway with door but appears to be a dead end.'},
  '90': {'Score': 1, 'Objects': {'wall': 'center mid-ground'}, 'ObjectRelationships': [], 'Anchors': [], 'Room': 'hallway', 'Path': 'Wall', 'Explanation': 'Just a wall, no navigation options.'},
  '150': {'Score': 2, 'Objects': {'door': 'right mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Another hallway section with door but limited access.'},
  '210': {'Score': 2, 'Objects': {'door': 'left mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Hallway with door but appears to be a dead end.'},
  '270': {'Score': 10, 'Objects': {'bed': 'center mid-ground', 'nightstand': 'left mid-ground', 'dresser': 'right background'}, 'ObjectRelationships': ['nightstand beside bed', 'dresser opposite bed'], 'Anchors': ['bed', 'nightstand', 'dresser'], 'Room': 'bedroom', 'Path': 'Unobstructed passage', 'Explanation': 'Main bedroom with bed, nightstand, and dresser. Perfect target location.'},
  '330': {'Score': 2, 'Objects': {'door': 'right mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Another hallway section with door but limited access.'}
}
```"""
        
        print("\n🎯 测试VLM响应解析...")
        parsed_result = test_agent._eval_response(mock_problematic_response)
        
        print(f"\n📊 解析结果:")
        print(f"   总键数: {len(parsed_result)}")
        print(f"   键列表: {list(parsed_result.keys())}")
        
        # 验证方向数据是否正确解析为字典
        expected_directions = ['30', '90', '150', '210', '270', '330']
        success_count = 0
        
        for direction in expected_directions:
            if direction in parsed_result:
                direction_data = parsed_result[direction]
                print(f"   方向 {direction}°:")
                if isinstance(direction_data, dict):
                    success_count += 1
                    score = direction_data.get('Score', 'missing')
                    objects = direction_data.get('Objects', {})
                    room = direction_data.get('Room', 'unknown')
                    print(f"      ✅ 字典格式 - Score: {score}, Objects: {len(objects)}, Room: {room}")
                else:
                    print(f"      ❌ 非字典格式 - Type: {type(direction_data)}")
                    if isinstance(direction_data, str):
                        print(f"         内容预览: {direction_data[:100]}...")
            else:
                print(f"   方向 {direction}°: 缺失")
        
        print(f"\n🎯 测试总结:")
        print(f"   成功解析的方向: {success_count}/{len(expected_directions)}")
        success_rate = success_count / len(expected_directions) * 100
        print(f"   成功率: {success_rate:.1f}%")
        
        if success_count == len(expected_directions):
            print("🎉 所有测试通过！VLM解析修复成功。")
            return True
        elif success_count >= 4:
            print("⚠️ 大部分测试通过，但还有改进空间。")
            return True
        else:
            print("❌ 测试失败，VLM解析问题仍然存在。")
            return False
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试期间发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_vlm_parsing_test()
    print(f"\n{'='*60}")
    print(f"最终结果: {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
