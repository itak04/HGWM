#!/usr/bin/env python3
"""
Test VLM parsing fixes to ensure direction data is properly parsed as dictionaries
"""

import sys
import os
sys.path.append('/home/ps/dqf/GoalNav/WMNavigation/src')

# Mock the VLM response that was causing issues
mock_vlm_response = """```json
{
  '30': {'Score': 2, 'Objects': {'wardrobe': 'far-left mid-ground', 'staircase': 'far-left background'}, 'ObjectRelationships': ['wardrobe beside staircase'], 'Anchors': ['wardrobe'], 'Room': 'landing_area', 'Path': 'Obstructed path', 'Explanation': 'Landing area with a wardrobe and staircase, but no clear path to the bedroom.'},
  '90': {'Score': 1, 'Objects': {'kitchen_island': 'center mid-ground', 'main_door': 'center background'}, 'ObjectRelationships': ['kitchen_island in front of main_door'], 'Anchors': ['kitchen_island'], 'Room': 'kitchen', 'Path': 'Clear doorway', 'Explanation': 'Kitchen area with island and door to other rooms.'},
  '150': {'Score': 6, 'Objects': {'bed': 'center mid-ground', 'nightstand': 'right foreground'}, 'ObjectRelationships': ['nightstand beside bed'], 'Anchors': ['bed', 'nightstand'], 'Room': 'bedroom', 'Path': 'Unobstructed passage', 'Explanation': 'Bedroom with bed clearly visible and accessible.'},
  '210': {'Score': 2, 'Objects': {'door': 'left mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Hallway with door but appears to be a dead end.'},
  '270': {'Score': 10, 'Objects': {'bed': 'center mid-ground', 'nightstand': 'left mid-ground', 'dresser': 'right background'}, 'ObjectRelationships': ['nightstand beside bed', 'dresser opposite bed'], 'Anchors': ['bed', 'nightstand', 'dresser'], 'Room': 'bedroom', 'Path': 'Unobstructed passage', 'Explanation': 'Main bedroom with bed, nightstand, and dresser. Perfect target location.'},
  '330': {'Score': 2, 'Objects': {'door': 'right mid-ground', 'wall': 'center mid-ground'}, 'ObjectRelationships': ['door in wall'], 'Anchors': ['door'], 'Room': 'hallway', 'Path': 'Dead end', 'Explanation': 'Another hallway section with door but limited access.'}
}
```"""

def test_vlm_parsing():
    print("🧪 Testing VLM parsing fixes...")
    
    # Create a minimal cotgraph agent for testing
    class MockCoTGraphAgent:
        def _extract_largest_json_section(self, text):
            # Copy the method we're testing
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
                            return eval_resp
                    except (ValueError, SyntaxError) as e:
                        print(f"   ast.literal_eval failed for {pattern_name}: {e}")
                    
                    # Try JSON parsing as backup
                    try:
                        json_str = extracted.replace("'", '"')
                        eval_resp = json.loads(json_str)
                        if isinstance(eval_resp, dict):
                            print(f"✅ Successfully parsed with {pattern_name} using json.loads")
                            return eval_resp
                    except json.JSONDecodeError as e:
                        print(f"   json.loads failed for {pattern_name}: {e}")
                        
                except (ValueError, IndexError) as e:
                    print(f"   Pattern {pattern_name} extraction failed: {e}")
                    continue
            
            # Fallback to regex extraction
            print("🔧 Attempting regex-based key-value extraction...")
            try:
                extracted_dict = self._regex_extract_dict(result)
                if extracted_dict:
                    print(f"✅ Successfully extracted via regex: {len(extracted_dict)} keys")
                    return extracted_dict
            except Exception as e:
                print(f"   Regex extraction failed: {e}")
            
            return {}
    
    # Test the parsing
    agent = MockCoTGraphAgent()
    parsed_result = agent._eval_response(mock_vlm_response)
    
    print(f"\n🎯 Parsing Test Results:")
    print(f"   Total keys parsed: {len(parsed_result)}")
    print(f"   Keys: {list(parsed_result.keys())}")
    
    # Check if direction keys are properly parsed as dictionaries
    expected_directions = ['30', '90', '150', '210', '270', '330']
    direction_success = 0
    
    for direction in expected_directions:
        if direction in parsed_result:
            direction_data = parsed_result[direction]
            print(f"   Direction {direction}: type={type(direction_data)}")
            if isinstance(direction_data, dict):
                direction_success += 1
                sample_keys = list(direction_data.keys())[:3]
                print(f"      ✅ Dict with keys: {sample_keys}")
                
                # Verify Score is numeric
                score = direction_data.get('Score')
                if isinstance(score, (int, float)):
                    print(f"      ✅ Score is numeric: {score}")
                else:
                    print(f"      ⚠️ Score is not numeric: {score} (type: {type(score)})")
                    
                # Verify Objects is dict
                objects = direction_data.get('Objects')
                if isinstance(objects, dict):
                    print(f"      ✅ Objects is dict with {len(objects)} items")
                else:
                    print(f"      ⚠️ Objects is not dict: {type(objects)}")
            else:
                print(f"      ❌ Not a dict - type: {type(direction_data)}")
                if isinstance(direction_data, str):
                    print(f"      ❌ String content: {direction_data[:100]}...")
        else:
            print(f"   Direction {direction}: MISSING")
    
    print(f"\n📊 Test Summary:")
    print(f"   Successful direction parsing: {direction_success}/{len(expected_directions)}")
    print(f"   Success rate: {direction_success/len(expected_directions)*100:.1f}%")
    
    if direction_success == len(expected_directions):
        print("🎉 All tests passed! VLM parsing fix is working correctly.")
        return True
    else:
        print("❌ Some tests failed. VLM parsing still has issues.")
        return False

if __name__ == "__main__":
    success = test_vlm_parsing()
    sys.exit(0 if success else 1)
