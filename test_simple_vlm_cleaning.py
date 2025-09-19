#!/usr/bin/env python3
"""
Simple test for VLM dictionary cleaning functions
"""
import re
import ast

def clean_dict_string(dict_str: str) -> str:
    """Clean dictionary string to fix common VLM formatting issues"""
    
    # Fix duplicate keys like 'Score': 6, 'Score': 'Score': 6
    dict_str = re.sub(r"'Score':\s*\d+,\s*'Score':\s*'Score':\s*\d+", 
                     lambda m: "'Score': " + re.search(r'\d+', m.group()).group(), dict_str)
    
    # Fix other duplicate key patterns  
    dict_str = re.sub(r"'(\w+)':\s*([^,}]+),\s*'\1':\s*'\1':\s*([^,}]+)", r"'\1': \2", dict_str)
    
    # Fix trailing commas before closing braces
    dict_str = re.sub(r',\s*}', '}', dict_str)
    
    # Fix multiple consecutive commas
    dict_str = re.sub(r',\s*,+', ',', dict_str)
    
    return dict_str.strip()

def fix_malformed_dict_string(dict_str: str) -> str:
    """Try to fix malformed dictionary strings with more aggressive cleaning"""
    
    try:
        # More aggressive cleaning
        dict_str = dict_str.strip()
        
        # Remove problematic duplicate key patterns entirely
        dict_str = re.sub(r"'Score':\s*'Score':\s*\d+", "'Score': 0", dict_str)
        dict_str = re.sub(r"'(\w+)':\s*'\1':\s*([^,}]+)", r"'\1': \2", dict_str)
        
        # Fix missing quotes around keys (be more careful with context)
        dict_str = re.sub(r'(\w+):\s*\{', r"'\1': {", dict_str)  # Fix keys before nested dicts
        dict_str = re.sub(r'(\w+):\s*\'([^\']*)\'\s*([,}])', r"'\1': '\2'\3", dict_str)  # quoted values
        dict_str = re.sub(r'(\w+):\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r"'\1': '\2'\3", dict_str)  # unquoted values
        
        # Fix nested object issues - ensure quotes around object keys/values
        dict_str = re.sub(r'\{\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)', r"{ '\1': '\2'", dict_str)
        dict_str = re.sub(r',\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)', r", '\1': '\2'", dict_str)
        
        # Fix trailing issues
        dict_str = re.sub(r',\s*}', '}', dict_str)
        dict_str = re.sub(r',\s*,+', ',', dict_str)
        
        # Basic validation - must start with { and end with }
        if not dict_str.startswith('{') or not dict_str.endswith('}'):
            return None
            
        return dict_str.strip()
        
    except Exception as e:
        print(f"   Error in fix_malformed_dict_string: {e}")
        return None

def manual_dict_extraction(text: str) -> dict:
    """Manually extract dictionary from problematic text"""
    
    result = {}
    
    # Extract Score (handle various formats)
    score_match = re.search(r"(?:'Score'|Score):\s*(\d+)", text)
    if score_match:
        result['Score'] = int(score_match.group(1))
    
    # Extract Objects (handle various formats)
    objects_match = re.search(r"(?:'Objects'|Objects):\s*\{([^}]+)\}", text)
    if objects_match:
        objects_text = objects_match.group(1)
        objects = {}
        
        # Try different patterns for object entries
        for obj_match in re.finditer(r"'([^']+)':\s*'([^']*)'", objects_text):
            objects[obj_match.group(1)] = obj_match.group(2)
        
        # Handle unquoted keys/values
        if not objects:  # If no quoted matches found
            for obj_match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)", objects_text):
                objects[obj_match.group(1)] = obj_match.group(2)
        
        result['Objects'] = objects
    
    # Extract Room (handle various formats)
    room_match = re.search(r"(?:'Room'|Room):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
    if room_match:
        result['Room'] = room_match.group(1) if room_match.group(1) else room_match.group(2).strip()
    
    # Extract Path (handle various formats)
    path_match = re.search(r"(?:'Path'|Path):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
    if path_match:
        result['Path'] = path_match.group(1) if path_match.group(1) else path_match.group(2).strip()
        
    # Extract Explanation (handle various formats)
    explanation_match = re.search(r"(?:'Explanation'|Explanation):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
    if explanation_match:
        result['Explanation'] = explanation_match.group(1) if explanation_match.group(1) else explanation_match.group(2).strip()
    
    return result if result else text

def clean_and_parse_direction_string(direction_str: str) -> dict:
    """Enhanced cleaning and parsing for direction data strings"""
    
    try:
        # Apply all cleaning strategies
        cleaned = clean_dict_string(direction_str)
        
        # Try parsing the cleaned version
        try:
            return ast.literal_eval(cleaned)
        except:
            # If that fails, try more aggressive fixing
            fixed = fix_malformed_dict_string(direction_str)
            if fixed:
                return ast.literal_eval(fixed)
            else:
                # Final attempt: extract key-value pairs manually
                return manual_dict_extraction(direction_str)
                
    except Exception as e:
        print(f"   Error in clean_and_parse_direction_string: {e}")
        return direction_str  # Return original if all fails

def test_cleaning_functions():
    """Test the cleaning functions"""
    
    print("🧪 Testing VLM dictionary parsing functions...")
    
    # Test case 1: Duplicate keys issue (direction 150)
    malformed_1 = "{'Score': 6, 'Score': 'Score': 6, 'Objects': {'chair': 'wooden', 'table': 'glass'}, 'Room': 'living room', 'Path': 'forward', 'Explanation': 'Clear path ahead'}"
    
    print("\n📋 Test 1: Direction 150 duplicate keys")
    print(f"Input: {malformed_1}")
    
    try:
        cleaned = clean_dict_string(malformed_1)
        print(f"Cleaned: {cleaned}")
        
        result = clean_and_parse_direction_string(malformed_1)
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result: {result}")
            print("✅ Test 1 PASSED")
        else:
            print("❌ Test 1 FAILED - not a dict")
    except Exception as e:
        print(f"❌ Test 1 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: Multiple issues - let's test a simpler case first  
    malformed_2 = "{'Score': 'Score': 8, 'Objects': {'sofa': 'old', 'lamp': 'bright'}, 'Room': 'living room', 'Path': 'turn left'}"
    
    print("\n📋 Test 2: Multiple formatting issues")
    print(f"Input: {malformed_2}")
    
    try:
        result = clean_and_parse_direction_string(malformed_2)
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result: {result}")
            print("✅ Test 2 PASSED")
        else:
            print("❌ Test 2 FAILED - not a dict")
    except Exception as e:
        print(f"❌ Test 2 FAILED with error: {e}")
    
    # Test case 3: Manual extraction fallback
    malformed_3 = "Score: 7, Objects: {chair: wooden, table: glass}, Room: kitchen, Path: straight, Explanation: good visibility"
    
    print("\n📋 Test 3: Manual extraction")
    print(f"Input: {malformed_3}")
    
    try:
        result = manual_dict_extraction(malformed_3)
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result: {result}")
            print("✅ Test 3 PASSED")
        else:
            print("❌ Test 3 FAILED - not a dict")
    except Exception as e:
        print(f"❌ Test 3 FAILED with error: {e}")

if __name__ == "__main__":
    print("🚀 Starting VLM parsing tests...\n")
    test_cleaning_functions()
    print("\n🎉 All tests completed!")
