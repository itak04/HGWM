#!/usr/bin/env python3
"""
Test enhanced VLM format parsing capabilities
"""
import sys
sys.path.append('src')

# Create a mock config for testing
class MockConfig:
    def __init__(self):
        pass

def test_malformed_dict_cleaning():
    """Test the new cleaning methods for malformed dictionary strings"""
    
    # Import and create agent with mock config
    from cotgraph_agent import CoTGraphAgent
    
    # Create dummy agent for testing
    try:
        mock_cfg = MockConfig()
        agent = CoTGraphAgent.__new__(CoTGraphAgent)  # Create without calling __init__
        
        # Initialize only the methods we need
        agent._clean_dict_string = CoTGraphAgent._clean_dict_string.__get__(agent, CoTGraphAgent)
        agent._fix_malformed_dict_string = CoTGraphAgent._fix_malformed_dict_string.__get__(agent, CoTGraphAgent)
        agent._clean_and_parse_direction_string = CoTGraphAgent._clean_and_parse_direction_string.__get__(agent, CoTGraphAgent)
        agent._manual_dict_extraction = CoTGraphAgent._manual_dict_extraction.__get__(agent, CoTGraphAgent)
        agent._regex_extract_dict = CoTGraphAgent._regex_extract_dict.__get__(agent, CoTGraphAgent)
        
    except Exception as e:
        print(f"Error setting up agent: {e}")
        return
    
    print("🧪 Testing enhanced VLM dictionary parsing...")
    
    # Test case 1: Duplicate keys issue
    malformed_1 = "{'Score': 6, 'Score': 'Score': 6, 'Objects': {'chair': 'wooden', 'table': 'glass'}, 'Room': 'living room', 'Path': 'forward', 'Explanation': 'Clear path ahead'}"
    
    print("\n📋 Test 1: Duplicate keys")
    print(f"Input: {malformed_1}")
    
    try:
        cleaned = agent._clean_dict_string(malformed_1)
        print(f"Cleaned: {cleaned}")
        
        result = agent._clean_and_parse_direction_string(malformed_1)
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result: {result}")
            print("✅ Test 1 PASSED")
        else:
            print("❌ Test 1 FAILED - not a dict")
    except Exception as e:
        print(f"❌ Test 1 FAILED with error: {e}")
    
    # Test case 2: Multiple issues
    malformed_2 = "{'Score': 'Score': 8, Objects: {'sofa': old, lamp: 'bright'}, 'Room': living room, 'Path': 'turn left',}"
    
    print("\n📋 Test 2: Multiple formatting issues")
    print(f"Input: {malformed_2}")
    
    try:
        result = agent._clean_and_parse_direction_string(malformed_2)
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
        result = agent._manual_dict_extraction(malformed_3)
        print(f"Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"Result: {result}")
            print("✅ Test 3 PASSED")
        else:
            print("❌ Test 3 FAILED - not a dict")
    except Exception as e:
        print(f"❌ Test 3 FAILED with error: {e}")

def test_regex_extract_dict():
    """Test the enhanced _regex_extract_dict method"""
    
    # Import and create agent with mock config
    from cotgraph_agent import CoTGraphAgent
    
    # Create dummy agent for testing
    try:
        agent = CoTGraphAgent.__new__(CoTGraphAgent)  # Create without calling __init__
        
        # Initialize only the methods we need
        agent._clean_dict_string = CoTGraphAgent._clean_dict_string.__get__(agent, CoTGraphAgent)
        agent._fix_malformed_dict_string = CoTGraphAgent._fix_malformed_dict_string.__get__(agent, CoTGraphAgent)
        agent._clean_and_parse_direction_string = CoTGraphAgent._clean_and_parse_direction_string.__get__(agent, CoTGraphAgent)
        agent._manual_dict_extraction = CoTGraphAgent._manual_dict_extraction.__get__(agent, CoTGraphAgent)
        agent._regex_extract_dict = CoTGraphAgent._regex_extract_dict.__get__(agent, CoTGraphAgent)
        
    except Exception as e:
        print(f"Error setting up agent: {e}")
        return
    
    print("\n🔍 Testing enhanced regex extraction...")
    
    # Simulate VLM response with direction data as problematic strings
    test_response = """
    Here's the analysis:
    
    '30': {'Score': 8, 'Objects': {'chair': 'wooden', 'table': 'round'}, 'Room': 'dining room', 'Path': 'forward', 'Explanation': 'Clear path'}
    '90': {'Score': 5, 'Objects': {'bookshelf': 'tall'}, 'Room': 'library', 'Path': 'right turn', 'Explanation': 'Some obstacles'}
    '150': {'Score': 6, 'Score': 'Score': 6, 'Objects': {'sofa': 'leather', 'tv': 'large'}, 'Room': 'living room', 'Path': 'slight right', 'Explanation': 'Furniture blocking'}
    
    Analysis complete.
    """
    
    print(f"Input response: {test_response}")
    
    try:
        result = agent._regex_extract_dict(test_response)
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {type(value)} - {value}")
                if key in ['30', '90', '150'] and isinstance(value, dict):
                    print(f"    ✅ Direction {key} successfully parsed as dict")
                elif key in ['30', '90', '150']:
                    print(f"    ❌ Direction {key} failed to parse as dict (type: {type(value)})")
        
        print("✅ Regex extraction test completed")
        
    except Exception as e:
        print(f"❌ Regex extraction test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting enhanced VLM parsing tests...\n")
    
    test_malformed_dict_cleaning()
    test_regex_extract_dict()
    
    print("\n🎉 All tests completed!")
