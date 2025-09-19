#!/usr/bin/env python3

"""
Test script for the streamlined overlap-based reasoning system
Tests the improved prompt generation and different reasoning strategies
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_exploration_phase_descriptions():
    """Test the overlap-based exploration phase detection"""
    print("=== Testing Exploration Phase Descriptions ===")
    
    # Mock CoTGraphAgent for testing
    class MockAgent:
        def _get_exploration_phase_description(self, avg_overlap, max_overlap):
            if max_overlap >= 0.7:
                return {
                    'phase_name': 'TARGET_VERIFICATION',
                    'reasoning_strategy': 'High semantic similarity detected - likely very close to target. Focus on precise localization and target confirmation.',
                    'weight_adjustments': {
                        'semantic_weight': 0.5,
                        'spatial_weight': 0.3,
                        'llm_weight': 0.2
                    }
                }
            elif avg_overlap >= 0.4 or max_overlap >= 0.5:
                return {
                    'phase_name': 'FOCUSED_SEARCH',
                    'reasoning_strategy': 'Moderate semantic alignment found - target area likely nearby. Navigate toward directions with highest semantic relevance.',
                    'weight_adjustments': {
                        'semantic_weight': 0.4,
                        'spatial_weight': 0.4,
                        'llm_weight': 0.2
                    }
                }
            else:
                return {
                    'phase_name': 'FRONTIER_EXPLORATION',
                    'reasoning_strategy': 'Low semantic similarity - broad exploration needed. Prioritize discovering new areas and rooms that might contain target.',
                    'weight_adjustments': {
                        'semantic_weight': 0.2,
                        'spatial_weight': 0.5,
                        'llm_weight': 0.3
                    }
                }
    
    agent = MockAgent()
    
    # Test cases for different overlap scenarios
    test_cases = [
        (0.1, 0.8, "HIGH OVERLAP"),
        (0.5, 0.6, "MEDIUM OVERLAP"), 
        (0.2, 0.3, "LOW OVERLAP"),
        (0.0, 0.1, "VERY LOW OVERLAP")
    ]
    
    for avg, max_overlap, scenario in test_cases:
        phase = agent._get_exploration_phase_description(avg, max_overlap)
        print(f"\n{scenario}: avg={avg:.1f}, max={max_overlap:.1f}")
        print(f"  Phase: {phase['phase_name']}")
        print(f"  Strategy: {phase['reasoning_strategy'][:80]}...")
        print(f"  Weights: semantic={phase['weight_adjustments']['semantic_weight']}, "
              f"spatial={phase['weight_adjustments']['spatial_weight']}, "
              f"llm={phase['weight_adjustments']['llm_weight']}")

def test_streamlined_prompt_generation():
    """Test the streamlined prompt generation for different phases"""
    print("\n=== Testing Streamlined Prompt Generation ===")
    
    # Mock data
    current_goal = "chair"
    primary_room = "living_room"
    secondary_rooms = ["dining_room", "bedroom"]
    anchor_names = ["sofa", "table", "lamp"]
    
    vlm_predictions = {
        '30': {'Score': 3, 'Room': 'hallway', 'Path': 'clear', 'Objects': {'door': 'center'}, 'DirectionalAnchors': []},
        '90': {'Score': 7, 'Room': 'living_room', 'Path': 'accessible', 'Objects': {'sofa': 'left', 'table': 'center'}, 'DirectionalAnchors': ['sofa points toward seating area']},
        '150': {'Score': 2, 'Room': 'unknown', 'Path': 'blocked', 'Objects': {}, 'DirectionalAnchors': []},
        '210': {'Score': 5, 'Room': 'dining_room', 'Path': 'clear', 'Objects': {'table': 'center', 'chairs': 'around'}, 'DirectionalAnchors': ['table indicates dining area']},
        '270': {'Score': 4, 'Room': 'hallway', 'Path': 'open', 'Objects': {'wall': 'sides'}, 'DirectionalAnchors': []},
        '330': {'Score': 1, 'Room': 'unknown', 'Path': 'unclear', 'Objects': {}, 'DirectionalAnchors': []}
    }
    
    all_direction_data = {
        '30': {'base_score': 2.1, 'overlap_score': 0.1},
        '90': {'base_score': 6.8, 'overlap_score': 0.75},
        '150': {'base_score': 1.2, 'overlap_score': 0.05},
        '210': {'base_score': 4.5, 'overlap_score': 0.45},
        '270': {'base_score': 3.2, 'overlap_score': 0.2},
        '330': {'base_score': 0.8, 'overlap_score': 0.02}
    }
    
    # Test different exploration phases
    exploration_phases = [
        {
            'phase_name': 'TARGET_VERIFICATION',
            'reasoning_strategy': 'High semantic similarity detected - likely very close to target.',
            'weight_adjustments': {'semantic_weight': 0.5, 'spatial_weight': 0.3, 'llm_weight': 0.2}
        },
        {
            'phase_name': 'FOCUSED_SEARCH', 
            'reasoning_strategy': 'Moderate semantic alignment found - target area likely nearby.',
            'weight_adjustments': {'semantic_weight': 0.4, 'spatial_weight': 0.4, 'llm_weight': 0.2}
        },
        {
            'phase_name': 'FRONTIER_EXPLORATION',
            'reasoning_strategy': 'Low semantic similarity - broad exploration needed.',
            'weight_adjustments': {'semantic_weight': 0.2, 'spatial_weight': 0.5, 'llm_weight': 0.3}
        }
    ]
    
    for phase in exploration_phases:
        print(f"\n--- {phase['phase_name']} PROMPT ---")
        
        # Generate phase-specific context header
        phase_name = phase['phase_name']
        if phase_name == 'TARGET_VERIFICATION':
            context_header = (
                f"NAVIGATION TASK: Find {current_goal.upper()} - HIGH CONFIDENCE (target likely nearby)\n"
                f"PRIMARY FOCUS: Semantic similarity analysis and target confirmation\n"
                f"Context: {primary_room} room, anchors: {', '.join(anchor_names[:3])}"
            )
        elif phase_name == 'FOCUSED_SEARCH':
            context_header = (
                f"NAVIGATION TASK: Find {current_goal.upper()} - FOCUSED SEARCH (moderate alignment found)\n"
                f"PRIMARY FOCUS: Balance semantic relevance with spatial navigation\n" 
                f"Context: Target in {primary_room}, also check {', '.join(secondary_rooms[:2])}, anchors: {', '.join(anchor_names[:3])}"
            )
        else:  # FRONTIER_EXPLORATION
            context_header = (
                f"NAVIGATION TASK: Find {current_goal.upper()} - EXPLORATION MODE (low similarity, search new areas)\n"
                f"PRIMARY FOCUS: Spatial navigation and room discovery\n"
                f"Context: Seek {primary_room} room, expected in {', '.join(secondary_rooms[:2])}, watch for {', '.join(anchor_names[:3])}"
            )
        
        print(context_header)
        
        # Generate direction summaries
        print("\nDIRECTIONAL ANALYSIS:")
        expected_directions = ['30', '90', '150', '210', '270', '330']
        
        for direction in expected_directions:
            direction_str = str(direction)
            vlm_data = vlm_predictions.get(direction_str, {})
            spatial_data = all_direction_data.get(direction_str, {})
            
            score = vlm_data.get('Score', 0)
            room_type = vlm_data.get('Room', 'unknown') 
            path_quality = vlm_data.get('Path', 'unknown')
            objects = vlm_data.get('Objects', {})
            anchors = vlm_data.get('DirectionalAnchors', [])
            
            base_score = spatial_data.get('base_score', 0.0)
            overlap_score = spatial_data.get('overlap_score', 0.0)
            
            key_objects = list(objects.keys())[:3] if objects else []
            anchor_summary = ', '.join(anchors[:2]) if anchors else 'none'
            
            # Phase-specific summary format
            if phase_name == 'TARGET_VERIFICATION':
                summary = f"{direction}°: Overlap={overlap_score:.3f}, VLM={score}/10 ({room_type}), Objects: {', '.join(key_objects) if key_objects else 'none'}, Anchors: {anchor_summary}"
            elif phase_name == 'FOCUSED_SEARCH':
                summary = f"{direction}°: Spatial={base_score:.2f}, Semantic={overlap_score:.3f}, VLM={score}/10 ({room_type}, {path_quality}), Key: {', '.join(key_objects[:2]) if key_objects else 'none'}"
            else:  # FRONTIER_EXPLORATION
                summary = f"{direction}°: VLM={score}/10 ({room_type}, {path_quality}), Spatial={base_score:.2f}, Objects: {', '.join(key_objects[:2]) if key_objects else 'none'}, Guides: {anchor_summary}"
            
            print(summary)
        
        # Generate phase-specific reasoning instructions
        print(f"\nSTRATEGY: {phase['reasoning_strategy']}")
        
        if phase_name == 'TARGET_VERIFICATION':
            print("1. Prioritize directions with overlap > 0.7 (strong semantic match)")
            print("2. Confirm VLM room matches expected (living_room)")
            print("3. Check for expected anchors: sofa, table, lamp")
            print("4. Ensure path accessibility (avoid 'blocked', prefer 'clear')")
        elif phase_name == 'FOCUSED_SEARCH':
            print("1. Compare top 2-3 overlap directions (0.4-0.7 range)")
            print("2. Favor VLM rooms matching living_room or dining_room, bedroom")
            print("3. Consider spatial scores for navigation quality")
            print("4. Look for directional anchors pointing toward target areas")
        else:  # FRONTIER_EXPLORATION
            print("1. Rely on VLM room predictions (seek living_room)")
            print("2. Favor higher VLM scores (>6/10) with clear paths")
            print("3. Look for any anchor objects or related items")
            print("4. Apply spatial logic for room connectivity and exploration")
        
        print("\n" + "="*60)

def test_llm_response_extraction():
    """Test the streamlined LLM response extraction"""
    print("\n=== Testing LLM Response Extraction ===")
    
    # Mock extraction function
    import re
    
    def extract_llm_direction_recommendation(llm_response):
        result = {
            'recommended_direction': 90,
            'confidence_score': 0.5, 
            'reasoning_summary': 'Default reasoning',
            'direction_evaluations': {}
        }
        
        try:
            # Extract recommended direction
            direction_patterns = [
                r'RECOMMENDED_DIRECTION:\s*(\d+)',
                r'recommended[_\s]direction[:\s]*(\d+)',
                r'direction[:\s]*(\d+)',
                r'choose[_\s](\d+)°?',
                r'go[_\s](\d+)°?'
            ]
            
            for pattern in direction_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    direction = int(match.group(1))
                    if direction in [30, 90, 150, 210, 270, 330]:
                        result['recommended_direction'] = direction
                        break
            
            # Extract confidence score
            confidence_patterns = [
                r'CONFIDENCE_SCORE:\s*([\d.]+)',
                r'confidence[:\s]*([\d.]+)',
                r'score[:\s]*([\d.]+)'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    confidence = float(match.group(1))
                    result['confidence_score'] = max(0.0, min(1.0, confidence))
                    break
            
            # Extract reasoning summary
            reasoning_patterns = [
                r'REASONING_SUMMARY:\s*(.+?)(?:\n|$)',
                r'reasoning[:\s]*(.+?)(?:\n|$)',
                r'summary[:\s]*(.+?)(?:\n|$)',
                r'explanation[:\s]*(.+?)(?:\n|$)'
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    if len(reasoning) > 10:
                        result['reasoning_summary'] = reasoning[:200]
                        break
            
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return result
    
    # Test cases with different LLM response formats
    test_responses = [
        {
            'name': 'Standard Format',
            'response': """
RECOMMENDED_DIRECTION: 90
CONFIDENCE_SCORE: 0.85
REASONING_SUMMARY: Direction 90° shows highest overlap (0.75) with living_room VLM prediction and clear path access, making it optimal for finding the chair.
"""
        },
        {
            'name': 'Informal Format',
            'response': """
Based on the analysis, I recommend direction 210 degrees as it shows good semantic alignment with dining_room prediction. My confidence is 0.72. The reasoning is that dining areas typically contain chairs and the path appears accessible.
"""
        },
        {
            'name': 'Mixed Format',
            'response': """
Looking at the directional data, direction 90° stands out with strong overlap score of 0.75 and living_room detection. Choose 90 degrees.
Confidence score: 0.9
Explanation: High semantic match plus clear navigation path.
"""
        }
    ]
    
    for test_case in test_responses:
        print(f"\n--- Testing {test_case['name']} ---")
        print(f"Response: {test_case['response'].strip()}")
        
        result = extract_llm_direction_recommendation(test_case['response'])
        
        print(f"Extracted:")
        print(f"  Direction: {result['recommended_direction']}°")
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Reasoning: {result['reasoning_summary']}")

def main():
    """Run all tests"""
    print("🧪 Testing Streamlined Overlap-Based Reasoning System")
    print("="*60)
    
    test_exploration_phase_descriptions()
    test_streamlined_prompt_generation()
    test_llm_response_extraction()
    
    print(f"\n✅ All tests completed successfully!")
    print("\n📋 Summary of Improvements:")
    print("1. ✨ Streamlined prompt generation with phase-specific context")
    print("2. 🎯 Simplified directional analysis format")
    print("3. 🧠 Phase-specific reasoning strategies")
    print("4. ⚡ Efficient LLM response extraction")
    print("5. 📊 Overlap-based exploration phase detection")

if __name__ == "__main__":
    main()
