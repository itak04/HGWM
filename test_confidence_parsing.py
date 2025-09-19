#!/usr/bin/env python3

"""
Test script for validating confidence score parsing robustness
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cotgraph_agent import CoTGraphAgent

def test_confidence_parsing():
    """Test various confidence score formats including malformed ones"""
    
    # Create agent instance (mock initialization for testing)
    agent = CoTGraphAgent("test")
    
    test_cases = [
        # Valid cases
        ("RECOMMENDED_DIRECTION: 90\nCONFIDENCE_SCORE: 0.8\nREASONING_SUMMARY: Good path", 0.8),
        ("Direction: 150\nConfidence: 0.65\nExplanation: Clear route", 0.65),
        ("Choose 30°\nScore: 85%\nReason: High certainty", 0.85),
        
        # Malformed cases that caused the original error
        ("RECOMMENDED_DIRECTION: 90\nCONFIDENCE_SCORE: .\nREASONING_SUMMARY: Default", 0.5),
        ("Direction: 210\nConfidence: ..\nExplanation: Empty dots", 0.5),
        ("Go 270°\nScore: 0.\nReason: Missing decimal", 0.5),
        ("Direction: 330\nConfidence: .5\nExplanation: Leading dot", 0.5),
        
        # Edge cases
        ("Direction: 90\nConfidence: 100%\nExplanation: Full confidence", 1.0),
        ("Direction: 150\nConfidence: 0\nExplanation: No confidence", 0.0),
        ("Direction: 210\nConfidence: 1.5\nExplanation: Over 100%", 1.0),  # Should be capped
        
        # Missing confidence (should use default)
        ("RECOMMENDED_DIRECTION: 30\nREASONING_SUMMARY: No confidence given", 0.5),
        
        # Multiple decimal points
        ("Direction: 270\nConfidence: 0.8.5\nExplanation: Double decimal", 0.8),  # Should take first valid part
    ]
    
    print("🧪 Testing confidence score parsing robustness...")
    print("=" * 60)
    
    all_passed = True
    
    for i, (llm_response, expected_confidence) in enumerate(test_cases, 1):
        try:
            result = agent._extract_llm_direction_recommendation(llm_response)
            actual_confidence = result['confidence_score']
            
            # Allow small floating point differences
            if abs(actual_confidence - expected_confidence) < 0.001:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                all_passed = False
            
            print(f"Test {i:2}: {status}")
            # Format the input for display
            formatted_input = repr(llm_response.replace('\n', '\\n'))
            print(f"   Input: {formatted_input}")
            print(f"   Expected: {expected_confidence}, Got: {actual_confidence}")
            print(f"   Direction: {result['recommended_direction']}°")
            print()
            
        except Exception as e:
            print(f"Test {i:2}: ❌ EXCEPTION - {e}")
            print(f"   Input: {repr(llm_response)}")
            all_passed = False
            print()
    
    # Test the specific error case that was reported
    print("🎯 Testing the specific error case...")
    problematic_response = """
    RECOMMENDED_DIRECTION: 90
    CONFIDENCE_SCORE: .
    REASONING_SUMMARY: This was causing float conversion error
    """
    
    try:
        result = agent._extract_llm_direction_recommendation(problematic_response)
        print(f"✅ Original error case handled successfully!")
        print(f"   Result: Direction={result['recommended_direction']}°, Confidence={result['confidence_score']}")
    except Exception as e:
        print(f"❌ Original error case still fails: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Confidence parsing is now robust.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    test_confidence_parsing()
