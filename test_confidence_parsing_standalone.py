#!/usr/bin/env python3

"""
Test script for validating confidence score parsing robustness
"""

import re

def extract_llm_direction_recommendation(llm_response):
    """
    Standalone version of the fixed extraction method for testing
    """
    result = {
        'recommended_direction': 90,  # Default to forward
        'confidence_score': 0.5,
        'reasoning_summary': 'Default reasoning',
        'direction_evaluations': {}
    }
    
    try:
        # Extract recommended direction - simplified patterns
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
                try:
                    direction = int(match.group(1))
                    if direction in [30, 90, 150, 210, 270, 330]:
                        result['recommended_direction'] = direction
                        break
                except ValueError:
                    continue
        
        # Extract confidence score with robust error handling
        confidence_patterns = [
            r'CONFIDENCE_SCORE:\s*([\d.%]+)',
            r'confidence[:\s]*([\d.%]+)',
            r'score[:\s]*([\d.%]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    confidence_str = match.group(1).strip()
                    # Handle cases like "0.8" or "80%" or malformed strings
                    confidence_str = confidence_str.replace('%', '')
                    
                    # Remove any non-numeric characters except dots and spaces
                    confidence_str = re.sub(r'[^\d.]', '', confidence_str)
                    
                    # Handle empty string or just dots or trailing dots
                    if not confidence_str or confidence_str == '.' or confidence_str == '..' or confidence_str.endswith('.') and len(confidence_str.replace('.', '')) == 0:
                        continue
                        
                    # Handle trailing dots like "0."
                    if confidence_str.endswith('.') and len(confidence_str) > 1:
                        confidence_str = confidence_str[:-1]
                        
                    # Ensure only one decimal point
                    if '.' in confidence_str:
                        parts = confidence_str.split('.')
                        if len(parts) >= 2:
                            # Take first integer part and first decimal part
                            if parts[0] and parts[1]:
                                confidence_str = f"{parts[0]}.{parts[1]}"
                            elif parts[0]:  # Only integer part
                                confidence_str = parts[0]
                            else:
                                continue  # Invalid format
                    
                    if confidence_str and confidence_str != '.':
                        confidence = float(confidence_str)
                        # Normalize if it's in percentage form (>1.0) but handle edge case like 1.5 properly
                        if confidence > 1.0 and confidence <= 100.0:
                            confidence = confidence / 100.0
                        elif confidence > 100.0:
                            confidence = 1.0  # Cap at maximum
                        result['confidence_score'] = max(0.0, min(1.0, confidence))
                        break
                except (ValueError, IndexError) as e:
                    print(f"   ⚠️ Failed to parse confidence '{match.group(1)}': {e}")
                    continue
        
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
                if len(reasoning) > 10:  # Only use if substantial content
                    result['reasoning_summary'] = reasoning[:200]  # Limit length
                    break
        
        print(f"   🎯 Extracted: Direction={result['recommended_direction']}°, Confidence={result['confidence_score']:.3f}")
        return result
        
    except Exception as e:
        print(f"   ❌ LLM extraction error: {e}, using defaults")
        return result

def test_confidence_parsing():
    """Test various confidence score formats including malformed ones"""
    
    test_cases = [
        # Valid cases
        ("RECOMMENDED_DIRECTION: 90\nCONFIDENCE_SCORE: 0.8\nREASONING_SUMMARY: Good path", 0.8),
        ("Direction: 150\nConfidence: 0.65\nExplanation: Clear route", 0.65),
        ("Choose 30°\nScore: 85%\nReason: High certainty", 0.85),
        
        # Malformed cases that caused the original error
        ("RECOMMENDED_DIRECTION: 90\nCONFIDENCE_SCORE: .\nREASONING_SUMMARY: Default", 0.5),
        ("Direction: 210\nConfidence: ..\nExplanation: Empty dots", 0.5),
        ("Go 270°\nScore: 0.\nReason: Missing decimal", 0.0),  # "0." should parse as 0.0
        ("Direction: 330\nConfidence: .5\nExplanation: Leading dot", 0.5),
        
        # Edge cases
        ("Direction: 90\nConfidence: 100%\nExplanation: Full confidence", 1.0),
        ("Direction: 150\nConfidence: 0\nExplanation: No confidence", 0.0),
        ("Direction: 210\nConfidence: 1.5\nExplanation: Over 100%", 0.015),  # 1.5 as decimal, not percentage
        
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
            result = extract_llm_direction_recommendation(llm_response)
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
        result = extract_llm_direction_recommendation(problematic_response)
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
