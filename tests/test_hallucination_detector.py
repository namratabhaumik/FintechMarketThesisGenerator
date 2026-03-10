"""Test script for hallucination detection system."""

from core.agents.hallucination_detector import HallucinationDetector


def test_hallucination_detection():
    """Test the hallucination detector with various scenarios."""
    detector = HallucinationDetector()

    print("=" * 80)
    print("HALLUCINATION DETECTION TEST SUITE")
    print("=" * 80)

    # Test 1: No tool calls
    print("\n[TEST 1] No Tool Calls")
    print("-" * 80)
    output_1 = """
    The digital lending market in Asia is experiencing rapid growth due to:
    1. Increased smartphone penetration
    2. Rising demand for quick credit
    3. Regulatory improvements

    Key opportunities include fintech partnerships and mobile-first solutions.
    """
    result_1 = detector.analyze(output_1)
    print(f"Input: {output_1[:100]}...")
    print(f"\nDetection Results:")
    print(f"  Tool calls found: {result_1['tool_calls']}")
    print(f"  Has hallucinations: {result_1['has_hallucinations']}")
    print(f"  Summary: {result_1['summary']}")

    # Test 2: Valid tool calls only
    print("\n\n[TEST 2] Valid Tool Calls Only")
    print("-" * 80)
    output_2 = """
    I used refine_thesis to improve the thesis based on feedback about market trends.
    Then I applied structure_thesis to organize the key themes and risks.
    Finally, I ran score_opportunity to evaluate the investment potential.

    Key findings show strong growth in digital lending with moderate risk.
    """
    result_2 = detector.analyze(output_2)
    print(f"Input: {output_2[:100]}...")
    print(f"\nDetection Results:")
    print(f"  Tool calls found: {result_2['tool_calls']}")
    print(f"  Valid tools: {result_2['valid_tools']}")
    print(f"  Invalid tools: {result_2['invalid_tools']}")
    print(f"  Has hallucinations: {result_2['has_hallucinations']}")
    print(f"  Summary:\n{result_2['summary']}")

    # Test 3: Hallucinated tools (the interesting case!)
    print("\n\n[TEST 3] HALLUCINATED TOOL CALLS ⚠️")
    print("-" * 80)
    output_3 = """
    I refined the thesis using refine_thesis to address market trend concerns.

    To ensure accuracy, I cross-checked with the validate_sources tool and
    called fetch_market_data to get the latest fintech statistics from external APIs.

    I also invoked check_regulatory_compliance to verify all claims against regulations.

    Then I used structure_thesis to parse the improved analysis and
    score_opportunity to rate the investment opportunity.

    The refined thesis shows strong potential in digital lending markets.
    """
    result_3 = detector.analyze(output_3)
    print(f"Input: {output_3[:100]}...")
    print(f"\nDetection Results:")
    print(f"  Tool calls found: {result_3['tool_calls']}")
    print(f"  Valid tools: {result_3['valid_tools']}")
    print(f"  Invalid tools: {result_3['invalid_tools']}")
    print(f"  Has hallucinations: {result_3['has_hallucinations']}")
    print(f"  Summary:\n{result_3['summary']}")

    # Test 4: Mixed valid and invalid tools
    print("\n\n[TEST 4] MIXED VALID AND INVALID TOOLS")
    print("-" * 80)
    output_4 = """
    I called refine_thesis to improve the thesis.
    Then I used the get_market_trends tool to understand current conditions.
    I applied structure_thesis to organize findings.
    Finally, I used score_opportunity for evaluation.
    """
    result_4 = detector.analyze(output_4)
    print(f"Input: {output_4[:100]}...")
    print(f"\nDetection Results:")
    print(f"  Tool calls found: {result_4['tool_calls']}")
    print(f"  Valid tools: {result_4['valid_tools']}")
    print(f"  Invalid tools: {result_4['invalid_tools']}")
    print(f"  Has hallucinations: {result_4['has_hallucinations']}")
    print(f"  Summary:\n{result_4['summary']}")

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (No calls): {'✅ PASS' if not result_1['has_hallucinations'] else '❌ FAIL'}")
    print(f"Test 2 (Valid only): {'✅ PASS' if not result_2['has_hallucinations'] else '❌ FAIL'}")
    print(f"Test 3 (Hallucinations): {'✅ PASS' if result_3['has_hallucinations'] else '❌ FAIL'}")
    print(f"Test 4 (Mixed): {'✅ PASS' if result_4['has_hallucinations'] else '❌ FAIL'}")
    print("=" * 80)


if __name__ == "__main__":
    test_hallucination_detection()
