"""Test script for execution tracker."""

from core.agents.execution_tracker import ExecutionTracker


def test_execution_tracker():
    """Test execution tracking functionality."""
    print("=" * 80)
    print("EXECUTION TRACKER TEST")
    print("=" * 80)

    tracker = ExecutionTracker()

    # Test 1: Log execution
    print("\n[TEST 1] Log Tool Execution")
    print("-" * 80)
    tracker.log_execution("refine_thesis", status="executed", details={"refinement_number": 1})
    tracker.log_execution("structure_thesis", status="executed", details={"success": True})
    tracker.log_execution("score_opportunity", status="executed")

    executed = tracker.get_executed_tools()
    print(f"Tools that executed: {executed}")
    print(f"Expected: ['refine_thesis', 'structure_thesis', 'score_opportunity']")
    assert executed == ["refine_thesis", "structure_thesis", "score_opportunity"], "❌ FAIL"
    print("✅ PASS")

    # Test 2: Check if tool executed
    print("\n[TEST 2] Check Tool Execution")
    print("-" * 80)
    has_refine = tracker.has_executed("refine_thesis")
    has_fake = tracker.has_executed("validate_sources")
    print(f"refine_thesis executed: {has_refine} (expected: True)")
    print(f"validate_sources executed: {has_fake} (expected: False)")
    assert has_refine and not has_fake, "❌ FAIL"
    print("✅ PASS")

    # Test 3: Log failed tool
    print("\n[TEST 3] Log Failed Tool")
    print("-" * 80)
    tracker.log_execution(
        "invalid_tool", status="failed", details={"error": "Tool does not exist"}
    )
    failed = tracker.get_failed_tools()
    print(f"Failed tools: {failed}")
    print(f"Expected: ['invalid_tool']")
    assert failed == ["invalid_tool"], "❌ FAIL"
    print("✅ PASS")

    # Test 4: Get all events
    print("\n[TEST 4] Get All Events")
    print("-" * 80)
    events = tracker.get_events()
    print(f"Total events logged: {len(events)}")
    print(f"Expected: 4")
    assert len(events) == 4, "❌ FAIL"
    print("✅ PASS")

    # Test 5: Serialize tracker
    print("\n[TEST 5] Serialize Tracker State")
    print("-" * 80)
    state = tracker.to_dict()
    print(f"Total events: {state['total_events']}")
    print(f"Executed count: {state['executed_count']}")
    print(f"Failed count: {state['failed_count']}")
    assert state["total_events"] == 4, "❌ FAIL"
    assert state["executed_count"] == 3, "❌ FAIL"
    assert state["failed_count"] == 1, "❌ FAIL"
    print("✅ PASS")

    # Test 6: Clear tracker
    print("\n[TEST 6] Clear Tracker")
    print("-" * 80)
    tracker.clear()
    events_after = tracker.get_events()
    print(f"Events after clear: {len(events_after)} (expected: 0)")
    assert len(events_after) == 0, "❌ FAIL"
    print("✅ PASS")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ All execution tracker tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_execution_tracker()
