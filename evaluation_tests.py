"""
SEASL+ Test Harness — evaluation_tests.py
Deterministic test stubs for all SEASL+ phases.
All tests must pass before production activation.
"""

import math
import sys
from seasl_evaluation import (
    compute_rolling_metrics,
    monte_carlo_batch_simulation,
    detect_variance_phase,
    compute_entropy,
)


def _assert(condition, test_name, detail=""):
    if condition:
        print(f"  [PASS] {test_name}")
    else:
        print(f"  [FAIL] {test_name}  -- {detail}")
        return False
    return True


def test_1_zero_history():
    """Test 1 -- Zero history: Input [], Expected status=ZERO_SAMPLE"""
    print("\nTest 1: Zero History")
    result = compute_rolling_metrics([])
    ok = True
    ok &= _assert(result["status"] == "ZERO_SAMPLE", "status == ZERO_SAMPLE", f"got {result['status']}")
    ok &= _assert(result["mean"] is None, "mean is None", f"got {result['mean']}")
    ok &= _assert(result["std"] is None, "std is None", f"got {result['std']}")
    ok &= _assert(result["ci_lower"] is None, "ci_lower is None")
    ok &= _assert(result["ci_upper"] is None, "ci_upper is None")

    # Also test None input
    result_none = compute_rolling_metrics(None)
    ok &= _assert(result_none["status"] == "NO_HISTORY", "None input -> NO_HISTORY", f"got {result_none['status']}")
    return ok


def test_2_single_value():
    """Test 2 -- Single value: Input [0.20], Expected mean=0.20, std=None, status=LOW_SAMPLE_WARNING"""
    print("\nTest 2: Single Value")
    result = compute_rolling_metrics([0.20])
    ok = True
    ok &= _assert(result["status"] == "LOW_SAMPLE_WARNING", "status == LOW_SAMPLE_WARNING", f"got {result['status']}")
    ok &= _assert(abs(result["mean"] - 0.20) < 1e-9, "mean == 0.20", f"got {result['mean']}")
    ok &= _assert(result["std"] is None, "std is None", f"got {result['std']}")
    ok &= _assert(result["ci_lower"] is None, "ci_lower is None")
    ok &= _assert(result["ci_upper"] is None, "ci_upper is None")
    return ok


def test_3_deterministic_monte_carlo():
    """Test 3 -- Deterministic Monte Carlo: Call twice with seed=42, expect identical results"""
    print("\nTest 3: Deterministic Monte Carlo")
    r1 = monte_carlo_batch_simulation(ticket_count=10, actual_rate=0.03, simulations=10000, seed=42)
    r2 = monte_carlo_batch_simulation(ticket_count=10, actual_rate=0.03, simulations=10000, seed=42)
    ok = True
    ok &= _assert(
        r1["baseline_mean"] == r2["baseline_mean"],
        "baseline_mean identical",
        f"{r1['baseline_mean']} vs {r2['baseline_mean']}"
    )
    ok &= _assert(
        r1["percentile_rank"] == r2["percentile_rank"],
        "percentile_rank identical",
        f"{r1['percentile_rank']} vs {r2['percentile_rank']}"
    )
    ok &= _assert(
        r1["p_value"] == r2["p_value"],
        "p_value identical",
        f"{r1['p_value']} vs {r2['p_value']}"
    )
    ok &= _assert(r1["seed_used"] == 42, "seed_used == 42")

    # Test NO_TICKETS edge case
    r0 = monte_carlo_batch_simulation(ticket_count=0, seed=42)
    ok &= _assert(r0["status"] == "NO_TICKETS", "ticket_count=0 -> NO_TICKETS")

    # Test minimum simulations guard
    try:
        monte_carlo_batch_simulation(ticket_count=10, simulations=500, seed=42)
        ok &= _assert(False, "simulations<1000 should raise ValueError")
    except ValueError:
        ok &= _assert(True, "simulations<1000 raises ValueError")

    return ok


def test_4_entropy_sanity():
    """Test 4 -- Entropy sanity: Uniform distribution test, H should equal log2(N)"""
    print("\nTest 4: Entropy Sanity")
    N = 40
    uniform_dist = {i: 1.0 / N for i in range(1, N + 1)}
    result = compute_entropy(uniform_dist, label="uniform_test")

    expected_H = math.log2(N)  # 5.321928...
    ok = True
    ok &= _assert(
        abs(result["entropy"] - expected_H) < 0.001,
        f"H == log2({N}) = {expected_H:.4f}",
        f"got {result['entropy']}"
    )
    ok &= _assert(result["n_symbols"] == N, f"n_symbols == {N}", f"got {result['n_symbols']}")
    ok &= _assert(result["status"] == "OK", "status == OK")

    # Edge: empty distribution
    empty_result = compute_entropy({}, label="empty")
    ok &= _assert(empty_result["status"] == "EMPTY_DISTRIBUTION", "empty dist -> EMPTY_DISTRIBUTION")
    ok &= _assert(empty_result["entropy"] == 0.0, "empty entropy == 0.0")

    return ok


def test_5_variance_detector():
    """Test 5 -- Variance detector: mean=0.03, std=0.01, current=0.05, Expected z=2.0, phase=STRONG_POSITIVE"""
    print("\nTest 5: Variance Detector")
    result = detect_variance_phase(current=0.05, mean=0.03, std=0.01)
    ok = True
    ok &= _assert(
        abs(result["z_score"] - 2.0) < 0.001,
        "z_score == 2.0",
        f"got {result['z_score']}"
    )
    ok &= _assert(
        result["phase"] == "STRONG_POSITIVE",
        "phase == STRONG_POSITIVE",
        f"got {result['phase']}"
    )

    # Test INSUFFICIENT_VARIANCE edge cases
    r_none = detect_variance_phase(current=0.05, mean=None, std=None)
    ok &= _assert(r_none["phase"] == "INSUFFICIENT_VARIANCE", "None mean/std -> INSUFFICIENT_VARIANCE")

    r_zero = detect_variance_phase(current=0.05, mean=0.03, std=0)
    ok &= _assert(r_zero["phase"] == "INSUFFICIENT_VARIANCE", "std=0 -> INSUFFICIENT_VARIANCE")

    # Test negative z
    r_neg = detect_variance_phase(current=0.01, mean=0.03, std=0.01)
    ok &= _assert(r_neg["phase"] == "STRONG_NEGATIVE", "z=-2.0 -> STRONG_NEGATIVE", f"got {r_neg['phase']}")

    return ok


def run_all_tests():
    """Run all 5 test stubs and report results."""
    print("=" * 50)
    print("SEASL+ EVALUATION TEST HARNESS")
    print("=" * 50)

    results = []
    results.append(("Test 1: Zero History", test_1_zero_history()))
    results.append(("Test 2: Single Value", test_2_single_value()))
    results.append(("Test 3: Deterministic MC", test_3_deterministic_monte_carlo()))
    results.append(("Test 4: Entropy Sanity", test_4_entropy_sanity()))
    results.append(("Test 5: Variance Detector", test_5_variance_detector()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print("")
    if all_pass:
        print("  ALL TESTS PASSED - Ready for production activation.")
    else:
        print("  SOME TESTS FAILED - Do NOT activate in production.")

    return all_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
