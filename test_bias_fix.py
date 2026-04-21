#!/usr/bin/env python3
"""Quick test for the new directional bias computation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_inference.realtime_trap_engine import compute_directional_bias_from_components


def test_bias_computation():
    """Test directional bias derivation from components."""
    
    # Test 1: High volume behavior + risk_score=50 → SHORT with MEDIUM confidence
    print("Test 1: High volume behavior (distribution trap)")
    components = {
        "volume_behavior": 0.75,      # Dominant
        "structure_failure": 0.2,
        "momentum_exhaustion": 0.15,
        "liquidity_intelligence": 0.1,
        "retail_behavior": 0.05,
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=50)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "SHORT", f"Expected SHORT, got {bias}"
    assert conf == "MEDIUM", f"Expected MEDIUM (risk_score=50), got {conf}"
    print("  ✓ PASSED\n")
    
    # Test 1b: High volume behavior + risk_score=65 → SHORT with HIGH confidence
    print("Test 1b: High volume behavior with high risk score")
    bias, conf = compute_directional_bias_from_components(components, risk_score=65)
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "SHORT", f"Expected SHORT, got {bias}"
    assert conf == "HIGH", f"Expected HIGH (risk_score=65), got {conf}"
    print("  ✓ PASSED\n")
    
    # Test 2: High retail behavior → LONG with MEDIUM confidence
    print("Test 2: High retail behavior (reversal setup)")
    components = {
        "volume_behavior": 0.2,
        "structure_failure": 0.15,
        "momentum_exhaustion": 0.1,
        "liquidity_intelligence": 0.15,
        "retail_behavior": 0.65,      # Dominant
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=45)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "LONG", f"Expected LONG, got {bias}"
    assert conf == "MEDIUM", f"Expected MEDIUM, got {conf}"
    print("  ✓ PASSED\n")
    
    # Test 3: Low confidence signals (weak dominance) but still directional
    print("Test 3: Weak dominance but directional bias")
    components = {
        "volume_behavior": 0.35,      # Slightly dominant
        "structure_failure": 0.28,
        "momentum_exhaustion": 0.25,
        "liquidity_intelligence": 0.2,
        "retail_behavior": 0.15,
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=35)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "SHORT", f"Expected SHORT, got {bias}"
    assert conf == "LOW", f"Expected LOW confidence, got {conf}"
    print("  ✓ PASSED - Bias shows direction even at LOW confidence!\n")
    
    # Test 4: All low components → NEUTRAL
    print("Test 4: All components very low")
    components = {
        "volume_behavior": 0.05,
        "structure_failure": 0.03,
        "momentum_exhaustion": 0.02,
        "liquidity_intelligence": 0.01,
        "retail_behavior": 0.02,
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=10)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "NEUTRAL", f"Expected NEUTRAL, got {bias}"
    print("  ✓ PASSED\n")
    
    # Test 5: Liquidity trap (HIGH liquidity_intelligence) + high risk_score
    print("Test 5: Liquidity trap setup (high risk)")
    components = {
        "volume_behavior": 0.15,
        "structure_failure": 0.1,
        "momentum_exhaustion": 0.12,
        "liquidity_intelligence": 0.72,    # Dominant
        "retail_behavior": 0.1,
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=65)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "SHORT", f"Expected SHORT, got {bias}"
    assert conf == "HIGH", f"Expected HIGH confidence, got {conf}"
    print("  ✓ PASSED\n")
    
    # Test 6: High momentum exhaustion + low risk_score still shows direction
    print("Test 6: Momentum exhaustion with low risk score")
    components = {
        "volume_behavior": 0.2,
        "structure_failure": 0.15,
        "momentum_exhaustion": 0.68,   # Dominant
        "liquidity_intelligence": 0.12,
        "retail_behavior": 0.08,
    }
    bias, conf = compute_directional_bias_from_components(components, risk_score=25)
    print(f"  Components: {components}")
    print(f"  Result: bias={bias}, confidence={conf}")
    assert bias == "SHORT", f"Expected SHORT, got {bias}"
    # At risk_score=25 with HIGH dominance, should get MEDIUM
    assert conf in ("LOW", "MEDIUM"), f"Expected LOW or MEDIUM, got {conf}"
    print("  ✓ PASSED - Shows direction even at low risk score!\n")
    
    print("=" * 60)
    print("All tests PASSED! ✓")
    print("Bias now reflects direction even at low confidence!")
    print("=" * 60)


if __name__ == "__main__":
    test_bias_computation()
