"""
Unit tests for FallbackRoutingGate (Phase 8, Task 8.1)
"""
from core.confidence.fallback_gate import FallbackRoutingGate

def test_fallback_gate_high_confidence():
    gate = FallbackRoutingGate(confidence_threshold=0.70)
    should_fallback, reasons, overall_conf = gate.evaluate(
        dealer_name="Renuka Motors",
        dealer_conf=0.92,
        model_name="Eicher 548",
        model_conf=0.88,
        horse_power=48.0,
        hp_conf=0.95,
        cost_val=860000.0,
        cost_conf=0.90,
        sig_present=True,
        stamp_present=True
    )
    assert should_fallback is False
    assert len(reasons) == 0
    assert overall_conf >= 0.85

def test_fallback_gate_low_confidence_trigger():
    gate = FallbackRoutingGate(confidence_threshold=0.70)
    should_fallback, reasons, overall_conf = gate.evaluate(
        dealer_name="UNKNOWN_DEALER",
        dealer_conf=0.30,
        model_name="Mahindra 575 DI SP Plus",
        model_conf=0.50,
        horse_power=47.0,
        hp_conf=0.50,
        cost_val=745000.0,
        cost_conf=0.50,
        sig_present=False,
        stamp_present=False
    )
    assert should_fallback is True
    assert len(reasons) > 0
    assert overall_conf < 0.70
