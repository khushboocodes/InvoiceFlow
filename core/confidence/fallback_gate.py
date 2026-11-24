"""
InvoiceFlow Confidence & Fallback Routing Gate (Phase 8, Task 8.1)
Evaluates multi-field extraction confidence and triggers surgical VLM fallback.
"""
from typing import Dict, Any, List, Tuple, Optional

class FallbackRoutingGate:
    """
    Evaluates extraction confidence scores and specification constraints
    to determine whether surgical Multimodal VLM fallback should be triggered.
    """
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        dealer_name: str,
        dealer_conf: float,
        model_name: str,
        model_conf: float,
        horse_power: float,
        hp_conf: float,
        cost_val: float,
        cost_conf: float,
        sig_present: bool,
        stamp_present: bool
    ) -> Tuple[bool, List[str], float]:
        """
        Evaluates extraction output.
        Returns (should_fallback, trigger_reasons, overall_document_confidence).
        """
        reasons: List[str] = []

        # 1. Dealer Name Check
        if dealer_conf < self.confidence_threshold or dealer_name in ["UNKNOWN_DEALER", ""]:
            reasons.append(f"Low dealer_name confidence ({dealer_conf:.2f}) or missing name")
        elif len(dealer_name.strip()) < 3:
            reasons.append(f"Dealer name too short ('{dealer_name}')")

        # 2. Model Name Check
        if model_conf < self.confidence_threshold or "Generic" in model_name:
            reasons.append(f"Low model_name confidence ({model_conf:.2f})")

        # 3. Horse Power Bounds Check
        if hp_conf < self.confidence_threshold or not (15.0 <= horse_power <= 120.0):
            reasons.append(f"Invalid or low confidence HP ({horse_power} HP, conf: {hp_conf:.2f})")

        # 4. Asset Cost Bounds Check
        if cost_conf < self.confidence_threshold or not (200000.0 <= cost_val <= 2500000.0):
            reasons.append(f"Invalid or low confidence Asset Cost (₹{cost_val:,.0f}, conf: {cost_conf:.2f})")

        # 5. Visual Stamp & Signature Check
        if not sig_present or not stamp_present:
            reasons.append(f"Missing visual entity (sig={sig_present}, stamp={stamp_present})")

        # Critical field check: If ANY core financial/tractor field defaulted or has low confidence
        is_default_cost = (cost_val == 745000.0 and cost_conf <= 0.65)
        is_default_hp = (horse_power == 47.0 and hp_conf <= 0.65)
        is_generic_model = ("Generic" in model_name or model_conf < self.confidence_threshold)
        is_bad_dealer = (dealer_conf < self.confidence_threshold or dealer_name in ["UNKNOWN_DEALER", ""])

        if is_default_cost:
            reasons.append("Default fallback Asset Cost detected (₹745,000)")
        if is_default_hp:
            reasons.append("Default fallback HP detected (47.0 HP)")

        # Composite document confidence
        overall_conf = round(
            (dealer_conf * 0.25) +
            (model_conf * 0.25) +
            (hp_conf * 0.20) +
            (cost_conf * 0.20) +
            ((0.90 if sig_present and stamp_present else 0.50) * 0.10),
            2
        )

        # Trigger VLM safety net if any critical field failed, or if any reason exists and doc confidence < 0.92
        should_fallback = (
            is_default_cost or
            is_default_hp or
            is_generic_model or
            is_bad_dealer or
            (len(reasons) > 0 and overall_conf < 0.92)
        )
        return should_fallback, reasons, overall_conf
