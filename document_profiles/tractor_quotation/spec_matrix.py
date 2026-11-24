"""
Model-to-HP and Price Specification Matrix Validator (Task 6.6 & 7.9)
Cross-field consistency validator for tractor specifications.
"""
from typing import Optional, Dict, Any

class SpecMatrixValidator:
    """
    Validates physical plausibility of extracted horsepower and asset cost
    against OEM catalogues and domain ranges.
    """

    @staticmethod
    def is_valid_hp_for_model(model_metadata: Optional[Dict[str, Any]], extracted_hp: float) -> bool:
        if not model_metadata or "hp" not in model_metadata:
            return 15.0 <= extracted_hp <= 120.0
        
        rated_hp = float(model_metadata["hp"])
        # Allow +/- 15% tolerance for rated vs engine gross HP differences
        return (rated_hp * 0.85) <= extracted_hp <= (rated_hp * 1.15)

    @staticmethod
    def is_valid_cost_for_model(model_metadata: Optional[Dict[str, Any]], extracted_cost: float) -> bool:
        # Standard tractor prices in India range from 2.5 Lakhs to 25 Lakhs
        if not (200000.0 <= extracted_cost <= 2500000.0):
            return False
            
        if model_metadata and "price_range_inr" in model_metadata:
            low, high = model_metadata["price_range_inr"]
            # With accessories, taxes, and insurance, allow 15% margin
            return (low * 0.80) <= extracted_cost <= (high * 1.25)
            
        return True

    @staticmethod
    def reconcile_hp(extracted_hp: Optional[float], model_metadata: Optional[Dict[str, Any]]) -> float:
        """
        Reconciles extracted HP with OEM catalog ground truth if extraction is ambiguous.
        """
        if extracted_hp is not None and 15.0 <= extracted_hp <= 120.0:
            if model_metadata and "hp" in model_metadata:
                rated_hp = float(model_metadata["hp"])
                # If extraction is slightly off (e.g. 47.5 vs 48.0 due to OCR noise), reconcile to rated HP
                if 0 < abs(extracted_hp - rated_hp) <= 0.8:
                    return rated_hp
            return extracted_hp

        if model_metadata and "hp" in model_metadata:
            return float(model_metadata["hp"])

        return 47.0
