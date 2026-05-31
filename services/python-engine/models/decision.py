"""
GreenScore decision aggregator.

Combines toxicity risk, synthesis route quality, SVHC regulatory risk,
and biodegradability into a composite GreenScore (0-100).
Final decision: APPROVE (>=70), REDESIGN (40-69), STOP (<40).
"""
from typing import Optional


WEIGHTS = {
    "toxicity": 0.35,
    "biodegradability": 0.20,
    "synthesis_quality": 0.25,
    "regulatory": 0.20,
}

APPROVE_THRESHOLD = 70
REDESIGN_THRESHOLD = 40


def _risk_to_score(risk_level: str) -> float:
    mapping = {"LOW": 85.0, "MEDIUM": 50.0, "HIGH": 15.0, "UNKNOWN": 40.0}
    return mapping.get(risk_level.upper(), 40.0)


def _biodeg_to_score(biodeg_label: str, biodeg_score: float) -> float:
    return biodeg_score * 100.0


def _synthesis_to_score(best_green_score: float) -> float:
    return best_green_score


def _regulatory_to_score(svhc_risk_level: str) -> float:
    mapping = {"LOW": 90.0, "MEDIUM": 45.0, "HIGH": 10.0, "UNKNOWN": 50.0}
    return mapping.get(svhc_risk_level.upper(), 50.0)


def compute_green_score(
    tox_risk_level: str,
    biodeg_score: float,
    biodeg_label: str,
    best_synthesis_green_score: float,
    svhc_risk_level: str,
    custom_weights: Optional[dict] = None,
) -> dict:
    weights = custom_weights if custom_weights else WEIGHTS

    component_scores = {
        "toxicity": _risk_to_score(tox_risk_level),
        "biodegradability": _biodeg_to_score(biodeg_label, biodeg_score),
        "synthesis_quality": _synthesis_to_score(best_synthesis_green_score),
        "regulatory": _regulatory_to_score(svhc_risk_level),
    }

    green_score = sum(
        component_scores[k] * weights.get(k, 0.0)
        for k in component_scores
    )
    green_score = max(0.0, min(100.0, green_score))

    if green_score >= APPROVE_THRESHOLD:
        decision = "APPROVE"
        color = "green"
        decision_text = "Molecule meets green chemistry criteria. Proceed with development."
    elif green_score >= REDESIGN_THRESHOLD:
        decision = "REDESIGN"
        color = "orange"
        decision_text = "Molecule requires optimization before proceeding. Review flagged criteria."
    else:
        decision = "STOP"
        color = "red"
        decision_text = "Molecule fails green chemistry criteria. Development not recommended without major redesign."

    recommendations = _generate_recommendations(
        tox_risk_level, biodeg_label, biodeg_score, best_synthesis_green_score, svhc_risk_level
    )

    return {
        "green_score": round(green_score, 1),
        "decision": decision,
        "decision_color": color,
        "decision_text": decision_text,
        "component_scores": {k: round(v, 1) for k, v in component_scores.items()},
        "weights_used": weights,
        "thresholds": {
            "approve": APPROVE_THRESHOLD,
            "redesign": REDESIGN_THRESHOLD,
        },
        "recommendations": recommendations,
        "methodology": "Weighted composite score. Weights: toxicity 35%, synthesis quality 25%, regulatory 20%, biodegradability 20%.",
    }


def _generate_recommendations(
    tox_risk: str,
    biodeg_label: str,
    biodeg_score: float,
    synth_score: float,
    svhc_risk: str,
) -> list[str]:
    recs = []

    if tox_risk == "HIGH":
        recs.append("Consider Safe-by-Design modifications: replace problematic functional groups (nitro → amide, aromatic amine → aliphatic).")
    elif tox_risk == "MEDIUM":
        recs.append("Run additional in vitro assays to confirm toxicity profile before scaling up.")

    if biodeg_score < 0.35:
        recs.append("Improve biodegradability: introduce ester or amide linkages, reduce halogen content, avoid polycyclic aromatic structures.")
    elif biodeg_label == "Medium":
        recs.append("Moderate biodegradability detected. Consider environmental fate studies before industrial scale-up.")

    if synth_score < 50:
        recs.append("Switch to greener synthesis route: use aqueous or bio-based solvents, minimize coupling reagent waste.")
    elif synth_score < 70:
        recs.append("Optimize solvent choice: replace halogenated solvents (DCM, CHCl₃) with EtOAc, 2-MeTHF, or water.")

    if svhc_risk == "HIGH":
        recs.append("REACH compliance required: molecule shows structural similarity to SVHC substances. Consult regulatory affairs before proceeding.")
    elif svhc_risk == "MEDIUM":
        recs.append("Regulatory watch: monitor ECHA candidate list updates. Document structural justification for non-restriction.")

    if not recs:
        recs.append("Molecule meets all green chemistry thresholds. Document datasets and methods for regulatory submission.")

    return recs


def format_decision_summary(green_score_result: dict, mol_info: dict) -> dict:
    return {
        "molecule": {
            "smiles": mol_info.get("smiles", ""),
            "formula": mol_info.get("formula", ""),
            "mw": mol_info.get("molecular_weight", 0),
        },
        "decision": green_score_result["decision"],
        "green_score": green_score_result["green_score"],
        "decision_text": green_score_result["decision_text"],
        "score_breakdown": green_score_result["component_scores"],
        "recommendations": green_score_result["recommendations"],
    }
