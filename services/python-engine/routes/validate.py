"""Step 5 — Human-in-the-Loop validation and report generation."""
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.molecule import parse_smiles, get_molecule_info
from models.qsar import get_qsar_model
from models.svhc_engine import compute_svhc_risk
from models.retrosynthesis import generate_synthesis_routes
from models.shap_explainer import explain_prediction
from models.decision import compute_green_score

router = APIRouter()

_VALIDATION_STORE: dict[str, dict] = {
    "E8098B70": {
        "validation_id": "E8098B70",
        "timestamp": "2026-05-30T22:00:00+00:00",
        "chemist_name": "Dr. Jane Smith",
        "molecule": {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "formula": "C9H8O4",
            "mw": 180.16,
            "logp": 1.25,
            "tpsa": 63.6,
            "hbd": 1,
            "hba": 4
        },
        "system_recommendation": "APPROVE",
        "chemist_decision": "APPROVE",
        "chemist_comment": "This compound is clean. The screening results show acceptable toxicity and good biodegradability. Synthesized route uses eco-friendly solvents (ethanol/water) with a low E-factor.",
        "green_score": 85,
        "score_breakdown": {
            "toxicity": 90,
            "synthesis": 80,
            "svhc": 100,
            "biodeg": 70
        },
        "recommendations": [
            "Verify batch purity before clinical screening.",
            "Monitor E-factor during scale-up."
        ],
        "toxicity_summary": {
            "risk_level": "LOW",
            "risk_score": 22.5,
            "confidence": 0.88,
            "structural_alerts": ["Aromatic ring"],
            "endpoints": {
                "acute_toxicity": {"score": 2.5, "label": "Low Risk"},
                "biodegradability": {"score": 0.72, "label": "Biodegradable"}
            }
        },
        "svhc_summary": {
            "risk_level": "LOW",
            "reach_flag": "SAFE",
            "regulation_note": "No matches found on the SVHC candidate list.",
            "top_matches": []
        },
        "selected_synthesis_route": {
            "rank": 1,
            "name": "Esterification",
            "conditions": "Reflux, 3h",
            "green_metrics": {"e_factor": 4.2, "atom_economy": 83.5, "pmi": 5.2, "green_score": 85.0},
            "solvents": ["Ethanol", "Water"]
        },
        "xai_top_features": [
            {"feature": "Carboxylic Acid", "shap_value": -0.15},
            {"feature": "Aromatic Ring", "shap_value": 0.08}
        ],
        "decision_agreement": True,
        "methodology": {
            "qsar_model": "AttentiveFP (Tox21)",
            "svhc_method": "Tanimoto similarity screen",
            "green_score_weights": "Tox (40%), Synth (35%), SVHC (25%)",
            "shap_method": "KernelSHAP"
        },
        "datasets": {
            "toxicity": "Tox21 reference set (NIH/NTP) + ChEMBL published literature",
            "svhc": "ECHA SVHC Candidate List",
            "solvents": "CHEM21 Solvent Selection Guide (Prat et al., Green Chem. 2016)",
            "green_metrics": "Sheldon E-factor (2007), Trost AE (1991), Jimenez-Gonzalez PMI (2011)"
        }
    }
}


class ValidateRequest(BaseModel):
    smiles: str
    decision: str
    comment: Optional[str] = ""
    selected_route_rank: Optional[int] = 1
    chemist_name: Optional[str] = "Anonymous Chemist"


@router.post("/validate")
def validate_molecule(req: ValidateRequest):
    if req.decision not in ("APPROVE", "REDESIGN", "STOP"):
        raise HTTPException(status_code=400, detail="Decision must be APPROVE, REDESIGN, or STOP.")

    mol = parse_smiles(req.smiles.strip())
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    mol_info = get_molecule_info(mol, req.smiles)
    qsar = get_qsar_model()
    tox = qsar.predict(mol)
    svhc = compute_svhc_risk(req.smiles)
    routes = generate_synthesis_routes(req.smiles)
    explanation = explain_prediction(mol, req.smiles)

    best_route = routes[0] if routes else {}
    selected_route = next((r for r in routes if r["rank"] == req.selected_route_rank), best_route)

    biodeg_score = tox["endpoints"]["biodegradability"]["score"]
    biodeg_label = tox["endpoints"]["biodegradability"]["label"]
    decision_result = compute_green_score(
        tox_risk_level=tox["risk_level"],
        biodeg_score=biodeg_score,
        biodeg_label=biodeg_label,
        best_synthesis_green_score=best_route.get("green_metrics", {}).get("green_score", 50.0),
        svhc_risk_level=svhc["risk_level"],
    )

    validation_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "validation_id": validation_id,
        "timestamp": timestamp,
        "chemist_name": req.chemist_name,
        "molecule": mol_info,
        "system_recommendation": decision_result["decision"],
        "chemist_decision": req.decision,
        "chemist_comment": req.comment or "",
        "green_score": decision_result["green_score"],
        "score_breakdown": decision_result["component_scores"],
        "recommendations": decision_result["recommendations"],
        "toxicity_summary": {
            "risk_level": tox["risk_level"],
            "risk_score": tox["risk_score"],
            "confidence": tox["confidence"],
            "structural_alerts": [a["name"] for a in tox["structural_alerts"]],
            "endpoints": {
                k: {"score": v.get("score", v.get("BCF", 0)), "label": v["label"]}
                for k, v in tox["endpoints"].items()
            },
        },
        "svhc_summary": {
            "risk_level": svhc["risk_level"],
            "reach_flag": svhc["reach_flag"],
            "regulation_note": svhc["regulation_note"],
            "top_matches": svhc["matches"][:3],
        },
        "selected_synthesis_route": {
            "rank": selected_route.get("rank"),
            "name": selected_route.get("reaction_type"),
            "conditions": selected_route.get("conditions"),
            "green_metrics": selected_route.get("green_metrics", {}),
            "solvents": [s["name"] for s in selected_route.get("solvents", [])],
        },
        "xai_top_features": explanation["top_features"][:5],
        "decision_agreement": req.decision == decision_result["decision"],
        "methodology": {
            "qsar_model": tox["model_info"],
            "svhc_method": svhc.get("similarity_method", ""),
            "green_score_weights": decision_result["weights_used"],
            "shap_method": explanation["method"],
        },
        "datasets": {
            "toxicity": "Tox21 reference set (NIH/NTP) + ChEMBL published literature",
            "svhc": "ECHA SVHC Candidate List",
            "solvents": "CHEM21 Solvent Selection Guide (Prat et al., Green Chem. 2016)",
            "green_metrics": "Sheldon E-factor (2007), Trost AE (1991), Jimenez-Gonzalez PMI (2011)",
        },
    }

    _VALIDATION_STORE[validation_id] = report

    # Active Adaptive Learning Feedback Loop: retrain the QSAR ML model on the expert's feedback
    try:
        qsar.add_feedback(req.smiles.strip(), req.decision)
    except Exception as e:
        print(f"Error during QSAR active learning feedback retraining: {e}")

    return {
        "success": True,
        "validation_id": validation_id,
        "report": report,
        "step": 5,
        "step_name": "Human-in-the-Loop Validation",
        "hitl_note": (
            "Chemist validation recorded. The human decision overrides system recommendation "
            "for audit purposes. This feedback is stored and can improve future scoring "
            "(adaptive learning loop)."
        ),
    }


@router.get("/report/{validation_id}")
def get_report(validation_id: str):
    report = _VALIDATION_STORE.get(validation_id.upper())
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {validation_id} not found.")
    return report


@router.get("/solvents")
def list_solvents():
    from models.solvent_db import get_all_solvents
    return {"solvents": get_all_solvents()}
