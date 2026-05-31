"""Step 1 — Early Risk Screening."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.molecule import parse_smiles, get_molecule_info
from models.qsar import get_qsar_model
from models.svhc_engine import compute_svhc_risk
from models.decision import compute_green_score

router = APIRouter()


class AnalyzeRequest(BaseModel):
    smiles: str


@router.post("/analyze")
def analyze_molecule(req: AnalyzeRequest):
    mol = parse_smiles(req.smiles.strip())
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string. Could not parse molecule.")

    mol_info = get_molecule_info(mol, req.smiles)
    qsar = get_qsar_model()
    try:
        tox_result = qsar.predict(mol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    svhc_result = compute_svhc_risk(req.smiles)

    biodeg_score = tox_result["endpoints"]["biodegradability"]["score"]
    biodeg_label = tox_result["endpoints"]["biodegradability"]["label"]

    decision = compute_green_score(
        tox_risk_level=tox_result["risk_level"],
        biodeg_score=biodeg_score,
        biodeg_label=biodeg_label,
        best_synthesis_green_score=60.0,
        svhc_risk_level=svhc_result["risk_level"],
    )

    return {
        "molecule": mol_info,
        "toxicity": tox_result,
        "svhc": svhc_result,
        "preliminary_decision": {
            "green_score": decision["green_score"],
            "decision": decision["decision"],
            "decision_text": decision["decision_text"],
            "decision_color": decision["decision_color"],
            "recommendations": decision["recommendations"],
        },
        "step": 1,
        "step_name": "Early Risk Screening",
    }
