"""Step 4 — XAI: SHAP-based explainability."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.molecule import parse_smiles
from models.shap_explainer import explain_prediction

router = APIRouter()


class ExplainRequest(BaseModel):
    smiles: str


@router.post("/explain")
def explain_molecule(req: ExplainRequest):
    mol = parse_smiles(req.smiles.strip())
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    explanation = explain_prediction(mol, req.smiles)

    return {
        **explanation,
        "step": 4,
        "step_name": "XAI — SHAP Atom-Level Explanations",
        "xai_note": (
            "SHAP (SHapley Additive exPlanations) assigns each molecular feature "
            "a contribution to the toxicity prediction. Red atoms increase predicted "
            "toxicity; green atoms decrease it. Values are exact Shapley values "
            "(TreeExplainer, Lundberg & Lee 2017)."
        ),
    }
