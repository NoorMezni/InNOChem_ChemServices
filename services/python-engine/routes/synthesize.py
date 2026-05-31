"""Step 2 — Synthesis Route Engine."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.molecule import parse_smiles, get_molecule_info
from models.retrosynthesis import generate_synthesis_routes

router = APIRouter()


class SynthesizeRequest(BaseModel):
    smiles: str


@router.post("/routes")
def get_synthesis_routes(req: SynthesizeRequest):
    mol = parse_smiles(req.smiles.strip())
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    mol_info = get_molecule_info(mol, req.smiles)
    routes = generate_synthesis_routes(req.smiles)

    if not routes:
        raise HTTPException(status_code=500, detail="No synthesis routes could be generated.")

    best = routes[0]
    worst = routes[-1]

    return {
        "molecule": {
            "smiles": mol_info["smiles"],
            "formula": mol_info["formula"],
            "molecular_weight": mol_info["molecular_weight"],
        },
        "routes": routes,
        "summary": {
            "n_routes": len(routes),
            "best_route": {
                "rank": 1,
                "name": best["reaction_type"],
                "green_score": best["green_metrics"]["green_score"],
                "e_factor": best["green_metrics"]["e_factor"],
                "atom_economy_pct": best["green_metrics"]["atom_economy_pct"],
            },
            "worst_route": {
                "rank": len(routes),
                "name": worst["reaction_type"],
                "green_score": worst["green_metrics"]["green_score"],
                "e_factor": worst["green_metrics"]["e_factor"],
            },
            "improvement_potential_pct": round(
                best["green_metrics"]["green_score"] - worst["green_metrics"]["green_score"], 1
            ),
        },
        "step": 2,
        "step_name": "Synthesis Route Engine",
    }
