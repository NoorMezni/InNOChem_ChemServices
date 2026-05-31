"""
SVHC (Substances of Very High Concern) regulatory risk engine.

Reference:
  ECHA Candidate List of SVHCs for authorisation
  https://echa.europa.eu/candidate-list-table

  Tanimoto similarity via RDKit Morgan fingerprints.
  Rogers & Hahn, J. Chem. Inf. Model. 2010, 50, 742-754.
"""
import json
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import Optional

_SVHC_DB: list[dict] = []
_SVHC_MOLS: list[tuple[dict, object]] = []

SIMILARITY_THRESHOLD_HIGH = 0.80
SIMILARITY_THRESHOLD_MEDIUM = 0.50


def load_svhc_db() -> None:
    global _SVHC_DB, _SVHC_MOLS
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "svhc.json")
    with open(data_path) as f:
        _SVHC_DB = json.load(f)

    _SVHC_MOLS = []
    for entry in _SVHC_DB:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            _SVHC_MOLS.append((entry, fp))


def compute_svhc_risk(smiles: str) -> dict:
    if not _SVHC_MOLS:
        load_svhc_db()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"risk_level": "UNKNOWN", "matches": [], "max_similarity": 0.0}

    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    matches = []
    max_sim = 0.0

    for entry, ref_fp in _SVHC_MOLS:
        sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
        if sim > SIMILARITY_THRESHOLD_MEDIUM:
            matches.append({
                "name": entry["name"],
                "cas": entry["cas"],
                "reason": entry["reason"],
                "category": entry["category"],
                "similarity": round(sim, 3),
                "risk_level": "HIGH" if sim >= SIMILARITY_THRESHOLD_HIGH else "MEDIUM",
            })
        if sim > max_sim:
            max_sim = sim

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    matches = matches[:5]

    if max_sim >= SIMILARITY_THRESHOLD_HIGH:
        risk_level = "HIGH"
        reach_flag = "SVHC_CANDIDATE"
        regulation_note = "Structural similarity to ECHA SVHC candidate(s). Authorisation may be required under REACH."
    elif max_sim >= SIMILARITY_THRESHOLD_MEDIUM:
        risk_level = "MEDIUM"
        reach_flag = "WATCH_LIST"
        regulation_note = "Moderate structural similarity to known SVHC substances. Regulatory monitoring recommended."
    else:
        risk_level = "LOW"
        reach_flag = "CLEAR"
        regulation_note = "No significant structural similarity to known SVHC substances found in current dataset."

    return {
        "risk_level": risk_level,
        "reach_flag": reach_flag,
        "regulation_note": regulation_note,
        "max_similarity": round(max_sim, 3),
        "matches": matches,
        "dataset_size": len(_SVHC_MOLS),
        "similarity_method": "Tanimoto / Morgan FP (radius=2, 2048 bits)",
        "source": "ECHA SVHC Candidate List",
    }


def check_reach_compliance(smiles: str) -> dict:
    risk = compute_svhc_risk(smiles)
    is_compliant = risk["risk_level"] == "LOW"
    return {
        "compliant": is_compliant,
        "svhc_risk": risk,
        "recommendation": (
            "Safe to proceed under current REACH regulations."
            if is_compliant
            else f"Review required. {risk['regulation_note']}"
        ),
    }
