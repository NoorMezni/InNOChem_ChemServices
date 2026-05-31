"""
SHAP-based explainability for QSAR predictions.

Uses TreeExplainer for the Random Forest model (exact Shapley values).
Maps feature contributions back to molecular atoms for visualization.

References:
  - SHAP: Lundberg & Lee, NeurIPS 2017. arXiv:1705.07874
  - Atom-level contributions: Rodríguez-Pérez & Bajorath, J. Med. Chem. 2020, 63, 8761
"""
import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from models.qsar import get_qsar_model, STRUCTURAL_ALERTS
from models.molecule import mol_to_svg_with_atom_colors
from typing import Optional


def explain_prediction(mol: Chem.Mol, smiles: str) -> dict:
    model = get_qsar_model()
    clf = model._xgb
    feature_names = model.get_feature_names()
    
    # Get scaled feature vector (what the model sees)
    feature_vector = model.get_scaled_feature_vector(mol).reshape(1, -1)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(feature_vector)

    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1][0]
    else:
        # For XGBoost binary, shap_values is usually just the log-odds for class 1
        shap_vals_class1 = shap_values[0]

    base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
    predicted_value = float(clf.predict_proba(feature_vector)[0][1])

    # Get raw feature vector to display unscaled values in UI
    raw_feature_vector = model.get_feature_vector(mol).reshape(1, -1)

    feature_contributions = []
    for i, (name, val) in enumerate(zip(feature_names, shap_vals_class1)):
        val_scalar = float(np.asarray(val).flat[0])
        # Skip displaying thousands of tiny 0-value FP bits in the UI
        if abs(val_scalar) < 1e-4 and name.startswith("fp_"):
            continue
            
        feat_scalar = float(np.asarray(raw_feature_vector[0][i]).flat[0])
        feature_contributions.append({
            "feature": name,
            "feature_value": round(feat_scalar, 4),
            "shap_value": round(val_scalar, 4),
            "direction": "increases_toxicity" if val_scalar > 0 else "decreases_toxicity",
        })

    feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    atom_contributions = _map_to_atoms(mol, shap_vals_class1, feature_names)

    atom_svg = mol_to_svg_with_atom_colors(mol, atom_contributions)

    waterfall_data = _build_waterfall(base_value, feature_contributions[:10], predicted_value)

    return {
        "base_value": round(base_value, 4),
        "predicted_value": round(predicted_value, 4),
        "shap_sum": round(float(np.sum(shap_vals_class1)), 4),
        "top_features": feature_contributions[:10],
        "atom_contributions": atom_contributions,
        "atom_svg": atom_svg,
        "waterfall": waterfall_data,
        "interpretation": _interpret_shap(feature_contributions[:5], predicted_value),
        "method": "SHAP TreeExplainer (XGBoost log-odds) — Lundberg & Lee, NeurIPS 2017",
        "model": "XGBoost Classifier — Ames/ClinTox/Tox21",
    }


def _map_to_atoms(mol: Chem.Mol, shap_vals: np.ndarray, feature_names: list[str]) -> list[dict]:
    n_atoms = mol.GetNumAtoms()
    atom_scores = [0.0] * n_atoms

    # 1. Map Morgan Fingerprint SHAP values accurately using bitInfo
    info = {}
    # Must match qsar.py: radius=2, nBits=1024
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=info)
    
    # FP features start after the 14 descriptors
    fp_start_idx = 14
    for bit, matches in info.items():
        bit_shap = float(np.asarray(shap_vals[fp_start_idx + bit]).flat[0])
        if bit_shap != 0:
            per_match = bit_shap / len(matches)
            for atom_idx, radius in matches:
                if atom_idx < n_atoms:
                    atom_scores[atom_idx] += per_match

    # 2. Map continuous RDKit descriptors (heuristic distribution)
    for i, name in enumerate(feature_names[:fp_start_idx]):
        shap_val = float(np.asarray(shap_vals[i]).flat[0])
        if shap_val == 0:
            continue
            
        aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
        polar_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 9, 15, 16, 17)]

        if name == "MolLogP" and aromatic_atoms:
            per_atom = shap_val / len(aromatic_atoms)
            for idx in aromatic_atoms:
                atom_scores[idx] += per_atom * 0.5
        elif name == "TPSA" and polar_atoms:
            per_atom = shap_val / len(polar_atoms)
            for idx in polar_atoms:
                atom_scores[idx] += per_atom * 0.5

    result = []
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        score = float(np.asarray(atom_scores[i]).flat[0])
        result.append({
            "atom_idx": i,
            "element": atom.GetSymbol(),
            "contribution": round(score, 4),
            "direction": "toxic" if score > 0 else ("safe" if score < 0 else "neutral"),
        })

    return result


def _build_waterfall(base_value: float, top_features: list[dict], final_value: float) -> list[dict]:
    waterfall = []
    running = base_value
    waterfall.append({
        "label": "Base value",
        "value": round(base_value, 4),
        "cumulative": round(running, 4),
        "type": "base",
    })

    for feat in top_features:
        delta = feat["shap_value"]
        running += delta
        waterfall.append({
            "label": feat["feature"],
            "value": round(delta, 4),
            "cumulative": round(running, 4),
            "type": "positive" if delta > 0 else "negative",
            "feature_value": feat["feature_value"],
        })

    waterfall.append({
        "label": "Predicted toxicity",
        "value": round(final_value, 4),
        "cumulative": round(final_value, 4),
        "type": "final",
    })
    return waterfall


def _interpret_shap(top_features: list[dict], predicted_value: float) -> str:
    if not top_features:
        return "No significant driving features identified."

    top_pos = [f for f in top_features if f["shap_value"] > 0]
    top_neg = [f for f in top_features if f["shap_value"] < 0]

    parts = []
    if top_pos:
        names = ", ".join(f["feature"] for f in top_pos[:3])
        parts.append(f"Key toxicity drivers: {names}")
    if top_neg:
        names = ", ".join(f["feature"] for f in top_neg[:2])
        parts.append(f"Mitigating factors: {names}")

    confidence_str = "high confidence" if abs(predicted_value - 0.5) > 0.3 else "moderate confidence"
    parts.append(f"Prediction made with {confidence_str} (score={predicted_value:.2f}).")

    return " | ".join(parts)
