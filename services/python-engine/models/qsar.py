"""
QSAR (Quantitative Structure-Activity Relationship) toxicity model.

Core classifier
---------------
  Algorithm : XGBoost (XGBClassifier, binary:logistic)
  Trained on: Ames mutagenicity dataset (combined Ames + ClinTox + Tox21,
               public domain via Kaggle / NIH)
  Features  : 14 RDKit physicochemical/structural/topological descriptors
               (StandardScaler-normalised) + 1024-bit Morgan ECFP4 fingerprint
               = 1 038 features total
  Artefacts : xgboost_ames.json  — serialised XGBoost model
               scaler.pkl        — fitted StandardScaler (descriptors only)
               metadata.json     — feature names, split metrics, hyperparams

Derived endpoints (rule-based, on top of the classifier output)
---------------
  Ecotoxicity    : composite of P(toxic) + LogP aquatic-transfer term
  Biodegradability: BIOWIN-inspired linear rule on polar-surface-area & LogP
  Bioaccumulation: BCF = 10^(0.77 × LogP − 0.70)  (Meylan & Howard, 1993)

Structural alerts: Brenk et al., ChemMedChem 2008, 3, 435-444

References
----------
  Ames dataset  : https://www.kaggle.com/datasets/heithembenmoussa2/ames-clintox-tox21
  Tox21         : https://tox21.gov/  (NIH/NTP, public domain)
  Brenk alerts  : Brenk et al. ChemMedChem 2008, 3, 435-444
  BCF model     : Meylan & Howard, Environ. Toxicol. Chem. 1993, 12, 2177
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Artifact paths — resolve relative to this file so callers can import from
# any working directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

def _find_artifact(name: str) -> Path:
    """Search for an artifact file next to this module or in a sub-folder."""
    candidates = [
        _HERE / name,
        _HERE / "combined_model_artefacts" / name,
        _HERE / "artefacts" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Artifact '{name}' not found. Expected locations:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# ---------------------------------------------------------------------------
# Featurisation constants (must match the training notebook exactly)
# ---------------------------------------------------------------------------
DESCRIPTOR_NAMES: list[str] = [
    # Physicochemical
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    # Structural
    "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NumHeteroatoms", "NumAromaticRings",
    # Topological
    "BertzCT", "BalabanJ", "HallKierAlpha",
]
N_DESC    : int = len(DESCRIPTOR_NAMES)   # 14
FP_BITS   : int = 1024
FP_RADIUS : int = 2
FP_NAMES  : list[str] = [f"fp_{i}" for i in range(FP_BITS)]
ALL_FEATURE_NAMES: list[str] = DESCRIPTOR_NAMES + FP_NAMES  # 1 038


# ---------------------------------------------------------------------------
# Structural alerts (Brenk et al. 2008)
# ---------------------------------------------------------------------------
STRUCTURAL_ALERTS: dict[str, str] = {
    "nitro_group":          "[N+](=O)[O-]",
    "aromatic_amine":       "[NH2,NH]c",
    "aldehyde":             "[CH]=O",
    "michael_acceptor":     "C=CC=O",
    "epoxide":              "C1OC1",
    "acyl_halide":          "C(=O)[Cl,Br,F,I]",
    "isocyanate":           "N=C=O",
    "diazo":                "[N]=[N+]=[C]",
    "peroxide":             "OO",
    "nitroso":              "N=O",
    "azo":                  "N=Nc",
    "quinone":              "O=C1C=CC(=O)C=C1",
    "halogenated_aromatic": "c[Cl,Br,I]",
    "polycyclic_aromatic":  "c1ccc2ccccc2c1",
    "thiol":                "[SH]",
    "heavy_halogen":        "[F,Cl,Br,I]",
}

_ALERT_DESCRIPTIONS: dict[str, str] = {
    "nitro_group":          "Potential mutagenicity via nitroreductase activation",
    "aromatic_amine":       "Known carcinogenicity risk via N-oxidation pathway",
    "aldehyde":             "Reactive electrophile; protein/DNA adduct formation",
    "michael_acceptor":     "Reactive electrophile; covalent binding to biomolecules",
    "epoxide":              "Highly reactive; DNA alkylation mutagenicity risk",
    "acyl_halide":          "Highly reactive; protein acylation",
    "isocyanate":           "Respiratory sensitiser; reactive with nucleophiles",
    "diazo":                "Highly reactive; potential DNA methylation agent",
    "peroxide":             "Oxidative stress; radical generation",
    "nitroso":              "Nitrosamine precursor; genotoxicity risk",
    "azo":                  "Potential reductive cleavage to aromatic amines",
    "quinone":              "Redox cycling; reactive oxygen species generation",
    "halogenated_aromatic": "Potential persistent pollutant; bioaccumulation concern",
    "polycyclic_aromatic":  "Carcinogenicity via metabolic activation (PAH class)",
    "thiol":                "Reactive towards disulfide bonds; oxidative stress",
    "heavy_halogen":        "Potential for bioaccumulation and environmental persistence",
}

# Pre-compile alert patterns once at import time
_COMPILED_ALERTS: dict[str, Optional[Chem.Mol]] = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in STRUCTURAL_ALERTS.items()
}


# ---------------------------------------------------------------------------
# Featurisation helpers
# ---------------------------------------------------------------------------

def _mol_to_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    Compute the 14 RDKit descriptors used during training.
    Returns float32 array of shape (14,).  All NaN on failure.
    """
    try:
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            BertzCT(mol),
            BalabanJ(mol),
            Descriptors.HallKierAlpha(mol),
        ], dtype=np.float32)
    except Exception:
        return np.full(N_DESC, np.nan, dtype=np.float32)


def _mol_to_morgan_fp(mol: Chem.Mol) -> np.ndarray:
    """
    Compute Morgan ECFP4 fingerprint.
    Returns uint8 binary array of shape (FP_BITS,).
    """
    arr = np.zeros(FP_BITS, dtype=np.uint8)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
        for bit in fp.GetOnBits():
            arr[bit] = 1
    except Exception:
        pass
    return arr


def _build_feature_vector(mol: Chem.Mol) -> tuple[np.ndarray, bool]:
    """
    Build the full 1 038-dimensional feature vector for a molecule.

    Returns
    -------
    x_raw : float32 array shape (1038,) — descriptors + fingerprint (unscaled)
    valid : bool — False if descriptors contain NaN (molecule should be skipped)
    """
    desc = _mol_to_descriptors(mol)
    if np.isnan(desc).any():
        return np.zeros(N_DESC + FP_BITS, dtype=np.float32), False
    fp = _mol_to_morgan_fp(mol)
    x_raw = np.hstack([desc, fp.astype(np.float32)])
    return x_raw, True


def _check_structural_alerts(mol: Chem.Mol) -> list[dict]:
    """Return list of matched structural alert dicts."""
    found = []
    for name, pattern in _COMPILED_ALERTS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append({
                "name":        name.replace("_", " ").title(),
                "smarts":      STRUCTURAL_ALERTS[name],
                "description": _ALERT_DESCRIPTIONS.get(
                    name, "Structural alert flagged by Brenk et al. (2008)"
                ),
            })
    return found


# ---------------------------------------------------------------------------
# QSARModel — public API
# ---------------------------------------------------------------------------

class QSARModel:
    """
    QSAR toxicity predictor backed by a pre-trained XGBoost classifier.

    Artifact loading
    ----------------
    Place the three artifact files next to this module (or in a sub-folder
    called ``combined_model_artefacts/`` or ``artefacts/``):

        xgboost_ames.json   — XGBoost model
        scaler.pkl          — fitted StandardScaler
        metadata.json       — feature/training metadata

    Usage
    -----
        from qsar import get_qsar_model
        model = get_qsar_model()
        result = model.predict(mol)   # mol is an rdkit.Chem.Mol object
    """

    def __init__(self) -> None:
        self._xgb: XGBClassifier = XGBClassifier()
        self._scaler = None
        self._metadata: dict = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model artifacts from disk.  Called automatically on first predict."""
        self._xgb.load_model(str(_find_artifact("xgboost_ames.json")))

        with open(_find_artifact("scaler.pkl"), "rb") as fh:
            self._scaler = pickle.load(fh)

        with open(_find_artifact("metadata.json")) as fh:
            self._metadata = json.load(fh)

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, x_raw: np.ndarray) -> np.ndarray:
        """Scale the descriptor block; leave fingerprint bits untouched."""
        x_proc = x_raw.copy().astype(np.float32)
        x_proc[:N_DESC] = self._scaler.transform(x_raw[:N_DESC].reshape(1, -1))[0]
        return x_proc

    # ------------------------------------------------------------------
    # Derived endpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _biodeg_score(logp: float, tpsa: float) -> float:
        """
        Lightweight BIOWIN-inspired biodegradability score [0, 1].
        Higher = more biodegradable.
        Molecules with low LogP and higher polar surface area tend to be
        more readily biodegraded.
        """
        score = 0.8 - 0.08 * max(0.0, logp) + 0.002 * tpsa
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _bcf(logp: float) -> float:
        """BCF = 10^(0.77 × LogP − 0.70)  (Meylan & Howard, 1993)."""
        return max(0.0, 10 ** (0.77 * logp - 0.70))

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(self, mol: Chem.Mol) -> dict:
        """
        Run full QSAR prediction for a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.Mol

        Returns
        -------
        dict with keys:
            risk_level        : 'LOW' | 'MEDIUM' | 'HIGH'
            risk_score        : float [0, 1]
            confidence        : float [0, 1]  — model certainty
            endpoints         : dict with acute_toxicity, ecotoxicity,
                                biodegradability, bioaccumulation sub-dicts
            structural_alerts : list of matched alert dicts
            top_descriptors   : list of {feature, value} for the 14 descriptors
            model_info        : provenance / metadata dict
        """
        self._ensure_loaded()

        # ── Featurise ────────────────────────────────────────────────────
        x_raw, valid = _build_feature_vector(mol)
        if not valid:
            raise ValueError("Could not compute RDKit descriptors for this molecule.")

        x_proc = self._preprocess(x_raw)

        # ── XGBoost inference ─────────────────────────────────────────────
        x_in  = x_proc.reshape(1, -1)
        tox_prob = float(self._xgb.predict_proba(x_in)[0, 1])

        # Confidence: distance from the 0.5 decision boundary, rescaled to [0.5, 1]
        confidence = float(np.clip(0.5 + abs(tox_prob - 0.5), 0.5, 1.0))

        # ── Derived physicochemical values ────────────────────────────────
        logp = float(x_raw[DESCRIPTOR_NAMES.index("MolLogP")])
        tpsa = float(x_raw[DESCRIPTOR_NAMES.index("TPSA")])

        biodeg_score = self._biodeg_score(logp, tpsa)
        bcf          = self._bcf(logp)
        ecotox_score = float(np.clip(tox_prob * 0.6 + max(0.0, logp - 3.0) * 0.1, 0.0, 1.0))

        # ── Endpoint labels ───────────────────────────────────────────────
        def _label(score: float) -> str:
            return "High" if score > 0.6 else ("Medium" if score > 0.3 else "Low")

        biodeg_label  = "High" if biodeg_score > 0.6 else ("Medium" if biodeg_score > 0.35 else "Low")
        bioaccum_label = "High" if bcf > 3.7 else ("Medium" if bcf > 2.0 else "Low")

        # ── Composite risk ────────────────────────────────────────────────
        composite = (
            tox_prob       * 0.40
            + ecotox_score * 0.25
            + (1 - biodeg_score) * 0.20
            + min(1.0, bcf / 5.0) * 0.15
        )
        if composite < 0.3:
            risk_level = "LOW"
        elif composite < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # ── Structural alerts ─────────────────────────────────────────────
        alerts_found = _check_structural_alerts(mol)

        # ── Top descriptors (by raw value, for interpretability) ──────────
        top_descriptors = [
            {"feature": name, "value": round(float(x_raw[i]), 4)}
            for i, name in enumerate(DESCRIPTOR_NAMES)
        ]

        return {
            "risk_level":  risk_level,
            "risk_score":  round(composite, 3),
            "confidence":  round(confidence, 3),
            "endpoints": {
                "acute_toxicity": {
                    "score":  round(tox_prob, 3),
                    "label":  _label(tox_prob),
                    "method": (
                        "XGBoost classifier — trained on Ames/ClinTox/Tox21 dataset "
                        "(14 RDKit descriptors + 1024-bit Morgan ECFP4)"
                    ),
                },
                "ecotoxicity": {
                    "score":  round(ecotox_score, 3),
                    "label":  _label(ecotox_score),
                    "method": "Composite: P(acute toxic) × 0.6 + max(0, LogP−3) × 0.1",
                },
                "biodegradability": {
                    "score":  round(biodeg_score, 3),
                    "label":  biodeg_label,
                    "method": "BIOWIN-inspired rule: f(LogP, TPSA)",
                },
                "bioaccumulation": {
                    "logP":   round(logp, 3),
                    "BCF":    round(bcf, 2),
                    "label":  bioaccum_label,
                    "method": "BCF = 10^(0.77 × LogP − 0.70)  (Meylan & Howard, 1993)",
                },
            },
            "structural_alerts": alerts_found,
            "top_descriptors":   top_descriptors,
            "model_info": {
                "algorithm":      "XGBoost (XGBClassifier, binary:logistic)",
                "n_features":     N_DESC + FP_BITS,
                "best_iteration": self._metadata.get("best_iteration"),
                "training_set":   "Ames mutagenicity + ClinTox + Tox21 (public domain)",
                "test_metrics":   self._metadata.get("test_metrics", {}),
                "version":        "2.0.0",
            },
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_smiles(self, smiles: str, threshold: float = 0.5) -> dict:
        """
        Parse a SMILES string and run predict().

        Returns the full result dict plus a top-level 'valid' flag and
        'label' / 'predicted_class' matching the notebook's inference API.
        Raises ValueError for unparseable SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "valid": False, "error": "Invalid SMILES"}

        result = self.predict(mol)
        tox_prob = result["endpoints"]["acute_toxicity"]["score"]
        cls      = int(tox_prob >= threshold)

        result["smiles"]          = smiles
        result["valid"]           = True
        result["predicted_class"] = cls
        result["label"]           = "Toxic (mutagenic)" if cls == 1 else "Non-toxic"
        result["probability"]     = round(tox_prob, 4)
        return result

    def get_feature_names(self) -> list[str]:
        """Return the full list of 1 038 feature names (descriptors + FP bits)."""
        return ALL_FEATURE_NAMES

    def get_descriptor_names(self) -> list[str]:
        """Return the 14 RDKit descriptor names used as continuous features."""
        return DESCRIPTOR_NAMES

    def get_feature_vector(self, mol: Chem.Mol) -> np.ndarray:
        """
        Return the raw (unscaled) 1 038-dim feature vector for a molecule.
        Useful for external explainability tools (SHAP, LIME, etc.).
        """
        x_raw, valid = _build_feature_vector(mol)
        if not valid:
            raise ValueError("Could not compute descriptors for this molecule.")
        return x_raw

    def get_scaled_feature_vector(self, mol: Chem.Mol) -> np.ndarray:
        """
        Return the preprocessed feature vector (descriptors scaled,
        fingerprint bits untouched) — exactly what the model sees.
        """
        self._ensure_loaded()
        x_raw, valid = _build_feature_vector(mol)
        if not valid:
            raise ValueError("Could not compute descriptors for this molecule.")
        return self._preprocess(x_raw)

    def add_feedback(self, smiles: str, decision: str) -> None:
        """
        Record human expert feedback for a molecule.

        Parameters
        ----------
        smiles   : SMILES string of the molecule
        decision : 'APPROVE' (non-toxic) or 'STOP' (toxic)

        Note: XGBoost does not support incremental learning; feedback is
        stored in-memory for audit/logging purposes.  To retrain with
        corrections, collect feedback entries and retrain from the notebook.
        """
        if decision not in ("APPROVE", "STOP"):
            raise ValueError("decision must be 'APPROVE' or 'STOP'")
        if not hasattr(self, "_feedback_log"):
            self._feedback_log: list[dict] = []
        self._feedback_log.append({"smiles": smiles, "decision": decision})

    def get_feedback_log(self) -> list[dict]:
        """Return all recorded feedback entries."""
        return getattr(self, "_feedback_log", [])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_MODEL_INSTANCE: Optional[QSARModel] = None


def get_qsar_model() -> QSARModel:
    """Return the shared, lazily-loaded QSARModel singleton."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = QSARModel()
    return _MODEL_INSTANCE


# ---------------------------------------------------------------------------
# Quick self-test (python qsar.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("Aspirin",         "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine",        "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Benzo[a]pyrene",  "C1=CC2=C3C=CC=CC3=C4C=CC=CC4=C2C=C1"),
        ("Aflatoxin B1",    "O=c1occc2c1cc1c(c2OC)c2c(cc1OC)OCO2"),
        ("NaCl",            "[Na+].[Cl-]"),
    ]

    model = get_qsar_model()
    print(f"{'Molecule':<20} {'P(toxic)':>9} {'Class':<22} {'Risk':>6}  Alerts")
    print("-" * 75)
    for name, smi in test_cases:
        r = model.predict_smiles(smi)
        if not r["valid"]:
            print(f"{name:<20}  {'INVALID SMILES':}")
            continue
        alerts = ", ".join(a["name"] for a in r["structural_alerts"]) or "none"
        print(
            f"{name:<20} {r['probability']:>9.4f}  {r['label']:<22} "
            f"{r['risk_level']:>6}  {alerts}"
        )