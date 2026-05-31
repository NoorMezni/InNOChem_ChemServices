import os
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

try:
    from rdkit.Chem.Draw import rdMolDraw2D
    _DRAW_AVAILABLE = True
except (ImportError, OSError):
    _DRAW_AVAILABLE = False


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    return mol


def _smiles_to_fallback_svg(smiles: str, mol: Chem.Mol) -> str:
    """
    Minimal inline SVG fallback when rdMolDraw2D is unavailable.
    Returns a text-based representation.
    """
    formula = rdMolDescriptors.CalcMolFormula(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    mw = round(Descriptors.MolWt(mol), 1)
    short_smi = smiles[:40] + ("…" if len(smiles) > 40 else "")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" '
        'style="background:#f8fafc;border-radius:8px;font-family:monospace;">'
        f'<text x="200" y="80" text-anchor="middle" font-size="18" fill="#1e293b" font-weight="bold">{formula}</text>'
        f'<text x="200" y="116" text-anchor="middle" font-size="11" fill="#64748b">MW: {mw} g/mol | Atoms: {n_atoms}</text>'
        f'<text x="200" y="148" text-anchor="middle" font-size="10" fill="#94a3b8">{short_smi}</text>'
        '<text x="200" y="200" text-anchor="middle" font-size="10" fill="#94a3b8">2D rendering requires libexpat</text>'
        "</svg>"
    )


def mol_to_svg(mol: Chem.Mol, width: int = 400, height: int = 300, smiles: str = "") -> str:
    if not _DRAW_AVAILABLE:
        return _smiles_to_fallback_svg(smiles or Chem.MolToSmiles(mol), mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_svg_with_atom_colors(
    mol: Chem.Mol,
    atom_contributions: list[dict],
    width: int = 500,
    height: int = 350,
) -> str:
    if not _DRAW_AVAILABLE:
        return _smiles_to_fallback_svg(Chem.MolToSmiles(mol), mol)

    highlight_atoms = {}
    for entry in atom_contributions:
        idx = entry["atom_idx"]
        contrib = entry["contribution"]
        if contrib > 0.05:
            r = min(1.0, contrib * 4)
            highlight_atoms[idx] = (r, 0.2, 0.2)
        elif contrib < -0.05:
            g = min(1.0, abs(contrib) * 4)
            highlight_atoms[idx] = (0.2, g, 0.2)
        else:
            highlight_atoms[idx] = (0.9, 0.9, 0.9)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = False
    atom_list = list(highlight_atoms.keys())
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atom_list,
        highlightAtomColors=highlight_atoms,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def get_molecule_info(mol: Chem.Mol, smiles: str) -> dict:
    canon_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    svg = mol_to_svg(mol, smiles=smiles)
    return {
        "smiles": canon_smiles,
        "formula": formula,
        "molecular_weight": round(mw, 2),
        "logP": round(logp, 3),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rot_bonds,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "svg": svg,
        "draw_available": _DRAW_AVAILABLE,
    }


def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> list:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)


def get_rdkit_descriptors(mol: Chem.Mol) -> dict:
    return {
        "MolLogP": Descriptors.MolLogP(mol),
        "MolMR": Descriptors.MolMR(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "MolWt": Descriptors.MolWt(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
    }
