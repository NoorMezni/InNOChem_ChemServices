"""
Green chemistry metrics calculator.

References:
- E-factor: Sheldon, R.A. Green Chem. 2007, 9, 1273
- Atom Economy: Trost, B.M. Science 1991, 254, 1471
- PMI: Jimenez-Gonzalez et al. Org. Process Res. Dev. 2011, 15, 912
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Optional


def calc_atom_economy(product_smiles: str, reactant_smiles_list: list[str]) -> float:
    """
    Atom Economy (%) = MW(desired product) / sum(MW(reactants)) * 100
    Trost 1991 definition.
    """
    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol is None:
        return 0.0
    prod_mw = Descriptors.MolWt(prod_mol)

    total_reactant_mw = 0.0
    for smi in reactant_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            total_reactant_mw += Descriptors.MolWt(mol)

    if total_reactant_mw == 0:
        return 0.0
    return round(min(100.0, (prod_mw / total_reactant_mw) * 100), 1)


def calc_e_factor(
    mass_product_g: float,
    mass_reactants_g: float,
    mass_solvents_g: float,
    mass_catalysts_g: float = 0.0,
    recycled_solvent_fraction: float = 0.0,
) -> float:
    """
    E-factor = total waste / mass of product
    Sheldon 2007 definition.
    Waste = reactants + solvents + catalysts - product
    """
    solvent_waste = mass_solvents_g * (1 - recycled_solvent_fraction)
    total_waste = mass_reactants_g + solvent_waste + mass_catalysts_g - mass_product_g
    total_waste = max(0.0, total_waste)
    if mass_product_g <= 0:
        return 999.0
    return round(total_waste / mass_product_g, 2)


def calc_pmi(
    mass_product_g: float,
    mass_reactants_g: float,
    mass_solvents_g: float,
    mass_catalysts_g: float = 0.0,
) -> float:
    """
    PMI (Process Mass Intensity) = total mass in / mass of product
    Jimenez-Gonzalez 2011 definition.
    PMI = E-factor + 1
    """
    total_mass_in = mass_reactants_g + mass_solvents_g + mass_catalysts_g
    if mass_product_g <= 0:
        return 999.0
    return round(total_mass_in / mass_product_g, 2)


def estimate_route_metrics(
    product_smiles: str,
    reactant_smiles_list: list[str],
    n_steps: int,
    solvent_volume_per_step_ml: float = 10.0,
    solvent_density: float = 0.9,
    catalyst_loading_pct: float = 5.0,
    recycled_solvent_pct: float = 0.0,
) -> dict:
    """
    Estimate green metrics for a synthesis route given product + reactants.
    Uses realistic approximations based on step count and molecular weights.
    """
    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol is None:
        return {}

    prod_mw = Descriptors.MolWt(prod_mol)
    mass_product_g = prod_mw / 1000.0 * 1.0

    total_reactant_mw = 0.0
    for smi in reactant_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            total_reactant_mw += Descriptors.MolWt(mol)

    mass_reactants_g = total_reactant_mw / 1000.0 * 1.1 * n_steps
    mass_solvents_g = solvent_volume_per_step_ml * solvent_density * n_steps
    mass_catalysts_g = mass_reactants_g * (catalyst_loading_pct / 100.0)
    recycled_fraction = recycled_solvent_pct / 100.0

    ae = calc_atom_economy(product_smiles, reactant_smiles_list)
    ef = calc_e_factor(mass_product_g, mass_reactants_g, mass_solvents_g, mass_catalysts_g, recycled_fraction)
    pmi = calc_pmi(mass_product_g, mass_reactants_g, mass_solvents_g, mass_catalysts_g)

    return {
        "atom_economy_pct": ae,
        "e_factor": ef,
        "pmi": pmi,
        "mass_product_g": round(mass_product_g, 4),
        "mass_reactants_g": round(mass_reactants_g, 4),
        "mass_solvents_g": round(mass_solvents_g, 4),
        "n_steps": n_steps,
    }


def classify_e_factor(e_factor: float) -> dict:
    """Map E-factor value to industry classification."""
    if e_factor < 1:
        return {"label": "Excellent", "sector": "Bulk chemicals", "color": "green"}
    elif e_factor < 5:
        return {"label": "Good", "sector": "Fine chemicals (target)", "color": "green"}
    elif e_factor < 25:
        return {"label": "Acceptable", "sector": "Fine chemicals (typical)", "color": "yellow"}
    elif e_factor < 100:
        return {"label": "High", "sector": "Pharmaceuticals (typical)", "color": "orange"}
    else:
        return {"label": "Very High", "sector": "Above pharma average", "color": "red"}
