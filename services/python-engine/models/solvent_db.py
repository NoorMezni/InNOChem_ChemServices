"""
Solvent scoring based on CHEM21 Solvent Selection Guide.

Reference:
  Prat et al. "CHEM21 selection guide of classical- and less classical-solvents."
  Green Chem. 2016, 18, 288-296. DOI: 10.1039/C5GC01008J

  Henderson et al. "Expanding GSK's solvent selection guide"
  Green Chem. 2011, 13, 854. DOI: 10.1039/C0GC00918K
"""
import json
import os
from typing import Optional

_SOLVENT_DB: list[dict] = []
_SOLVENT_BY_NAME: dict[str, dict] = {}


def load_solvent_db() -> None:
    global _SOLVENT_DB, _SOLVENT_BY_NAME
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "solvents.json")
    with open(data_path) as f:
        _SOLVENT_DB = json.load(f)
    _SOLVENT_BY_NAME = {s["name"].lower(): s for s in _SOLVENT_DB}


def get_all_solvents() -> list[dict]:
    if not _SOLVENT_DB:
        load_solvent_db()
    return _SOLVENT_DB


def lookup_solvent(name: str) -> Optional[dict]:
    if not _SOLVENT_BY_NAME:
        load_solvent_db()
    return _SOLVENT_BY_NAME.get(name.lower())


def score_solvent(name: str) -> dict:
    solvent = lookup_solvent(name)
    if solvent is None:
        return {
            "name": name,
            "found": False,
            "score": 5.0,
            "class": "Unknown",
            "safety": 5,
            "health": 5,
            "environment": 5,
            "alternatives": [],
        }
    return {
        "name": solvent["name"],
        "found": True,
        "score": solvent["score"],
        "class": solvent["class"],
        "safety": solvent["safety"],
        "health": solvent["health"],
        "environment": solvent["environment"],
        "alternatives": solvent.get("alternatives", []),
        "source": "CHEM21 Solvent Selection Guide (Prat et al., 2016)",
    }


def get_e_factor_reduction(original_solvent: str, replacement_solvent: str) -> dict:
    orig = lookup_solvent(original_solvent)
    repl = lookup_solvent(replacement_solvent)
    if not orig or not repl:
        return {"reduction_pct": 0.0, "explanation": "Unknown solvent"}

    score_diff = repl["score"] - orig["score"]
    reduction_pct = (score_diff / 10.0) * 30.0

    return {
        "original": orig["name"],
        "replacement": repl["name"],
        "original_score": orig["score"],
        "replacement_score": repl["score"],
        "score_improvement": round(score_diff, 1),
        "estimated_e_factor_reduction_pct": round(reduction_pct, 1),
        "class_change": f"{orig['class']} → {repl['class']}",
    }


def suggest_green_solvents(hazardous_names: list[str]) -> list[dict]:
    if not _SOLVENT_BY_NAME:
        load_solvent_db()
    suggestions = []
    for name in hazardous_names:
        solvent = lookup_solvent(name)
        if solvent and solvent.get("alternatives"):
            alternatives = []
            for alt_name in solvent["alternatives"]:
                alt = lookup_solvent(alt_name)
                if alt:
                    alternatives.append({
                        "name": alt["name"],
                        "score": alt["score"],
                        "class": alt["class"],
                    })
            suggestions.append({
                "original": solvent["name"],
                "original_class": solvent["class"],
                "original_score": solvent["score"],
                "alternatives": alternatives,
            })
    return suggestions
