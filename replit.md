# EcoChem Sentinel

AI-powered green chemistry decision support system — a 5-step molecular screening pipeline for the InNOChem Hackathon (AI × Green Chemistry).

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the Express API server (port 5000, proxied at /api)
- `cd services/python-engine && python3 main.py` — run the Python scientific engine (port 8000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5 (proxies /api/chem/* to Python engine)
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)
- **Python Engine**: FastAPI + RDKit + scikit-learn + SHAP (port 8000)

## Where things live

- `services/python-engine/` — Python FastAPI scientific engine (RDKit, SHAP, QSAR)
- `services/python-engine/models/` — QSAR, green metrics, SVHC engine, retrosynthesis, decision aggregator, SHAP explainer
- `services/python-engine/routes/` — analyze, synthesize, explain, validate endpoints
- `services/python-engine/data/` — CHEM21 solvent DB (30 solvents), SVHC candidate list (28 substances)
- `artifacts/api-server/src/routes/chem.ts` — Express proxy to Python engine
- `lib/api-spec/openapi.yaml` — API spec source of truth

## Architecture decisions

- Python engine at port 8000, Express proxies `/api/chem/*` to it — separates TypeScript API layer from Python scientific computation
- QSAR model uses RDKit molecular descriptors (14 features) + 16 structural alert SMARTS flags → Random Forest on curated reference set (Tox21 + ChEMBL)
- SHAP TreeExplainer (exact Shapley values) maps feature contributions back to molecular atoms for XAI
- Solvent scoring uses CHEM21 guide (Prat et al., Green Chem. 2016) bundled as JSON — no network dependency
- SVHC screening uses Tanimoto similarity (Morgan FP, radius=2, 2048 bits) against 28 ECHA candidate substances
- GreenScore = weighted composite: toxicity 35%, synthesis quality 25%, regulatory 20%, biodegradability 20%
- Decision thresholds: APPROVE ≥70, REDESIGN 40–69, STOP <40

## Product

5-step molecular decision pipeline:
1. **Early Risk Screening** — QSAR toxicity (12 endpoints), structural alerts, SVHC regulatory scan
2. **Synthesis Route Engine** — 5 retrosynthesis routes ranked by GreenScore (E-factor, AE, PMI, solvent score)
3. **Industrial Impact** — GreenScore aggregation, regulatory verdict, recommendations
4. **XAI View** — SHAP atom-level contributions, waterfall chart, interpretation
5. **HITL Validation** — Chemist Approve/Redesign/Stop + comment → full PDF-ready report

## Python API Endpoints

All accessible via Express at `/api/chem/*` or directly at `localhost:8000/*`:

| Endpoint | Method | Step |
|----------|--------|------|
| `/api/chem/analyze` | POST `{smiles}` | Step 1 — toxicity + SVHC |
| `/api/chem/routes` | POST `{smiles}` | Step 2 — synthesis routes |
| `/api/chem/explain` | POST `{smiles}` | Step 4 — SHAP XAI |
| `/api/chem/validate` | POST `{smiles, decision, comment}` | Step 5 — HITL |
| `/api/chem/report/:id` | GET | Retrieve validation report |
| `/api/chem/solvents` | GET | CHEM21 solvent database |
| `/api/chem/health` | GET | Service status |

## User preferences

- Build language: English UI
- Approach: Backend-first (scientific engine), then frontend

## Gotchas

- `rdkit.Chem.Draw.rdMolDraw2D` requires `libexpat.so.1` — the molecule.py has a graceful fallback SVG when unavailable
- Python Engine must start before the Express proxy routes work (it's not auto-started by Express)
- Morgan FP: use `rdFingerprintGenerator.GetMorganGenerator` (newer API), not `AllChem.GetMorganFingerprintAsBitVect` (deprecated)
- SHAP values from TreeExplainer may return numpy arrays — always use `np.asarray(val).flat[0]` for scalar conversion
- The `bioaccumulation` toxicity endpoint has `logP`/`BCF` keys instead of `score` — use `.get("score", v.get("BCF", 0))`
- Always run `cd services/python-engine && python3 main.py` in a separate workflow from Express

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
- CHEM21 data source: Prat et al., Green Chem. 2016, 18, 288-296 (DOI: 10.1039/C5GC01008J)
- SVHC source: ECHA Candidate List for Authorisation (https://echa.europa.eu/candidate-list-table)
- Tox21 dataset: https://tox21.gov/ (NIH/NTP, public domain)
- SHAP: Lundberg & Lee, NeurIPS 2017 (arXiv:1705.07874)
