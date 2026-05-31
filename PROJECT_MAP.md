# PROJECT_MAP.md - SynthGuard

## [TECH_STACK]
**System Date Verified**: May 2026
- **Backend**:
  - FastAPI (REST API framework)
  - LLM Framework (LangChain / LlamaIndex / OpenAI API) - "Chef d'Orchestre"
  - RDKit (Cheminformatics, Molecular Descriptors: MW, TPSA, LogP)
  - DeepChem (QSAR Toxicity Predictions trained on Tox21/ChEMBL/PubChem)
  - AiZynthFinder (Retrosynthesis via Monte Carlo Tree Search)
  - SHAP (Explainable AI - Heat maps & feature attribution)
  - scikit-learn (Random Forest, Scoring engines)
  - Celery (Async task queue for AiZynthFinder)
  - PostgreSQL (Relational DB for persistence, HITL feedback, molecule history)
  - Redis (Message broker for Celery)
  - Jinja2 + WeasyPrint (PDF Report Generation)
- **Frontend**:
  - React (UI Framework)
  - Ketcher.js (Molecular drawing interface - SMILES / Structure)
  - 3Dmol.js (2D/3D visualization & SHAP mapping)
  - Recharts (Radar, Bar, Line charts)
  - jsPDF + html2canvas (Frontend PDF export)
- **Infrastructure**:
  - Docker Compose (fastapi, postgres, redis, react/nginx, aizynthfinder)

## [ARCHITECTURE]
**Principle**: Surgical Architecture & Simplicity First (Domain-Driven Design)
- **Core/Shared Layer**:
  - `core/config`: System variables, thresholds.
  - `core/db`: DB session management.
  - `core/logger`: Async non-blocking structured logging.
- **Domains**:
  - `domains/orchestrator`: LLM Router, prompt management, and narrative generator (The "Chef").
  - `domains/screening`: RDKit descriptor extraction & DeepChem toxicity prediction (Acute, Aquatic, Bioaccumulation).
  - `domains/synthesis`: AiZynthFinder adapter, MCTS policies, Material DB queries (PubChem/ChEMBL).
  - `domains/impact`: Green Chemistry Engine (E-factor, Atom Economy, PMI, Solvent Toxicity).
  - `domains/xai`: SHAP integration for visual explanations (Red/Green mapping).
  - `domains/hitl`: Human-in-the-Loop Validation workflows & PDF Report Generation.
- **Frontend App**:
  - `components/shared`: UI Primitives.
  - `features/drawing`: Ketcher integration.
  - `features/dashboard`: LLM Narrative View, Risk Classification, Route Ranking, XAI View, HITL Panel.

## [SYSTEM_FLOW]
1. **Input (Scenes 1-2)**: User draws molecule (Ketcher) or pastes SMILES and asks a question in natural language. LLM parses the intent (e.g., "Is it safe?").
2. **Descriptor Extraction (Scene 3)**: LLM calls RDKit to extract MW, LogP, TPSA, etc.
3. **Toxicity Prediction (Scenes 4-6)**: LLM passes data to DeepChem. DeepChem compares vs Tox21/ChEMBL and computes LD50, LC50, and Bioaccumulation risk.
4. **Explainability (Scene 7-8)**: SHAP explains the risk (e.g., Nitro group +20 risk) producing Heat Maps. System generates Global Risk score (🟢 🟠 🔴).
5. **Retrosynthesis (Scenes 9-12)**: If accepted, AiZynthFinder runs MCTS to generate routes. Connects to material databases to verify availability.
6. **Green Engine (Scenes 13-14)**: Computes E-factor, Atom Economy, PMI, and Solvent Toxicity. Ranks routes by GreenScore.
7. **HITL & LLM Narrative (Scenes 15-17)**: Chemist reviews routes. LLM summarizes all computational findings into human-readable text. Chemist validates, and a 6-section PDF Report is generated.

## [ORPHANS & PENDING]
- [ ] Initialize Backend scaffolding (FastAPI + Core setup).
- [ ] Initialize Frontend scaffolding (React).
- [ ] Implement Async Logger.
- [ ] Set up Docker Compose network.
- [ ] Integrate RDKit SMILES parser.
- [ ] Setup DeepChem Tox21 inference pipeline.
- [ ] Configure AiZynthFinder Celery workers.
- [ ] Build Solvent DB parsing & scoring logic.
- [ ] Implement SHAP to 3Dmol.js mapping logic.
- [ ] Implement HITL API endpoints & Postgres schema.
