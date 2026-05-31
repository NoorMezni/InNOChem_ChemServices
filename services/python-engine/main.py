"""
EcoChem Sentinel — Python Scientific Engine
FastAPI service providing QSAR toxicity prediction, synthesis route generation,
SHAP explainability, and HITL validation for green chemistry decision support.
"""
import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.analyze import router as analyze_router
from routes.synthesize import router as synth_router
from routes.explain import router as explain_router
from routes.validate import router as validate_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ecochem")

_startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    t0 = time.time()
    log.info("EcoChem Sentinel engine starting up...")

    log.info("Loading QSAR model (training on reference set)...")
    from models.qsar import get_qsar_model
    get_qsar_model()
    log.info("QSAR model ready.")

    log.info("Loading SVHC database...")
    from models.svhc_engine import load_svhc_db
    load_svhc_db()
    log.info("SVHC database ready.")

    log.info("Loading solvent database...")
    from models.solvent_db import load_solvent_db
    load_solvent_db()
    log.info("Solvent database ready.")

    elapsed = time.time() - t0
    _startup_time = elapsed
    log.info(f"EcoChem Sentinel ready in {elapsed:.1f}s — listening on port {os.environ.get('PORT', 8000)}")
    yield
    log.info("EcoChem Sentinel shutting down.")


app = FastAPI(
    title="EcoChem Sentinel API",
    description=(
        "AI-powered green chemistry decision support system. "
        "Provides QSAR toxicity prediction, retrosynthesis route scoring, "
        "SHAP-based XAI, SVHC regulatory screening, and human-in-the-loop validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    log.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f}ms)")
    return response


app.include_router(analyze_router)
app.include_router(synth_router)
app.include_router(explain_router)
app.include_router(validate_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "EcoChem Sentinel Scientific Engine",
        "version": "1.0.0",
        "startup_time_s": round(_startup_time, 2) if _startup_time else None,
        "endpoints": ["/analyze", "/routes", "/explain", "/validate", "/report/{id}", "/solvents"],
    }


@app.get("/")
def root():
    return {
        "name": "EcoChem Sentinel",
        "description": "Green chemistry AI decision support — InNOChem Hackathon",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
