"""
Package: backend
Purpose: FastAPI backend layer connecting SIH 2026 ML models to the frontend dashboard.

Architecture:
    backend/main.py             — FastAPI app, lifespan, CORS, routers
    backend/schemas.py          — Pydantic v2 request/response models
    backend/services/           — Business services (model registry, prediction, component lookup)
    backend/utils/              — Preprocessing helpers (drift computation, row assembly)

Reuses:
    src.inference.predict       — load_models(), predict_24h(), predict_96h()
    src.decision.screening_decision — make_screening_decision(), run_screening_pipeline()
    src.features.build_features — FEATURES_24H_GATE, FEATURES_96H_GATE
    src.data.load_data          — load_ml_ready_data()
    src.data.split_data         — split_components()
"""
