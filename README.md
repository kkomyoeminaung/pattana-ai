# Pattana AI v2.0 — PRD Causal AI Engine

**By Myo Min Aung — Independent Researcher, Yangon, Myanmar**

> World's first Geometry-based Causal AI — SU(5) Non-Abelian Gauge Fields + Logical Curvature

## Theory (Papers)
- **Ad1.pdf** — PRD AI Engine: SU(5) Framework (Phase 1-6)
- **Ad2_1.pdf** — Non-Abelian Gauge Fields & Logical Curvature (v1.8)
- **Advance_1.pdf** — Geometric Approach to Causal AI (complete)

## What's New in v2.0

| Feature | Theory | Status |
|---------|--------|--------|
| Field Strength Tensor F^a_uv | v1.8 Eq.3 | ✅ Implemented |
| Wilson Loop W(γ) | v1.8 Eq.6 | ✅ Implemented |
| Yang-Mills Action S_YM | v1.8 Eq.7 | ✅ Implemented |
| Parallel Transport Coherence | v1.8 Eq.2 | ✅ Implemented |
| Counterfactual via Gauge Transform | v1.8 Eq.4-5 | ✅ Implemented |
| Phase 4 Self-Interaction | Paper 1 §5.4 | ✅ Implemented |
| /api/counterfactual endpoint | Phase 5 | ✅ New |
| /api/prd/geometry endpoint | Visualization | ✅ New |
| Geometry Panel in UI | Frontend | ✅ New |

## Architecture

```
Text → |Ψ⟩ ∈ C^24     Phase 1: SU(5) state embedding
     → Propagate       Phase 2: Causal graph A_hat |Ψ⟩
     → A^a_mu          Phase 3: Gauge field
     → F^a_uv          Phase 3: Logical Curvature (hallucination metric)
     → W(γ)            Phase 3: Wilson loop consistency check
     → S_YM            Phase 6: Yang-Mills action (min = best reasoning)
     → |Ψ_cf⟩          Phase 5: Counterfactual via exp(i α^a T^a)
```

## Quick Start

```bash
# Terminal 1
ollama serve

# Terminal 2
cd pattana_ai/backend
pip install -r requirements.txt
python server.py
```

Open: http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/chat | POST | Chat with PRD analysis |
| /api/counterfactual | POST | Phase 5 what-if analysis |
| /api/prd/geometry | GET | Raw geometric data |
| /api/stats | GET | Self-learning stats |
| /api/feedback | POST | Train PRD engine |

## PRD Scores Explained

- **Confidence** — Parallel transport coherence (not probability)
- **Curvature ||F||²** — Logical tension (0=flat/consistent, high=contradiction)
- **Wilson W(γ)** — Holonomy check (0=no paradox, 1=logical loop paradox)
- **Yang-Mills S_YM** — Reasoning quality (-1/4 × curvature)
- **Self-Interaction** — Non-abelian meta-cognition strength

## License
AGPL-3.0 — Open Source

## Citation
```
Myo Min Aung (2026). Pattana-Relational Dynamics (PRD) AI Engine v2.0.
GitHub: https://github.com/kkomyoeminaung/pattana-ai
```
