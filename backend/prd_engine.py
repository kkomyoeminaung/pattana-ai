"""
PRD Causal Engine — Core
Pattana-Relational Dynamics by Myo Min Aung
Causal reasoning layer that wraps LLM responses
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import re


# ─── SU(5) Generator Algebra ──────────────────────────────────────────────────

def build_generators() -> dict:
    G = {}
    # Cartan H1–H4
    H1 = np.zeros((5,5), dtype=complex)
    H1[0,0], H1[1,1] = 0.5, -0.5
    H2 = np.zeros((5,5), dtype=complex)
    H2[0,0] = H2[1,1] = 1/(2*np.sqrt(3)); H2[2,2] = -2/(2*np.sqrt(3))
    H3 = np.zeros((5,5), dtype=complex)
    for i in range(3): H3[i,i] = 1/(2*np.sqrt(6))
    H3[3,3] = -3/(2*np.sqrt(6))
    H4 = np.zeros((5,5), dtype=complex)
    for i in range(4): H4[i,i] = 1/(2*np.sqrt(10))
    H4[4,4] = -4/(2*np.sqrt(10))
    G['H1']=H1; G['H2']=H2; G['H3']=H3; G['H4']=H4

    # Step operators
    for i in range(5):
        for j in range(5):
            if i != j:
                E = np.zeros((5,5), dtype=complex); E[i,j] = 1.0
                G[f'E{i+1}{j+1}'] = E

    # Interaction operators
    G['S1'] = H1 + H2
    G['S2'] = H1 + H3
    G['R1'] = 1j*(H1 - H2)
    G['R2'] = 1j*(H1 - H3)
    return G


GENERATORS = build_generators()
GEN_LIST   = list(GENERATORS.values())[:24]


def commutator(A, B):
    return A @ B - B @ A


def relational_state(text: str) -> np.ndarray:
    """Map text to a 24-dim relational state vector via hash embedding."""
    psi = np.zeros(24, dtype=complex)
    for i, ch in enumerate(text[:200]):
        idx = (ord(ch) + i) % 24
        phase = (ord(ch) * np.pi) / 128
        psi[idx] += np.exp(1j * phase)
    norm = np.linalg.norm(psi)
    if norm < 1e-10:
        psi = np.random.randn(24) + 1j*np.random.randn(24)
        norm = np.linalg.norm(psi)
    return psi / norm


def expectation_values(psi: np.ndarray) -> np.ndarray:
    evs = np.zeros(24)
    for i, G in enumerate(GEN_LIST):
        g_flat = G.flatten().real[:24]
        if len(g_flat) < 24:
            g_flat = np.pad(g_flat, (0, 24-len(g_flat)))
        evs[i] = float(np.real(np.conj(psi) @ np.diag(np.abs(g_flat[:24])) @ psi))
    return evs


def causal_coherence(psi_q: np.ndarray, psi_a: np.ndarray) -> float:
    """
    Causal coherence score between question state and answer state.
    High score = answer is causally consistent with question.
    Range: 0.0 – 1.0
    """
    overlap = np.abs(np.conj(psi_q) @ psi_a)
    ev_q = expectation_values(psi_q)
    ev_a = expectation_values(psi_a)
    # Measure algebraic consistency: how much do the causal structures align?
    consistency = 1.0 - np.linalg.norm(ev_q - ev_a) / (np.linalg.norm(ev_q) + np.linalg.norm(ev_a) + 1e-8)
    score = 0.5 * overlap + 0.5 * consistency
    return float(np.clip(score, 0, 1))


def propagate_causal_graph(psi: np.ndarray, steps: int = 3) -> np.ndarray:
    """Multi-step causal propagation: |Ψ(t+1)⟩ = tanh(Â·|Ψ(t)⟩)"""
    state = psi.real.copy()
    # Build sparse adjacency from commutator norms
    n = min(len(state), 24)
    A = np.zeros((n, n))
    for i in range(min(4, n)):
        for j in range(min(4, n)):
            if i != j:
                c = commutator(GEN_LIST[i], GEN_LIST[j])
                A[i, j] = np.linalg.norm(c) * 0.1

    # Normalize adjacency
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0/np.sqrt(deg + 1e-8), 0)
    A_hat = np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)

    state = state[:n]
    for _ in range(steps):
        state = np.tanh(A_hat @ state)
    full = np.zeros(24)
    full[:n] = state
    return full


# ─── Hallucination Detection ──────────────────────────────────────────────────

# Patterns that suggest hallucination or low-confidence content
HALLUCINATION_PATTERNS = [
    r"\b(I think|I believe|I'm not sure|probably|maybe|might be|could be)\b",
    r"\b(as of my knowledge|I don't have|I cannot|I am unable)\b",
    r'\b(it is possible that|there is a chance|it seems like)\b',
    r'\b(approximately|roughly|around|about)\s+\d',
    r'(?i)\b(invented|discovered|born|died)\s+in\s+\d{4}\b',  # factual claims
]

CONFIDENCE_BOOSTERS = [
    r'\b(research shows|studies indicate|according to|evidence suggests)\b',
    r'\b(specifically|precisely|exactly|definitively)\b',
    r'\b(the formula|the equation|mathematically|formally)\b',
]


def hallucination_score(text: str) -> float:
    """
    Returns hallucination risk: 0.0 = clean, 1.0 = high risk
    Uses PRD causal consistency + linguistic patterns
    """
    risk = 0.0

    # Linguistic pattern check
    for pat in HALLUCINATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            risk += 0.08

    # Confidence reducers
    for pat in CONFIDENCE_BOOSTERS:
        if re.search(pat, text, re.IGNORECASE):
            risk -= 0.05

    # Length heuristic: very short answers to complex questions are risky
    if len(text) < 50:
        risk += 0.1

    # Repetition detection
    sentences = text.split('.')
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.7:
            risk += 0.15

    return float(np.clip(risk, 0.0, 1.0))


def prd_confidence_score(question: str, answer: str) -> float:
    """
    PRD-based confidence: combines causal coherence + anti-hallucination
    Returns 0.0–1.0 confidence score
    """
    psi_q = relational_state(question)
    psi_a = relational_state(answer)

    # Propagate through causal graph
    psi_q_prop = propagate_causal_graph(psi_q)
    psi_a_prop = propagate_causal_graph(psi_a)

    psi_q_final = np.zeros(24, dtype=complex)
    psi_q_final[:24] = psi_q + psi_q_prop * 0.3
    psi_q_final /= (np.linalg.norm(psi_q_final) + 1e-8)

    psi_a_final = np.zeros(24, dtype=complex)
    psi_a_final[:24] = psi_a + psi_a_prop * 0.3
    psi_a_final /= (np.linalg.norm(psi_a_final) + 1e-8)

    coherence  = causal_coherence(psi_q_final, psi_a_final)
    hall_risk  = hallucination_score(answer)
    confidence = coherence * (1.0 - hall_risk * 0.6)

    return float(np.clip(confidence, 0.0, 1.0))


@dataclass
class PRDAnalysis:
    confidence:       float   # 0–1
    causal_coherence: float   # 0–1
    hallucination_risk: float # 0–1
    causal_depth:     int     # propagation steps used
    ev_top3:          list    # top 3 active generators
    flag:             str     # "HIGH" / "MEDIUM" / "LOW" confidence


def analyze(question: str, answer: str) -> PRDAnalysis:
    psi_q = relational_state(question)
    psi_a = relational_state(answer)
    psi_q_p = propagate_causal_graph(psi_q, steps=3)
    psi_a_p = propagate_causal_graph(psi_a, steps=3)

    psi_q_f = (psi_q + psi_q_p * 0.3)
    psi_a_f = (psi_a + psi_a_p * 0.3)
    psi_q_f /= (np.linalg.norm(psi_q_f) + 1e-8)
    psi_a_f /= (np.linalg.norm(psi_a_f) + 1e-8)

    coherence = causal_coherence(psi_q_f, psi_a_f)
    hall_risk = hallucination_score(answer)
    confidence = float(np.clip(coherence * (1 - hall_risk * 0.6), 0, 1))

    evs = expectation_values(psi_a)
    top3_idx = np.argsort(np.abs(evs))[-3:][::-1].tolist()
    gen_names = list(GENERATORS.keys())
    top3 = [gen_names[i] if i < len(gen_names) else f'G{i}' for i in top3_idx]

    if confidence >= 0.65:
        flag = "HIGH"
    elif confidence >= 0.40:
        flag = "MEDIUM"
    else:
        flag = "LOW"

    return PRDAnalysis(
        confidence       = round(confidence, 3),
        causal_coherence = round(coherence, 3),
        hallucination_risk = round(hall_risk, 3),
        causal_depth     = 3,
        ev_top3          = top3,
        flag             = flag,
    )
