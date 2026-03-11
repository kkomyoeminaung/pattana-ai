"""
PRD Causal AI Engine v2.0 — Full v1.8 Theory
Pattana-Relational Dynamics by Myo Min Aung
Yangon, Myanmar — March 2026

Phases 1-6 from papers:
  Ad1.pdf  : SU(5) Framework
  Ad2_1.pdf: Non-Abelian Gauge Fields + Logical Curvature
  Advance_1: Geometric Causal AI (complete)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import re

try:
    from scipy.linalg import expm as _expm
    def matrix_exp(H): return _expm(H)
except ImportError:
    def matrix_exp(H):
        # Fallback: first-order approximation
        return np.eye(H.shape[0]) + H + H@H/2

# ═══════════════════════════════════════════════
# PHASE 1 — SU(5) Generator Algebra
# [Ta,Tb] = i*f_abc*Tc
# ═══════════════════════════════════════════════

def build_generators():
    G = {}
    # Cartan H1-H4: diagonal, traceless Hermitian, Tr(Hi Hj)=0.5 delta_ij
    H1 = np.zeros((5,5), dtype=complex); H1[0,0]=0.5; H1[1,1]=-0.5
    H2 = np.zeros((5,5), dtype=complex)
    H2[0,0]=H2[1,1]=1/(2*np.sqrt(3)); H2[2,2]=-2/(2*np.sqrt(3))
    H3 = np.zeros((5,5), dtype=complex)
    for i in range(3): H3[i,i]=1/(2*np.sqrt(6))
    H3[3,3]=-3/(2*np.sqrt(6))
    H4 = np.zeros((5,5), dtype=complex)
    for i in range(4): H4[i,i]=1/(2*np.sqrt(10))
    H4[4,4]=-4/(2*np.sqrt(10))
    G['H1']=H1; G['H2']=H2; G['H3']=H3; G['H4']=H4
    # Step operators: off-diagonal transitions
    for i in range(5):
        for j in range(5):
            if i!=j:
                E=np.zeros((5,5),dtype=complex); E[i,j]=1.0
                G[f'E{i+1}{j+1}']=E
    # Interaction operators: mutual causality
    G['S1']=(H1+H2)/np.sqrt(2); G['S2']=(H1+H3)/np.sqrt(2)
    G['R1']=1j*(H1-H2)/np.sqrt(2); G['R2']=1j*(H1-H3)/np.sqrt(2)
    return G

GENERATORS = build_generators()
GEN_NAMES  = list(GENERATORS.keys())[:24]
GEN_LIST   = [GENERATORS[k] for k in GEN_NAMES]

def compute_structure_constants():
    """f_abc = -2i Tr([Ta,Tb]Tc)"""
    n=len(GEN_LIST); f=np.zeros((n,n,n))
    for a in range(min(n,12)):
        for b in range(min(n,12)):
            comm=GEN_LIST[a]@GEN_LIST[b]-GEN_LIST[b]@GEN_LIST[a]
            for c in range(min(n,12)):
                f[a,b,c]=float((-2j*np.trace(comm@GEN_LIST[c])).real)
    return f

_F_ABC = None
def get_fabc():
    global _F_ABC
    if _F_ABC is None: _F_ABC=compute_structure_constants()
    return _F_ABC

# ═══════════════════════════════════════════════
# PHASE 2 — State Vectors & Propagation
# ═══════════════════════════════════════════════

def text_to_state(text):
    """Embed text -> |Psi> in C^24"""
    psi=np.zeros(24,dtype=complex)
    for i,ch in enumerate(text[:300]):
        idx=(ord(ch)*7+i*13)%24
        phase=(ord(ch)*np.pi)/128.0
        psi[idx]+=np.exp(1j*phase)
    norm=np.linalg.norm(psi)
    if norm<1e-10:
        psi=np.ones(24,dtype=complex); norm=np.linalg.norm(psi)
    return psi/norm

def propagate_state(psi, steps=3):
    """Causal graph propagation: |Psi(t+1)> = sigma(A_hat |Psi(t)>)"""
    n=24; A=np.zeros((n,n))
    for i in range(min(6,n)):
        for j in range(min(6,n)):
            if i!=j:
                comm=GEN_LIST[i]@GEN_LIST[j]-GEN_LIST[j]@GEN_LIST[i]
                A[i,j]=np.linalg.norm(comm)*0.04
    deg=A.sum(axis=1)+1e-8
    dinv=np.diag(1/np.sqrt(deg))
    Ahat=dinv@A@dinv
    state=psi.real.copy()
    for _ in range(steps): state=np.tanh(Ahat@state)
    res=np.zeros(24,dtype=complex); res.real=state; res.imag=psi.imag*0.4
    norm=np.linalg.norm(res); return res/(norm+1e-10)

def expectation_vals(psi):
    """<Gi> = <Psi|Gi|Psi>"""
    evs=np.zeros(24)
    for i,G in enumerate(GEN_LIST):
        gd=np.abs(np.diag(G)); p=psi[:5]
        evs[i]=float(np.real(np.conj(p)@(gd*p)))
    return evs

# ═══════════════════════════════════════════════
# PHASE 3 — Field Strength Tensor (Logical Curvature)
# F^a_uv = d_u A^a_v - d_v A^a_u + g*f^abc*A^b_u*A^c_v
# ═══════════════════════════════════════════════

def compute_gauge_field(psi, g=0.5):
    """A^a_mu from state expectations. Shape (4,24)"""
    evs=expectation_vals(psi); A=np.zeros((4,24))
    for mu in range(4):
        A[mu,:]=g*evs*np.cos(mu*np.pi/4)
    return A

def compute_field_strength(A, g=0.5):
    """
    F^a_uv = dA_kin + g*f^abc*A^b_u*A^c_v
    LOGICAL CURVATURE: F=0 -> flat logic, consistent reasoning
                       F!=0 -> logical tension, hallucination risk
    Shape (4,4,24)
    """
    f_abc=get_fabc(); n_mu=4; n_a=24
    F=np.zeros((n_mu,n_mu,n_a))
    for mu in range(n_mu):
        for nu in range(mu+1,n_mu):
            for a in range(n_a):
                dA=A[mu,a]-A[nu,a]
                nonab=sum(g*f_abc[a,b,c]*A[mu,b]*A[nu,c]
                         for b in range(min(6,n_a)) for c in range(min(6,n_a)))
                F[mu,nu,a]=dA+nonab; F[nu,mu,a]=-F[mu,nu,a]
    return F

def curvature_norm(F):
    """||F||^2 — scalar logical tension. 0=consistent, high=hallucination risk"""
    return float(np.sum(F**2))

def wilson_loop(A):
    """
    W(gamma) = P exp(i*contour A*dx)
    W=I -> consistent; W!=I -> logical paradox/hallucination
    Returns deviation from identity in [0,1]
    """
    n=min(8,A.shape[1]); W=np.eye(n,dtype=complex)
    for mu in [0,1,0,1]:  # square loop
        H=np.zeros((n,n),dtype=complex)
        for i in range(n):
            for j in range(n):
                H[i,j]=A[mu,(i-j)%A.shape[1]]*0.08
        W=W@matrix_exp(1j*H)
    dev=float(np.linalg.norm(W-np.eye(n,dtype=complex))/n)
    return min(dev,1.0)

# ═══════════════════════════════════════════════
# PHASE 4 — Self-Interaction (Meta-Cognition)
# ═══════════════════════════════════════════════

def self_interaction(F):
    """Non-abelian self-interaction strength: g*f^abc*A^b*A^c contribution"""
    return float(np.var(F.reshape(-1,F.shape[-1])))

# ═══════════════════════════════════════════════
# PHASE 5 — Counterfactual via Gauge Transform
# |Psi_cf> = exp(i*alpha^a*Ta) |Psi_fact>
# ═══════════════════════════════════════════════

def parallel_transport_coherence(psi_q, psi_a, A):
    """
    gauge-corrected coherence = base_coherence * exp(-transport_error)
    From v1.8 paper: transport along geodesic in logical manifold
    """
    H=np.zeros((5,5),dtype=complex)
    for mu in range(min(4,A.shape[0])):
        for a in range(min(4,A.shape[1])):
            H+=A[mu,a]*GEN_LIST[a]*0.04
    U=matrix_exp(1j*H)
    psi_transported=U@psi_q[:5]
    transport_err=float(np.linalg.norm(psi_transported-psi_a[:5]))
    base=float(np.abs(np.conj(psi_q)@psi_a))
    return float(np.clip(base*np.exp(-transport_err),0,1))

def do_counterfactual(question, answer, intervention):
    """
    Phase 5 What-if: intervention as gauge transformation
    intervention = {"variable": str, "change": float}
    """
    psi_fact=text_to_state(answer)
    alpha=np.zeros(24); change=intervention.get("change",0.5)
    psi_var=text_to_state(intervention.get("variable",""))
    evs=expectation_vals(psi_var)
    top=np.argsort(np.abs(evs))[-4:]
    for idx in top: alpha[idx]=change*evs[idx]

    H=np.zeros((5,5),dtype=complex)
    for a,al in enumerate(alpha[:len(GEN_LIST)]):
        H+=al*GEN_LIST[a]
    U=matrix_exp(1j*H); psi_cf=psi_fact.copy()
    psi_cf[:5]=U@psi_fact[:5]
    psi_cf/=(np.linalg.norm(psi_cf)+1e-10)

    A_cf=compute_gauge_field(psi_cf); F_cf=compute_field_strength(A_cf)
    A_orig=compute_gauge_field(psi_fact); F_orig=compute_field_strength(A_orig)
    w_cf=wilson_loop(A_cf); consistency=1.0-w_cf
    curv_change=curvature_norm(F_cf)-curvature_norm(F_orig)

    return {
        "curvature_change": round(curv_change,4),
        "consistency_score": round(consistency,3),
        "wilson_cf": round(w_cf,4),
        "intervention_generators": [GEN_NAMES[i] for i in top if i<len(GEN_NAMES)],
        "interpretation": (
            "Logically consistent counterfactual" if consistency>0.7 else
            "Moderate logical tension" if consistency>0.4 else
            "High tension — potentially paradoxical"
        )
    }

# ═══════════════════════════════════════════════
# PHASE 6 — Yang-Mills Action + Reinforcement
# S = -1/4 * F^a_uv * F^a^uv
# L = -R + lambda*||F||^2
# ═══════════════════════════════════════════════

def yang_mills_action(F):
    """S_YM = -1/4 ||F||^2. Minimum S = minimum logical distortion"""
    return -0.25*curvature_norm(F)

def reinforcement_loss(reward, F, lam=0.01):
    """L = -E[R] + lambda*||F||^2"""
    return -reward + lam*curvature_norm(F)

# ═══════════════════════════════════════════════
# HALLUCINATION — Geometric + Linguistic
# ═══════════════════════════════════════════════

_UNCERTAIN = [
    r"\b(I think|I believe|I'm not sure|probably|maybe|might be)\b",
    r"\b(as of my knowledge|I don't have|I cannot)\b",
    r"\b(it is possible|there is a chance|it seems)\b",
]
_CONFIDENT = [
    r"\b(research shows|studies indicate|evidence suggests)\b",
    r"\b(specifically|precisely|exactly|mathematically)\b",
]

def linguistic_risk(text):
    r=0.0
    for p in _UNCERTAIN:
        if re.search(p,text,re.IGNORECASE): r+=0.07
    for p in _CONFIDENT:
        if re.search(p,text,re.IGNORECASE): r-=0.04
    if len(text)<40: r+=0.1
    sents=[s for s in text.split('.') if s.strip()]
    if len(sents)>3 and len(set(sents))/len(sents)<0.7: r+=0.15
    return float(np.clip(r,0,1))

def geometric_risk(F, wilson_val):
    """v1.8 geometric hallucination: high curvature + non-trivial Wilson loop"""
    cn=curvature_norm(F)
    cr=float(2/(1+np.exp(-cn*0.1))-1)
    return float(np.clip(0.6*cr+0.4*wilson_val,0,1))

# ═══════════════════════════════════════════════
# MAIN ANALYSIS DATACLASS
# ═══════════════════════════════════════════════

@dataclass
class PRDAnalysis:
    # Scores
    confidence:         float
    causal_coherence:   float
    hallucination_risk: float
    # v1.8 geometric
    logical_curvature:  float
    wilson_loop_val:    float
    yang_mills_action:  float
    self_interaction:   float
    transport_error:    float
    # Meta
    flag:               str
    causal_depth:       int
    ev_top3:            list
    intervention_basis: list

def analyze(question: str, answer: str, g: float = 0.5) -> PRDAnalysis:
    """Full PRD v2.0 pipeline — Phases 1-6"""
    # Phase 1+2: state + propagation
    psi_q=text_to_state(question); psi_a=text_to_state(answer)
    psi_q=psi_q+0.3*propagate_state(psi_q); psi_q/=(np.linalg.norm(psi_q)+1e-10)
    psi_a=psi_a+0.3*propagate_state(psi_a); psi_a/=(np.linalg.norm(psi_a)+1e-10)

    # Phase 3: gauge field + curvature
    A_q=compute_gauge_field(psi_q,g); A_a=compute_gauge_field(psi_a,g)
    A=(A_q+A_a)/2; F=compute_field_strength(A,g)
    cn=curvature_norm(F); wl=wilson_loop(A)

    # Phase 4
    si=self_interaction(F)

    # Phase 5: parallel transport coherence
    pt_coh=parallel_transport_coherence(psi_q,psi_a,A)
    base_coh=float(np.abs(np.conj(psi_q)@psi_a))
    terr=float(abs(base_coh-pt_coh))

    # Phase 6
    ym=yang_mills_action(F)

    # Hallucination
    lr=linguistic_risk(answer); gr=geometric_risk(F,wl)
    hall=float(np.clip(0.5*lr+0.5*gr,0,1))

    # Confidence
    conf=float(np.clip(pt_coh*(1-hall*0.6),0,1))

    # Generators
    evs=expectation_vals(psi_a)
    top3=[GEN_NAMES[i] for i in np.argsort(np.abs(evs))[-3:][::-1] if i<len(GEN_NAMES)]
    inter=[GEN_NAMES[i] for i in np.argsort(np.abs(evs))[-5:][::-1] if i<len(GEN_NAMES)]

    flag="HIGH" if conf>=0.65 else "MEDIUM" if conf>=0.40 else "LOW"

    return PRDAnalysis(
        confidence=round(conf,3), causal_coherence=round(pt_coh,3),
        hallucination_risk=round(hall,3), logical_curvature=round(cn,4),
        wilson_loop_val=round(wl,4), yang_mills_action=round(ym,4),
        self_interaction=round(si,4), transport_error=round(terr,4),
        flag=flag, causal_depth=3, ev_top3=top3, intervention_basis=inter,
    )
