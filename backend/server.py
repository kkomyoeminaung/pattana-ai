"""
Pattana AI — FastAPI Backend (v2 with Self-Learning)
Run: python server.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx
import json
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from prd_engine import analyze
from learning_engine import get_memory, get_kb, get_web_learner, get_self_improve

app = FastAPI(title="Pattana AI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL    = os.getenv("OLLAMA_URL",   "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PRD_SYSTEM = """You are Pattana AI — a highly intelligent general-purpose assistant built on the Pattana-Relational Dynamics (PRD) causal reasoning framework by Myo Min Aung, Independent Researcher, Yangon, Myanmar.

## Core Identity
You are NOT just a chatbot. You are a causal reasoning engine that:
- Reasons through CAUSES and EFFECTS, not just statistical pattern matching
- Distinguishes OBSERVATION from INTERVENTION from COUNTERFACTUAL
- Acknowledges uncertainty honestly rather than hallucinating
- Continuously learns from every conversation
- Can answer ANY topic: science, math, coding, history, philosophy, creative writing, life advice, Myanmar culture, and more

## PRD Causal Reasoning (apply to every answer)
Before responding, mentally apply:
1. HETU (Root cause): What is the fundamental cause here?
2. ARAMMANA (Context): What is the full context of this question?
3. UPANISSAYA (Supporting conditions): What background knowledge applies?
4. SAHAJATA (Co-arising): What related concepts arise together?
5. CONFIDENCE: How certain am I? Be honest.

## Response Style
- Clear, direct, genuinely helpful
- For complex topics: structure with headers and examples
- For code: provide working, commented code
- For uncertain topics: say "I'm not certain, but..." rather than hallucinating
- Respond in the same language the user uses (Burmese = Burmese, English = English)
- Be warm and respectful — you understand Myanmar culture

## What Makes You Different from Standard AI
- You reason causally, not just statistically
- Your PRD engine scores every response for hallucination risk
- You learn from feedback and improve over time
- You can pull in recent knowledge from the web when needed
- Low hallucination: if you don't know, you say so clearly

Always aim to be the most helpful, honest, and accurate AI assistant possible."""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages:   List[ChatMessage]
    model:      Optional[str]  = None
    stream:     Optional[bool] = True
    use_memory: Optional[bool] = True
    use_web:    Optional[bool] = True

class FeedbackRequest(BaseModel):
    memory_id: str
    score:     int

class LearnRequest(BaseModel):
    content: str
    topic:   Optional[str] = "general"


@app.get("/api/health")
async def health():
    si = get_self_improve()
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except:
        models, ollama_ok = [], False
    return {
        "status": "ok", "version": "2.0",
        "ollama": ollama_ok, "models": models,
        "prd_engine": True, "learning": True,
        "stats": si.improvement_summary(),
    }


@app.get("/api/models")
async def get_models():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            return {"models": [{"name": m["name"], "size": str(round(m.get("size",0)/1e9,1))+"GB"}
                               for m in r.json().get("models",[])]}
    except:
        return {"models": []}


@app.post("/api/chat")
async def chat(req: ChatRequest, bg: BackgroundTasks):
    model    = req.model or DEFAULT_MODEL
    question = req.messages[-1].content if req.messages else ""
    memory   = get_memory()
    kb       = get_kb()
    web      = get_web_learner()
    si       = get_self_improve()

    # Web search context
    web_context = ""
    if req.use_web and await web.should_search(question):
        result = await web.search(question)
        if result:
            web_context = f"\n\n## Live web knowledge:\n{result}"

    # Memory context
    mem_context = si.get_context_injection(question, memory, kb) if req.use_memory else ""

    system = PRD_SYSTEM
    if mem_context: system += f"\n\n{mem_context}"
    if web_context: system += web_context

    ollama_msgs = [{"role": "system", "content": system}]
    for m in req.messages[-20:]:
        ollama_msgs.append({"role": m.role, "content": m.content})

    if req.stream:
        return StreamingResponse(
            _stream(model, ollama_msgs, question, memory, si),
            media_type="text/event-stream"
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"{OLLAMA_URL}/api/chat",
                    json={"model": model, "messages": ollama_msgs, "stream": False})
                answer = r.json().get("message", {}).get("content", "")
                prd    = analyze(question, answer)
                mid    = memory.add(question, answer, prd.confidence)
                si.record_chat(question, answer, prd.confidence)
                return {"answer": answer, "memory_id": mid,
                        "prd": {"confidence": prd.confidence,
                                "causal_coherence": prd.causal_coherence,
                                "hallucination_risk": prd.hallucination_risk,
                                "flag": prd.flag, "top_generators": prd.ev_top3}}
        except httpx.ConnectError:
            raise HTTPException(503, "Ollama offline")
        except Exception as e:
            raise HTTPException(500, str(e))


async def _stream(model, messages, question, memory, si):
    full = ""
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True}) as resp:
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'type':'error','content':f'Ollama error {resp.status_code}'})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.strip(): continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full += token
                            yield f"data: {json.dumps({'type':'token','content':token})}\n\n"
                        if chunk.get("done"): break
                    except: continue

        if full:
            prd = analyze(question, full)
            mid = memory.add(question, full, prd.confidence)
            si.record_chat(question, full, prd.confidence)
            yield f"data: {json.dumps({'type':'prd','confidence':prd.confidence,'causal_coherence':prd.causal_coherence,'hallucination_risk':prd.hallucination_risk,'flag':prd.flag,'memory_id':mid,'top_generators':prd.ev_top3})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    except httpx.ConnectError:
        err = "Ollama မရှိသေးဘူး။\n\n**Terminal မှာ run ပါ:**\n```\nollama serve\n```"
        yield f"data: {json.dumps({'type':'error','content':err})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type':'error','content':str(e)})}\n\n"


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    get_memory().feedback(req.memory_id, req.score)
    return {"ok": True}

@app.get("/api/stats")
async def stats():
    return get_self_improve().improvement_summary()

@app.get("/api/memory")
async def get_mems(limit: int = 20):
    mems = get_memory().memories[-limit:]
    return {"memories": mems, "total": len(get_memory().memories)}

@app.get("/api/knowledge")
async def knowledge(limit: int = 20):
    return {"facts": get_kb().facts[-limit:], "topics": get_kb().get_all_topics(),
            "total": len(get_kb().facts)}

@app.post("/api/learn")
async def learn(req: LearnRequest):
    fid = get_kb().add_fact(req.content, "manual", req.topic, quality=0.9)
    return {"ok": True, "id": fid}

@app.delete("/api/memory/clear")
async def clear_mem():
    m = get_memory(); m.memories = []; m.save()
    return {"ok": True}


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/manifest.json")
    async def manifest():
        return FileResponse(os.path.join(frontend_dir, "manifest.json"))

    @app.get("/sw.js")
    async def sw():
        return FileResponse(os.path.join(frontend_dir, "sw.js"))


if __name__ == "__main__":
    import uvicorn, socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"

    print("\n" + "="*60)
    print("  ✦  Pattana AI Server  v2.0")
    print("="*60)
    print(f"  PC:    http://localhost:8000")
    print(f"  Phone: http://{local_ip}:8000")
    print("="*60)
    print("  PRD Engine + Self-Learning + Web Search")
    print("  Ensure Ollama is running: ollama serve")
    print("="*60 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


# ── Counterfactual endpoint (Phase 5) ─────────────────────────
class CounterfactualRequest(BaseModel):
    question:     str
    answer:       str
    variable:     str
    change:       Optional[float] = 0.5

@app.post("/api/counterfactual")
async def counterfactual(req: CounterfactualRequest):
    """Phase 5: What-if analysis via gauge transformation"""
    from prd_engine import do_counterfactual
    result = do_counterfactual(
        req.question, req.answer,
        {"variable": req.variable, "change": req.change}
    )
    return result

# ── PRD Visualization data ─────────────────────────────────────
@app.get("/api/prd/geometry")
async def prd_geometry(text: str = ""):
    """Return geometric data for visualization"""
    from prd_engine import (text_to_state, compute_gauge_field,
                            compute_field_strength, curvature_norm,
                            wilson_loop, yang_mills_action, expectation_vals)
    if not text:
        return {"error": "text required"}
    psi = text_to_state(text)
    A   = compute_gauge_field(psi)
    F   = compute_field_strength(A)
    evs = expectation_vals(psi)
    return {
        "state_real":        psi.real.tolist(),
        "state_imag":        psi.imag.tolist(),
        "expectation_vals":  evs.tolist(),
        "curvature_norm":    round(curvature_norm(F), 6),
        "wilson_loop":       round(wilson_loop(A), 6),
        "yang_mills_action": round(yang_mills_action(F), 6),
        "gauge_field_mu0":   A[0].tolist(),
    }
