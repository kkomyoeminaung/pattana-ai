# Pattana AI — Setup Guide

PRD Causal AI Engine by Myo Min Aung, Yangon, Myanmar

---

## Project Structure

```
pattana_ai/
├── backend/
│   ├── server.py        ← FastAPI server (ဒါ run တာ)
│   └── prd_engine.py    ← PRD Causal Engine core
├── frontend/
│   ├── index.html       ← Gemini-style PWA UI
│   ├── manifest.json    ← PWA manifest
│   └── sw.js            ← Service worker
└── requirements.txt
```

---

## Step 1 — Ollama Install

**Windows:**
```
https://ollama.com/download
```
Download ဆွဲပြီး install လုပ်

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Step 2 — Model Download

```bash
# အကောင်းဆုံး (4GB)
ollama pull llama3.2

# ပေါ့တဲ့ model (2GB) — PC ကြိုးနည်းရင်
ollama pull qwen2.5:3b

# Reasoning အကောင်းဆုံး (4GB)
ollama pull deepseek-r1:7b
```

---

## Step 3 — Python Dependencies

```bash
cd pattana_ai
pip install -r requirements.txt
```

---

## Step 4 — Run

Terminal 1 — Ollama start:
```bash
ollama serve
```

Terminal 2 — Backend start:
```bash
cd pattana_ai/backend
python server.py
```

ထွက်လာမယ်:
```
============================================================
  Pattana AI Server
============================================================
  Local:   http://localhost:8000
  Phone:   http://192.168.1.x:8000    ← ဒါကို ဖုန်းမှာ ဖွင့်
============================================================
```

---

## Phone မှာ သုံးနည်း

1. PC နဲ့ Phone တစ်ထဲ WiFi ချိတ်ထားပါ
2. Phone browser မှာ `http://192.168.1.x:8000` ဖွင့်ပါ
   (x ကို server terminal ထဲက IP နဲ့ ပြောင်းပါ)
3. Browser → Share → "Add to Home Screen" နှိပ်ရင် app icon ရမယ်

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 PRD Causal Engine | SU(5) algebra ကို သုံးပြီး causal reasoning |
| 📊 Confidence Score | အဖြေတိုင်းမှာ confidence % ပြတယ် |
| ⚠️ Hallucination Detection | မသေချာတဲ့ answer တွေကို flag လုပ်တယ် |
| 🔄 Streaming | Gemini လို token-by-token stream |
| 📱 PWA | Phone home screen မှာ install လုပ်လို့ရ |
| 🤖 Multi-model | Llama, Qwen, Mistral, DeepSeek ရွေးလို့ရ |

---

## PRD Engine ဘာလုပ်သလဲ

မင်းမေးတဲ့ question ကို relational state vector |Ψ⟩ ∈ ℂ²⁴ အဖြစ်
SU(5) generator algebra ထဲ map လုပ်တယ်။

AI ဖြေတဲ့ answer ကိုလည်း state vector အဖြစ် map ပြီး
causal coherence စစ်တယ်:

- **Confidence** — question နဲ့ answer ရဲ့ causal coherence
- **Hallucination Risk** — မသေချာတဲ့ ဘာသာစကား patterns စစ်
- **Flag** — HIGH / MEDIUM / LOW confidence

ဒါကြောင့် standard LLM ထက် hallucination နည်းတယ်။

---

## Troubleshooting

**"Ollama မရှိဘူး" error:**
```bash
ollama serve   # terminal မှာ run ထားရမယ်
```

**Phone ကနေ ချိတ်မရဘူး:**
- PC firewall မှာ port 8000 ဖွင့်ပေးပါ
- Windows Defender → Allow app through firewall → Python ကို ✓

**Model slow:**
```bash
ollama pull qwen2.5:3b   # ပေါ့တဲ့ model သုံး
```
