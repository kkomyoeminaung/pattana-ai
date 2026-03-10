"""
Pattana AI — Self-Learning Engine
- Unsupervised learning from conversation history
- Internet knowledge fetching (web search)
- Self-improvement via feedback loops
- PRD causal memory bank
"""

import json
import os
import re
import time
import hashlib
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

MEMORY_FILE   = os.path.join(DATA_DIR, "memory.json")
LEARNING_FILE = os.path.join(DATA_DIR, "learned_facts.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")
STATS_FILE    = os.path.join(DATA_DIR, "stats.json")


# ── Data Structures ───────────────────────────────────────────────────────────
@dataclass
class Memory:
    id:         str
    question:   str
    answer:     str
    confidence: float
    feedback:   int     # -1 bad, 0 neutral, +1 good
    timestamp:  str
    source:     str     # "user" | "web" | "self"
    tags:       list


@dataclass
class LearnedFact:
    id:       str
    content:  str
    source:   str
    topic:    str
    quality:  float
    uses:     int
    timestamp: str


# ── Persistence ───────────────────────────────────────────────────────────────
def load_json(path: str, default) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return default


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Memory Bank ───────────────────────────────────────────────────────────────
class MemoryBank:
    """
    Stores conversation history + learned facts.
    Unsupervised clustering to find related memories.
    """

    def __init__(self):
        raw = load_json(MEMORY_FILE, {"memories": [], "version": 1})
        self.memories: list[dict] = raw.get("memories", [])
        self._prune_old()

    def _prune_old(self):
        # Keep max 2000 memories, remove lowest confidence
        if len(self.memories) > 2000:
            self.memories.sort(key=lambda m: m.get("confidence", 0) + m.get("feedback", 0))
            self.memories = self.memories[-2000:]

    def add(self, question: str, answer: str, confidence: float,
            source: str = "user", tags: list = None) -> str:
        mid = hashlib.md5(f"{question}{time.time()}".encode()).hexdigest()[:12]
        mem = {
            "id": mid,
            "question": question[:500],
            "answer": answer[:2000],
            "confidence": confidence,
            "feedback": 0,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "tags": tags or self._extract_tags(question + " " + answer),
        }
        self.memories.append(mem)
        self.save()
        return mid

    def feedback(self, mem_id: str, score: int):
        """score: +1 = good, -1 = bad"""
        for m in self.memories:
            if m["id"] == mem_id:
                m["feedback"] = score
                m["confidence"] = min(1.0, m["confidence"] + score * 0.05)
                break
        self.save()

    def find_similar(self, query: str, top_k: int = 3) -> list:
        """Simple TF-IDF-like similarity for memory retrieval."""
        if not self.memories:
            return []
        q_words = set(query.lower().split())
        scored = []
        for m in self.memories:
            m_words = set((m["question"] + " " + m["answer"]).lower().split())
            overlap = len(q_words & m_words) / (len(q_words | m_words) + 1)
            score = overlap + m.get("feedback", 0) * 0.1 + m.get("confidence", 0) * 0.05
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k] if _ > 0.05]

    def _extract_tags(self, text: str) -> list:
        # Simple keyword extraction
        stopwords = {"the","a","an","is","are","was","were","be","to","of","and",
                     "in","it","for","on","with","as","at","by","from","that","this"}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:5]

    def get_stats(self) -> dict:
        total = len(self.memories)
        good  = sum(1 for m in self.memories if m.get("feedback", 0) > 0)
        bad   = sum(1 for m in self.memories if m.get("feedback", 0) < 0)
        avg_conf = np.mean([m.get("confidence", 0.5) for m in self.memories]) if self.memories else 0
        return {
            "total_memories": total,
            "good_feedback": good,
            "bad_feedback": bad,
            "avg_confidence": round(float(avg_conf), 3),
        }

    def save(self):
        save_json(MEMORY_FILE, {"memories": self.memories, "version": 1})


# ── Knowledge Base ────────────────────────────────────────────────────────────
class KnowledgeBase:
    """
    Stores learned facts from internet + self-improvement.
    Unsupervised topic clustering.
    """

    def __init__(self):
        raw = load_json(LEARNING_FILE, {"facts": [], "topics": {}})
        self.facts: list[dict] = raw.get("facts", [])
        self.topics: dict = raw.get("topics", {})

    def add_fact(self, content: str, source: str, topic: str, quality: float = 0.5) -> str:
        fid = hashlib.md5(content[:100].encode()).hexdigest()[:12]
        # Don't duplicate
        for f in self.facts:
            if f["id"] == fid:
                f["uses"] = f.get("uses", 0) + 1
                self.save()
                return fid

        fact = {
            "id": fid,
            "content": content[:1000],
            "source": source,
            "topic": topic,
            "quality": quality,
            "uses": 0,
            "timestamp": datetime.now().isoformat(),
        }
        self.facts.append(fact)

        # Update topic index
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(fid)

        # Keep max 500 facts
        if len(self.facts) > 500:
            self.facts.sort(key=lambda f: f.get("quality", 0) + f.get("uses", 0) * 0.1)
            self.facts = self.facts[-500:]

        self.save()
        return fid

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.facts:
            return []
        q_words = set(query.lower().split())
        scored = []
        for f in self.facts:
            f_words = set(f["content"].lower().split())
            overlap = len(q_words & f_words) / (len(q_words | f_words) + 1)
            score = overlap * (1 + f.get("quality", 0.5)) * (1 + f.get("uses", 0) * 0.02)
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k] if _ > 0.01]

    def get_all_topics(self) -> list:
        return sorted(self.topics.keys())

    def save(self):
        save_json(LEARNING_FILE, {"facts": self.facts, "topics": self.topics})


# ── Web Learner ───────────────────────────────────────────────────────────────
class WebLearner:
    """
    Fetches knowledge from internet to augment LLM responses.
    Uses DuckDuckGo instant answers (no API key needed).
    """

    SEARCH_URL = "https://api.duckduckgo.com/"

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._cache: dict = {}

    async def search(self, query: str) -> Optional[str]:
        if not HAS_HTTPX:
            return None

        # Cache check
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(self.SEARCH_URL, params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                })
                if r.status_code != 200:
                    return None

                data = r.json()
                result_parts = []

                # Abstract (Wikipedia-style summary)
                if data.get("Abstract"):
                    result_parts.append(data["Abstract"])

                # Answer (instant answer)
                if data.get("Answer"):
                    result_parts.append(f"Answer: {data['Answer']}")

                # Related topics
                for topic in data.get("RelatedTopics", [])[:2]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        result_parts.append(topic["Text"])

                if result_parts:
                    result = "\n".join(result_parts)[:800]
                    self._cache[cache_key] = result

                    # Save to knowledge base
                    topic = self._detect_topic(query)
                    self.kb.add_fact(result, "web:duckduckgo", topic, quality=0.7)

                    return result
        except Exception:
            pass
        return None

    def _detect_topic(self, text: str) -> str:
        text_lower = text.lower()
        topics = {
            "science": ["physics","chemistry","biology","quantum","atom","molecule"],
            "math": ["calculus","algebra","geometry","equation","theorem","proof"],
            "technology": ["computer","software","hardware","internet","ai","machine learning"],
            "history": ["war","century","ancient","empire","revolution","king","dynasty"],
            "medicine": ["disease","treatment","symptom","medicine","health","doctor"],
            "philosophy": ["ethics","logic","consciousness","existence","mind","truth"],
        }
        for topic, keywords in topics.items():
            if any(k in text_lower for k in keywords):
                return topic
        return "general"

    async def should_search(self, question: str) -> bool:
        """Decide if web search would help this question."""
        # Questions about current events, facts, definitions benefit from web
        triggers = [
            r'\b(what is|who is|when did|where is|how does|define|explain)\b',
            r'\b(latest|recent|current|news|today|2024|2025|2026)\b',
            r'\b(invented|discovered|born|founded|created)\b',
        ]
        for t in triggers:
            if re.search(t, question, re.IGNORECASE):
                return True
        return False


# ── Self-Improvement Engine ────────────────────────────────────────────────────
class SelfImprovementEngine:
    """
    Analyzes conversation patterns to improve future responses.
    Unsupervised learning: clusters bad responses, improves system prompt additions.
    """

    def __init__(self, memory: MemoryBank, kb: KnowledgeBase):
        self.memory = memory
        self.kb = kb
        self.stats = load_json(STATS_FILE, {
            "total_chats": 0,
            "good_responses": 0,
            "bad_responses": 0,
            "learned_topics": [],
            "improvement_cycles": 0,
            "last_improvement": None,
        })

    def record_chat(self, question: str, answer: str, confidence: float):
        self.stats["total_chats"] = self.stats.get("total_chats", 0) + 1
        if confidence >= 0.65:
            self.stats["good_responses"] = self.stats.get("good_responses", 0) + 1
        elif confidence < 0.4:
            self.stats["bad_responses"] = self.stats.get("bad_responses", 0) + 1
        self._save_stats()

    def get_context_injection(self, question: str, memory: MemoryBank, kb: KnowledgeBase) -> str:
        """
        Build a context string from memory + knowledge base
        to inject into the LLM prompt for this question.
        """
        parts = []

        # Retrieve similar past conversations
        similar = memory.find_similar(question, top_k=2)
        if similar:
            parts.append("## Relevant past knowledge:")
            for m in similar:
                if m.get("feedback", 0) >= 0:  # only good/neutral memories
                    parts.append(f"Q: {m['question'][:200]}")
                    parts.append(f"A: {m['answer'][:300]}")

        # Retrieve relevant learned facts
        facts = kb.search(question, top_k=2)
        if facts:
            parts.append("\n## Learned facts:")
            for f in facts:
                parts.append(f"[{f['topic']}] {f['content'][:300]}")

        return "\n".join(parts) if parts else ""

    def analyze_weaknesses(self) -> list:
        """Find topics where confidence is low — unsupervised weak-spot detection."""
        weak = []
        from collections import Counter
        bad_mems = [m for m in self.memory.memories if m.get("feedback", 0) < 0 or m.get("confidence", 1) < 0.4]
        if bad_mems:
            all_tags = []
            for m in bad_mems:
                all_tags.extend(m.get("tags", []))
            if all_tags:
                common = Counter(all_tags).most_common(5)
                weak = [tag for tag, _ in common]
        return weak

    def improvement_summary(self) -> dict:
        weak = self.analyze_weaknesses()
        return {
            "total_chats": self.stats.get("total_chats", 0),
            "accuracy_rate": round(
                self.stats.get("good_responses", 0) /
                max(self.stats.get("total_chats", 1), 1), 3),
            "weak_topics": weak,
            "memories": self.memory.get_stats(),
            "learned_facts": len(self.kb.facts),
            "improvement_cycles": self.stats.get("improvement_cycles", 0),
        }

    def _save_stats(self):
        save_json(STATS_FILE, self.stats)


# ── Global instances ───────────────────────────────────────────────────────────
_memory = None
_kb     = None
_web    = None
_self_improve = None


def get_memory() -> MemoryBank:
    global _memory
    if _memory is None:
        _memory = MemoryBank()
    return _memory


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def get_web_learner() -> WebLearner:
    global _web
    if _web is None:
        _web = WebLearner(get_kb())
    return _web


def get_self_improve() -> SelfImprovementEngine:
    global _self_improve
    if _self_improve is None:
        _self_improve = SelfImprovementEngine(get_memory(), get_kb())
    return _self_improve
