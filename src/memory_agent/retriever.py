"""Retriever: LLM decision → dual-channel search → format for injection."""
import json
import logging
from memory_agent.config import Config
from memory_agent.debug import is_enabled as _debug_enabled
from memory_agent.debug import log_request as _log_req
from memory_agent.debug import log_response as _log_resp
from memory_agent.storage import MemoryStore
from memory_agent.prompts import (
    RETRIEVAL_DECISION_SYSTEM_PROMPT, RETRIEVAL_DECISION_USER_TEMPLATE,
    format_memories_for_injection,
)

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: Config, store: MemoryStore):
        self.config = config
        self.store = store

    def retrieve(self, user_query: str) -> tuple[list[dict], str]:
        decision = self._llm_decision(user_query)
        if not decision.get("need_retrieve"):
            return [], ""

        raw_results = []

        queries = decision.get("semantic_queries", []) or []
        for query in queries:
            raw_results.extend(self._semantic_search(query))

        recent_range = decision.get("recent_range")
        if recent_range:
            start = recent_range.get("start", 1)
            end = recent_range.get("end", 10)
            limit = end - start + 1
            offset = start - 1
            raw_results.extend(self._time_range_search(limit, offset))

        # Dedup by memory_id
        seen = set()
        deduped = []
        for r in raw_results:
            mid = r.get("memory_id") or r.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            deduped.append(r)

        # Hydrate semantic results: ChromaDB returns {memory_id, text, distance}
        # but format_memory_for_injection needs {summary, key_points, tags, ...}
        hydrated = []
        for r in deduped:
            mid = r.get("memory_id") or r.get("id")
            if "summary" not in r or r.get("summary") is None:
                full = self.store.get_memory(mid)
                if full:
                    hydrated.append(full)
                else:
                    hydrated.append({
                        "id": mid, "memory_id": mid,
                        "summary": r.get("text", ""),
                        "key_points": [], "tags": [],
                        "conversation_at": None,
                        "entities": [], "decisions": [],
                    })
            else:
                hydrated.append(r)

        context = format_memories_for_injection(hydrated)
        return hydrated, context

    def _llm_decision(self, user_query: str) -> dict:
        import httpx
        url = f"{self.config.llm_api_base}/chat/completions"
        req_headers = {"Authorization": f"Bearer {self.config.llm_api_key}", "Content-Type": "application/json"}
        req_body = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": RETRIEVAL_DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": RETRIEVAL_DECISION_USER_TEMPLATE.format(user_query=user_query)},
            ],
            "temperature": 0, "max_tokens": 200,
        }
        if _debug_enabled():
            rid = _log_req("retriever", "POST", url, req_headers, req_body)
        response = httpx.post(url, headers=req_headers, json=req_body, timeout=30)
        response.raise_for_status()
        data = response.json()
        if _debug_enabled():
            _log_resp(rid, response.status_code, data)
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse retrieval decision JSON: %s", e)
            return {"need_retrieve": False, "semantic_queries": [], "recent_range": None}

    def _semantic_search(self, query: str) -> list[dict]:
        return self.store.query_chroma(query_text=query, top_k=self.config.retrieval_top_k, min_distance=self.config.retrieval_similarity_threshold)

    def _time_range_search(self, limit: int, offset: int) -> list[dict]:
        return self.store.get_recent_memories(limit=limit, offset=offset)
