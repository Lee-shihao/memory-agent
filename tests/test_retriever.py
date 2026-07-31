from unittest.mock import patch, MagicMock
from memory_agent.config import Config
from memory_agent.storage import MemoryStore
from memory_agent.retriever import Retriever


def make_config(**kwargs):
    defaults = {
        "llm_api_base": "https://test.com/v1", "llm_api_key": "sk-test", "llm_model": "test-model",
        "embedding_api_base": "https://test.com/v1", "embedding_api_key": "sk-test", "embedding_model": "test-embed",
        "retrieval_top_k": 10, "retrieval_similarity_threshold": 0.5,
    }
    defaults.update(kwargs)
    return Config(**defaults)


class TestRetrievalDecision:
    def test_no_retrieval_needed(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        retriever = Retriever(cfg, store)
        decision = {"need_retrieve": False, "semantic_queries": [], "recent_range": None}
        with patch.object(retriever, "_llm_decision", return_value=decision):
            memories, context = retriever.retrieve("hello world")
            assert memories == []
            assert context == ""

    def test_semantic_retrieval_only(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        retriever = Retriever(cfg, store)
        decision = {"need_retrieve": True, "semantic_queries": ["python async"], "recent_range": None}
        mock_results = [{"memory_id": "mem-1", "text": "...", "distance": 0.2, "metadata": {}}]
        with patch.object(retriever, "_llm_decision", return_value=decision):
            with patch.object(retriever, "_semantic_search", return_value=mock_results):
                with patch.object(retriever, "_time_range_search", return_value=[]) as trs:
                    memories, context = retriever.retrieve("python async")
                    assert len(memories) == 1
                    assert "Relevant Memories" in context
                    trs.assert_not_called()

    def test_time_range_retrieval_only(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        retriever = Retriever(cfg, store)
        decision = {"need_retrieve": True, "semantic_queries": [], "recent_range": {"start": 1, "end": 3}}
        mock_results = [{"id": "mem-1", "summary": "Recent", "key_points": [], "tags": [], "conversation_at": "2026-07-30T00:00:00Z"}]
        with patch.object(retriever, "_llm_decision", return_value=decision):
            with patch.object(retriever, "_time_range_search", return_value=mock_results) as trs:
                with patch.object(retriever, "_semantic_search", return_value=[]) as ss:
                    memories, context = retriever.retrieve("recent stuff")
                    trs.assert_called_once_with(3, 0)
                    ss.assert_not_called()

    def test_dual_channel_dedup(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        retriever = Retriever(cfg, store)
        decision = {"need_retrieve": True, "semantic_queries": ["test"], "recent_range": {"start": 1, "end": 3}}
        shared = {"memory_id": "mem-shared", "text": "...", "distance": 0.1, "metadata": {}}
        unique_a = {"memory_id": "mem-a", "text": "...", "distance": 0.3, "metadata": {}}
        with patch.object(retriever, "_llm_decision", return_value=decision):
            with patch.object(retriever, "_semantic_search", return_value=[shared, unique_a]):
                with patch.object(retriever, "_time_range_search", return_value=[
                    {"id": "mem-shared", "summary": "...", "key_points": [], "tags": [], "conversation_at": "2026-07-30T00:00:00Z"},
                    {"id": "mem-b", "summary": "...", "key_points": [], "tags": [], "conversation_at": "2026-07-29T00:00:00Z"},
                ]):
                    memories, context = retriever.retrieve("test query")
                    memory_ids = [m.get("memory_id") or m.get("id") for m in memories]
                    assert memory_ids.count("mem-shared") == 1
                    assert len(memories) == 3
