from unittest.mock import patch, MagicMock
from memory_agent.config import Config
from memory_agent.storage import MemoryStore
from memory_agent.extractor import extract_and_store, ExtractionResult


def make_config(**kwargs):
    defaults = {
        "llm_api_base": "https://test.com/v1", "llm_api_key": "sk-test", "llm_model": "test-model",
        "embedding_api_base": "https://test.com/v1", "embedding_api_key": "sk-test", "embedding_model": "test-embed",
        "retrieval_top_k": 10, "retrieval_similarity_threshold": 0.5,
        "extractor_auto_confirm": False, "extractor_keep_full_transcript": True,
    }
    defaults.update(kwargs)
    return Config(**defaults)


def make_llm_response(content):
    """Build an httpx-like response mock whose .json() returns an OpenAI-style payload."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


class TestExtractionResult:
    def test_from_llm_response(self):
        response = {"summary": "Discussed Python async", "key_points": ["Use create_task"], "tags": ["python"], "entities": [{"name": "src/worker.py", "type": "file", "description": "Worker"}], "decisions": ["Adopt create_task"]}
        result = ExtractionResult.from_dict(response)
        assert result.summary == "Discussed Python async"
        assert len(result.key_points) == 1
        assert result.tags == ["python"]


class TestExtractAndStore:
    def test_extraction_and_store(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        with patch.object(store, "init_chroma"):
            store._chroma_collection = MagicMock()
            store._chroma_collection.count.return_value = 0
            store._get_embedding = MagicMock(return_value=[0.1] * 1536)
            mock_resp = make_llm_response('{"summary":"Test","key_points":["K1"],"tags":["t1"],"entities":[],"decisions":[]}')
            with patch("httpx.post", return_value=mock_resp):
                with patch("builtins.input", return_value="y"):
                    result = extract_and_store(transcript="User: test\nAssistant: ok", config=cfg, store=store)
            assert result is True

    def test_user_discards(self, temp_project):
        cfg = make_config()
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        with patch.object(store, "init_chroma"):
            store._chroma_collection = MagicMock()
            store._chroma_collection.count.return_value = 0
            store._get_embedding = MagicMock(return_value=[0.1] * 1536)
            mock_resp = make_llm_response('{"summary":"T","key_points":[],"tags":[],"entities":[],"decisions":[]}')
            with patch("httpx.post", return_value=mock_resp):
                with patch("builtins.input", return_value="n"):
                    result = extract_and_store(transcript="test", config=cfg, store=store)
            assert result is False

    def test_auto_confirm(self, temp_project):
        cfg = make_config(extractor_auto_confirm=True)
        store = MemoryStore(temp_project / "test.db")
        store.init_schema()
        with patch.object(store, "init_chroma"):
            store._chroma_collection = MagicMock()
            store._chroma_collection.count.return_value = 0
            store._get_embedding = MagicMock(return_value=[0.1] * 1536)
            mock_resp = make_llm_response('{"summary":"T","key_points":[],"tags":[],"entities":[],"decisions":[]}')
            with patch("httpx.post", return_value=mock_resp):
                result = extract_and_store(transcript="test", config=cfg, store=store)
            assert result is True
