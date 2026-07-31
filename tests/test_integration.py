"""Smoke test: full pipeline with mocked LLM API."""
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone
from memory_agent.config import load_config, Config
from memory_agent.storage import MemoryStore
from memory_agent.retriever import Retriever
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store


def setup_config(temp_project: Path) -> Config:
    config_dir = temp_project / ".agent-memory"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("""
llm:
  api_base: https://api.deepseek.com/v1
  api_key: sk-test
  model: deepseek-chat
embedding:
  api_base: https://api.openai.com/v1
  api_key: sk-test
  model: text-embedding-3-small
retrieval:
  top_k: 5
  similarity_threshold: 0.5
extractor:
  auto_confirm: true
  keep_full_transcript: false
""")
    return load_config(temp_project)


class TestFullPipeline:
    def test_first_conversation_no_prior_memories(self, temp_project):
        cfg = setup_config(temp_project)
        store = MemoryStore(cfg.memory_dir / "memories.db")
        store.init_schema()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        call_count = [0]

        def fake_json():
            call_count[0] += 1
            if call_count[0] == 1:
                # Retrieval decision: no retrieval
                return {"choices": [{"message": {"content": '{"need_retrieve":false,"semantic_queries":[],"recent_range":null}'}}]}
            elif call_count[0] == 2:
                # Agent response
                return {"choices": [{"message": {"content": "I've created the file.", "tool_calls": None}}]}
            else:
                # Extraction
                return {"choices": [{"message": {"content": '{"summary":"Created file","key_points":["File created"],"tags":["file-ops"],"entities":[],"decisions":[]}'}}]}

        with patch("httpx.post") as mock_post:
            mock_post.return_value = mock_resp
            mock_post.return_value.json.side_effect = fake_json
            with patch.object(store, "init_chroma"):
                store._chroma_collection = MagicMock()
                store._chroma_collection.count.return_value = 0
                store._get_embedding = MagicMock(return_value=[0.0] * 1536)

                retriever = Retriever(cfg, store)
                memories, context = retriever.retrieve("Create a test file")
                assert memories == []
                assert context == ""

                transcript = run_agent_loop(config=cfg, user_query="Create a test file", memory_context="")
                assert "I've created" in transcript

                result = extract_and_store(transcript=transcript, config=cfg, store=store)
                assert result is True

    def test_second_conversation_finds_previous(self, temp_project):
        cfg = setup_config(temp_project)
        store = MemoryStore(cfg.memory_dir / "memories.db")
        store.init_schema()
        now = datetime.now(timezone.utc)

        with patch.object(store, "init_chroma"):
            store._chroma_collection = MagicMock()
            store._chroma_collection.count.return_value = 0
            store._get_embedding = MagicMock(return_value=[0.0] * 1536)
            store.insert_memory(
                summary="Discussed Python async patterns", conversation_at=now,
                conversation_json=None, chroma_doc_id="chroma-1",
                key_points=["Use create_task"], tags=["python", "async"],
                entities=[], decisions=[],
            )

        mock_resp = MagicMock()
        call_count = [0]

        def fake_json():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"choices": [{"message": {"content": '{"need_retrieve":true,"semantic_queries":["python async"],"recent_range":null}'}}]}
            elif call_count[0] == 2:
                return {"choices": [{"message": {"content": "Based on our previous discussion...", "tool_calls": None}}]}
            else:
                return {"choices": [{"message": {"content": '{"summary":"Continued async","key_points":[],"tags":["python"],"entities":[],"decisions":[]}'}}]}

        mock_semantic = [{"memory_id": "mem-1", "text": "Discussed Python async", "distance": 0.2, "metadata": {}}]

        with patch("httpx.post") as mock_post:
            mock_post.return_value = mock_resp
            mock_post.return_value.json.side_effect = fake_json
            with patch.object(store, "_chroma_collection", MagicMock()):
                with patch.object(store, "query_chroma", return_value=mock_semantic):
                    retriever = Retriever(cfg, store)
                    memories, context = retriever.retrieve("Continue the Python async discussion")
                    assert len(memories) > 0
                    assert "Relevant Memories" in context
