"""Smoke test: full pipeline with tool-driven retrieval — no pre-loop injection."""
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone
from memory_agent.config import load_config, Config
from memory_agent.storage import MemoryStore
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store
from memory_agent.tools import reset_session_state


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


class TestAgentLoopToolDriven:
    """Agent loop now receives no pre-injected memory_context."""

    def test_agent_loop_no_memory_context_parameter(self, temp_project):
        cfg = setup_config(temp_project)

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Task completed.", "tool_calls": None}}]
            }
            # New signature: no memory_context parameter
            transcript = run_agent_loop(config=cfg, user_query="Simple task")
            assert "Task completed" in transcript

    def test_agent_loop_uses_round1_prompt(self, temp_project):
        cfg = setup_config(temp_project)

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "OK", "tool_calls": None}}]
            }
            run_agent_loop(config=cfg, user_query="test")

            call_args = mock_post.call_args
            req_body = call_args[1]["json"]
            system_msg = req_body["messages"][0]["content"]
            assert "Before diving into the task" in system_msg
            assert "search_memory" in system_msg
            assert "search_skills" in system_msg

    def test_session_state_reset(self, temp_project):
        """reset_session_state clears dedup sets."""
        from memory_agent.tools import _returned_memory_ids, _returned_skill_names

        _returned_memory_ids.add("test-1")
        _returned_skill_names.add("test-skill")

        reset_session_state()

        assert len(_returned_memory_ids) == 0
        assert len(_returned_skill_names) == 0


class TestExtractionStillWorks:
    """Memory extraction after agent loop is unchanged."""

    def test_extract_after_agent_loop(self, temp_project):
        cfg = setup_config(temp_project)
        store = MemoryStore(cfg.memory_dir / "memories.db")
        store.init_schema()

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()

            call_count = [0]
            def fake_json():
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"choices": [{"message": {"content": "Done.", "tool_calls": None}}]}
                else:
                    return {"choices": [{"message": {"content": '{"summary":"Task done","key_points":["Done"],"tags":["test"],"entities":[],"decisions":[]}'}}]}

            mock_post.return_value.json.side_effect = fake_json

            with patch.object(store, "init_chroma"):
                store._chroma_collection = MagicMock()
                store._chroma_collection.count.return_value = 0
                store._get_embedding = MagicMock(return_value=[0.0] * 1536)

                transcript = run_agent_loop(config=cfg, user_query="Do something")

                result = extract_and_store(transcript=transcript, config=cfg, store=store)
                assert result is True
