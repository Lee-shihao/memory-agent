from unittest.mock import patch, MagicMock
from memory_agent.config import Config
from memory_agent.agent_loop import run_agent_loop


def make_config(**kwargs):
    defaults = {
        "llm_api_base": "https://test.com/v1", "llm_api_key": "sk-test", "llm_model": "test-model",
        "embedding_api_base": "https://test.com/v1", "embedding_api_key": "sk-test", "embedding_model": "test-embed",
        "retrieval_top_k": 10, "retrieval_similarity_threshold": 0.5,
    }
    defaults.update(kwargs)
    return Config(**defaults)


class TestAgentLoop:
    def test_simple_text_response(self):
        cfg = make_config()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Hello!", "tool_calls": None}}]
            }
            transcript = run_agent_loop(config=cfg, user_query="Hi", memory_context="", tools=[])
            assert "Hello" in transcript

    def test_tool_call_loop(self):
        cfg = make_config()
        call_count = [0]

        def fake_json():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"t.txt"}'}}
                ]}}]}
            return {"choices": [{"message": {"content": "File contents: hello", "tool_calls": None}}]}

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.side_effect = fake_json
            with patch("memory_agent.tools.tool_read_file", return_value="hello"):
                transcript = run_agent_loop(config=cfg, user_query="Read t.txt", memory_context="", tools=[])
            assert call_count[0] == 2

    def test_respects_max_iterations(self):
        cfg = make_config()
        def fake_json():
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"t.txt"}'}}
            ]}}]}
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock()
            mock_post.return_value.raise_for_status = MagicMock()
            mock_post.return_value.json.side_effect = fake_json
            with patch("memory_agent.tools.tool_read_file", return_value="contents"):
                transcript = run_agent_loop(config=cfg, user_query="Read", memory_context="", tools=[], max_iterations=5)
            assert "Max tool call iterations" in transcript
