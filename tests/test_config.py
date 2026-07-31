# tests/test_config.py
from memory_agent.config import Config, load_config


class TestConfigFromDict:
    def test_parses_minimal_dict(self):
        cfg = Config.from_dict({
            "llm": {"api_base": "https://x.com/v1", "api_key": "k", "model": "m"},
            "embedding": {"api_base": "https://x.com/v1", "api_key": "k", "model": "m"},
            "retrieval": {},
            "extractor": {},
        })
        assert cfg.llm_api_base == "https://x.com/v1"
        assert cfg.llm_api_key == "k"
        assert cfg.llm_model == "m"

    def test_uses_defaults_for_missing_fields(self):
        cfg = Config.from_dict({
            "llm": {"api_base": "https://x.com/v1", "api_key": "k", "model": "m"},
            "embedding": {"api_base": "https://x.com/v1", "api_key": "k", "model": "m"},
        })
        assert cfg.retrieval_top_k == 10
        assert cfg.retrieval_similarity_threshold == 0.5
        assert cfg.extractor_auto_confirm is False
        assert cfg.extractor_keep_full_transcript is True

    def test_env_var_substitution(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "secret-123")
        cfg = Config.from_dict({
            "llm": {"api_base": "https://x.com/v1", "api_key": "${TEST_KEY}", "model": "m"},
            "embedding": {"api_base": "https://x.com/v1", "api_key": "k", "model": "m"},
        })
        assert cfg.llm_api_key == "secret-123"


class TestLoadConfig:
    def test_loads_from_project_root(self, temp_project_with_config):
        cfg = load_config(temp_project_with_config)
        assert cfg.llm_model == "deepseek-chat"
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.retrieval_top_k == 10

    def test_creates_default_config_if_missing(self, temp_project):
        cfg = load_config(temp_project)
        config_file = temp_project / ".agent-memory" / "config.yaml"
        assert config_file.exists()
        assert cfg.llm_model == "deepseek-chat"

    def test_memory_dir_returns_path(self, temp_project):
        cfg = load_config(temp_project)
        expected = temp_project / ".agent-memory"
        assert cfg.memory_dir == expected
