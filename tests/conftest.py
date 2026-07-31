# tests/conftest.py
import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_project():
    """Create a temporary project directory with no existing .agent-memory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_project_with_config(temp_project):
    """Create a temporary project with a config.yaml file."""
    config_dir = temp_project / ".agent-memory"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("""
llm:
  api_base: https://api.deepseek.com/v1
  api_key: sk-test-key
  model: deepseek-chat
embedding:
  api_base: https://api.siliconflow.cn/v1
  api_key: ${SF_API_KEY}
  model: BAAI/bge-m3
retrieval:
  top_k: 10
  similarity_threshold: 0.5
extractor:
  auto_confirm: false
  keep_full_transcript: true
""")
    return temp_project
