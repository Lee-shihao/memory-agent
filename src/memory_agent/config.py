"""Configuration loading with env var substitution."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CONFIG_YAML = """\
# Memory Agent Configuration
llm:
  api_base: https://api.deepseek.com/v1
  api_key: ${DEEPSEEK_API_KEY}
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
"""

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return _ENV_VAR_RE.sub(_replace, value)


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve env vars in all string values of a dict."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _resolve_dict(value)
        else:
            result[key] = value
    return result


@dataclass
class Config:
    """Memory Agent configuration."""

    llm_api_base: str
    llm_api_key: str
    llm_model: str

    embedding_api_base: str
    embedding_api_key: str
    embedding_model: str

    retrieval_top_k: int = 10
    retrieval_similarity_threshold: float = 0.5

    extractor_auto_confirm: bool = False
    extractor_keep_full_transcript: bool = True

    memory_dir: Path = field(default_factory=Path)

    @classmethod
    def from_dict(cls, raw: dict, project_root: Path | None = None) -> "Config":
        """Create Config from a raw dictionary, resolving env vars."""
        resolved = _resolve_dict(raw)
        llm = resolved.get("llm", {})
        embedding = resolved.get("embedding", {})
        retrieval = resolved.get("retrieval", {})
        extractor = resolved.get("extractor", {})
        memory_dir = (project_root or Path.cwd()) / ".agent-memory"

        return cls(
            llm_api_base=llm.get("api_base", ""),
            llm_api_key=llm.get("api_key", ""),
            llm_model=llm.get("model", ""),
            embedding_api_base=embedding.get("api_base", ""),
            embedding_api_key=embedding.get("api_key", ""),
            embedding_model=embedding.get("model", ""),
            retrieval_top_k=retrieval.get("top_k", 10),
            retrieval_similarity_threshold=retrieval.get("similarity_threshold", 0.5),
            extractor_auto_confirm=extractor.get("auto_confirm", False),
            extractor_keep_full_transcript=extractor.get("keep_full_transcript", True),
            memory_dir=memory_dir,
        )


def load_config(project_root: Path) -> Config:
    """Load configuration from a project's .agent-memory/config.yaml."""
    config_dir = project_root / ".agent-memory"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"

    if not config_file.exists():
        config_file.write_text(DEFAULT_CONFIG_YAML)

    with open(config_file) as f:
        raw = yaml.safe_load(f) or {}

    return Config.from_dict(raw, project_root=project_root)
