# Memory Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI Agent with a built-in memory system — clean context per invocation, LLM-decision-based dual-channel memory retrieval, tool-calling agent loop, and post-conversation memory extraction with user review.

**Architecture:** Python CLI tool using SQLite + ChromaDB for storage, OpenAI-compatible API for both LLM and embedding. Each invocation is a standalone 3-step pipeline: Retrieve → Agent Loop → Extract. No sessions, no shared context between invocations.

**Tech Stack:** Python 3.10+, ChromaDB (embedded), SQLite (stdlib), httpx (HTTP client), PyYAML, pytest

## Global Constraints

- Python 3.10+ required
- Zero external services — ChromaDB embedded mode, SQLite file-based
- OpenAI-compatible API for all LLM and embedding calls
- Per-project memory storage under `<project-root>/.agent-memory/`
- Default LLM: deepseek-chat via api.deepseek.com
- Default embedding: text-embedding-3-small via api.openai.com
- Config file at `.agent-memory/config.yaml` with env var substitution (`${VAR}`)

---

## File Structure

```
src/memory_agent/
├── __init__.py          # Empty
├── config.py            # Config loading, env var substitution
├── storage.py           # SQLite schema + ChromaDB collection management
├── prompts.py           # Prompt templates for retriever decision + extractor
├── retriever.py         # LLM decision → dual-channel retrieval → format
├── tools.py             # Built-in tools (read_file, write_file, bash)
├── agent_loop.py        # OpenAI-compatible API loop with tool calling
├── extractor.py         # LLM extraction → user review → store
├── commands.py          # /memory slash commands
├── cli.py               # Entry point: argparse, orchestrates 3-step flow

tests/
├── __init__.py
├── conftest.py          # Shared fixtures (temp dir, in-memory DB, mock API)
├── test_config.py
├── test_storage.py
├── test_retriever.py
├── test_agent_loop.py
├── test_extractor.py
├── test_commands.py
├── test_integration.py
```

**Dependency Graph:**
```
config ──→ storage ──→ retriever ──→ agent_loop ──→ cli
                  ├──→ extractor ──→ cli
                  ├──→ commands ──→ cli
prompts ──→ retriever
prompts ──→ extractor
tools ──→ agent_loop
```

---

### Task 1: Project Scaffold and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/memory_agent/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: Installable package `memory-agent` with dependencies

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "memory-agent"
version = "0.1.0"
description = "CLI Agent with built-in vector memory system"
requires-python = ">=3.10"
dependencies = [
    "chromadb>=0.4.0",
    "httpx>=0.25.0",
    "pyyaml>=6.0",
    "openai>=1.0.0",
]

[project.scripts]
memory-agent = "memory_agent.cli:main"

[tool.setuptools.package-dir]
"" = "src"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty init files**

```bash
touch src/memory_agent/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: Install package in dev mode**

Run: `cd /home/leo/workspace/code-agent && .venv/bin/pip install -e .`
Expected: Package installs without errors

- [ ] **Step 4: Verify dependencies import**

Run: `.venv/bin/python -c "import chromadb; import httpx; import yaml; import openai; print('OK')"`
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/memory_agent/__init__.py tests/__init__.py
git commit -m "chore: project scaffold with dependencies"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/memory_agent/config.py`
- Create: `tests/test_config.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `Config` dataclass, `load_config(project_root: Path) -> Config`, `Config.from_dict(d: dict) -> Config`
- Produces: `DEFAULT_CONFIG_YAML` constant string

- [ ] **Step 1: Write tests (conftest.py)**

```python
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
  api_base: https://api.openai.com/v1
  api_key: sk-embed-key
  model: text-embedding-3-small
retrieval:
  top_k: 10
  similarity_threshold: 0.5
extractor:
  auto_confirm: false
  keep_full_transcript: true
""")
    return temp_project
```

- [ ] **Step 2: Write tests (test_config.py)**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement config.py**

```python
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
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-small

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/memory_agent/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: add config module with env var substitution"
```

---

### Task 3: Storage Module — SQLite

**Files:**
- Create: `src/memory_agent/storage.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces: `MemoryStore` class with `init_schema()`, `insert_memory(...)`, `get_memory(id)`, `get_recent_memories(limit, offset)`, `delete_memory(id)`, `search_by_tag(tag)`, `get_status()`, `get_all_tags()`

- [ ] **Step 1: Write tests (test_storage.py — SQLite part only)**

```python
# tests/test_storage.py
from datetime import datetime, timezone
from memory_agent.storage import MemoryStore


class TestInitSchema:
    def test_creates_tables(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        store._conn.execute("SELECT 1 FROM memories")
        store._conn.execute("SELECT 1 FROM key_points")
        store._conn.execute("SELECT 1 FROM tags")
        store._conn.execute("SELECT 1 FROM memory_tags")
        store._conn.execute("SELECT 1 FROM entities")
        store._conn.execute("SELECT 1 FROM decisions")

    def test_idempotent(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        store.init_schema()  # Should not raise


class TestInsertAndGet:
    def test_insert_and_retrieve_memory(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        now = datetime.now(timezone.utc)
        mid = store.insert_memory(
            summary="Test summary", conversation_at=now,
            conversation_json='{"messages": []}', chroma_doc_id="chroma-123",
            key_points=["Point 1", "Point 2"], tags=["python", "testing"],
            entities=[{"name": "src/main.py", "type": "file", "description": "Main"}],
            decisions=["Use pytest"],
        )
        mem = store.get_memory(mid)
        assert mem is not None
        assert mem["summary"] == "Test summary"
        assert mem["chroma_doc_id"] == "chroma-123"
        assert len(mem["key_points"]) == 2
        assert "python" in mem["tags"]
        assert len(mem["entities"]) == 1
        assert len(mem["decisions"]) == 1

    def test_get_nonexistent_returns_none(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        assert store.get_memory("nonexistent") is None


class TestRecentMemories:
    def test_returns_ordered_by_created_at(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        now = datetime.now(timezone.utc)
        ids = []
        for i in range(5):
            mid = store.insert_memory(
                summary=f"Memory {i}", conversation_at=now,
                conversation_json=None, chroma_doc_id=f"c-{i}",
                key_points=[], tags=[], entities=[], decisions=[],
            )
            ids.append(mid)
        recent = store.get_recent_memories(limit=3, offset=0)
        assert len(recent) == 3
        assert recent[0]["id"] == ids[-1]

    def test_offset_works(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        now = datetime.now(timezone.utc)
        for i in range(5):
            store.insert_memory(
                summary=f"Memory {i}", conversation_at=now,
                conversation_json=None, chroma_doc_id=f"c-{i}",
                key_points=[], tags=[], entities=[], decisions=[],
            )
        page1 = store.get_recent_memories(limit=2, offset=0)
        page2 = store.get_recent_memories(limit=2, offset=2)
        ids_page1 = {m["id"] for m in page1}
        ids_page2 = {m["id"] for m in page2}
        assert ids_page1.isdisjoint(ids_page2)


class TestDelete:
    def test_cascading_delete(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        now = datetime.now(timezone.utc)
        mid = store.insert_memory(
            summary="To delete", conversation_at=now,
            conversation_json=None, chroma_doc_id="c-del",
            key_points=["KP"], tags=["tag1"],
            entities=[{"name": "f", "type": "file", "description": "d"}],
            decisions=["D"],
        )
        store.delete_memory(mid)
        assert store.get_memory(mid) is None
        rows = store._conn.execute("SELECT 1 FROM key_points WHERE memory_id = ?", (mid,)).fetchall()
        assert len(rows) == 0


class TestStatus:
    def test_returns_counts(self, temp_project):
        db_path = temp_project / "test.db"
        store = MemoryStore(db_path)
        store.init_schema()
        now = datetime.now(timezone.utc)
        for i in range(3):
            store.insert_memory(
                summary=f"M{i}", conversation_at=now,
                conversation_json=None, chroma_doc_id=f"c-{i}",
                key_points=[f"KP{i}"], tags=["common", f"tag{i}"],
                entities=[], decisions=[],
            )
        status = store.get_status()
        assert status["total_memories"] == 3
        assert status["total_tags"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage.py -v`
Expected: FAIL — module import error

- [ ] **Step 3: Implement storage.py (SQLite part)**

```python
"""Storage layer: SQLite for metadata, ChromaDB for vectors."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    conversation_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_json TEXT,
    chroma_doc_id TEXT
);

CREATE TABLE IF NOT EXISTS key_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, tag_id)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    content TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_conversation_at ON memories(conversation_at DESC);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag_id ON memory_tags(tag_id);
"""


class MemoryStore:
    """SQLite-backed store for memory metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert_memory(
        self, summary, conversation_at, conversation_json,
        chroma_doc_id, key_points, tags, entities, decisions,
    ) -> str:
        conn = self._get_conn()
        memory_id = uuid.uuid4().hex[:12]

        conn.execute(
            """INSERT INTO memories (id, summary, conversation_at, conversation_json, chroma_doc_id)
               VALUES (?, ?, ?, ?, ?)""",
            (memory_id, summary, conversation_at, conversation_json, chroma_doc_id),
        )

        for i, kp in enumerate(key_points):
            conn.execute(
                "INSERT INTO key_points (memory_id, content, sort_order) VALUES (?, ?, ?)",
                (memory_id, kp, i),
            )

        for tag_name in tags:
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            row = conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
            if row:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
                    (memory_id, row["id"]),
                )

        for entity in entities:
            conn.execute(
                "INSERT INTO entities (memory_id, name, type, description) VALUES (?, ?, ?, ?)",
                (memory_id, entity["name"], entity["type"], entity.get("description", "")),
            )

        for decision in decisions:
            conn.execute(
                "INSERT INTO decisions (memory_id, content) VALUES (?, ?)",
                (memory_id, decision),
            )

        conn.commit()
        return memory_id

    def get_memory(self, memory_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._hydrate_memory(dict(row))

    def get_recent_memories(self, limit: int, offset: int = 0) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._hydrate_memory(dict(r)) for r in rows]

    def delete_memory(self, memory_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

    def search_by_tag(self, tag: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT m.* FROM memories m
               JOIN memory_tags mt ON m.id = mt.memory_id
               JOIN tags t ON mt.tag_id = t.id
               WHERE t.name = ? ORDER BY m.created_at DESC""",
            (tag,),
        ).fetchall()
        return [self._hydrate_memory(dict(r)) for r in rows]

    def get_status(self) -> dict:
        conn = self._get_conn()
        total_memories = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        total_tags = conn.execute("SELECT COUNT(*) as c FROM tags").fetchone()["c"]
        last_insert = conn.execute(
            "SELECT created_at FROM memories ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_memories": total_memories,
            "total_tags": total_tags,
            "last_insert_at": last_insert["created_at"] if last_insert else None,
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
        }

    def get_all_tags(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT name FROM tags ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    def _hydrate_memory(self, row: dict) -> dict:
        conn = self._get_conn()
        mid = row["id"]

        kp_rows = conn.execute(
            "SELECT content FROM key_points WHERE memory_id = ? ORDER BY sort_order", (mid,),
        ).fetchall()
        row["key_points"] = [r["content"] for r in kp_rows]

        tag_rows = conn.execute(
            "SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id WHERE mt.memory_id = ?", (mid,),
        ).fetchall()
        row["tags"] = [r["name"] for r in tag_rows]

        ent_rows = conn.execute(
            "SELECT name, type, description FROM entities WHERE memory_id = ?", (mid,),
        ).fetchall()
        row["entities"] = [{"name": r["name"], "type": r["type"], "description": r["description"]} for r in ent_rows]

        dec_rows = conn.execute(
            "SELECT content FROM decisions WHERE memory_id = ?", (mid,),
        ).fetchall()
        row["decisions"] = [r["content"] for r in dec_rows]

        return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage layer for memory metadata"
```

---

### Task 4: Storage Module — ChromaDB

**Files:**
- Modify: `src/memory_agent/storage.py` (add ChromaDB methods and imports)
- Modify: `tests/test_storage.py` (add ChromaDB test class)

**Interfaces:**
- Adds to `MemoryStore`: `init_chroma(persist_dir, embedding_api_base, embedding_api_key, embedding_model)`, `add_to_chroma(memory_id, text, metadata) -> str`, `query_chroma(query_text, top_k, min_distance=None) -> list[dict]`, `delete_from_chroma(doc_id)`, `count_chroma() -> int`

- [ ] **Step 1: Add ChromaDB tests to test_storage.py**

Append to tests/test_storage.py:

```python
class TestChromaDB:
    def test_init_creates_collection(self, temp_project):
        db_path = temp_project / "test.db"
        chroma_dir = temp_project / "chroma"
        store = MemoryStore(db_path)
        store.init_schema()
        store.init_chroma(
            persist_dir=chroma_dir,
            embedding_api_base="https://api.openai.com/v1",
            embedding_api_key="sk-test",
            embedding_model="text-embedding-3-small",
        )
        assert store._chroma_client is not None
        assert store._chroma_collection is not None
        assert store.count_chroma() == 0

    def test_add_and_query(self, temp_project):
        db_path = temp_project / "test.db"
        chroma_dir = temp_project / "chroma"
        store = MemoryStore(db_path)
        store.init_schema()
        store.init_chroma(
            persist_dir=chroma_dir,
            embedding_api_base="https://api.openai.com/v1",
            embedding_api_key="sk-test",
            embedding_model="text-embedding-3-small",
        )
        store.add_to_chroma(
            memory_id="mem-1", text="Python async patterns",
            metadata={"tags": "python", "conversation_at": "2026-07-28T00:00:00Z"},
        )
        store.add_to_chroma(
            memory_id="mem-2", text="Database connection pooling",
            metadata={"tags": "database", "conversation_at": "2026-07-29T00:00:00Z"},
        )
        assert store.count_chroma() == 2
        results = store.query_chroma("Python async programming", top_k=2)
        assert len(results) > 0

    def test_delete(self, temp_project):
        db_path = temp_project / "test.db"
        chroma_dir = temp_project / "chroma"
        store = MemoryStore(db_path)
        store.init_schema()
        store.init_chroma(
            persist_dir=chroma_dir,
            embedding_api_base="https://api.openai.com/v1",
            embedding_api_key="sk-test",
            embedding_model="text-embedding-3-small",
        )
        doc_id = store.add_to_chroma(memory_id="mem-del", text="Temporary", metadata={})
        assert store.count_chroma() == 1
        store.delete_from_chroma(doc_id)
        assert store.count_chroma() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage.py::TestChromaDB -v`
Expected: FAIL — AttributeError

- [ ] **Step 3: Add ChromaDB methods to storage.py**

Add imports at the top of storage.py:

```python
import chromadb
from chromadb.config import Settings as ChromaSettings
```

Add methods to MemoryStore class (before `_get_conn`):

```python
def init_chroma(self, persist_dir, embedding_api_base, embedding_api_key, embedding_model):
    persist_dir.mkdir(parents=True, exist_ok=True)
    self._chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    self._chroma_collection = self._chroma_client.get_or_create_collection(
        name="memories",
        metadata={"hnsw:space": "cosine"},
    )
    self._embedding_api_base = embedding_api_base
    self._embedding_api_key = embedding_api_key
    self._embedding_model = embedding_model

def _get_embedding(self, text: str) -> list[float]:
    import httpx
    response = httpx.post(
        f"{self._embedding_api_base}/embeddings",
        headers={
            "Authorization": f"Bearer {self._embedding_api_key}",
            "Content-Type": "application/json",
        },
        json={"model": self._embedding_model, "input": text},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]

def add_to_chroma(self, memory_id, text, metadata) -> str:
    embedding = self._get_embedding(text)
    doc_id = f"mem-{memory_id}"
    self._chroma_collection.add(
        ids=[doc_id], embeddings=[embedding],
        documents=[text], metadatas=[{**metadata, "memory_id": memory_id}],
    )
    return doc_id

def query_chroma(self, query_text, top_k, min_distance=None) -> list[dict]:
    embedding = self._get_embedding(query_text)
    results = self._chroma_collection.query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    memories = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else None
            if min_distance is not None and distance is not None and distance > min_distance:
                continue
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            memories.append({
                "chroma_doc_id": doc_id,
                "memory_id": metadata.get("memory_id", ""),
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": metadata,
                "distance": distance,
            })
    return memories

def delete_from_chroma(self, doc_id):
    self._chroma_collection.delete(ids=[doc_id])

def count_chroma(self) -> int:
    return self._chroma_collection.count()
```

- [ ] **Step 4: Run all storage tests**

Run: `.venv/bin/python -m pytest tests/test_storage.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/storage.py tests/test_storage.py
git commit -m "feat: add ChromaDB vector storage with embedding API"
```

---

### Task 5: Prompt Templates

**Files:**
- Create: `src/memory_agent/prompts.py`

**Interfaces:**
- Produces: `RETRIEVAL_DECISION_SYSTEM_PROMPT`, `RETRIEVAL_DECISION_USER_TEMPLATE`, `EXTRACTOR_SYSTEM_PROMPT`, `EXTRACTOR_USER_TEMPLATE`, `BASE_AGENT_SYSTEM_PROMPT`, `MEMORY_CONTEXT_HEADER`, `format_memory_for_injection(memory) -> str`, `format_memories_for_injection(memories) -> str`

- [ ] **Step 1: Implement prompts.py**

```python
"""Prompt templates for the memory agent."""

RETRIEVAL_DECISION_SYSTEM_PROMPT = """\
You are a memory retrieval decision engine. You have access to a history memory database
containing past conversations between a user and an AI assistant.

Given the user's query, decide whether to retrieve relevant past memories.
If needed, generate 1-3 semantic search queries AND/OR specify a recent range (N through M)
of the most recent memories.

Rules:
- If the user's query references past work, previous discussions, or prior context,
  retrieve relevant memories.
- For phrases like "just now", "last time", "previous", "a moment ago", use recent_range.
- For topical references like "Python async we discussed", use semantic_queries.
- For simple, self-contained questions (e.g., "hello", "write a hello world"),
  return need_retrieve: false.
- You can use both semantic_queries and recent_range together.

Output ONLY a JSON object, no other text:
{"need_retrieve": true/false, "semantic_queries": ["q1","q2"], "recent_range": {"start":N,"end":M} or null}
"""

RETRIEVAL_DECISION_USER_TEMPLATE = "User query: {user_query}"


EXTRACTOR_SYSTEM_PROMPT = """\
You are a conversation memory extractor. Given a complete transcript of a conversation
between a user and an AI assistant, extract the key information as structured data.

Output ONLY a JSON object:
{
  "summary": "Concise summary in <=200 characters, in the conversation's language",
  "key_points": ["Key conclusion 1", "Key conclusion 2", ...],
  "tags": ["tag1", "tag2", ...],
  "entities": [{"name":"...", "type":"file|function|class|concept|dependency|config", "description":"..."}],
  "decisions": ["Decision 1", "Decision 2", ...]
}

Guidelines:
- summary: <=200 chars, captures the essence of the conversation
- key_points: 3-8 items, each a single sentence
- tags: 3-6 lowercase tags for categorization
- entities: type must be one of file/function/class/concept/dependency/config
- decisions: explicit choices made. Can be empty array.
"""

EXTRACTOR_USER_TEMPLATE = "Conversation transcript:\n\n{transcript}"


BASE_AGENT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools. You can read files, write files,
and execute shell commands to help the user accomplish their tasks.

Work step by step. Use tools when needed. When you have completed the user's request,
provide a clear summary of what was done.
"""


MEMORY_CONTEXT_HEADER = "## Relevant Memories (from past conversations)\n"

_MEMORY_ENTRY_TEMPLATE = """\
### [{date}] {summary}
- Key Points:
{key_points}
- Tags: {tags}
"""


def format_memory_for_injection(memory: dict) -> str:
    date = memory.get("conversation_at", "unknown")
    if isinstance(date, str) and len(date) >= 10:
        date = date[:10]
    key_points = memory.get("key_points", [])
    kp_lines = "\n".join(f"  - {kp}" for kp in key_points) if key_points else "  (none)"
    tags = ", ".join(memory.get("tags", [])) or "none"
    return _MEMORY_ENTRY_TEMPLATE.format(date=date, summary=memory.get("summary", ""), key_points=kp_lines, tags=tags)


def format_memories_for_injection(memories: list[dict]) -> str:
    if not memories:
        return ""
    entries = [format_memory_for_injection(m) for m in memories]
    return MEMORY_CONTEXT_HEADER + "\n".join(entries)
```

- [ ] **Step 2: Commit**

```bash
git add src/memory_agent/prompts.py
git commit -m "feat: add prompt templates for retriever, extractor, and agent"
```

---

### Task 6: Retriever

**Files:**
- Create: `src/memory_agent/retriever.py`
- Create: `tests/test_retriever.py`

**Interfaces:**
- Produces: `Retriever` class with `retrieve(user_query) -> tuple[list[dict], str]`
- Consumes: `Config`, `MemoryStore`, prompts

- [ ] **Step 1: Write tests/test_retriever.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Implement retriever.py**

```python
"""Retriever: LLM decision → dual-channel search → format for injection."""
import json
from memory_agent.config import Config
from memory_agent.storage import MemoryStore
from memory_agent.prompts import (
    RETRIEVAL_DECISION_SYSTEM_PROMPT, RETRIEVAL_DECISION_USER_TEMPLATE,
    format_memories_for_injection,
)


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

        context = format_memories_for_injection(deduped)
        return deduped, context

    def _llm_decision(self, user_query: str) -> dict:
        import httpx
        response = httpx.post(
            f"{self.config.llm_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": RETRIEVAL_DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": RETRIEVAL_DECISION_USER_TEMPLATE.format(user_query=user_query)},
                ],
                "temperature": 0, "max_tokens": 200,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def _semantic_search(self, query: str) -> list[dict]:
        return self.store.query_chroma(query_text=query, top_k=self.config.retrieval_top_k)

    def _time_range_search(self, limit: int, offset: int) -> list[dict]:
        return self.store.get_recent_memories(limit=limit, offset=offset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever with LLM decision and dual-channel search"
```

---

### Task 7: Built-in Tools

**Files:**
- Create: `src/memory_agent/tools.py`

**Interfaces:**
- Produces: `TOOL_DEFINITIONS: list[dict]`, `execute_tool(name, arguments) -> str`

- [ ] **Step 1: Implement tools.py**

```python
"""Built-in tools: read_file, write_file, run_bash."""
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path.cwd()


def _resolve_path(file_path: str) -> Path:
    p = Path(file_path)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def tool_read_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    path = _resolve_path(file_path)
    if not path.exists():
        return f"Error: File not found: {path}"
    try:
        with open(path) as f:
            lines = f.readlines()
        total = len(lines)
        if limit is None:
            limit = total
        selected = lines[offset : offset + limit]
        result = "".join(selected)
        return f"File: {path} (lines {offset+1}-{min(offset+limit, total)} of {total})\n\n{result}"
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(file_path: str, content: str) -> str:
    path = _resolve_path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_run_bash(command: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(WORKSPACE_ROOT))
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. Use offset and limit for line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "offset": {"type": "integer", "description": "Start line (0-indexed)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command in workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_EXECUTORS = {"read_file": tool_read_file, "write_file": tool_write_file, "run_bash": tool_run_bash}


def execute_tool(name: str, arguments: dict) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return f"Error: Unknown tool '{name}'"
    return executor(**arguments)
```

- [ ] **Step 2: Commit**

```bash
git add src/memory_agent/tools.py
git commit -m "feat: add built-in tools (read_file, write_file, run_bash)"
```

---

### Task 8: Agent Loop

**Files:**
- Create: `src/memory_agent/agent_loop.py`
- Create: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `run_agent_loop(config, user_query, memory_context, tools=None, max_iterations=50) -> str`
- Consumes: `Config`, `tools.py`, `prompts.py`

- [ ] **Step 1: Write tests/test_agent_loop.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py -v`
Expected: FAIL

- [ ] **Step 3: Implement agent_loop.py**

```python
"""Agent loop: OpenAI-compatible API with tool calling iteration."""
import json
import httpx
from memory_agent.config import Config
from memory_agent.prompts import BASE_AGENT_SYSTEM_PROMPT
from memory_agent.tools import TOOL_DEFINITIONS, execute_tool


def run_agent_loop(config: Config, user_query: str, memory_context: str, tools=None, max_iterations=50) -> str:
    if tools is None:
        tools = TOOL_DEFINITIONS

    system_content = BASE_AGENT_SYSTEM_PROMPT
    if memory_context:
        system_content += "\n\n" + memory_context

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    transcript_parts = [f"User: {user_query}"]

    for _ in range(max_iterations):
        response = httpx.post(
            f"{config.llm_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {config.llm_api_key}", "Content-Type": "application/json"},
            json={"model": config.llm_model, "messages": messages, "tools": tools, "tool_choice": "auto"},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls:
            messages.append({
                "role": "assistant", "content": message.get("content"),
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                tool_result = execute_tool(tool_name, args)
                transcript_parts.append(f"Tool [{tool_name}]: {tc['function']['arguments']}\nResult: {tool_result[:500]}")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})
        else:
            assistant_content = message.get("content", "")
            transcript_parts.append(f"Assistant: {assistant_content}")
            return "\n\n".join(transcript_parts)

    transcript_parts.append("[Max tool call iterations reached]")
    return "\n\n".join(transcript_parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: add agent loop with tool calling iteration"
```

---

### Task 9: Extractor

**Files:**
- Create: `src/memory_agent/extractor.py`
- Create: `tests/test_extractor.py`

**Interfaces:**
- Produces: `extract_and_store(transcript, config, store) -> bool`
- Produces: `ExtractionResult` dataclass
- Consumes: `Config`, `MemoryStore`, prompts

- [ ] **Step 1: Write tests/test_extractor.py**

```python
from datetime import datetime, timezone
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
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='{"summary":"Test","key_points":["K1"],"tags":["t1"],"entities":[],"decisions":[]}'))]
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
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='{"summary":"T","key_points":[],"tags":[],"entities":[],"decisions":[]}'))]
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
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='{"summary":"T","key_points":[],"tags":[],"entities":[],"decisions":[]}'))]
            with patch("httpx.post", return_value=mock_resp):
                result = extract_and_store(transcript="test", config=cfg, store=store)
            assert result is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement extractor.py**

```python
"""Extractor: post-conversation memory extraction with user review."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
import httpx
from memory_agent.config import Config
from memory_agent.storage import MemoryStore
from memory_agent.prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_TEMPLATE


@dataclass
class ExtractionResult:
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            summary=data.get("summary", ""), key_points=data.get("key_points", []),
            tags=data.get("tags", []), entities=data.get("entities", []),
            decisions=data.get("decisions", []),
        )


def _call_extraction_llm(config: Config, transcript: str) -> ExtractionResult:
    response = httpx.post(
        f"{config.llm_api_base}/chat/completions",
        headers={"Authorization": f"Bearer {config.llm_api_key}", "Content-Type": "application/json"},
        json={
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTOR_USER_TEMPLATE.format(transcript=transcript)},
            ],
            "temperature": 0.3, "max_tokens": 1000,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return ExtractionResult.from_dict(json.loads(content))


def _display_preview(result: ExtractionResult) -> None:
    print("\n" + "=" * 50)
    print("📝 Memory Preview")
    print("=" * 50)
    print(f"\nSummary: {result.summary}")
    print(f"\nTags: {', '.join(result.tags) or '(none)'}")
    print(f"\nKey Points:")
    for kp in result.key_points:
        print(f"  • {kp}")
    if not result.key_points:
        print("  (none)")
    print(f"\nEntities:")
    for ent in result.entities:
        print(f"  • {ent['name']} ({ent['type']}): {ent.get('description', '')}")
    if not result.entities:
        print("  (none)")
    print(f"\nDecisions:")
    for dec in result.decisions:
        print(f"  • {dec}")
    if not result.decisions:
        print("  (none)")
    print()


def _get_user_choice() -> str:
    while True:
        choice = input("[S]ave  [E]dit  [D]iscard: ").strip().lower()
        if choice in ("s", "save"): return "save"
        elif choice in ("d", "discard"): return "discard"
        elif choice in ("e", "edit"): return "edit"
        else: print("Please enter S, E, or D")


def _open_editor(result: ExtractionResult) -> ExtractionResult:
    import subprocess, tempfile, os
    editor = os.environ.get("EDITOR", "vim")
    data = {"summary": result.summary, "key_points": result.key_points, "tags": result.tags, "entities": result.entities, "decisions": result.decisions}
    content = json.dumps(data, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path) as f:
            edited = json.loads(f.read())
        return ExtractionResult.from_dict(edited)
    finally:
        os.unlink(tmp_path)


def extract_and_store(transcript: str, config: Config, store: MemoryStore) -> bool:
    print("\nExtracting memory from conversation...")
    try:
        result = _call_extraction_llm(config, transcript)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    if config.extractor_auto_confirm:
        _store_result(result, transcript, config, store)
        print("Memory saved (auto-confirm).")
        return True

    while True:
        _display_preview(result)
        choice = _get_user_choice()
        if choice == "save":
            _store_result(result, transcript, config, store)
            print("Memory saved.")
            return True
        elif choice == "edit":
            result = _open_editor(result)
        elif choice == "discard":
            print("Memory discarded.")
            return False


def _store_result(result: ExtractionResult, transcript: str, config: Config, store: MemoryStore) -> None:
    now = datetime.now(timezone.utc)
    chroma_dir = config.memory_dir / "chroma"
    if not hasattr(store, "_chroma_collection") or store._chroma_collection is None:
        store.init_chroma(persist_dir=chroma_dir, embedding_api_base=config.embedding_api_base, embedding_api_key=config.embedding_api_key, embedding_model=config.embedding_model)
    embedding_text = result.summary
    if result.key_points:
        embedding_text += "\n" + "\n".join(result.key_points)
    chroma_doc_id = store.add_to_chroma(memory_id="pending", text=embedding_text, metadata={"tags": ",".join(result.tags), "conversation_at": now.isoformat()})
    conversation_json = json.dumps({"transcript": transcript}) if config.extractor_keep_full_transcript else None
    memory_id = store.insert_memory(
        summary=result.summary, conversation_at=now, conversation_json=conversation_json,
        chroma_doc_id=chroma_doc_id, key_points=result.key_points, tags=result.tags,
        entities=result.entities, decisions=result.decisions,
    )
    print(f"  Memory ID: {memory_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/extractor.py tests/test_extractor.py
git commit -m "feat: add extractor with LLM extraction and user review"
```

---

### Task 10: CLI Commands

**Files:**
- Create: `src/memory_agent/commands.py`
- Create: `tests/test_commands.py`

**Interfaces:**
- Produces: `handle_slash_command(message, store, injected_memories) -> tuple[bool, str]`

- [ ] **Step 1: Write tests/test_commands.py**

```python
from datetime import datetime, timezone
from memory_agent.storage import MemoryStore
from memory_agent.commands import handle_slash_command


class TestHandleSlashCommand:
    def test_not_a_command(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        was_cmd, response = handle_slash_command("hello world", store, [])
        assert was_cmd is False

    def test_memory_status(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        now = datetime.now(timezone.utc)
        store.insert_memory(summary="Test", conversation_at=now, conversation_json=None, chroma_doc_id="c1", key_points=[], tags=[], entities=[], decisions=[])
        was_cmd, response = handle_slash_command("/memory status", store, [])
        assert was_cmd is True
        assert "total" in response.lower()

    def test_memory_recent(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        now = datetime.now(timezone.utc)
        for i in range(3):
            store.insert_memory(summary=f"M{i}", conversation_at=now, conversation_json=None, chroma_doc_id=f"c{i}", key_points=[], tags=[], entities=[], decisions=[])
        was_cmd, response = handle_slash_command("/memory recent 2", store, [])
        assert was_cmd is True

    def test_memory_with_injected(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        injected = [{"summary": "Injected", "key_points": [], "tags": [], "conversation_at": "2026-07-30T00:00:00Z"}]
        was_cmd, response = handle_slash_command("/memory", store, injected)
        assert was_cmd is True
        assert "Injected" in response

    def test_memory_show_not_found(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        was_cmd, response = handle_slash_command("/memory show abc123", store, [])
        assert was_cmd is True
        assert "not found" in response.lower()

    def test_memory_delete(self, temp_project):
        store = MemoryStore(temp_project / "test.db"); store.init_schema()
        now = datetime.now(timezone.utc)
        mid = store.insert_memory(summary="Del", conversation_at=now, conversation_json=None, chroma_doc_id="cd", key_points=[], tags=[], entities=[], decisions=[])
        was_cmd, response = handle_slash_command(f"/memory delete {mid}", store, [])
        assert was_cmd is True
        assert store.get_memory(mid) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: FAIL

- [ ] **Step 3: Implement commands.py**

```python
"""Slash command handlers for /memory operations."""
from memory_agent.storage import MemoryStore


def handle_slash_command(message: str, store: MemoryStore, injected_memories: list[dict]) -> tuple[bool, str]:
    stripped = message.strip()
    if not stripped.startswith("/memory"):
        return False, ""

    parts = stripped.split(maxsplit=2)
    subcommand = parts[1] if len(parts) > 1 else ""
    args = parts[2] if len(parts) > 2 else ""

    if not subcommand:
        return True, _cmd_show_injected(injected_memories)
    elif subcommand == "recent":
        n = int(args) if args.isdigit() else 10
        return True, _cmd_recent(store, n)
    elif subcommand == "search":
        if not args: return True, "Usage: /memory search <query>"
        return True, _cmd_search(store, args)
    elif subcommand == "show":
        if not args: return True, "Usage: /memory show <id>"
        return True, _cmd_show(store, args)
    elif subcommand == "delete":
        if not args: return True, "Usage: /memory delete <id>"
        return True, _cmd_delete(store, args)
    elif subcommand == "status":
        return True, _cmd_status(store)
    else:
        return True, _cmd_usage()


def _cmd_show_injected(injected):
    if not injected:
        return "No memories were injected for this conversation."
    lines = ["Memories injected for this conversation:"]
    for i, mem in enumerate(injected, 1):
        lines.append(f"  {i}. [{mem.get('memory_id') or mem.get('id', '?')}] {mem.get('summary', '(no summary)')}")
    return "\n".join(lines)


def _cmd_recent(store, n):
    memories = store.get_recent_memories(limit=n)
    if not memories: return "No memories in database."
    lines = [f"Recent {len(memories)} memories:"]
    for mem in memories:
        lines.append(f"  [{mem['id']}] {mem['summary'][:80]}")
    return "\n".join(lines)


def _cmd_search(store, query):
    if not hasattr(store, "_chroma_collection") or store._chroma_collection is None:
        return "Vector search not available (ChromaDB not initialized)."
    try:
        results = store.query_chroma(query, top_k=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results: return f"No memories found matching: {query}"
    lines = [f"Search results for '{query}':"]
    for r in results:
        lines.append(f"  [{r.get('memory_id','?')}] (distance:{r.get('distance',0):.3f}) {r.get('text','')[:100]}")
    return "\n".join(lines)


def _cmd_show(store, memory_id):
    mem = store.get_memory(memory_id)
    if mem is None: return f"Memory not found: {memory_id}"
    lines = [f"=== Memory: {memory_id} ===", f"Summary: {mem['summary']}", f"Tags: {', '.join(mem.get('tags',[])) or '(none)'}", f"Conversation at: {mem.get('conversation_at','unknown')}", f"Created at: {mem.get('created_at','unknown')}", "", "Key Points:"]
    for kp in mem.get("key_points", []): lines.append(f"  • {kp}")
    lines.append(""); lines.append("Entities:")
    for ent in mem.get("entities", []): lines.append(f"  • {ent['name']} ({ent['type']}): {ent.get('description','')}")
    if not mem.get("entities"): lines.append("  (none)")
    lines.append(""); lines.append("Decisions:")
    for dec in mem.get("decisions", []): lines.append(f"  • {dec}")
    if not mem.get("decisions"): lines.append("  (none)")
    return "\n".join(lines)


def _cmd_delete(store, memory_id):
    mem = store.get_memory(memory_id)
    if mem is None: return f"Memory not found: {memory_id}"
    chroma_doc_id = mem.get("chroma_doc_id")
    if chroma_doc_id and hasattr(store, "_chroma_collection") and store._chroma_collection:
        try: store.delete_from_chroma(chroma_doc_id)
        except Exception: pass
    store.delete_memory(memory_id)
    return f"Memory deleted: {memory_id}"


def _cmd_status(store):
    status = store.get_status()
    lines = ["=== Memory Database Status ===", f"Total memories: {status['total_memories']}", f"Total tags: {status['total_tags']}", f"Last insert: {status['last_insert_at'] or 'never'}", f"DB path: {status['db_path']}", f"DB size: {status['db_size_bytes']} bytes"]
    tags = store.get_all_tags()
    if tags: lines.append(f"\nTags: {', '.join(tags)}")
    return "\n".join(lines)


def _cmd_usage():
    return """Usage:
  /memory                  Show injected memories
  /memory recent [N]       Show recent N memories (default 10)
  /memory search <query>   Semantic search
  /memory show <id>        Show memory details
  /memory delete <id>      Delete a memory
  /memory status           Database statistics"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_agent/commands.py tests/test_commands.py
git commit -m "feat: add /memory CLI commands"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `src/memory_agent/cli.py`

**Interfaces:**
- Produces: `main()` — argparse entry point orchestrating 3-step pipeline

- [ ] **Step 1: Implement cli.py**

```python
#!/usr/bin/env python3
"""Memory Agent CLI — 3-step pipeline: Retrieve → Agent Loop → Extract."""
import argparse
import sys
from pathlib import Path
from memory_agent.config import load_config
from memory_agent.storage import MemoryStore
from memory_agent.retriever import Retriever
from memory_agent.agent_loop import run_agent_loop
from memory_agent.extractor import extract_and_store


def main():
    parser = argparse.ArgumentParser(description="Memory Agent — AI assistant with persistent memory")
    parser.add_argument("query", nargs="*", help="Your query or task for the agent")
    parser.add_argument("-p", "--project", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--no-memory", action="store_true", help="Skip memory retrieval")
    parser.add_argument("--no-extract", action="store_true", help="Skip memory extraction")
    args = parser.parse_args()

    if args.query:
        user_query = " ".join(args.query)
    elif not sys.stdin.isatty():
        user_query = sys.stdin.read().strip()
    else:
        parser.print_help(); sys.exit(1)

    project_root = args.project.resolve()
    config = load_config(project_root)

    db_path = config.memory_dir / "memories.db"
    store = MemoryStore(db_path)
    store.init_schema()

    # Step 1: Memory Retrieval
    memory_context = ""
    injected_memories = []
    if not args.no_memory:
        print("Checking memory...", file=sys.stderr)
        try:
            retriever = Retriever(config, store)
            injected_memories, memory_context = retriever.retrieve(user_query)
            if memory_context:
                print(f"  Injected {len(injected_memories)} memory/memories.", file=sys.stderr)
        except Exception as e:
            print(f"  Memory retrieval failed: {e}", file=sys.stderr)

    # Step 2: Agent Loop
    print(file=sys.stderr)
    transcript = run_agent_loop(config=config, user_query=user_query, memory_context=memory_context)
    print(transcript)

    # Step 3: Memory Extraction
    if not args.no_extract:
        print(file=sys.stderr)
        try:
            extract_and_store(transcript=transcript, config=config, store=store)
        except Exception as e:
            print(f"Memory extraction failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the entry point is callable**

Run: `.venv/bin/python -m memory_agent.cli --help`
Expected: Help text with --project, --no-memory, --no-extract flags

- [ ] **Step 3: Commit**

```bash
git add src/memory_agent/cli.py
git commit -m "feat: add CLI entry point orchestrating 3-step pipeline"
```

---

### Task 12: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke test for full pipeline"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Retriever LLM decision + dual-channel (semantic + time-range) → Task 6
- ✅ Agent Loop with tool call iteration → Task 8
- ✅ Extractor with user review (Save/Edit/Discard) → Task 9
- ✅ SQLite storage with full schema → Task 3
- ✅ ChromaDB vector storage with embedding → Task 4
- ✅ Config with env var substitution → Task 2
- ✅ `/memory` CLI commands → Task 10
- ✅ CLI entry point with --no-memory, --no-extract flags → Task 11
- ✅ System Prompt structure (base + memory + tools) → Tasks 5, 8
- ✅ Per-project `.agent-memory/` directory → Task 2
- ✅ Embedding strategy (summary + key_points) → Task 9
- ✅ Dedup by memory_id when merging channels → Task 6
- ❌ Graph features → explicitly deferred in spec

**2. Placeholder scan:** No TBD, TODO, or vague instructions found.

**3. Type consistency:**
- `Config` fields consistent across all tasks
- `MemoryStore` methods: `insert_memory()`, `get_memory()`, `get_recent_memories()`, `delete_memory()`, `query_chroma()`, `init_chroma()`, `add_to_chroma()` — used consistently
- `Retriever.retrieve()` returns `tuple[list[dict], str]` — consistent with consumers
- `extract_and_store()` returns `bool` — consistent with cli.py
- `handle_slash_command()` returns `tuple[bool, str]` — interface is consistent
