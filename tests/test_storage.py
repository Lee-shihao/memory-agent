# tests/test_storage.py
import hashlib
import os
from datetime import datetime, timezone

import pytest

from memory_agent.storage import MemoryStore


@pytest.fixture(autouse=True)
def fake_embedding(monkeypatch):
    """Deterministic fake embeddings so ChromaDB tests never hit the embedding API.

    ChromaDB tests pass a fake API base/key; the real `_get_embedding` would
    make an HTTP call (401 for the fake key). We stub it with a hash-based
    embedding instead.
    """

    def _fake_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    monkeypatch.setattr(MemoryStore, "_get_embedding", _fake_embedding)


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


class TestChromaDB:
    def test_init_creates_collection(self, temp_project):
        db_path = temp_project / "test.db"
        chroma_dir = temp_project / "chroma"
        store = MemoryStore(db_path)
        store.init_schema()
        store.init_chroma(
            persist_dir=chroma_dir,
            embedding_api_base="https://api.siliconflow.cn/v1",
            embedding_api_key=os.environ.get("SF_API_KEY", "test-key"),
            embedding_model="BAAI/bge-m3",
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
            embedding_api_base="https://api.siliconflow.cn/v1",
            embedding_api_key=os.environ.get("SF_API_KEY", "test-key"),
            embedding_model="BAAI/bge-m3",
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
            embedding_api_base="https://api.siliconflow.cn/v1",
            embedding_api_key=os.environ.get("SF_API_KEY", "test-key"),
            embedding_model="BAAI/bge-m3",
        )
        doc_id = store.add_to_chroma(memory_id="mem-del", text="Temporary", metadata={})
        assert store.count_chroma() == 1
        store.delete_from_chroma(doc_id)
        assert store.count_chroma() == 0
