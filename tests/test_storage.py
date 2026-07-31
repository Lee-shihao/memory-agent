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
