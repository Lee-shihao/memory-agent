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
