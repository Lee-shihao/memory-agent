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
            "SELECT * FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
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
               WHERE t.name = ? ORDER BY m.created_at DESC, m.rowid DESC""",
            (tag,),
        ).fetchall()
        return [self._hydrate_memory(dict(r)) for r in rows]

    def get_status(self) -> dict:
        conn = self._get_conn()
        total_memories = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        total_tags = conn.execute("SELECT COUNT(*) as c FROM tags").fetchone()["c"]
        last_insert = conn.execute(
            "SELECT created_at FROM memories ORDER BY created_at DESC, rowid DESC LIMIT 1"
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
