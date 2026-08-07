//! Relational storage: SQLite CRUD for memories and their child records.

use crate::prompts::{Entity, Memory};
use crate::storage::MemoryStore;
use anyhow::Result;
use rusqlite::params;
use serde_json::Value as JsonValue;

/// A `memories` row projected for hydration.
type MemoryRow = (String, String, Option<String>, Option<String>, Option<String>, Option<i64>);

impl MemoryStore {
    /// Insert a memory and its child records. Returns the memory id (either the
    /// caller-provided one or a freshly generated short id).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_memory(
        &self,
        summary: &str,
        conversation_at: &chrono::DateTime<chrono::Utc>,
        conversation_json: Option<&str>,
        key_points: &[String],
        tags: &[String],
        entities: &[JsonValue],
        decisions: &[String],
        memory_id: Option<&str>,
    ) -> Result<String> {
        let memory_id = memory_id.map(|s| s.to_string()).unwrap_or_else(|| {
            uuid::Uuid::new_v4()
                .to_string()
                .chars()
                .take(12)
                .collect::<String>()
        });

        self.conn.execute(
            "INSERT INTO memories (id, summary, conversation_at, conversation_json)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                memory_id,
                summary,
                conversation_at.to_rfc3339(),
                conversation_json
            ],
        )?;

        for (i, kp) in key_points.iter().enumerate() {
            self.conn.execute(
                "INSERT INTO key_points (memory_id, content, sort_order) VALUES (?1, ?2, ?3)",
                params![memory_id, kp, i as i32],
            )?;
        }

        for tag_name in tags {
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (name) VALUES (?1)",
                params![tag_name],
            )?;
            let tag_id: i64 = self.conn.query_row(
                "SELECT id FROM tags WHERE name = ?1",
                params![tag_name],
                |row| row.get(0),
            )?;
            self.conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?1, ?2)",
                params![memory_id, tag_id],
            )?;
        }

        for entity in entities {
            self.conn.execute(
                "INSERT INTO entities (memory_id, name, type, description) VALUES (?1, ?2, ?3, ?4)",
                params![
                    memory_id,
                    entity["name"].as_str().unwrap_or(""),
                    entity["type"].as_str().unwrap_or(""),
                    entity["description"].as_str().unwrap_or(""),
                ],
            )?;
        }

        for decision in decisions {
            self.conn.execute(
                "INSERT INTO decisions (memory_id, content) VALUES (?1, ?2)",
                params![memory_id, decision],
            )?;
        }

        Ok(memory_id)
    }

    pub fn get_memory(&self, memory_id: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, conversation_at, created_at, conversation_json, vec_rowid
             FROM memories WHERE id = ?1",
        )?;
        let row = stmt
            .query_row(params![memory_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                ))
            })
            .ok();

        match row {
            Some((id, summary, conv_at, created_at, conv_json, vec_rowid)) => Ok(Some(
                self.hydrate_memory(id, summary, conv_at, created_at, conv_json, vec_rowid)?,
            )),
            None => Ok(None),
        }
    }

    pub fn get_recent_memories(&self, limit: usize, offset: usize) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, conversation_at, created_at, conversation_json, vec_rowid
             FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ?1 OFFSET ?2",
        )?;
        let rows: Vec<MemoryRow> = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut memories = Vec::new();
        for (id, summary, conv_at, created_at, conv_json, vec_rowid) in rows {
            memories.push(self.hydrate_memory(
                id,
                summary,
                conv_at,
                created_at,
                conv_json,
                vec_rowid,
            )?);
        }
        Ok(memories)
    }

    /// Delete a memory and its vector: first drop the vector row (which needs
    /// the metadata row to resolve `vec_rowid`), then the metadata (child rows
    /// cascade via foreign keys).
    pub fn delete_memory(&self, memory_id: &str) -> Result<()> {
        self.remove_memory_vector(memory_id)?;
        self.conn
            .execute("DELETE FROM memories WHERE id = ?1", params![memory_id])?;
        Ok(())
    }

    pub fn search_by_tag(&self, tag: &str) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.summary, m.conversation_at, m.created_at, m.conversation_json, m.vec_rowid
             FROM memories m
             JOIN memory_tags mt ON m.id = mt.memory_id
             JOIN tags t ON mt.tag_id = t.id
             WHERE t.name = ?1 ORDER BY m.created_at DESC, m.rowid DESC",
        )?;
        let rows: Vec<MemoryRow> = stmt
            .query_map(params![tag], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut memories = Vec::new();
        for (id, summary, conv_at, created_at, conv_json, vec_rowid) in rows {
            memories.push(self.hydrate_memory(
                id,
                summary,
                conv_at,
                created_at,
                conv_json,
                vec_rowid,
            )?);
        }
        Ok(memories)
    }

    pub fn get_status(&self) -> Result<JsonValue> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        let total_tags: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM tags", [], |row| row.get(0))?;
        let last_insert: Option<String> = self
            .conn
            .query_row(
                "SELECT created_at FROM memories ORDER BY created_at DESC, rowid DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .ok();

        let db_size = std::fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0);

        Ok(serde_json::json!({
            "total_memories": total,
            "total_tags": total_tags,
            "last_insert_at": last_insert,
            "db_path": self.db_path.to_string_lossy(),
            "db_size_bytes": db_size,
        }))
    }

    pub fn get_all_tags(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT name FROM tags ORDER BY name")?;
        let tags = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(tags)
    }

    #[allow(clippy::too_many_arguments)]
    fn hydrate_memory(
        &self,
        id: String,
        summary: String,
        conversation_at: Option<String>,
        created_at: Option<String>,
        conversation_json: Option<String>,
        vec_rowid: Option<i64>,
    ) -> Result<Memory> {
        // Key points
        let key_points: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT content FROM key_points WHERE memory_id = ?1 ORDER BY sort_order",
            )?;
            let rows = stmt.query_map(params![id], |row| row.get(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        // Tags
        let tags: Vec<String> = {
            let mut stmt = self.conn.prepare(
                "SELECT t.name FROM tags t
                 JOIN memory_tags mt ON t.id = mt.tag_id
                 WHERE mt.memory_id = ?1",
            )?;
            let rows = stmt.query_map(params![id], |row| row.get(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        // Entities
        let entities: Vec<Entity> = {
            let mut stmt = self
                .conn
                .prepare("SELECT name, type, description FROM entities WHERE memory_id = ?1")?;
            let rows = stmt.query_map(params![id], |row| {
                Ok(Entity {
                    name: row.get(0)?,
                    entity_type: row.get(1)?,
                    description: row.get(2)?,
                })
            })?;
            rows.filter_map(|r| r.ok()).collect()
        };

        // Decisions
        let decisions: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare("SELECT content FROM decisions WHERE memory_id = ?1")?;
            let rows = stmt.query_map(params![id], |row| row.get(0))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        Ok(Memory {
            id,
            summary,
            conversation_at,
            created_at,
            key_points,
            tags,
            entities,
            decisions,
            vec_rowid,
            conversation_json,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_sample(store: &MemoryStore, summary: &str, tags: &[&str]) -> String {
        store
            .insert_memory(
                summary,
                &chrono::Utc::now(),
                None,
                &["KP1".into(), "KP2".into()],
                &tags.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                &[serde_json::json!({"name": "main.rs", "type": "file", "description": "entry"})],
                &["Decision 1".into()],
                None,
            )
            .unwrap()
    }

    #[test]
    fn test_init_schema_creates_tables() {
        let tmp = std::env::temp_dir().join("test_storage_relation.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();
        let count: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('memories','skill_vectors')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_insert_and_get_memory() {
        let tmp = std::env::temp_dir().join("test_memory_relation.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        let mid = insert_sample(&store, "Test summary", &["rust", "test"]);
        assert!(!mid.is_empty());

        let mem = store.get_memory(&mid).unwrap().unwrap();
        assert_eq!(mem.summary, "Test summary");
        assert_eq!(mem.tags.len(), 2);
        assert_eq!(mem.key_points.len(), 2);
        assert_eq!(mem.entities.len(), 1);
        assert_eq!(mem.decisions.len(), 1);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_get_recent_and_delete() {
        let tmp = std::env::temp_dir().join("test_recent_relation.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        for i in 0..5 {
            insert_sample(&store, &format!("Summary {i}"), &[]);
        }

        let recent = store.get_recent_memories(3, 0).unwrap();
        assert_eq!(recent.len(), 3);

        let first_id = recent[0].id.clone();
        store.delete_memory(&first_id).unwrap();
        assert!(store.get_memory(&first_id).unwrap().is_none());

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_tags_and_entities() {
        let tmp = std::env::temp_dir().join("test_tags_relation.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        insert_sample(&store, "Tag test", &["rust", "database"]);

        let all_tags = store.get_all_tags().unwrap();
        assert!(all_tags.contains(&"rust".to_string()));
        assert!(all_tags.contains(&"database".to_string()));

        let by_tag = store.search_by_tag("rust").unwrap();
        assert!(!by_tag.is_empty());
        assert_eq!(by_tag[0].entities[0].name, "main.rs");

        let _ = std::fs::remove_file(&tmp);
    }
}
