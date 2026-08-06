use crate::prompts::{Entity, Memory};
use anyhow::{Context, Result};
use instant_distance::{Builder, HnswMap, Point as HnswPoint, Search};
use parking_lot::RwLock;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::fs;
/// Storage layer: SQLite for metadata, HNSW for vectors.
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SCHEMA_SQL: &str = r#"
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
"#;
/// A vector point wrapping a 1024-d embedding.
/// Distance metric: cosine distance (1 - cosine_similarity on unit-normalized vectors).
#[derive(Clone, Serialize, Deserialize)]
pub struct EmbeddingPoint {
    pub data: Vec<f32>,
}

impl EmbeddingPoint {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl HnswPoint for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot.max(-1.0).min(1.0)
    }
}

/// Metadata stored alongside a memory vector.
#[derive(Clone, Serialize, Deserialize)]
struct MemoryMeta {
    doc_id: String,
    text: String,
    metadata_json: String,
}

/// Metadata stored alongside a skill vector.
#[derive(Clone, Serialize, Deserialize)]
struct SkillMeta {
    name: String,
    description: String,
    source: String,
}

type MemoryIndex = HnswMap<EmbeddingPoint, MemoryMeta>;
type SkillIndex = HnswMap<EmbeddingPoint, SkillMeta>;

pub struct MemoryStore {
    conn: Connection,
    db_path: PathBuf,
    // HNSW vector indices (Arc<RwLock<>> for concurrent reads, exclusive writes)
    memories_idx: Arc<RwLock<Option<MemoryIndex>>>,
    skills_idx: Arc<RwLock<Option<SkillIndex>>>,
    persist_dir: PathBuf,
    // Embedding config
    embedding_api_base: String,
    embedding_api_key: String,
    embedding_model: String,
    vector_store_initialized: bool,
}

impl MemoryStore {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON")?;
        Ok(MemoryStore {
            conn,
            db_path: db_path.to_path_buf(),
            memories_idx: Arc::new(RwLock::new(None)),
            skills_idx: Arc::new(RwLock::new(None)),
            persist_dir: PathBuf::new(),
            embedding_api_base: String::new(),
            embedding_api_key: String::new(),
            embedding_model: String::new(),
            vector_store_initialized: false,
        })
    }

    pub fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    /// Initialize vector store: load existing indices from disk, or create empty ones.
    pub async fn init_vector_store(
        &mut self,
        persist_dir: &Path,
        embedding_api_base: &str,
        embedding_api_key: &str,
        embedding_model: &str,
    ) -> Result<()> {
        fs::create_dir_all(persist_dir)?;
        self.embedding_api_base = embedding_api_base.to_string();
        self.embedding_api_key = embedding_api_key.to_string();
        self.embedding_model = embedding_model.to_string();
        self.persist_dir = persist_dir.to_path_buf();

        let mem_path = persist_dir.join("memories.hnsw");
        let skill_path = persist_dir.join("skills.hnsw");

        // Load or create memory index
        let mem_idx = if mem_path.exists() {
            let data = fs::read(&mem_path)?;
            bincode::deserialize(&data).context("Failed to deserialize memories index")?
        } else {
            Builder::default()
                .ef_search(200)
                .ef_construction(200)
                .build(Vec::new(), Vec::new())
        };
        *self.memories_idx.write() = Some(mem_idx);

        // Load or create skill index
        let skill_idx = if skill_path.exists() {
            let data = fs::read(&skill_path)?;
            bincode::deserialize(&data).context("Failed to deserialize skills index")?
        } else {
            Builder::default()
                .ef_search(100)
                .ef_construction(100)
                .build(Vec::new(), Vec::new())
        };
        *self.skills_idx.write() = Some(skill_idx);

        self.vector_store_initialized = true;
        Ok(())
    }

    pub fn is_vector_store_initialized(&self) -> bool {
        self.vector_store_initialized
    }

    /// Compatibility alias for callers that still reference the old name.
    pub async fn init_lancedb(
        &mut self,
        persist_dir: &Path,
        embedding_api_base: &str,
        embedding_api_key: &str,
        embedding_model: &str,
    ) -> Result<()> {
        self.init_vector_store(persist_dir, embedding_api_base, embedding_api_key, embedding_model)
            .await
    }

    /// Compatibility alias.
    pub fn is_lancedb_initialized(&self) -> bool {
        self.is_vector_store_initialized()
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if !self.vector_store_initialized {
            anyhow::bail!("Vector store not initialized — embedding API not configured");
        }
        let client = reqwest::Client::new();
        let url = format!("{}/embeddings", self.embedding_api_base);
        let resp = client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.embedding_api_key),
            )
            .json(&serde_json::json!({
                "model": self.embedding_model,
                "input": text,
            }))
            .send()
            .await?
            .error_for_status()?;
        let data: JsonValue = resp.json().await?;
        let embedding: Vec<f32> = data["data"][0]["embedding"]
            .as_array()
            .context("Missing embedding in API response")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        Ok(embedding)
    }

    /// Persist the memory index to disk.
    fn save_memories(&self) -> Result<()> {
        let idx = self.memories_idx.read();
        if let Some(ref idx) = *idx {
            let data = bincode::serialize(idx).context("Failed to serialize memories index")?;
            fs::write(self.persist_dir.join("memories.hnsw"), data)?;
        }
        Ok(())
    }

    /// Persist the skill index to disk.
    fn save_skills(&self) -> Result<()> {
        let idx = self.skills_idx.read();
        if let Some(ref idx) = *idx {
            let data = bincode::serialize(idx).context("Failed to serialize skills index")?;
            fs::write(self.persist_dir.join("skills.hnsw"), data)?;
        }
        Ok(())
    }

    /// Rebuild the memory index from current data, adding a new entry.
    fn rebuild_memories_with(
        &self,
        new_point: EmbeddingPoint,
        new_meta: MemoryMeta,
    ) -> Result<()> {
        let mut idx = self.memories_idx.write();
        let current = idx.take().unwrap();
        let mut points: Vec<EmbeddingPoint> = current
            .iter()
            .map(|(_, p)| p.clone())
            .collect();
        let mut metas: Vec<MemoryMeta> = current.values.clone();
        points.push(new_point);
        metas.push(new_meta);
        *idx = Some(Builder::default().ef_search(200).ef_construction(200).build(points, metas));
        Ok(())
    }

    /// Rebuild the memory index without a specific entry.
    fn rebuild_memories_without(&self, doc_id: &str) -> Result<()> {
        let mut idx = self.memories_idx.write();
        let current = idx.take().unwrap();
        let mut points: Vec<EmbeddingPoint> = Vec::new();
        let mut metas: Vec<MemoryMeta> = Vec::new();
        for (i, (_, point)) in current.iter().enumerate() {
            if current.values[i].doc_id != doc_id {
                points.push(point.clone());
                metas.push(current.values[i].clone());
            }
        }
        *idx = Some(Builder::default().ef_search(200).ef_construction(200).build(points, metas));
        Ok(())
    }

    /// Rebuild the skill index with an upserted entry.
    fn rebuild_skills_with(
        &self,
        new_point: EmbeddingPoint,
        new_meta: SkillMeta,
    ) -> Result<()> {
        let mut idx = self.skills_idx.write();
        let current = idx.take().unwrap();
        let mut points: Vec<EmbeddingPoint> = Vec::new();
        let mut metas: Vec<SkillMeta> = Vec::new();
        // Copy all entries except one with the same name (upsert)
        for (i, (_, point)) in current.iter().enumerate() {
            if current.values[i].name != new_meta.name {
                points.push(point.clone());
                metas.push(current.values[i].clone());
            }
        }
        points.push(new_point);
        metas.push(new_meta);
        *idx = Some(Builder::default().ef_search(100).ef_construction(100).build(points, metas));
        Ok(())
    }

    // -- Memory vector operations --

    pub async fn add_to_lancedb(
        &self,
        memory_id: &str,
        text: &str,
        metadata: &JsonValue,
    ) -> Result<String> {
        let embedding = self.get_embedding(text).await?;
        let doc_id = format!("mem-{memory_id}");
        let meta = MemoryMeta {
            doc_id: doc_id.clone(),
            text: text.to_string(),
            metadata_json: metadata.to_string(),
        };
        self.rebuild_memories_with(EmbeddingPoint::new(embedding), meta)?;
        self.save_memories()?;
        Ok(doc_id)
    }

    pub async fn query_lancedb(&self, query_text: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let embedding = self.get_embedding(query_text).await?;
        let query_point = EmbeddingPoint::new(embedding);

        let idx = self.memories_idx.read();
        let idx = idx.as_ref().context("Memory index not initialized")?;

        let mut search = Search::default();
        let results: Vec<_> = idx
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| {
                let meta = item.value;
                let m: JsonValue =
                    serde_json::from_str(&meta.metadata_json).unwrap_or_default();
                let memory_id = m
                    .get("memory_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&meta.doc_id)
                    .to_string();
                serde_json::json!({
                    "chroma_doc_id": meta.doc_id,
                    "memory_id": memory_id,
                    "text": meta.text,
                    "metadata": m,
                })
            })
            .collect();

        Ok(results)
    }

    pub async fn delete_from_lancedb(&self, doc_id: &str) -> Result<()> {
        self.rebuild_memories_without(doc_id)?;
        self.save_memories()?;
        Ok(())
    }

    // -- Skill vector operations --

    pub async fn add_skill_to_lancedb(
        &self,
        name: &str,
        text: &str,
        description: &str,
        source: &str,
    ) -> Result<()> {
        let embedding = self.get_embedding(text).await?;
        let meta = SkillMeta {
            name: name.to_string(),
            description: description.to_string(),
            source: source.to_string(),
        };
        self.rebuild_skills_with(EmbeddingPoint::new(embedding), meta)?;
        self.save_skills()?;
        Ok(())
    }

    pub async fn search_skills_lancedb(&self, query: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let embedding = self.get_embedding(query).await?;
        let query_point = EmbeddingPoint::new(embedding);

        let idx = self.skills_idx.read();
        let idx = idx.as_ref().context("Skill index not initialized")?;

        let mut search = Search::default();
        let results: Vec<_> = idx
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| {
                let meta = item.value;
                serde_json::json!({
                    "name": meta.name,
                    "description": meta.description,
                    "source": meta.source,
                })
            })
            .collect();

        Ok(results)
    }

    // -- SQLite CRUD (unchanged) --

    pub fn insert_memory(
        &self,
        summary: &str,
        conversation_at: &chrono::DateTime<chrono::Utc>,
        conversation_json: Option<&str>,
        chroma_doc_id: &str,
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
            "INSERT INTO memories (id, summary, conversation_at, conversation_json, chroma_doc_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                memory_id,
                summary,
                conversation_at.to_rfc3339(),
                conversation_json,
                chroma_doc_id
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
            "SELECT id, summary, conversation_at, created_at, conversation_json, chroma_doc_id
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
                    row.get::<_, Option<String>>(5)?,
                ))
            })
            .ok();

        match row {
            Some((id, summary, conv_at, created_at, conv_json, chroma_doc_id)) => Ok(Some(
                self.hydrate_memory(id, summary, conv_at, created_at, conv_json, chroma_doc_id)?,
            )),
            None => Ok(None),
        }
    }

    pub fn get_recent_memories(&self, limit: usize, offset: usize) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, conversation_at, created_at, conversation_json, chroma_doc_id
             FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ?1 OFFSET ?2",
        )?;
        let rows: Vec<(
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        )> = stmt
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
        for (id, summary, conv_at, created_at, conv_json, chroma_doc_id) in rows {
            memories.push(self.hydrate_memory(
                id,
                summary,
                conv_at,
                created_at,
                conv_json,
                chroma_doc_id,
            )?);
        }
        Ok(memories)
    }

    pub fn delete_memory(&self, memory_id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM memories WHERE id = ?1", params![memory_id])?;
        Ok(())
    }

    pub fn search_by_tag(&self, tag: &str) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.summary, m.conversation_at, m.created_at, m.conversation_json, m.chroma_doc_id
             FROM memories m
             JOIN memory_tags mt ON m.id = mt.memory_id
             JOIN tags t ON mt.tag_id = t.id
             WHERE t.name = ?1 ORDER BY m.created_at DESC, m.rowid DESC"
        )?;
        let rows: Vec<(
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        )> = stmt
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
        for (id, summary, conv_at, created_at, conv_json, chroma_doc_id) in rows {
            memories.push(self.hydrate_memory(
                id,
                summary,
                conv_at,
                created_at,
                conv_json,
                chroma_doc_id,
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

        let db_size = fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0);

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

    fn hydrate_memory(
        &self,
        id: String,
        summary: String,
        conversation_at: Option<String>,
        created_at: Option<String>,
        conversation_json: Option<String>,
        chroma_doc_id: Option<String>,
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
            chroma_doc_id,
            conversation_json,
        })
    }

    pub fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_schema_creates_tables() {
        let tmp = std::env::temp_dir().join("test_storage_rs.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();
        let count: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_insert_and_get_memory() {
        let tmp = std::env::temp_dir().join("test_memory_rs.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        let now = chrono::Utc::now();
        let mid = store.insert_memory(
            "Test summary", &now, None, "chroma-1",
            &["KP1".into(), "KP2".into()],
            &["rust".into(), "test".into()],
            &[serde_json::json!({"name": "main.rs", "type": "file", "description": "entry point"})],
            &["Decision 1".into()],
            None,
        ).unwrap();

        assert!(!mid.is_empty());
        let mem = store.get_memory(&mid).unwrap();
        assert!(mem.is_some());
        let mem = mem.unwrap();
        assert_eq!(mem.summary, "Test summary");
        assert_eq!(mem.tags.len(), 2);
        assert_eq!(mem.key_points.len(), 2);
        assert_eq!(mem.entities.len(), 1);
        assert_eq!(mem.decisions.len(), 1);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_get_recent_and_delete() {
        let tmp = std::env::temp_dir().join("test_recent_rs.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        let now = chrono::Utc::now();
        for i in 0..5 {
            store
                .insert_memory(
                    &format!("Summary {i}"),
                    &now,
                    None,
                    &format!("chroma-{i}"),
                    &[],
                    &[],
                    &[],
                    &[],
                    None,
                )
                .unwrap();
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
        let tmp = std::env::temp_dir().join("test_tags_rs.db");
        let _ = std::fs::remove_file(&tmp);
        let store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();

        let now = chrono::Utc::now();
        let mid = store.insert_memory(
            "Tag test", &now, None, "chroma-tag",
            &["key point".into()],
            &["rust".into(), "database".into()],
            &[serde_json::json!({"name": "lib.rs", "type": "file", "description": "library root"})],
            &["Use SQLite".into()],
            None,
        ).unwrap();

        let mem = store.get_memory(&mid).unwrap().unwrap();
        assert_eq!(mem.tags.len(), 2);
        assert!(mem.tags.contains(&"rust".to_string()));
        assert_eq!(mem.entities.len(), 1);
        assert_eq!(mem.entities[0].name, "lib.rs");

        let all_tags = store.get_all_tags().unwrap();
        assert!(all_tags.len() >= 2);

        let by_tag = store.search_by_tag("rust").unwrap();
        assert!(!by_tag.is_empty());

        let _ = std::fs::remove_file(&tmp);
    }
}