use crate::prompts::{Entity, Memory};
use anyhow::{Context, Result};
use arrow::array::{Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatchIterator;
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use rusqlite::{params, Connection};
use serde_json::Value as JsonValue;
use std::fs;
/// Storage layer: SQLite for metadata, LanceDB for vectors.
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

// Default embedding dimension for BAAI/bge-m3
const EMBEDDING_DIM: i32 = 1024;

pub struct MemoryStore {
    conn: Connection,
    db_path: PathBuf,
    // LanceDB state
    lancedb_conn: Option<Arc<lancedb::Connection>>,
    embedding_api_base: String,
    embedding_api_key: String,
    embedding_model: String,
    lancedb_initialized: bool,
}

impl MemoryStore {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON")?;
        Ok(MemoryStore {
            conn,
            db_path: db_path.to_path_buf(),
            lancedb_conn: None,
            embedding_api_base: String::new(),
            embedding_api_key: String::new(),
            embedding_model: String::new(),
            lancedb_initialized: false,
        })
    }

    pub fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    pub async fn init_lancedb(
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

        let db = connect(persist_dir.to_str().unwrap()).execute().await?;
        let db = Arc::new(db);

        // Ensure memories table exists
        let existing_tables = db.table_names().execute().await?;
        if !existing_tables.iter().any(|t| t == "memories") {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        EMBEDDING_DIM,
                    ),
                    true,
                ),
                Field::new("text", DataType::Utf8, true),
                Field::new("metadata_json", DataType::Utf8, true),
            ]));
            db.create_empty_table("memories", schema).execute().await?;
        }

        // Ensure skills table exists
        if !existing_tables.iter().any(|t| t == "skills") {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        EMBEDDING_DIM,
                    ),
                    true,
                ),
                Field::new("description", DataType::Utf8, true),
                Field::new("source", DataType::Utf8, true),
            ]));
            db.create_empty_table("skills", schema).execute().await?;
        }

        self.lancedb_conn = Some(db);
        self.lancedb_initialized = true;
        Ok(())
    }

    pub fn is_lancedb_initialized(&self) -> bool {
        self.lancedb_initialized
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if !self.lancedb_initialized {
            anyhow::bail!("LanceDB not initialized — embedding API not configured");
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

    pub async fn add_to_lancedb(
        &self,
        memory_id: &str,
        text: &str,
        metadata: &JsonValue,
    ) -> Result<String> {
        let db = self
            .lancedb_conn
            .as_ref()
            .context("LanceDB not initialized")?;
        let embedding = self.get_embedding(text).await?;
        let doc_id = format!("mem-{memory_id}");

        let table = db.open_table("memories").execute().await?;

        // Build a simple record batch with Arrow arrays
        let id_array = StringArray::from(vec![doc_id.clone()]);
        let text_array = StringArray::from(vec![text.to_string()]);
        let metadata_array = StringArray::from(vec![metadata.to_string()]);

        // Build FixedSizeList for vector
        let flat: Vec<f32> = embedding.clone();
        let vector_values = Float32Array::from(flat);
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));

        use arrow::array::FixedSizeListArray;
        let vector_array =
            FixedSizeListArray::new(vector_field, EMBEDDING_DIM, Arc::new(vector_values), None);

        let schema = table.schema().await?;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(text_array),
                Arc::new(metadata_array),
            ],
        )?;

        let schema_ref = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema_ref);
        table.add(Box::new(reader)).execute().await?;

        Ok(doc_id)
    }

    pub async fn query_lancedb(&self, query_text: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let db = self
            .lancedb_conn
            .as_ref()
            .context("LanceDB not initialized")?;
        let embedding = self.get_embedding(query_text).await?;
        let table = db.open_table("memories").execute().await?;

        let results = table
            .query()
            .nearest_to(embedding.clone())?
            .limit(top_k)
            .execute()
            .await?;

        // results is a RecordBatchStream — collect into batches
        let mut memories = Vec::new();
        let mut stream = results;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let schema = batch.schema();
            let id_idx = schema.index_of("id").unwrap_or(0);
            let text_idx = schema.index_of("text").unwrap_or(2);
            let meta_idx = schema.index_of("metadata_json").unwrap_or(3);

            for row in 0..batch.num_rows() {
                let id = batch
                    .column(id_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();
                let text = batch
                    .column(text_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();
                let metadata_str = batch
                    .column(meta_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();
                let metadata: JsonValue = serde_json::from_str(&metadata_str).unwrap_or_default();
                let memory_id = metadata
                    .get("memory_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&id)
                    .to_string();

                memories.push(serde_json::json!({
                    "chroma_doc_id": id,
                    "memory_id": memory_id,
                    "text": text,
                    "metadata": metadata,
                }));
            }
        }
        Ok(memories)
    }

    pub async fn delete_from_lancedb(&self, doc_id: &str) -> Result<()> {
        let db = self
            .lancedb_conn
            .as_ref()
            .context("LanceDB not initialized")?;
        let table = db.open_table("memories").execute().await?;
        table.delete(format!("id = '{doc_id}'").as_str()).await?;
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
        let db = self
            .lancedb_conn
            .as_ref()
            .context("LanceDB not initialized")?;
        let embedding = self.get_embedding(text).await?;

        // Delete existing entry for this skill if it exists
        let table = db.open_table("skills").execute().await?;
        table.delete(format!("name = '{name}'").as_str()).await?;

        let name_array = StringArray::from(vec![name.to_string()]);
        let desc_array = StringArray::from(vec![description.to_string()]);
        let source_array = StringArray::from(vec![source.to_string()]);
        let flat: Vec<f32> = embedding;
        let vector_values = Float32Array::from(flat);
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = arrow::array::FixedSizeListArray::new(
            vector_field,
            EMBEDDING_DIM,
            Arc::new(vector_values),
            None,
        );

        let schema = table.schema().await?;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(name_array),
                Arc::new(vector_array),
                Arc::new(desc_array),
                Arc::new(source_array),
            ],
        )?;
        let schema_ref = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema_ref);
        table.add(Box::new(reader)).execute().await?;
        Ok(())
    }

    pub async fn search_skills_lancedb(&self, query: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let db = self
            .lancedb_conn
            .as_ref()
            .context("LanceDB not initialized")?;
        let embedding = self.get_embedding(query).await?;
        let table = db.open_table("skills").execute().await?;

        let results = table
            .query()
            .nearest_to(embedding.clone())?
            .limit(top_k)
            .execute()
            .await?;

        let mut skills = Vec::new();
        let mut stream = results;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let schema = batch.schema();
            let name_idx = schema.index_of("name").unwrap_or(0);
            let desc_idx = schema.index_of("description").unwrap_or(2);
            let source_idx = schema.index_of("source").unwrap_or(3);

            for row in 0..batch.num_rows() {
                let name = batch
                    .column(name_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();
                let description = batch
                    .column(desc_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();
                let source = batch
                    .column(source_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(row).to_string())
                    .unwrap_or_default();

                skills.push(serde_json::json!({
                    "name": name,
                    "description": description,
                    "source": source,
                }));
            }
        }
        Ok(skills)
    }

    // -- SQLite CRUD --

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
        // Connection is dropped when store is dropped
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
