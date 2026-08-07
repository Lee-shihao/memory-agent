//! Vector storage: embedding API client + sqlite-vec (`vec0`) virtual tables.
//!
//! Vectors live in the same SQLite database as the relational metadata, linked
//! by `memories.vec_rowid` / `skill_vectors.vec_rowid`. Inserts and deletes are
//! plain SQL — no index rebuild or sidecar files.

use crate::storage::MemoryStore;
use anyhow::{Context, Result};
use rusqlite::{params, OptionalExtension};
use serde_json::Value as JsonValue;
use zerocopy::IntoBytes;

const MEMORIES_VEC: &str = "memories_vec";
const SKILLS_VEC: &str = "skills_vec";

impl MemoryStore {
    /// Store the embedding config. Vectors persist in the SQLite file itself,
    /// so there is nothing to load; the `vec0` tables are created lazily on the
    /// first insert once the embedding dimension is known.
    pub async fn init_vector_store(
        &mut self,
        embedding_api_base: &str,
        embedding_api_key: &str,
        embedding_model: &str,
    ) -> Result<()> {
        self.embedding_api_base = embedding_api_base.to_string();
        self.embedding_api_key = embedding_api_key.to_string();
        self.embedding_model = embedding_model.to_string();
        // Build the HTTP client once and reuse it. Local endpoints (Ollama, LM
        // Studio, test mocks) must not be routed through an HTTP proxy, which
        // would fail for loopback addresses.
        let mut builder = reqwest::Client::builder();
        let url = format!("{embedding_api_base}/embeddings");
        if let Ok(parsed) = reqwest::Url::parse(&url) {
            if let Some(host) = parsed.host_str() {
                if matches!(host, "localhost" | "127.0.0.1" | "::1") {
                    builder = builder.no_proxy();
                }
            }
        }
        self.embedding_http = Some(builder.build()?);
        self.vector_store_initialized = true;
        Ok(())
    }

    pub fn is_vector_store_initialized(&self) -> bool {
        self.vector_store_initialized
    }

    /// Fetch an embedding from the configured OpenAI-compatible endpoint and
    /// L2-normalize it so cosine distance (`1 - dot`) is well-defined.
    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if !self.vector_store_initialized {
            anyhow::bail!("Vector store not initialized — embedding API not configured");
        }
        let url = format!("{}/embeddings", self.embedding_api_base);
        let client = self
            .embedding_http
            .as_ref()
            .context("Embedding HTTP client not initialized")?;
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
        let mut embedding: Vec<f32> = data["data"][0]["embedding"]
            .as_array()
            .context("Missing embedding in API response")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        Ok(embedding)
    }

    /// Create the `vec0` tables with the given dimension (needed before any
    /// insert/query, since `float[N]` fixes N at creation time).
    fn ensure_vec_tables(&self, dim: usize) -> Result<()> {
        self.conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {MEMORIES_VEC} \
                USING vec0(embedding float[{dim}] distance_metric=cosine);
             CREATE VIRTUAL TABLE IF NOT EXISTS {SKILLS_VEC} \
                USING vec0(embedding float[{dim}] distance_metric=cosine);"
        ))?;
        Ok(())
    }

    fn vec_table_exists(&self, name: &str) -> bool {
        self.conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                params![name],
                |_| Ok(()),
            )
            .is_ok()
    }

    // -- Memory vectors ------------------------------------------------------

    /// Embed `text` and store the vector, linking it to the memory row via
    /// `memories.vec_rowid`.
    pub async fn add_memory_vector(&self, memory_id: &str, text: &str) -> Result<()> {
        let embedding = self.get_embedding(text).await?;
        if !self.vec_table_exists(MEMORIES_VEC) {
            self.ensure_vec_tables(embedding.len())?;
        }
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| -> Result<()> {
            let existing: Option<i64> = self
                .conn
                .query_row(
                    "SELECT vec_rowid FROM memories WHERE id = ?1",
                    params![memory_id],
                    |row| row.get(0),
                )
                .optional()?;
            if existing.is_none() {
                anyhow::bail!("Memory {memory_id} not found");
            }
            // Re-embedding is an upsert: drop the old vector row so it can't
            // become an orphan.
            if let Some(old_rowid) = existing {
                self.conn.execute(
                    &format!("DELETE FROM {MEMORIES_VEC} WHERE rowid = ?1"),
                    params![old_rowid],
                )?;
            }
            self.conn.execute(
                &format!("INSERT INTO {MEMORIES_VEC}(embedding) VALUES (?1)"),
                params![embedding.as_bytes()],
            )?;
            let vec_rowid = self.conn.last_insert_rowid();
            self.conn.execute(
                "UPDATE memories SET vec_rowid = ?1 WHERE id = ?2",
                params![vec_rowid, memory_id],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => self.conn.execute_batch("COMMIT")?,
            Err(_) => {
                let _ = self.conn.execute_batch("ROLLBACK");
            }
        }
        result
    }

    /// KNN search over memories. Returns the top-k `{memory_id, text, distance}`
    /// records, with `memory_id` ready to use for SQLite hydration.
    pub async fn search_memory_vectors(&self, query: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let embedding = self.get_embedding(query).await?;
        if !self.vec_table_exists(MEMORIES_VEC) {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(&format!(
            "SELECT m.id, m.summary, v.distance
             FROM {MEMORIES_VEC} v
             JOIN memories m ON m.vec_rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2"
        ))?;
        let rows = stmt.query_map(
            params![embedding.as_bytes(), top_k as i64],
            |row| {
                Ok(serde_json::json!({
                    "memory_id": row.get::<_, String>(0)?,
                    "text": row.get::<_, String>(1)?,
                    "distance": row.get::<_, f64>(2)?,
                }))
            },
        )?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Drop the memory's vector row. Called by `delete_memory`.
    pub fn remove_memory_vector(&self, memory_id: &str) -> Result<()> {
        let vec_rowid: Option<i64> = self
            .conn
            .query_row(
                "SELECT vec_rowid FROM memories WHERE id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .ok();
        if let Some(rowid) = vec_rowid {
            self.conn.execute(
                &format!("DELETE FROM {MEMORIES_VEC} WHERE rowid = ?1"),
                params![rowid],
            )?;
        }
        Ok(())
    }

    /// Drop a skill's vector and metadata rows. Safe to call when the vector
    /// tables don't exist yet (e.g. a fresh DB): the metadata row is deleted
    /// unconditionally, the vec row only if the table is present.
    pub fn remove_skill_vector(&self, name: &str) -> Result<()> {
        if self.vec_table_exists(SKILLS_VEC) {
            self.conn.execute(
                &format!(
                    "DELETE FROM {SKILLS_VEC} WHERE rowid IN \
                     (SELECT vec_rowid FROM skill_vectors WHERE name = ?1)"
                ),
                params![name],
            )?;
        }
        self.conn
            .execute("DELETE FROM skill_vectors WHERE name = ?1", params![name])?;
        Ok(())
    }

    // -- Skill vectors -------------------------------------------------------

    /// Embed a skill (upsert by name) so repeated indexing is idempotent.
    pub async fn add_skill_vector(
        &self,
        name: &str,
        text: &str,
        description: &str,
        source: &str,
    ) -> Result<()> {
        let embedding = self.get_embedding(text).await?;
        if !self.vec_table_exists(SKILLS_VEC) {
            self.ensure_vec_tables(embedding.len())?;
        }
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| -> Result<()> {
            self.remove_skill_vector(name)?;
            self.conn.execute(
                &format!("INSERT INTO {SKILLS_VEC}(embedding) VALUES (?1)"),
                params![embedding.as_bytes()],
            )?;
            let vec_rowid = self.conn.last_insert_rowid();
            self.conn.execute(
                "INSERT INTO skill_vectors (vec_rowid, name, description, source) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![vec_rowid, name, description, source],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => self.conn.execute_batch("COMMIT")?,
            Err(_) => {
                let _ = self.conn.execute_batch("ROLLBACK");
            }
        }
        result
    }

    /// KNN search over skills. Returns the top-k `{name, description, source}`.
    pub async fn search_skill_vectors(&self, query: &str, top_k: usize) -> Result<Vec<JsonValue>> {
        let embedding = self.get_embedding(query).await?;
        if !self.vec_table_exists(SKILLS_VEC) {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(&format!(
            "SELECT s.name, s.description, s.source
             FROM {SKILLS_VEC} v
             JOIN skill_vectors s ON s.vec_rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2"
        ))?;
        let rows = stmt.query_map(
            params![embedding.as_bytes(), top_k as i64],
            |row| {
                Ok(serde_json::json!({
                    "name": row.get::<_, String>(0)?,
                    "description": row.get::<_, String>(1)?,
                    "source": row.get::<_, Option<String>>(2)?,
                }))
            },
        )?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// A fixed, already-normalized embedding so insert and query vectors match.
    fn embedding_response() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0] }]
        }))
    }

    async fn setup_store(
        server: &MockServer,
    ) -> (MemoryStore, std::path::PathBuf) {
        let tmp = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        let mut store = MemoryStore::new(&tmp).unwrap();
        store.init_schema().unwrap();
        store
            .init_vector_store(&server.uri(), "key", "model")
            .await
            .unwrap();
        (store, tmp)
    }

    #[tokio::test]
    async fn test_memory_vector_roundtrip() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(embedding_response())
            .mount(&server)
            .await;

        let (store, db_path) = setup_store(&server).await;

        let mid = store
            .insert_memory(
                "vector memory",
                &chrono::Utc::now(),
                None,
                &[],
                &[],
                &[],
                &[],
                None,
            )
            .unwrap();
        store.add_memory_vector(&mid, "some text to embed").await.unwrap();

        // The inserted vector is now reachable via semantic search.
        let results = store.search_memory_vectors("query", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["memory_id"], mid);

        // delete_memory removes both the vector and the metadata.
        store.delete_memory(&mid).unwrap();
        assert!(store.get_memory(&mid).unwrap().is_none());
        let results = store.search_memory_vectors("query", 5).await.unwrap();
        assert!(results.is_empty());

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn test_skill_vector_upsert() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(embedding_response())
            .mount(&server)
            .await;

        let (store, db_path) = setup_store(&server).await;

        store
            .add_skill_vector("my-skill", "name: desc", "desc v1", "src")
            .await
            .unwrap();
        // Re-indexing the same skill must replace, not duplicate.
        store
            .add_skill_vector("my-skill", "name: desc", "desc v2", "src")
            .await
            .unwrap();

        let results = store.search_skill_vectors("query", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], "my-skill");
        assert_eq!(results[0]["description"], "desc v2");

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn test_remove_skill_vector_cleans_rows() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(embedding_response())
            .mount(&server)
            .await;

        let (store, db_path) = setup_store(&server).await;

        store
            .add_skill_vector("my-skill", "name: desc", "desc", "src")
            .await
            .unwrap();
        store.remove_skill_vector("my-skill").unwrap();

        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM skill_vectors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let results = store.search_skill_vectors("query", 5).await.unwrap();
        assert!(results.is_empty());

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn test_add_memory_vector_rejects_missing_memory() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(embedding_response())
            .mount(&server)
            .await;

        let (store, db_path) = setup_store(&server).await;

        assert!(store.add_memory_vector("no-such-memory", "text").await.is_err());

        // No vector row should have been left behind.
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM memories_vec", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn test_memory_vector_embed_twice_is_upsert() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(embedding_response())
            .mount(&server)
            .await;

        let (store, db_path) = setup_store(&server).await;

        let mid = store
            .insert_memory(
                "vector memory",
                &chrono::Utc::now(),
                None,
                &[],
                &[],
                &[],
                &[],
                None,
            )
            .unwrap();
        store.add_memory_vector(&mid, "text").await.unwrap();
        store.add_memory_vector(&mid, "text").await.unwrap();

        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM memories_vec", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let _ = std::fs::remove_file(&db_path);
    }
}
