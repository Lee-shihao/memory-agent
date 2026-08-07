//! Storage layer: SQLite relational tables for metadata + sqlite-vec `vec0`
//! virtual tables for embedding vectors, all in a single database file.
//!
//! Split by concern:
//! - `schema.rs`   — schema DDL
//! - `relation.rs` — relational CRUD (`impl MemoryStore`)
//! - `vector.rs`   — embedding + vector operations (`impl MemoryStore`)

pub mod relation;
pub mod schema;
pub mod vector;

use anyhow::Result;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use schema::SCHEMA_SQL;

/// Register the sqlite-vec extension before any connection is opened.
/// `sqlite3_auto_extension` makes the extension available to every future
/// connection in the process, so this only needs to run once.
static VEC_EXT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_vec_extension() {
    VEC_EXT_INIT.get_or_init(|| {
        // sqlite-vec declares the symbol as `fn()`, but SQLite's extension-init
        // ABI takes (db, err, api). Transmute to the real signature; the target
        // type is spelled out so clippy's annotation lint is satisfied.
        let init: unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *const std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int =
            unsafe { std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()) };
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(init));
        }
    });
}

pub struct MemoryStore {
    conn: Connection,
    db_path: PathBuf,
    embedding_api_base: String,
    embedding_api_key: String,
    embedding_model: String,
    embedding_http: Option<reqwest::Client>,
    vector_store_initialized: bool,
}

impl MemoryStore {
    pub fn new(db_path: &Path) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON")?;
        // Multiple connections may be open at once (main store + per-tool-call
        // stores); give writers a window instead of failing immediately with
        // SQLITE_BUSY.
        conn.execute_batch("PRAGMA busy_timeout = 5000")?;
        Ok(MemoryStore {
            conn,
            db_path: db_path.to_path_buf(),
            embedding_api_base: String::new(),
            embedding_api_key: String::new(),
            embedding_model: String::new(),
            embedding_http: None,
            vector_store_initialized: false,
        })
    }

    pub fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        // Migrate pre-sqlite-vec databases: add the vec_rowid column that
        // replaces the old chroma_doc_id. Fails silently when it exists.
        let _ = self
            .conn
            .execute_batch("ALTER TABLE memories ADD COLUMN vec_rowid INTEGER;");
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_vec_rowid ON memories(vec_rowid);",
        )?;
        Ok(())
    }

    /// Vectors persist in the SQLite file (vec0 virtual tables), so there is
    /// nothing to flush. Kept for API stability.
    pub fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
