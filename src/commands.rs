use crate::prompts::Memory;
/// Slash command handlers for /memory operations.
use crate::storage::MemoryStore;

/// Truncate a string to at most `max_chars` characters on a valid UTF-8 boundary.
fn truncate_str(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

pub async fn handle_slash_command(
    message: &str,
    store: &MemoryStore,
    injected_memories: &[Memory],
) -> Option<String> {
    let stripped = message.trim();
    if !stripped.starts_with("/memory") {
        return None;
    }

    let parts: Vec<&str> = stripped.splitn(3, |c: char| c.is_whitespace()).collect();
    let subcommand = parts.get(1).copied().unwrap_or("");
    let args = parts.get(2).copied().unwrap_or("");

    Some(match subcommand {
        "" => cmd_show_injected(injected_memories),
        "recent" => {
            let n: usize = args.parse().unwrap_or(10);
            cmd_recent(store, n)
        }
        "search" => {
            if args.is_empty() {
                "Usage: /memory search <query>".to_string()
            } else {
                cmd_search(store, args).await
            }
        }
        "show" => {
            if args.is_empty() {
                "Usage: /memory show <id>".to_string()
            } else {
                cmd_show(store, args)
            }
        }
        "delete" => {
            if args.is_empty() {
                "Usage: /memory delete <id>".to_string()
            } else {
                cmd_delete(store, args).await
            }
        }
        "status" => cmd_status(store),
        _ => cmd_usage(),
    })
}

fn cmd_show_injected(injected: &[Memory]) -> String {
    if injected.is_empty() {
        return "No memories were injected for this conversation.".to_string();
    }
    let mut lines = vec!["Memories injected for this conversation:".to_string()];
    for (i, mem) in injected.iter().enumerate() {
        lines.push(format!("  {}. [{}] {}", i + 1, mem.id, mem.summary));
    }
    lines.join("\n")
}

fn cmd_recent(store: &MemoryStore, n: usize) -> String {
    match store.get_recent_memories(n, 0) {
        Ok(memories) if !memories.is_empty() => {
            let mut lines = vec![format!("Recent {} memories:", memories.len())];
            for mem in &memories {
                let summary = truncate_str(&mem.summary, 80);
                lines.push(format!("  [{}] {}", mem.id, summary));
            }
            lines.join("\n")
        }
        _ => "No memories in database.".to_string(),
    }
}

async fn cmd_search(store: &MemoryStore, query: &str) -> String {
    match store.query_lancedb(query, 5).await {
        Ok(results) if !results.is_empty() => {
            let mut lines = vec![format!("Search results for '{}':", query)];
            for r in results {
                let mid = r.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
                let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let text = truncate_str(text, 100);
                lines.push(format!("  [{mid}] {text}"));
            }
            lines.join("\n")
        }
        _ => format!("No memories found matching: {query}"),
    }
}

fn cmd_show(store: &MemoryStore, memory_id: &str) -> String {
    match store.get_memory(memory_id) {
        Ok(Some(mem)) => {
            let mut lines = vec![
                format!("=== Memory: {memory_id} ==="),
                format!("Summary: {}", mem.summary),
                format!(
                    "Tags: {}",
                    if mem.tags.is_empty() {
                        "(none)".to_string()
                    } else {
                        mem.tags.join(", ")
                    }
                ),
                format!(
                    "Conversation at: {}",
                    mem.conversation_at.as_deref().unwrap_or("unknown")
                ),
                format!(
                    "Created at: {}",
                    mem.created_at.as_deref().unwrap_or("unknown")
                ),
                String::new(),
                "Key Points:".to_string(),
            ];
            for kp in &mem.key_points {
                lines.push(format!("  • {kp}"));
            }
            lines.push(String::new());
            lines.push("Entities:".to_string());
            if mem.entities.is_empty() {
                lines.push("  (none)".to_string());
            } else {
                for ent in &mem.entities {
                    lines.push(format!(
                        "  • {} ({}): {}",
                        ent.name,
                        ent.entity_type,
                        ent.description.as_deref().unwrap_or("")
                    ));
                }
            }
            lines.push(String::new());
            lines.push("Decisions:".to_string());
            for dec in &mem.decisions {
                lines.push(format!("  • {dec}"));
            }
            if mem.decisions.is_empty() {
                lines.push("  (none)".to_string());
            }
            lines.join("\n")
        }
        _ => format!("Memory not found: {memory_id}"),
    }
}

async fn cmd_delete(store: &MemoryStore, memory_id: &str) -> String {
    match store.get_memory(memory_id) {
        Ok(Some(mem)) => {
            if let Some(ref doc_id) = mem.chroma_doc_id {
                store.delete_from_lancedb(doc_id).await.ok();
            }
            store.delete_memory(memory_id).ok();
            format!("Memory deleted: {memory_id}")
        }
        _ => format!("Memory not found: {memory_id}"),
    }
}

fn cmd_status(store: &MemoryStore) -> String {
    match store.get_status() {
        Ok(status) => {
            let mut lines = vec![
                "=== Memory Database Status ===".to_string(),
                format!("Total memories: {}", status["total_memories"]),
                format!("Total tags: {}", status["total_tags"]),
                format!(
                    "Last insert: {}",
                    status["last_insert_at"].as_str().unwrap_or("never")
                ),
                format!("DB path: {}", status["db_path"].as_str().unwrap_or("")),
                format!("DB size: {} bytes", status["db_size_bytes"]),
            ];
            if let Ok(tags) = store.get_all_tags() {
                if !tags.is_empty() {
                    lines.push(format!("\nTags: {}", tags.join(", ")));
                }
            }
            lines.join("\n")
        }
        Err(e) => format!("Error getting status: {e}"),
    }
}

fn cmd_usage() -> String {
    r#"Usage:
  /memory                  Show injected memories
  /memory recent [N]       Show recent N memories (default 10)
  /memory search <query>   Semantic search
  /memory show <id>        Show memory details
  /memory delete <id>      Delete a memory
  /memory status           Database statistics"#
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
        assert_eq!(truncate_str("short", 80), "short"); // shorter than limit
    }

    #[test]
    fn test_truncate_cjk() {
        // '可' is 3 bytes in UTF-8
        let s = "用户询问模型身份，助手回答为DeepSeek";
        let truncated = truncate_str(s, 5);
        assert!(truncated.len() <= s.len());
        assert!(!truncated.is_empty());
        // must be valid UTF-8 and on a char boundary
        assert_eq!(truncated, "用户询问模"); // first 5 chars
    }

    #[test]
    fn test_truncate_cjk_single_char_boundary() {
        // 3-byte char: 可 (E5 8F AF)
        let s = "助手回答为DeepSeek开发的AI助手，运行在可访问工具的环境中";
        let truncated = truncate_str(s, 22);
        // "可" starts at byte 79 and spans 79..82, byte 80 is inside it
        // truncate_str should stop before "可"
        assert!(!truncated.contains('可'), "should not include partial char: '{truncated}'");
    }

    #[test]
    fn test_handle_slash_command_non_memory() {
        let path = std::env::temp_dir().join("test_cmd_nomem.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(handle_slash_command("/help", &store, &[]));
        assert!(result.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_slash_command_help_text_is_async() {
        // Just verify the functions compile as async
        let path = std::env::temp_dir().join("test_cmd2.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(handle_slash_command("/memory recent 1", &store, &[]));
        assert!(result.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cmd_recent_empty() {
        let path = std::env::temp_dir().join("test_cmd3.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();
        assert_eq!(cmd_recent(&store, 10), "No memories in database.");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cmd_show_not_found() {
        let path = std::env::temp_dir().join("test_cmd4.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();
        assert_eq!(cmd_show(&store, "nonexistent"), "Memory not found: nonexistent");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cmd_status_empty() {
        let path = std::env::temp_dir().join("test_cmd5.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();
        let status = cmd_status(&store);
        assert!(status.contains("Total memories: 0"), "status: {status}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cmd_show_injected() {
        let empty: &[Memory] = &[];
        let result = cmd_show_injected(empty);
        assert!(result.contains("No memories"), "result: {result}");

        let mem = Memory {
            id: "test-id".into(),
            summary: "test summary".into(),
            conversation_at: None,
            created_at: None,
            key_points: vec![],
            tags: vec![],
            entities: vec![],
            decisions: vec![],
            chroma_doc_id: None,
            conversation_json: None,
        };
        let result = cmd_show_injected(&[mem]);
        assert!(result.contains("test-id"), "result: {result}");
    }

    #[test]
    fn test_cmd_search_runs_without_nested_runtime_panic() {
        // This test verifies the fix: cmd_search must not create a nested runtime
        let path = std::env::temp_dir().join("test_cmd_search.db");
        let _ = std::fs::remove_file(&path);
        let mut store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();

        // Create a runtime and call the async command within it
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            // init vector store so query_lancedb doesn't bail early
            let tmp = std::env::temp_dir().join("test_cmd_search_hnsw");
            let _ = std::fs::create_dir_all(&tmp);
            store
                .init_vector_store(&tmp, "https://localhost", "key", "model")
                .await
                .ok();
            cmd_search(&store, "test").await
        });
        // Should not panic with "Cannot start a runtime from within a runtime"
        // Will fail on embedding API call, but that's OK
        assert!(
            result.contains("No memories found") || result.contains("Search results"),
            "result: {result}"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("test_cmd_search_hnsw"));
    }

    #[test]
    fn test_cmd_delete_runs_without_nested_runtime_panic() {
        let path = std::env::temp_dir().join("test_cmd_delete.db");
        let _ = std::fs::remove_file(&path);
        let mut store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            let tmp = std::env::temp_dir().join("test_cmd_delete_hnsw");
            let _ = std::fs::create_dir_all(&tmp);
            store
                .init_vector_store(&tmp, "https://localhost", "key", "model")
                .await
                .ok();
            cmd_delete(&store, "nonexistent").await
        });
        assert!(result.contains("Memory not found"), "result: {result}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("test_cmd_delete_hnsw"));
    }

    #[test]
    fn test_cmd_usage() {
        let u = cmd_usage();
        assert!(u.contains("/memory"), "usage: {u}");
    }

    #[test]
    fn test_handle_slash_command_parsing() {
        let path = std::env::temp_dir().join("test_cmd_parse.db");
        let _ = std::fs::remove_file(&path);
        let store = MemoryStore::new(&path).unwrap();
        store.init_schema().unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();

        // Test status (sync subcommand)
        let r = rt.block_on(handle_slash_command("/memory status", &store, &[]));
        assert!(r.unwrap().contains("Total memories"));

        // Test show not found (sync subcommand)
        let r = rt.block_on(handle_slash_command("/memory show abc", &store, &[]));
        assert!(r.unwrap().contains("not found"));

        // Test recent empty (sync subcommand)
        let r = rt.block_on(handle_slash_command("/memory recent 5", &store, &[]));
        assert!(r.unwrap().contains("No memories"));

        // Test search missing args
        let r = rt.block_on(handle_slash_command("/memory search", &store, &[]));
        assert!(r.unwrap().contains("Usage"));

        // Test unknown subcommand
        let r = rt.block_on(handle_slash_command("/memory unknown", &store, &[]));
        assert!(r.unwrap().contains("Usage"));

        let _ = std::fs::remove_file(&path);
    }
}
