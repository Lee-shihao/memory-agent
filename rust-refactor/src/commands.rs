use crate::prompts::Memory;
/// Slash command handlers for /memory operations.
use crate::storage::MemoryStore;

pub fn handle_slash_command(
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
                cmd_search(store, args)
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
                cmd_delete(store, args)
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
                let summary = if mem.summary.len() > 80 {
                    &mem.summary[..80]
                } else {
                    &mem.summary
                };
                lines.push(format!("  [{}] {}", mem.id, summary));
            }
            lines.join("\n")
        }
        _ => "No memories in database.".to_string(),
    }
}

fn cmd_search(store: &MemoryStore, query: &str) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        match store.query_lancedb(query, 5).await {
            Ok(results) if !results.is_empty() => {
                let mut lines = vec![format!("Search results for '{}':", query)];
                for r in results {
                    let mid = r.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
                    let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let text = if text.len() > 100 { &text[..100] } else { text };
                    lines.push(format!("  [{mid}] {text}"));
                }
                lines.join("\n")
            }
            _ => format!("No memories found matching: {query}"),
        }
    })
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

fn cmd_delete(store: &MemoryStore, memory_id: &str) -> String {
    match store.get_memory(memory_id) {
        Ok(Some(mem)) => {
            if let Some(ref doc_id) = mem.chroma_doc_id {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    store.delete_from_lancedb(doc_id).await.ok();
                });
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
