use super::{workspace_root, get_returned_memory_ids, add_returned_memory_id,
             get_returned_skill_names, add_returned_skill_name, JsonValue, HashMap};

// ── ask_user ──────────────────────────────────────────────────────────────────

pub fn ask_user_definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user for input when you need clarification or choices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question"},
                    "header": {"type": "string", "description": "Short label (max 12 chars)"},
                    "options": {
                        "type": "array", "minItems": 2, "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["label", "description"],
                        },
                    },
                    "multi_select": {"type": "boolean"},
                },
                "required": ["question", "header"],
            },
        },
    })
}

pub fn ask_user(args: &HashMap<String, JsonValue>) -> String {
    let options = args.get("options").and_then(|v| v.as_array());
    let multi_select = args.get("multi_select").and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(opts) = options {
        if !opts.is_empty() {
            let selected = if multi_select {
                opts.iter().filter_map(|o| o["label"].as_str()).collect::<Vec<_>>().join(", ")
            } else {
                opts[0]["label"].as_str().unwrap_or("").to_string()
            };
            return format!("[Selected] {selected}");
        }
    }
    String::new()
}

// ── load_skill ────────────────────────────────────────────────────────────────

pub fn load_skill(args: &HashMap<String, JsonValue>) -> String {
    let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("");
    crate::skills::load_skill_content(name)
}

pub fn search_memory_definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search past conversation memories for relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Results count (default: 5)"},
                },
                "required": ["query"],
            },
        },
    })
}

// ── search_memory ─────────────────────────────────────────────────────────────

pub fn search_memory(args: &HashMap<String, JsonValue>) -> String {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let ws = workspace_root();
        let config = match crate::config::load_config(&ws) {
            Ok(c) => c, Err(e) => return format!("Config error: {e}"),
        };
        let db_path = config.memory_dir.join("memories.db");
        let mut store = match crate::storage::MemoryStore::new(&db_path) {
            Ok(s) => s, Err(e) => return format!("Store error: {e}"),
        };
        if let Err(e) = store.init_schema() { return format!("Schema error: {e}"); }
        if !store.is_lancedb_initialized() {
            if let Err(e) = store.init_lancedb(
                &config.memory_dir, &config.embedding_api_base,
                &config.embedding_api_key, &config.embedding_model,
            ).await { return format!("LanceDB error: {e}"); }
        }
        let results = match store.query_lancedb(query, top_k).await {
            Ok(r) => r, Err(e) => return format!("Search failed: {e}"),
        };
        if results.is_empty() { return format!("No memories found matching: {query}"); }

        let returned = get_returned_memory_ids();
        let new: Vec<_> = results.iter().filter(|r| {
            let mid = r.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            mid == "?" || !returned.contains(mid)
        }).collect();
        if new.is_empty() {
            return "No new memories found. Try a different query.".to_string();
        }
        for r in &new {
            if let Some(mid) = r.get("memory_id").and_then(|v| v.as_str()) {
                add_returned_memory_id(mid.to_string());
            }
        }
        let mut lines = vec![format!("Memory search results for '{query}':\n")];
        for r in &new {
            let mid = r.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
            let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let text = if text.len() > 200 { &text[..200] } else { text };
            lines.push(format!("[{mid}] {text}"));
        }
        lines.join("\n")
    })
}

pub fn search_skills_definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": "Search for relevant skills via semantic matching. Returns full skill content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What capability you need"},
                    "top_k": {"type": "integer", "description": "Results count (default: 3)"},
                },
                "required": ["query"],
            },
        },
    })
}

// ── search_skills ─────────────────────────────────────────────────────────────

pub fn search_skills(args: &HashMap<String, JsonValue>) -> String {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let ws = workspace_root();
        let config = match crate::config::load_config(&ws) {
            Ok(c) => c, Err(e) => return format!("Config error: {e}"),
        };
        let skills = crate::skills::cached_skills();
        if skills.is_empty() { return "No skills installed.".to_string(); }
        let mut router = crate::skills::SkillRouter::new(
            &config.memory_dir, &config.embedding_api_base,
            &config.embedding_api_key, &config.embedding_model,
        );
        if let Err(e) = router.index_skills(&skills).await { return format!("Index error: {e}"); }
        let raw = match router.search(query, top_k).await {
            Ok(r) => r, Err(e) => return format!("Search failed: {e}"),
        };
        let returned = get_returned_skill_names();
        let new: Vec<_> = raw.iter().filter(|r| {
            let name = r.get("name").and_then(|v| v.as_str()).unwrap_or("");
            !returned.contains(name)
        }).collect();
        if new.is_empty() { return "No new skills found. Try a different query.".to_string(); }
        for r in &new {
            if let Some(name) = r.get("name").and_then(|v| v.as_str()) {
                add_returned_skill_name(name.to_string());
            }
        }
        let mut lines = vec![format!("Skill search results for '{query}':\n")];
        for (i, r) in new.iter().enumerate() {
            let name = r.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = r.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let src = r.get("source").and_then(|v| v.as_str()).unwrap_or("");
            lines.push(format!("## {name}\n**Description:** {desc}\n**Source:** {src}"));
            if let Some(s) = crate::skills::get_skill(name) {
                lines.push(format!("\n{}", s.load()));
            }
            if i < new.len() - 1 { lines.push("\n---\n".to_string()); }
        }
        lines.join("\n")
    })
}

// ── pre_index_skills ──────────────────────────────────────────────────────────

pub fn pre_index_skills(config: &crate::config::Config) {
    let skills = crate::skills::cached_skills();
    if skills.is_empty() { return; }
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut router = crate::skills::SkillRouter::new(
        &config.memory_dir, &config.embedding_api_base,
        &config.embedding_api_key, &config.embedding_model,
    );
    rt.block_on(async { router.index_skills(&skills).await }).ok();
}
