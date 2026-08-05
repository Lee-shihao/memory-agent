use regex::Regex;
use super::{resolve_path, workspace_root, JsonValue, HashMap};
use std::fs;

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents for a regex pattern. Returns [{file, line, context}]. Use to locate code, functions, types, or errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search (default: '.')"},
                    "file_pattern": {"type": "string", "description": "Filter by extension, e.g. '*.rs', '*.py'"},
                    "max_results": {"type": "integer", "description": "Max results (default: 30)"},
                },
                "required": ["query"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let search_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let file_pattern = args.get("file_pattern").and_then(|v| v.as_str());
    let max_results = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(30) as usize;

    if query.is_empty() {
        return "Error: query is required".to_string();
    }

    let root = resolve_path(search_path);
    if !root.exists() {
        return format!("Error: Path not found: {}", root.display());
    }

    let re = match Regex::new(query) {
        Ok(r) => r,
        Err(e) => return format!("Error: Invalid regex: {e}"),
    };

    let mut results: Vec<JsonValue> = Vec::new();
    let ws = workspace_root();
    let ext_filter: Option<String> = file_pattern.map(|p| p.trim_start_matches("*.").to_string());

    let walker = walkdir::WalkDir::new(&root).max_depth(10);
    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let fp = entry.path();
        if fp.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.')) { continue; }
        if let Some(ext) = fp.extension() {
            if matches!(ext.to_string_lossy().as_ref(), "pyc" | "pyo" | "so" | "o" | "a" | "exe" | "bin" | "png" | "jpg") { continue; }
        }
        if let Some(ref ext) = ext_filter {
            if fp.extension().map_or(true, |e| e.to_string_lossy() != *ext) { continue; }
        }

        let content = match fs::read_to_string(fp) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for (lineno, line) in content.lines().enumerate() {
            if re.is_match(line) {
                let rel = fp.strip_prefix(&ws).unwrap_or(fp);
                let ctx = line.trim();
                let ctx = if ctx.len() > 200 { &ctx[..200] } else { ctx };
                results.push(serde_json::json!({
                    "file": rel.to_string_lossy(),
                    "line": lineno + 1,
                    "context": ctx,
                }));
                if results.len() >= max_results { break; }
            }
        }
        if results.len() >= max_results { break; }
    }

    if results.is_empty() {
        format!("No matches for '{query}'")
    } else {
        serde_json::to_string_pretty(&results).unwrap_or_default()
    }
}
