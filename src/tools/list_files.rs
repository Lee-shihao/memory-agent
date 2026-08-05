use super::{resolve_path, JsonValue, HashMap};

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a workspace path. Use to understand project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')"},
                    "recursive": {"type": "boolean", "description": "Search recursively (default: true)"},
                    "max_depth": {"type": "integer", "description": "Max directory depth (default: 3)"},
                    "include_hidden": {"type": "boolean", "description": "Include hidden files (default: false)"},
                },
                "required": [],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let search_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(true);
    let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let include_hidden = args.get("include_hidden").and_then(|v| v.as_bool()).unwrap_or(false);

    let root = resolve_path(search_path);
    if !root.exists() {
        return format!("Error: Path not found: {}", root.display());
    }

    let mut files: Vec<String> = Vec::new();
    let max_entries = 200;

    let walker = if recursive {
        walkdir::WalkDir::new(&root).max_depth(max_depth)
    } else {
        walkdir::WalkDir::new(&root).max_depth(1)
    };

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        if !include_hidden {
            if p.file_name().map_or(false, |n| n.to_string_lossy().starts_with('.')) {
                if p != root { continue; }
            }
        }
        let skip_dirs = [".git", "node_modules", "target", "__pycache__", ".venv"];
        if p.is_dir() && p.file_name().map_or(false, |n| skip_dirs.contains(&n.to_string_lossy().as_ref())) {
            if p != root { continue; }
        }
        let rel = p.strip_prefix(&root).unwrap_or(p);
        let prefix = if p.is_dir() { "\u{1F4C1}" } else { "\u{1F4C4}" };
        files.push(format!("{prefix} {}", rel.display()));
        if files.len() >= max_entries {
            files.push(format!("... (truncated at {max_entries} entries)"));
            break;
        }
    }

    if files.is_empty() {
        format!("No files found in {}", root.display())
    } else {
        files.join("\n")
    }
}
