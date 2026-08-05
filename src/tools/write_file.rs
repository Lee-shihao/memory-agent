use super::{resolve_path, JsonValue, HashMap};
use std::fs;

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Set overwrite=true to replace existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "File content"},
                    "overwrite": {"type": "boolean", "description": "Allow overwrite (default: false)"},
                },
                "required": ["path", "content"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let overwrite = args.get("overwrite").and_then(|v| v.as_bool()).unwrap_or(false);

    let path = resolve_path(file_path);

    if path.exists() && !overwrite {
        return format!(
            "Error: {} already exists. Set overwrite=true to replace, \
             or use replace_text for targeted edits.",
            path.display()
        );
    }

    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return format!("Error creating directories: {e}");
        }
    }

    match fs::write(&path, content) {
        Ok(_) => format!("File written: {} ({} bytes)", path.display(), content.len()),
        Err(e) => format!("Error writing file: {e}"),
    }
}
