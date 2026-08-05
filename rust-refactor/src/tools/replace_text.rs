use super::{resolve_path, read_error, JsonValue, HashMap};
use std::fs;

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace an exact string in a file. Fails if old_text appears multiple times unless expected_count is set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "old_text": {"type": "string", "description": "Exact text to replace"},
                    "new_text": {"type": "string", "description": "Replacement (empty string to delete)"},
                    "expected_count": {"type": "integer", "description": "Expected occurrences (prevents ambiguity)"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let old_text = args.get("old_text").and_then(|v| v.as_str()).unwrap_or("");
    let new_text = args.get("new_text").and_then(|v| v.as_str()).unwrap_or("");
    let expected_count = args.get("expected_count").and_then(|v| v.as_u64()).map(|n| n as usize);

    if old_text.is_empty() {
        return "Error: old_text must not be empty".to_string();
    }

    let path = resolve_path(file_path);
    if !path.exists() {
        return read_error(&path);
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return format!("Error reading file: {e}"),
    };

    let count = content.matches(old_text).count();
    if count == 0 {
        return format!("Error: old_text not found in {}", path.display());
    }
    if let Some(expected) = expected_count {
        if count != expected {
            return format!(
                "Error: ambiguous — old_text appears {count} times, expected {expected}. \
                 Use more specific old_text or set expected_count={count}.",
            );
        }
    } else if count > 1 {
        return format!(
            "Error: old_text appears {count} times in {path}. \
             Set expected_count={count} to proceed, or narrow old_text.",
            path = path.display()
        );
    }

    let new_content = content.replacen(old_text, new_text, 1);

    let temp_path = path.with_extension("tmp.edit");
    if let Err(e) = fs::write(&temp_path, &new_content) {
        let _ = fs::remove_file(&temp_path);
        return format!("Error writing file: {e}");
    }
    if let Err(e) = fs::rename(&temp_path, &path) {
        let _ = fs::remove_file(&temp_path);
        return format!("Error replacing file: {e}");
    }

    format!(
        "Replaced in {}: {}",
        path.display(),
        if old_text.len() > 40 { format!("{}...", &old_text[..40]) } else { old_text.to_string() }
    )
}
