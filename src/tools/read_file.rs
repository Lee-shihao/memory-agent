use super::{resolve_path, read_error, JsonValue, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents with optional line range. Use start_line/end_line to avoid loading entire files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "start_line": {"type": "integer", "description": "First line number (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line number (inclusive)"},
                },
                "required": ["path"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|n| n.saturating_sub(1) as usize);
    let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

    let path = resolve_path(file_path);
    if !path.exists() {
        return read_error(&path);
    }

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => return format!("Error reading file: {e}"),
    };
    let reader = BufReader::new(file);

    let start = start_line.unwrap_or(0);
    let mut result_lines: Vec<String> = Vec::new();
    let mut line_count: usize = 0;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => return format!("Error reading file: {e}"),
        };
        if line_count >= start {
            result_lines.push(line.clone());
            if let Some(end) = end_line {
                if line_count + 1 >= end { break; }
            }
        }
        line_count += 1;
        if result_lines.len() > 2000 {
            result_lines.push("... (truncated at 2000 lines)".to_string());
            break;
        }
    }

    let total = line_count;
    let shown_start = if line_count == 0 { 0 } else { start.min(line_count - 1) + 1 };
    let shown_end = shown_start + result_lines.len() - 1;

    format!(
        "File: {} (lines {}-{} of {})\n\n{}",
        path.display(),
        shown_start,
        shown_end.min(total),
        total,
        result_lines.join("\n")
    )
}
