use regex::Regex;
use super::{resolve_path, read_error, JsonValue, HashMap};
use std::fs;

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch. Verifies context before applying. Fails if context mismatches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "patch": {"type": "string", "description": "Unified diff in @@ -old +new @@ format"},
                    "dry_run": {"type": "boolean", "description": "Validate without applying (default: false)"},
                },
                "required": ["path", "patch"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let patch = args.get("patch").and_then(|v| v.as_str()).unwrap_or("");
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);

    let path = resolve_path(file_path);
    if !path.exists() {
        return read_error(&path);
    }

    let hunk_re = Regex::new(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@").unwrap();

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return format!("Error reading file: {e}"),
    };
    let lines: Vec<&str> = content.lines().collect();

    if let Some(caps) = hunk_re.captures(patch) {
        let old_start: usize = caps[1].parse().unwrap_or(1);
        let old_count: usize = caps.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(1);
        let new_count: usize = caps.get(4).and_then(|m| m.as_str().parse().ok()).unwrap_or(1);

        let patch_lines: Vec<&str> = patch.lines().skip_while(|l| !l.starts_with("@@")).collect();
        let mut old_lines: Vec<String> = Vec::new();
        let mut new_lines: Vec<String> = Vec::new();

        for pl in &patch_lines[1..] {
            if pl.starts_with('-') {
                old_lines.push(pl[1..].to_string());
            } else if pl.starts_with('+') {
                new_lines.push(pl[1..].to_string());
            } else if pl.starts_with(' ') {
                old_lines.push(pl[1..].to_string());
                new_lines.push(pl[1..].to_string());
            }
        }

        // Verify context
        let actual_start = old_start.saturating_sub(1);
        let actual_end = (actual_start + old_count).min(lines.len());
        let actual_slice: Vec<String> = lines[actual_start..actual_end]
            .iter().map(|s| s.to_string()).collect();

        if actual_slice != old_lines {
            return format!(
                "Error: context mismatch.\nPatch expects:\n  {}\nbut file has:\n  {}\n\
                 Re-read the file and regenerate the patch.",
                old_lines.join("\n  "),
                actual_slice.join("\n  "),
            );
        }

        if dry_run {
            return format!(
                "Dry run — patch would apply cleanly. {} lines would change.",
                old_count.max(new_count)
            );
        }

        // Apply
        let mut new_content: Vec<String> = lines[..actual_start].iter().map(|s| s.to_string()).collect();
        new_content.extend(new_lines);
        new_content.extend(lines[actual_end..].iter().map(|s| s.to_string()));

        let temp_path = path.with_extension("tmp.patch");
        if let Err(e) = fs::write(&temp_path, new_content.join("\n")) {
            let _ = fs::remove_file(&temp_path);
            return format!("Error writing file: {e}");
        }
        if let Err(e) = fs::rename(&temp_path, &path) {
            let _ = fs::remove_file(&temp_path);
            return format!("Error replacing file: {e}");
        }

        let changed = old_count.max(new_count);
        format!("Patch applied: {} ({} lines changed)", path.display(), changed)
    } else {
        "Error: Could not parse patch. Use unified diff format:\n\
         @@ -line,count +line,count @@\n  context\n-old\n+new\n  context".to_string()
    }
}
