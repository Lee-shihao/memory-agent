use super::{workspace_root, JsonValue, HashMap};
use std::process::Command as Process;

pub fn diff_definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff. Use after edits to review changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to diff (omit for all)"},
                    "staged": {"type": "boolean", "description": "Show staged changes (default: false)"},
                },
                "required": [],
            },
        },
    })
}

pub fn status_definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show git status — modified, staged, untracked files.",
            "parameters": { "type": "object", "properties": {}, "required": [] },
        },
    })
}

fn run_git(args: &[&str]) -> String {
    match Process::new("git").args(args).current_dir(workspace_root()).output() {
        Ok(output) => {
            let mut result = String::from_utf8_lossy(&output.stdout).to_string();
            if !output.stderr.is_empty() {
                result.push_str(&format!("\n[stderr]\n{}", String::from_utf8_lossy(&output.stderr)));
            }
            if result.trim().is_empty() { "(no output)".to_string() } else { result }
        }
        Err(e) => format!("Error: {e}"),
    }
}

pub fn diff(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str());
    let staged = args.get("staged").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut cmd = vec!["diff"];
    if staged { cmd.push("--staged"); }
    if let Some(p) = file_path {
        cmd.push("--");
        cmd.push(p);
    }
    run_git(&cmd)
}

pub fn status(_args: &HashMap<String, JsonValue>) -> String {
    run_git(&["status", "--short"])
}
