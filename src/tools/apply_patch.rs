use regex::Regex;
use super::{resolve_path, read_error, JsonValue, HashMap};
use std::fs;

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch. Verifies context before applying. Fails if context mismatches. Supports multiple hunks.",
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

struct Hunk<'a> {
    /// 1-based start line in the original file
    old_start: usize,
    /// Number of context + removal lines in this hunk
    old_count: usize,
    /// Body lines after the @@ header, with prefix chars: '-', '+', ' '
    body_lines: Vec<&'a str>,
    /// Raw @@ header line for error messages
    header: &'a str,
}

/// Parse a unified diff patch string into a list of hunks.
fn parse_hunks(patch: &str) -> Result<Vec<Hunk<'_>>, String> {
    let hunk_re = Regex::new(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@").unwrap();

    // Collect all hunk header positions and captures
    let caps: Vec<_> = hunk_re.captures_iter(patch).collect();

    if caps.is_empty() {
        return Err("Could not parse patch: no hunk headers found.".to_string());
    }

    let mut hunks = Vec::with_capacity(caps.len());

    for i in 0..caps.len() {
        let cap = &caps[i];
        let full_match = cap.get(0).unwrap();
        let header = full_match.as_str();

        let old_start: usize = cap[1].parse().unwrap_or(1);
        let old_count: usize = cap
            .get(2)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(1);

        // Body: from after this header to the start of the next header (or end of patch)
        let body_start = full_match.end();
        let body_end = caps
            .get(i + 1)
            .and_then(|c| c.get(0))
            .map(|m| m.start())
            .unwrap_or(patch.len());

        // Extract and trim the body text, skip leading/trailing blank lines
        let body = patch[body_start..body_end].trim_matches('\n');
        let body_lines: Vec<&str> = if body.is_empty() {
            Vec::new()
        } else {
            body.lines().collect()
        };

        hunks.push(Hunk {
            old_start,
            old_count,
            body_lines,
            header,
        });
    }

    Ok(hunks)
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let file_path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
    let patch = args.get("patch").and_then(|v| v.as_str()).unwrap_or("");
    let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);

    let path = resolve_path(file_path);
    if !path.exists() {
        return read_error(&path);
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return format!("Error reading file: {e}"),
    };
    let lines: Vec<&str> = content.lines().collect();

    let hunks = match parse_hunks(patch) {
        Ok(h) => h,
        Err(e) => return format!("Error: {e}\nUse unified diff format:\n  @@ -line,count +line,count @@\n   context\n  -old\n  +new\n   context"),
    };

    let total = hunks.len();

    // --- Dry run: verify all hunks independently ---
    if dry_run {
        for (i, hunk) in hunks.iter().enumerate() {
            let (old_lines, _) = split_body(&hunk.body_lines);
            let start = hunk.old_start.saturating_sub(1);
            let end = (start + hunk.old_count).min(lines.len());
            let actual: Vec<String> = lines[start..end].iter().map(|s| s.to_string()).collect();

            if actual != old_lines {
                return format!(
                    "Error: context mismatch in hunk {}/{}.\n  {}\nPatch expects:\n  {}\nbut file has:\n  {}\n\
                     Re-read the file and regenerate the patch.",
                    i + 1,
                    total,
                    hunk.header,
                    old_lines.join("\n  "),
                    actual.join("\n  "),
                );
            }
        }
        return format!("Dry run — {} hunk(s) would apply cleanly.", total);
    }

    // --- Apply: verify and apply hunks in order ---
    let mut pos: usize = 0; // current position in original lines (0-based)
    let mut new_content: Vec<String> = Vec::new();

    for (i, hunk) in hunks.iter().enumerate() {
        let hunk_label = format!("Hunk {}/{}", i + 1, total);
        let (old_lines, new_lines) = split_body(&hunk.body_lines);

        let start = hunk.old_start.saturating_sub(1);
        let end = (start + hunk.old_count).min(lines.len());

        // Detect out-of-order or overlapping hunks
        if start < pos {
            return format!(
                "Error: {} overlaps with previous hunk.\n  {}\n\
                 Expected hunk to start at or after line {} (original), got line {}.\n\
                 Hunks must not overlap. Re-read the file and regenerate the patch.",
                hunk_label,
                hunk.header,
                pos + 1,
                hunk.old_start,
            );
        }

        // Copy unchanged lines between previous position and this hunk
        new_content.extend(lines[pos..start].iter().map(|s| s.to_string()));

        // Verify context
        let actual: Vec<String> = lines[start..end].iter().map(|s| s.to_string()).collect();
        if actual != old_lines {
            return format!(
                "Error: context mismatch in {}.\n  {}\nPatch expects:\n  {}\nbut file has:\n  {}\n\
                 Re-read the file and regenerate the patch.",
                hunk_label,
                hunk.header,
                old_lines.join("\n  "),
                actual.join("\n  "),
            );
        }

        // Apply this hunk: keep '+' and ' ' lines (skip '-' lines)
        for line in &new_lines {
            new_content.push(line.clone());
        }

        pos = end;
    }

    // Copy remaining lines after the last hunk
    new_content.extend(lines[pos..].iter().map(|s| s.to_string()));

    // Atomic write: temp file then rename
    let temp_path = path.with_extension("tmp.patch");
    if let Err(e) = fs::write(&temp_path, new_content.join("\n")) {
        let _ = fs::remove_file(&temp_path);
        return format!("Error writing file: {e}");
    }
    if let Err(e) = fs::rename(&temp_path, &path) {
        let _ = fs::remove_file(&temp_path);
        return format!("Error replacing file: {e}");
    }

    format!(
        "Patch applied: {} ({} hunk(s) changed)",
        path.display(),
        total,
    )
}

/// Split hunk body lines into (old_lines, new_lines).
/// - Old lines: context (' ') and removals ('-') without prefix
/// - New lines: context (' ') and additions ('+') without prefix
fn split_body(body_lines: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut old_lines: Vec<String> = Vec::new();
    let mut new_lines: Vec<String> = Vec::new();

    for line in body_lines {
        match line.chars().next() {
            Some('-') => old_lines.push(line[1..].to_string()),
            Some('+') => new_lines.push(line[1..].to_string()),
            Some(' ') => {
                old_lines.push(line[1..].to_string());
                new_lines.push(line[1..].to_string());
            }
            _ => {} // ignore unexpected lines
        }
    }

    (old_lines, new_lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn apply(file_content: &str, patch: &str) -> String {
        let dir = std::env::temp_dir().join("apply_patch_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_file.txt");

        // Write test file
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(file_content.as_bytes()).unwrap();

        // Override workspace root to temp dir
        crate::tools::set_workspace_root(&dir);

        let mut args = HashMap::new();
        args.insert("path".to_string(), serde_json::json!("test_file.txt"));
        args.insert("patch".to_string(), serde_json::json!(patch));
        args.insert("dry_run".to_string(), serde_json::json!(false));

        let result = run(&args);

        // Read back modified file
        let modified = std::fs::read_to_string(&path).unwrap_or_default();

        let _ = std::fs::remove_dir_all(&dir);
        format!("result: {result}\nfile:\n{modified}")
    }

    #[test]
    fn test_single_hunk_modify() {
        let original = "line 1\nline 2\nline 3\nline 4\nline 5\n";
        let patch = "@@ -2,3 +2,3 @@\n line 2\n-line 3\n+line three modified\n line 4\n";
        let output = apply(original, patch);
        assert!(output.contains("Patch applied"), "Should succeed: {output}");
        assert!(output.contains("line three modified"), "Should have new line: {output}");
        assert!(!output.contains("line 3\n"), "Old line 3 should be removed: {output}");
    }

    #[test]
    fn test_multi_hunk_modify() {
        let original = "\
fn main() {
    let x = 1;
    println!(\"x\");  // old comment
    let y = 2;
    println!(\"y\");  // another old
}
";
        let patch = "\
@@ -3,1 +3,1 @@
-    println!(\"x\");  // old comment
+    println!(\"x\");  // new comment
@@ -5,1 +5,1 @@
-    println!(\"y\");  // another old
+    println!(\"y\");  // another new
";
        let output = apply(original, patch);
        assert!(output.contains("Patch applied"), "Should succeed: {output}");
        assert!(output.contains("new comment"), "Hunk 1 should apply: {output}");
        assert!(output.contains("another new"), "Hunk 2 should apply: {output}");
        assert!(!output.contains("old comment"), "Old line 1 should be gone: {output}");
        assert!(!output.contains("another old"), "Old line 2 should be gone: {output}");
    }

    #[test]
    fn test_context_mismatch_reported_per_hunk() {
        let original = "line 1\nline 2\nline 3\nline 4\nline 5\n";
        // Patch expects "line 2" at line 2 but file has it; second hunk expects wrong context
        let patch = "\
@@ -2,2 +2,2 @@\n line 2\n line 3\n\
@@ -4,1 +4,1 @@\n-wrong context\n+new line\n";
        let output = apply(original, patch);
        assert!(output.contains("context mismatch"), "Should report context mismatch: {output}");
        assert!(output.contains("Hunk 2/2"), "Should report which hunk failed: {output}");
    }

    #[test]
    fn test_overlapping_hunks_rejected() {
        let original = "line 1\nline 2\nline 3\nline 4\n";
        let patch = "\
@@ -1,2 +1,2 @@\n line 1\n line 2\n\
@@ -1,2 +1,2 @@\n line 1\n line 2\n";
        let output = apply(original, patch);
        assert!(output.contains("overlaps"), "Should reject overlapping hunks: {output}");
    }

    #[test]
    fn test_dry_run_multi_hunk() {
        let dir = std::env::temp_dir().join("apply_patch_test_dry");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.txt");
        std::fs::write(&path, "line 1\nline 2\nline 3\nline 4\n").unwrap();

        crate::tools::set_workspace_root(&dir);

        let mut args = HashMap::new();
        args.insert("path".to_string(), serde_json::json!("test.txt"));
        args.insert(
            "patch".to_string(),
            serde_json::json!("@@ -1,2 +1,2 @@\n line 1\n line 2\n@@ -4,1 +4,1 @@\n line 4\n"),
        );
        args.insert("dry_run".to_string(), serde_json::json!(true));

        let result = run(&args);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            result.contains("Dry run") && result.contains("would apply cleanly"),
            "Dry run should validate all hunks: {result}"
        );
    }

    #[test]
    fn test_empty_patch_error() {
        let output = apply("anything\n", "");
        assert!(output.contains("Error"), "Should error on empty patch: {output}");
    }
}
