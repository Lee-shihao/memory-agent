use once_cell::sync::Lazy;
use regex::Regex;
use super::{workspace_root, JsonValue, HashMap};
use std::collections::HashSet;
use std::process::Command as Process;

// ── Bash classification ──────────────────────────────────────────────────────

fn safe_bash_commands() -> &'static HashSet<&'static str> {
    static S: Lazy<HashSet<&str>> = Lazy::new(|| {
        HashSet::from([
            "ls", "cat", "file", "head", "tail", "less", "more", "find", "grep", "wc", "stat",
            "du", "df", "sort", "uniq", "pwd", "which", "type", "env", "printenv", "uname",
            "whoami", "date", "id", "hostname", "tree", "awk", "sed", "cut", "tr", "tee", "echo",
            "true", "false", "diff", "cmp", "dirname", "basename", "realpath", "readlink", "xargs",
            "mkdir", "touch",
        ])
    });
    &S
}

fn dangerous_bash_commands() -> &'static HashSet<&'static str> {
    static D: Lazy<HashSet<&str>> = Lazy::new(|| {
        HashSet::from([
            "rm", "rmdir", "dd", "chmod", "chown", "chgrp", "sudo", "su", "kill", "killall",
            "pkill", "shutdown", "reboot", "halt", "systemctl", "service", "mount", "umount",
            "mkfs", "fdisk", "apt", "apt-get", "yum", "dnf", "pacman", "pip", "pip3", "npm",
            "yarn", "npx", "cargo", "go", "curl", "wget", "ssh", "scp", "rsync", "eval",
            "exec", "source",
        ])
    });
    &D
}

#[derive(Debug, PartialEq)]
pub enum BashTier { Safe, Dangerous, Unknown }

pub fn classify_bash_command(command: &str) -> BashTier {
    let stripped = command.trim();
    if stripped.is_empty() { return BashTier::Safe; }
    let pipe_re = Regex::new(r"(curl|wget)\s+.*\|\s*(sh|bash)").unwrap();
    if pipe_re.is_match(stripped) { return BashTier::Dangerous; }
    let parts: Vec<&str> = stripped.split_whitespace().collect();
    let base = parts[0];
    if base == "git" && parts.len() > 1 {
        return match parts[1] {
            "push" | "fetch" | "pull" => BashTier::Dangerous,
            _ => BashTier::Safe,
        };
    }
    if dangerous_bash_commands().contains(base) { return BashTier::Dangerous; }
    if safe_bash_commands().contains(base) { return BashTier::Safe; }
    BashTier::Unknown
}

// ── Safety checks ─────────────────────────────────────────────────────────────

fn is_catastrophic(command: &str) -> Option<&'static str> {
    let catastrophic = [
        ("rm -rf /", "rm -rf /"),
        ("rm -rf /*", "rm -rf /*"),
        ("rm -rf ~", "rm -rf ~"),
        ("mkfs.", "mkfs — filesystem format"),
        ("fdisk ", "fdisk — disk partition"),
        ("dd if=", "dd — raw disk write"),
        ("shutdown", "shutdown"),
        ("reboot", "reboot"),
        ("poweroff", "poweroff"),
        (":(){ :|:& };:", "fork bomb"),
        (">/dev/sda", "raw block device write"),
    ];
    let lower = command.to_lowercase();
    for (pattern, reason) in &catastrophic {
        if lower.contains(pattern) { return Some(reason); }
    }
    None
}

pub fn is_shell_escape(command: &str) -> Option<&'static str> {
    let escapes = [
        ("python -c", "python -c — code execution"),
        ("python3 -c", "python3 -c — code execution"),
        ("perl -e", "perl -e — code execution"),
        ("ruby -e", "ruby -e — code execution"),
        ("node -e", "node -e — code execution"),
        ("sh -c", "sh -c — nested shell"),
        ("bash -c", "bash -c — nested shell"),
    ];
    let lower = command.to_lowercase();
    for (pattern, reason) in &escapes {
        if lower.contains(pattern) { return Some(reason); }
    }
    None
}

// ── run_bash tool ─────────────────────────────────────────────────────────────

pub fn definition() -> JsonValue {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command inside workspace. Returns {success, exit_code, stdout, stderr, duration_ms}. Max 5min, max 10000 chars output. Catastrophic commands blocked; shell escapes require confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout_seconds": {"type": "integer", "description": "Timeout (default: 30, max: 300)"},
                    "max_output_chars": {"type": "integer", "description": "Max output chars (default: 10000)"},
                },
                "required": ["command"],
            },
        },
    })
}

pub fn run(args: &HashMap<String, JsonValue>) -> String {
    let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let timeout_secs = args.get("timeout_seconds").and_then(|v| v.as_u64()).unwrap_or(30) as u64;
    let max_output = args.get("max_output_chars").and_then(|v| v.as_u64()).unwrap_or(10000) as usize;

    if command.is_empty() {
        return serde_json::json!({"success": false, "exit_code": -1, "error": "command is empty"}).to_string();
    }

    // Catastrophic block
    if let Some(reason) = is_catastrophic(command) {
        return serde_json::json!({"success": false, "exit_code": -1, "error": format!("Blocked: {reason}")}).to_string();
    }

    let _timeout = timeout_secs.min(300);
    let start = std::time::Instant::now();
    let ws = workspace_root();

    let result = Process::new("bash")
        .args(["-c", command])
        .current_dir(&ws)
        .output();

    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            let success = output.status.success();
            let exit_code = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            let (stdout_str, stdout_trunc) = if stdout.len() > max_output {
                (format!("{}\n... (truncated at {max_output} chars)", &stdout[..max_output]), true)
            } else {
                (stdout.to_string(), false)
            };
            let (stderr_str, _) = if stderr.len() > max_output {
                (stderr[..max_output].to_string(), true)
            } else {
                (stderr.to_string(), false)
            };

            let mut resp = serde_json::json!({
                "success": success,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "stdout": stdout_str,
                "stderr": stderr_str,
            });
            if stdout.is_empty() && stderr.is_empty() {
                resp["stdout"] = serde_json::json!("(no output)");
            }
            if stdout_trunc { resp["stdout_truncated"] = serde_json::json!(true); }
            serde_json::to_string(&resp).unwrap_or_default()
        }
        Err(e) => serde_json::json!({
            "success": false, "exit_code": -1, "duration_ms": duration_ms,
            "error": format!("{e}"),
        }).to_string(),
    }
}
