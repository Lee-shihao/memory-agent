use chrono::Utc;
use serde_json::Value as JsonValue;
use std::fs::{self, OpenOptions};
use std::io::Write;
/// Debug logging for HTTP API calls. Enable via --debug flag.
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static DEBUG_STATE: Mutex<Option<PathBuf>> = Mutex::new(None);
static SESSION_STATS: Mutex<SessionStats> = Mutex::new(SessionStats::new());

const SEPARATOR: &str = "──────────────────────────────────────────────────────────────────────";

#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub prompt_cache_hit_tokens: u64,
    pub prompt_cache_miss_tokens: u64,
    pub llm_call_count: u64,
}

impl SessionStats {
    const fn new() -> Self {
        SessionStats {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: 0,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: 0,
            llm_call_count: 0,
        }
    }
}

pub fn enable(memory_dir: &Path) {
    let dir = memory_dir.to_path_buf();
    fs::create_dir_all(&dir).ok();
    let log_file = dir.join("debug.log");
    {
        let mut f = match OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&log_file)
        {
            Ok(f) => f,
            Err(_) => return,
        };
        let sep = "═".repeat(70);
        let _ = writeln!(
            f,
            "{sep}\n  Debug session: {}\n{sep}\n",
            Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ")
        );
    }
    *DEBUG_STATE.lock().unwrap() = Some(log_file);
}

pub fn disable() {
    *DEBUG_STATE.lock().unwrap() = None;
}

pub fn is_enabled() -> bool {
    DEBUG_STATE.lock().unwrap().is_some()
}

fn get_debug_file() -> Option<PathBuf> {
    DEBUG_STATE.lock().unwrap().clone()
}

fn write_raw(text: &str) {
    let debug_file = get_debug_file();
    if let Some(ref path) = debug_file {
        if let Ok(mut f) = OpenOptions::new().append(true).open(path) {
            let _ = f.write_all(text.as_bytes());
        }
    }
}

fn pretty_json(obj: &JsonValue) -> String {
    serde_json::to_string_pretty(obj).unwrap_or_else(|_| format!("{:?}", obj))
}

fn sanitize_headers(headers: &JsonValue) -> JsonValue {
    let mut h = headers.clone();
    if let Some(obj) = h.as_object_mut() {
        for key in ["authorization", "Authorization"] {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                if v.starts_with("Bearer ") {
                    let truncated = if v.len() > 8 {
                        format!("Bearer ...{}", &v[v.len() - 8..])
                    } else {
                        "Bearer ...".to_string()
                    };
                    obj.insert(key.to_string(), JsonValue::String(truncated));
                }
            }
        }
    }
    h
}

pub fn log_request(
    module: &str,
    method: &str,
    url: &str,
    headers: Option<&JsonValue>,
    body: Option<&JsonValue>,
) -> String {
    if !is_enabled() {
        return String::new();
    }
    let request_id = Utc::now().format("%H%M%S-%f").to_string();
    let request_id = request_id.chars().take(15).collect::<String>();
    let ts = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();

    let mut lines = vec![
        SEPARATOR.to_string(),
        format!("[{ts}]  REQUEST  {request_id}  module={module}"),
        format!("{method}  {url}"),
    ];
    if let Some(h) = headers {
        lines.push(format!("Headers: {}", pretty_json(&sanitize_headers(h))));
    }
    if let Some(b) = body {
        lines.push(format!("Body:\n{}", pretty_json(b)));
    }
    lines.push(String::new());
    write_raw(&lines.join("\n"));
    request_id
}

pub fn log_response(
    request_id: &str,
    status_code: u16,
    body: Option<&JsonValue>,
    error: Option<&str>,
) {
    if !is_enabled() {
        return;
    }
    let ts = Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    let icon = if (200..300).contains(&status_code) {
        "✓"
    } else {
        "✗"
    };

    let mut lines = vec![format!(
        "[{ts}]  RESPONSE  {request_id}  {icon} HTTP {status_code}"
    )];
    if let Some(e) = error {
        lines.push(format!("ERROR: {e}"));
    }
    if let Some(b) = body {
        let body_str = pretty_json(b);
        if body_str.len() > 16384 {
            lines.push(format!("Body:\n{}... (truncated)", &body_str[..16384]));
        } else {
            lines.push(format!("Body:\n{}", body_str));
        }
    }
    lines.push(format!("{SEPARATOR}\n"));
    write_raw(&lines.join("\n"));
}

pub fn accumulate_usage(usage: &JsonValue) {
    if usage.is_null() {
        return;
    }
    let mut stats = SESSION_STATS.lock().unwrap();
    stats.prompt_tokens += usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    stats.completion_tokens += usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    stats.total_tokens += usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if let Some(details) = usage.get("prompt_tokens_details") {
        stats.cached_tokens += details
            .get("cached_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    }
    stats.prompt_cache_hit_tokens += usage
        .get("prompt_cache_hit_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    stats.prompt_cache_miss_tokens += usage
        .get("prompt_cache_miss_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    stats.llm_call_count += 1;
}

pub fn get_session_stats() -> SessionStats {
    SESSION_STATS.lock().unwrap().clone()
}

pub fn reset_session_stats() {
    *SESSION_STATS.lock().unwrap() = SessionStats::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disable_by_default() {
        assert!(!is_enabled());
    }

    #[test]
    fn test_enable_disable_cycle() {
        let tmp = std::env::temp_dir().join("test_debug_rs");
        fs::create_dir_all(&tmp).ok();
        enable(&tmp);
        assert!(is_enabled());
        disable();
        assert!(!is_enabled());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_session_stats_accumulate() {
        reset_session_stats();
        let usage = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": { "cached_tokens": 20 }
        });
        accumulate_usage(&usage);
        let stats = get_session_stats();
        assert_eq!(stats.prompt_tokens, 100);
        assert_eq!(stats.completion_tokens, 50);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.cached_tokens, 20);
        assert_eq!(stats.llm_call_count, 1);
    }
}
