use anyhow::Result;
use clap::Parser;
use memory_agent::*;
use rustyline::completion::Completer;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{MatchingBracketValidator, ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Helper};
use std::borrow::Cow;
/// Memory Agent CLI — 3-step pipeline: Retrieve → Agent Loop → Extract.
use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "memory-agent",
    version,
    about = "AI assistant with persistent memory",
    long_about = None,
)]
struct Cli {
    /// Your query or task for the agent. Omit to enter interactive mode.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    query: Vec<String>,

    /// Project root directory
    #[arg(short = 'p', long, default_value = ".")]
    project: PathBuf,

    /// Skip memory retrieval for this invocation
    #[arg(long)]
    no_memory: bool,

    /// Skip memory extraction after the conversation
    #[arg(long)]
    no_extract: bool,

    /// Prompt for save/edit/discard on each extracted memory
    #[arg(long)]
    manual_extract: bool,

    /// Log all HTTP API calls to .agent-memory/debug.log
    #[arg(long)]
    debug: bool,

    /// List installed skills and exit
    #[arg(long)]
    skill_list: bool,

    /// Install a skill from a local directory or git URL
    #[arg(long, value_name = "SOURCE")]
    skill_install: Option<String>,

    /// Additional skill directory to search
    #[arg(long, value_name = "DIR")]
    skill_dir: Option<String>,
}

const BANNER: &str = r#"
╔══════════════════════════════════════════════╗
║            🧠  Memory Agent                  ║
║                                              ║
║  自带向量记忆的 AI 助手                        ║
║  输入问题开始对话，/memory 查看和管理记忆       ║
║  Ctrl+D 或 /exit 退出                        ║
╚══════════════════════════════════════════════╝
"#;

fn print_token_stats() {
    let stats = debug::get_session_stats();
    if stats.llm_call_count == 0 {
        return;
    }
    let mut cache_rate = String::new();
    if stats.prompt_tokens > 0 && stats.cached_tokens > 0 {
        let rate = stats.cached_tokens as f64 / stats.prompt_tokens as f64 * 100.0;
        cache_rate = format!("\n  Cache hit rate:    {rate:.1}%");
    }
    eprintln!(
        "\n{}\n📊 Token usage this conversation:\n  LLM calls:         {}\n  Prompt tokens:     {}\n  Completion tokens: {}\n  Total tokens:      {}\n  Cached tokens:     {}{}\n{}\n",
        "=".repeat(50),
        stats.llm_call_count,
        stats.prompt_tokens,
        stats.completion_tokens,
        stats.total_tokens,
        stats.cached_tokens,
        cache_rate,
        "=".repeat(50),
    );
}

/// Confirmation callback for tools — handles ask_user and run_bash.
fn tool_confirm(tool_name: &str, args: &HashMap<String, serde_json::Value>) -> (bool, String) {
    // --- ask_user: handle entirely here, block execution ---
    if tool_name == "ask_user" {
        let question = args.get("question").and_then(|v| v.as_str()).unwrap_or("");
        let header = args
            .get("header")
            .and_then(|v| v.as_str())
            .unwrap_or("Question");
        let options = args.get("options").and_then(|v| v.as_array());
        let multi_select = args
            .get("multi_select")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Display question
        eprintln!("\n  ❓ {header}");
        eprintln!("  {question}");

        if let Some(opts) = options {
            for (i, opt) in opts.iter().enumerate() {
                let label = opt.get("label").and_then(|v| v.as_str()).unwrap_or("?");
                let desc = opt
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                eprintln!("  [{i}] {label}: {desc}", i = i + 1);
            }
            if multi_select {
                eprint!("  Enter numbers (e.g. 1,3) or type custom (60s timeout): ");
            } else {
                eprint!("  Enter number or type custom (60s timeout): ");
            }
        } else {
            eprint!("  Type your response (60s timeout): ");
        }
        let _ = io::stderr().flush();

        let mut input = String::new();
        // Simple read — timeout would require async I/O
        if io::stdin().read_line(&mut input).is_err() {
            input.clear();
        }
        let input = input.trim().to_string();

        eprintln!();

        if input.is_empty() {
            if let Some(opts) = options {
                let label = opts[0].get("label").and_then(|v| v.as_str()).unwrap_or("");
                return (false, format!("[Selected] {label}"));
            }
            return (false, String::new());
        }

        // Try to parse as option numbers
        if let Some(opts) = options {
            let parts: Vec<&str> = input.split([',', ' ']).collect();
            let mut numbers = Vec::new();
            for p in &parts {
                if let Ok(n) = p.trim().parse::<usize>() {
                    if n >= 1 && n <= opts.len() {
                        numbers.push(n);
                    }
                }
            }
            if !numbers.is_empty() {
                let labels: Vec<&str> = numbers
                    .iter()
                    .map(|&n| {
                        opts[n - 1]
                            .get("label")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                    })
                    .collect();
                if multi_select {
                    return (false, format!("[Selected] {}", labels.join(", ")));
                }
                return (false, format!("[Selected] {}", labels[0]));
            }
        }

        // Free text response
        return (false, input);
    }

    // --- run_bash: classify and confirm ---
    if tool_name == "run_bash" {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let tier = tools::classify_bash_command(command);

        // Silent auto-allow for safe commands
        if tier == tools::BashTier::Safe {
            return (true, String::new());
        }

        let display_cmd = if command.len() > 200 {
            format!("{}...", &command[..197])
        } else {
            command.to_string()
        };

        // Shell escape detection — always require confirmation
        let escape_reason = tools::is_shell_escape(command);

        // Dangerous / escape / unknown — require confirmation
        let label = if let Some(reason) = escape_reason {
            format!("  ⚠️  run_bash [ESCAPE: {reason}]")
        } else if tier == tools::BashTier::Dangerous {
            format!("  ⚠️  run_bash [DANGEROUS]")
        } else {
            format!("  🔧 run_bash")
        };

        eprintln!("\n{label}");
        eprintln!("  {display_cmd}");
        eprint!("  [y] allow  [n] deny: ");
        let _ = io::stderr().flush();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            eprintln!("\n  ⛔ (denied)");
            return (false, "Blocked: no confirmation".to_string());
        }
        let input = input.trim().to_lowercase();

        eprintln!();

        return if input == "y" || input == "yes" {
            (true, String::new())
        } else {
            (false, format!("Blocked: {}", if input.is_empty() || input == "n" || input == "no" {
                "denied by user".to_string()
            } else {
                input
            }))
        };
    }

    // All other tools: allow
    (true, String::new())
}

/// Parse a "/skill_name rest" command.
///
/// If `input` starts with `/` and the first token matches an installed skill,
/// returns `(rest_query, Some(skill_content))`. Otherwise returns
/// `(input.to_string(), None)` — the input is passed through unchanged.
fn parse_skill_command(input: &str) -> (String, Option<String>) {
    if !input.starts_with('/') {
        return (input.to_string(), None);
    }
    let mut parts = input[1..].splitn(2, ' ');
    let skill_name = parts.next().unwrap_or("").trim();
    if skill_name.is_empty() {
        return (input.to_string(), None);
    }
    if let Some(skill) = skills::get_skill(skill_name) {
        let rest = parts.next().unwrap_or("").trim().to_string();
        (rest, Some(skill.load()))
    } else {
        (input.to_string(), None)
    }
}

/// Execute the full 3-step pipeline for a single conversation turn.
async fn run_pipeline(
    user_query: &str,
    config: &config::Config,
    store: &mut storage::MemoryStore,
    skip_memory: bool,
    skip_extract: bool,
    manual_extract: bool,
) -> Result<()> {
    // Reset per-conversation state
    if debug::is_enabled() {
        debug::reset_session_stats();
    }

    tools::reset_session_state();

    // Pre-index skills (only if memory is enabled)
    if !skip_memory {
        tools::pre_index_skills(config).await;
    }

    // Handle /memory slash commands
    if user_query.starts_with("/memory") {
        if let Some(response) = commands::handle_slash_command(user_query, store, &[]).await {
            println!("{response}");
            return Ok(());
        }
    }

    // Handle "/skill_name query" commands: load the skill and inject its
    // content into the system prompt, using the rest as the user query.
    let (effective_query, skill_context) = parse_skill_command(user_query);

    // Step 2: Agent Loop
    eprintln!();
    let confirm_cb: agent_loop::ConfirmCallback = Arc::new(tool_confirm);
    let transcript = agent_loop::run_agent_loop(
        config,
        &effective_query,
        None,
        50,
        Some(confirm_cb),
        skill_context.as_deref(),
    )
    .await?;
    println!("{transcript}");

    // Step 3: Memory Extraction
    if !skip_extract {
        eprintln!();
        let auto = !manual_extract;
        if let Err(e) = extractor::extract_and_store(&transcript, config, store, Some(auto)).await {
            eprintln!("Memory extraction failed: {e}");
        }
    }

    // Print token stats in debug mode
    if debug::is_enabled() {
        print_token_stats();
    }

    Ok(())
}

/// Custom rustyline Helper that provides tab-completion for slash commands.
struct ReplHelper {
    completer: ReplCompleter,
    highlighter: MatchingBracketHighlighter,
    validator: MatchingBracketValidator,
    hinter: HistoryHinter,
}

impl Completer for ReplHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        self.completer.complete(line, pos, ctx)
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for ReplHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        use std::borrow::Cow::Owned;
        Owned(format!("\x1b[90m{hint}\x1b[0m"))
    }

    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        self.highlighter.highlight_char(line, pos, forced)
    }
}

impl Validator for ReplHelper {
    fn validate(
        &self,
        ctx: &mut ValidationContext,
    ) -> rustyline::Result<ValidationResult> {
        self.validator.validate(ctx)
    }

    fn validate_while_typing(&self) -> bool {
        self.validator.validate_while_typing()
    }
}

impl Helper for ReplHelper {}

/// Top-level slash commands and their subcommands.
struct ReplCompleter;

impl Completer for ReplCompleter {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        // Only complete when line starts with "/"
        if !line.starts_with('/') {
            return Ok((pos, vec![]));
        }

        // Built-in commands plus installed skills (as "/name").
        let mut top_level: Vec<String> = vec![
            "/memory".to_string(),
            "/exit".to_string(),
            "/quit".to_string(),
            "/q".to_string(),
            "/help".to_string(),
        ];
        for skill in crate::skills::cached_skills() {
            top_level.push(format!("/{}", skill.name));
        }
        let memory_subs: &[&str] = &[
            "recent", "search", "show", "delete", "status",
        ];

        // How many complete words (space-separated segments)?
        let segments: Vec<&str> = line[..pos].split(' ').collect();

        match segments.len() {
            1 => {
                // Completing top-level command: find matches
                let partial = segments[0];
                let candidates: Vec<String> = top_level
                    .iter()
                    .filter(|cmd| cmd.starts_with(partial))
                    .map(|s| s.to_string())
                    .collect();
                // Replace from the "/" character
                let start = line[..pos].rfind('/').unwrap_or(pos);
                Ok((start, candidates))
            }
            _ => {
                // Completing subcommand — only for "/memory"
                if segments[0] == "/memory" {
                    let partial = segments.get(1).copied().unwrap_or("");
                    let candidates: Vec<String> = memory_subs
                        .iter()
                        .filter(|sub| sub.starts_with(partial))
                        .map(|s| s.to_string())
                        .collect();
                    // Start position: right after "/memory "
                    let cmd_end = line[..pos].find(' ').map(|i| i + 1).unwrap_or(pos);
                    Ok((cmd_end, candidates))
                } else {
                    Ok((pos, vec![]))
                }
            }
        }
    }
}

/// Interactive REPL mode.
fn interactive_loop(
    config: &config::Config,
    store: &mut storage::MemoryStore,
    skip_memory: bool,
    skip_extract: bool,
    manual_extract: bool,
) -> Result<()> {
    eprintln!("{BANNER}");

    let helper = ReplHelper {
        completer: ReplCompleter,
        highlighter: MatchingBracketHighlighter::new(),
        validator: MatchingBracketValidator::new(),
        hinter: HistoryHinter::new(),
    };
    let mut rl = rustyline::Editor::<ReplHelper, _>::new()?;
    rl.set_helper(Some(helper));
    let history_file = dirs::home_dir()
        .unwrap_or_default()
        .join(".memory_agent_history");

    // Load history
    let _ = rl.load_history(history_file.as_path());

    let rt = tokio::runtime::Runtime::new()?;

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let input = line.trim().to_string();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(&input);

                if input == "/exit" || input == "/quit" || input == "/q" {
                    eprintln!("Goodbye.");
                    break;
                }
                if input == "/help" {
                    eprintln!(concat!(
                        "  Enter a question or task to start a conversation.\n",
                        "  /memory              Show injected memories\n",
                        "  /memory recent [N]   Show recent N memories\n",
                        "  /memory search <q>   Semantic search\n",
                        "  /memory show <id>    Show memory details\n",
                        "  /memory delete <id>  Delete a memory\n",
                        "  /memory status       Database statistics\n",
                        "  /exit, /quit, /q     Exit\n",
                        "  /help                Show this help\n",
                        "  Ctrl+D               Exit",
                    ));
                    continue;
                }

                rt.block_on(run_pipeline(
                    &input,
                    config,
                    store,
                    skip_memory,
                    skip_extract,
                    manual_extract,
                ))?;
            }
            Err(rustyline::error::ReadlineError::Eof)
            | Err(rustyline::error::ReadlineError::Interrupted) => {
                eprintln!("\nGoodbye.");
                break;
            }
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(history_file.as_path());

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Skill management commands (exit after processing)
    if cli.skill_list {
        println!("{}", skills::list_installed_skills(None));
        return Ok(());
    }
    if let Some(ref source) = cli.skill_install {
        println!("{}", skills::install_skill(source, Some(&cli.project)));
        return Ok(());
    }

    // Register extra skill directory
    if let Some(ref dir) = cli.skill_dir {
        for d in dir.split(':') {
            skills::add_search_path(d);
        }
    }

    // Determine user query
    let user_query = if !cli.query.is_empty() {
        Some(cli.query.join(" "))
    } else if !io::stdin().is_terminal() {
        // Read from pipe
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let s = input.trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    } else {
        None // interactive mode
    };

    // Set up project root
    let project_root = cli.project.canonicalize()?;
    tools::set_workspace_root(&project_root);
    let config = config::load_config(&project_root)?;

    // Initialize skill cache (scan once, serve from memory thereafter)
    skills::init_skills(&project_root);

    // Debug logging
    if cli.debug {
        debug::enable(&config.memory_dir);
        eprintln!(
            "Debug logging enabled → {}",
            config.memory_dir.join("debug.log").display()
        );
    }

    // Create runtime for async initialization and single-shot mode.
    // For interactive mode, interactive_loop creates its own runtime internally,
    // so we drop this one first to avoid nested runtime errors.
    let rt = tokio::runtime::Runtime::new()?;

    // Initialize storage
    let db_path = config.memory_dir.join("memories.db");
    let mut store = storage::MemoryStore::new(&db_path)?;
    store.init_schema()?;
    rt.block_on(store.init_vector_store(
        &config.embedding_api_base,
        &config.embedding_api_key,
        &config.embedding_model,
    ))?;

    if let Some(query) = user_query {
        // Single-shot mode
        rt.block_on(run_pipeline(
            &query,
            &config,
            &mut store,
            cli.no_memory,
            cli.no_extract,
            cli.manual_extract,
        ))?;
    } else {
        // Interactive REPL — drop the runtime first so interactive_loop
        // can create its own without nesting.
        drop(rt);
        interactive_loop(
            &config,
            &mut store,
            cli.no_memory,
            cli.no_extract,
            cli.manual_extract,
        )?;
    }

    Ok(())
}
