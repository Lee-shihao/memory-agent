"""Extractor: post-conversation memory extraction with user review."""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
import httpx
from memory_agent.config import Config
from memory_agent.storage import MemoryStore
from memory_agent.prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_TEMPLATE


@dataclass
class ExtractionResult:
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            summary=data.get("summary", ""), key_points=data.get("key_points", []),
            tags=data.get("tags", []), entities=data.get("entities", []),
            decisions=data.get("decisions", []),
        )


def _call_extraction_llm(config: Config, transcript: str) -> ExtractionResult:
    response = httpx.post(
        f"{config.llm_api_base}/chat/completions",
        headers={"Authorization": f"Bearer {config.llm_api_key}", "Content-Type": "application/json"},
        json={
            "model": config.llm_model,
            "messages": [
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTOR_USER_TEMPLATE.format(transcript=transcript)},
            ],
            "temperature": 0.3, "max_tokens": 1000,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return ExtractionResult.from_dict(json.loads(content))


def _display_preview(result: ExtractionResult) -> None:
    print("\n" + "=" * 50)
    print("📝 Memory Preview")
    print("=" * 50)
    print(f"\nSummary: {result.summary}")
    print(f"\nTags: {', '.join(result.tags) or '(none)'}")
    print(f"\nKey Points:")
    for kp in result.key_points:
        print(f"  • {kp}")
    if not result.key_points:
        print("  (none)")
    print(f"\nEntities:")
    for ent in result.entities:
        print(f"  • {ent['name']} ({ent['type']}): {ent.get('description', '')}")
    if not result.entities:
        print("  (none)")
    print(f"\nDecisions:")
    for dec in result.decisions:
        print(f"  • {dec}")
    if not result.decisions:
        print("  (none)")
    print()


def _get_user_choice() -> str:
    while True:
        choice = input("[S]ave  [E]dit  [D]iscard: ").strip().lower()
        if choice in ("s", "save", "y", "yes"): return "save"
        elif choice in ("d", "discard", "n", "no"): return "discard"
        elif choice in ("e", "edit"): return "edit"
        else: print("Please enter S, E, or D")


def _open_editor(result: ExtractionResult) -> ExtractionResult:
    import subprocess, tempfile, os
    editor = os.environ.get("EDITOR", "vim")
    data = {"summary": result.summary, "key_points": result.key_points, "tags": result.tags, "entities": result.entities, "decisions": result.decisions}
    content = json.dumps(data, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path) as f:
            edited = json.loads(f.read())
        return ExtractionResult.from_dict(edited)
    finally:
        os.unlink(tmp_path)


def extract_and_store(transcript: str, config: Config, store: MemoryStore) -> bool:
    print("\nExtracting memory from conversation...")
    try:
        result = _call_extraction_llm(config, transcript)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    if config.extractor_auto_confirm:
        _store_result(result, transcript, config, store)
        print("Memory saved (auto-confirm).")
        return True

    while True:
        _display_preview(result)
        choice = _get_user_choice()
        if choice == "save":
            _store_result(result, transcript, config, store)
            print("Memory saved.")
            return True
        elif choice == "edit":
            result = _open_editor(result)
        elif choice == "discard":
            print("Memory discarded.")
            return False


def _store_result(result: ExtractionResult, transcript: str, config: Config, store: MemoryStore) -> None:
    now = datetime.now(timezone.utc)
    chroma_dir = config.memory_dir / "chroma"
    if not hasattr(store, "_chroma_collection") or store._chroma_collection is None:
        store.init_chroma(persist_dir=chroma_dir, embedding_api_base=config.embedding_api_base, embedding_api_key=config.embedding_api_key, embedding_model=config.embedding_model)
    embedding_text = result.summary
    if result.key_points:
        embedding_text += "\n" + "\n".join(result.key_points)
    memory_id = uuid.uuid4().hex[:12]
    chroma_doc_id = store.add_to_chroma(memory_id=memory_id, text=embedding_text, metadata={"tags": ",".join(result.tags), "conversation_at": now.isoformat()})
    conversation_json = json.dumps({"transcript": transcript}) if config.extractor_keep_full_transcript else None
    memory_id = store.insert_memory(
        summary=result.summary, conversation_at=now, conversation_json=conversation_json,
        chroma_doc_id=chroma_doc_id, key_points=result.key_points, tags=result.tags,
        entities=result.entities, decisions=result.decisions,
        memory_id=memory_id,
    )
    print(f"  Memory ID: {memory_id}")
