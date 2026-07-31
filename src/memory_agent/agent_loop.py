"""Agent loop: OpenAI-compatible API with tool calling iteration."""
import json
from collections.abc import Callable

import httpx

from memory_agent.config import Config
from memory_agent.prompts import BASE_AGENT_SYSTEM_PROMPT
from memory_agent.tools import TOOL_DEFINITIONS, execute_tool

# Callback: (tool_name, arguments) -> (allowed: bool, feedback: str)
# - allowed=True  → execute tool, append feedback (if any) to result
# - allowed=False → skip tool, use feedback as tool result
ConfirmCallback = Callable[[str, dict], tuple[bool, str]]


def run_agent_loop(
    config: Config,
    user_query: str,
    memory_context: str,
    tools: list[dict] | None = None,
    max_iterations: int = 50,
    confirm_callback: ConfirmCallback | None = None,
) -> str:
    if tools is None:
        tools = TOOL_DEFINITIONS

    system_content = BASE_AGENT_SYSTEM_PROMPT
    if memory_context:
        system_content += "\n\n" + memory_context

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    transcript_parts = [f"User: {user_query}"]

    for _ in range(max_iterations):
        response = httpx.post(
            f"{config.llm_api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.llm_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Record assistant message with tool call requests
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                # --- confirmation hook ---
                if confirm_callback is not None:
                    allowed, feedback = confirm_callback(tool_name, args)
                else:
                    allowed, feedback = True, ""

                if allowed:
                    tool_result = execute_tool(tool_name, args)
                    if feedback:
                        tool_result += "\n\n[User note: " + feedback + "]"
                else:
                    tool_result = "[Blocked by user]" + (f" {feedback}" if feedback else "")

                transcript_parts.append(
                    f"Tool [{tool_name}]: {tc['function']['arguments']}\nResult: {tool_result[:500]}"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
        else:
            assistant_content = message.get("content", "")
            transcript_parts.append(f"Assistant: {assistant_content}")
            return "\n\n".join(transcript_parts)

    transcript_parts.append("[Max tool call iterations reached]")
    return "\n\n".join(transcript_parts)
