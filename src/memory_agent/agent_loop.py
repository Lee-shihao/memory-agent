"""Agent loop: OpenAI-compatible API with tool calling iteration."""
import json
from collections.abc import Callable

import httpx

from memory_agent.config import Config
from memory_agent.debug import is_enabled as _debug_enabled
from memory_agent.debug import log_request as _log_req
from memory_agent.debug import log_response as _log_resp
from memory_agent.prompts import ROUND_1_SYSTEM_PROMPT, ROUND_2_PLUS_PROMPT
from memory_agent.tools import TOOL_DEFINITIONS, execute_tool

# Callback: (tool_name, arguments) -> (allowed: bool, feedback: str)
# - allowed=True  → execute tool, append feedback (if any) to result
# - allowed=False → skip tool, use feedback as tool result
ConfirmCallback = Callable[[str, dict], tuple[bool, str]]


def run_agent_loop(
    config: Config,
    user_query: str,
    tools: list[dict] | None = None,
    max_iterations: int = 50,
    confirm_callback: ConfirmCallback | None = None,
) -> str:
    if tools is None:
        tools = TOOL_DEFINITIONS

    # Start with Round 1 prompt; it will be updated to Round 2+ after first iteration
    system_content = ROUND_1_SYSTEM_PROMPT

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query},
    ]

    transcript_parts = [f"User: {user_query}"]

    for iteration in range(max_iterations):
        # --- dynamic prompt switching ---
        if iteration >= 1:
            messages[0]["content"] = ROUND_1_SYSTEM_PROMPT + "\n\n" + ROUND_2_PLUS_PROMPT

        url = f"{config.llm_api_base}/chat/completions"
        req_body = {
            "model": config.llm_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        req_headers = {
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json",
        }
        if _debug_enabled():
            rid = _log_req("agent_loop", "POST", url, req_headers, req_body)

        try:
            response = httpx.post(url, headers=req_headers, json=req_body, timeout=120)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            transcript_parts.append(
                f"Assistant: [API error {e.response.status_code}] "
                f"LLM returned an error — please check your API key and network, then retry."
            )
            return "\n\n".join(transcript_parts)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
            transcript_parts.append(
                f"Assistant: [Connection error] "
                f"Cannot reach {config.llm_api_base} — {e}"
            )
            return "\n\n".join(transcript_parts)

        data = response.json()
        if _debug_enabled():
            _log_resp(rid, response.status_code, data)
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
