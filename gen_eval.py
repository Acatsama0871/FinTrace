"""
CLI for tool-calling trajectory generation and schema extraction.

Usage:
    python gen_eval.py generate --input data/testset/selected_800.json --output data/trajectory/testset_trajectory --model gpt-5-mini
    python gen_eval.py generate --input data/testset/selected_800.json --output data/trajectory/testset_trajectory --model claude-opus-4-6
    python gen_eval.py extract-schemas --output-dir mcp_tool_call_schema_extract
"""

import asyncio
import json
import re
import sys
from typing import Any

import typer  # type: ignore[import-untyped]
from dotenv import load_dotenv

load_dotenv()

# Handle BaseExceptionGroup for Python < 3.11
if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup  # noqa: F821
else:
    _BaseExceptionGroup = Exception  # type: ignore


_logfire = None


def get_logfire():
    """Lazy-load and configure logfire (deferred to avoid segfault on Python 3.14a5).

    Returns the configured logfire module. Safe to call multiple times — only
    initializes once.
    """
    global _logfire
    if _logfire is None:
        import logfire  # type: ignore[import-untyped]

        logfire.configure()
        _logfire = logfire
    return _logfire


# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

FMP_MCP_URL = "https://financialmodelingprep.com/mcp"

SYSTEM_PROMPT = """\
You are a financial data assistant with access to Financial Modeling Prep (FMP) API tools.

Your job is to solve the user's financial query step by step by deciding whether to call a tool or provide the final answer.

You must strictly follow this protocol on EVERY assistant turn:

1. Before making any function call or giving the final answer, you must output exactly one <think>...</think> block.
2. The <think> block must be concise, factual, and based only on:
    - the user query
- tool outputs already observed
3. The <think> block must follow this exact structure:

<think>
step_goal: ...
known_information: ...
missing_information: ...
decision: call_tool / answer
justification: ...
tool_name: ...              # only if decision=call_tool
tool_argument_plan: ...     # only if decision=call_tool
expected_result: ...        # only if decision=call_tool
answer_plan: ...            # only if decision=answer
</think>

4. After the <think> block:
    - if more information is needed, emit the function call
    - otherwise provide the final answer inside <answer>...</answer>

Additional rules:
- Never skip the <think> block.
- Never put the final answer inside <think>.
- Never fabricate tool results.
- Only use tools when necessary.
- When enough information is available, stop calling tools and answer directly.
- Keep the final answer concise and directly responsive to the user's question.

CRITICAL — Do NOT ask the user for clarification. You must always attempt to answer:
- If the query is ambiguous (e.g., missing a specific ticker, date range, or metric), make a reasonable default assumption (e.g., use the most recent available data, pick the most common interpretation, choose a well-known company as an example) and state your assumption briefly in the answer.
- If a tool call returns an error or empty result, try alternative tools, parameters, or approaches before giving up. Never respond with "Could you provide...?" or "Which ticker did you mean?"
- If you cannot fully answer, provide the best partial answer you can with the data you have, and note what is missing — but do NOT ask the user to supply it.
- Your role is to maximize the completeness of the answer using available tools, not to defer back to the user."""

USER_PROMPT_TEMPLATE = """\
Please answer the following financial question using the available FMP API tools when needed.

Question:
{query}

Remember:
- On every assistant turn, first output exactly one <think> block.
- Then either call a tool or provide the final answer in <answer>...</answer>.
- NEVER ask the user for clarification or additional information. If anything is ambiguous, make a reasonable assumption, state it, and proceed to answer."""

# ---------------------------------------------------------------------------
# Schema conversion utilities
# ---------------------------------------------------------------------------


def mcp_tool_to_raw(tool) -> dict:
    """Convert an MCP tool object to a plain dict."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.inputSchema
        if tool.inputSchema
        else {
            "type": "object",
            "properties": {},
        },
    }


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
            if tool.inputSchema
            else {
                "type": "object",
                "properties": {},
            },
        },
    }


def mcp_tool_to_claude(tool) -> dict:
    """Convert an MCP tool to Anthropic Claude tool format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema
        if tool.inputSchema
        else {
            "type": "object",
            "properties": {},
        },
    }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def is_client_request_error(exc: Exception) -> bool:
    """Check if an exception is a client-side request error (4xx HTTP or connection issue)."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if isinstance(exc, APIStatusError) and 400 <= exc.status_code < 500:
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    error_msg = str(exc).lower()
    if any(
        keyword in error_msg
        for keyword in [
            "rate limit",
            "429",
            "400",
            "401",
            "403",
            "timeout",
            "connection",
        ]
    ):
        return True
    return False


def clean_step_goal(text: str) -> str:
    """Remove 'step_goal:' / 'step goal:' / 'Step Goal:' prefixes from text."""
    return re.sub(r"(?i)step[_ ]goal\s*:\s*", "", text).strip()


def extract_think_tags(text: str) -> tuple[str, str]:
    """Extract content between <think>...</think> and return (reasoning, cleaned_text)."""
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(text)
    reasoning = "\n".join(m.strip() for m in matches) if matches else ""
    reasoning = clean_step_goal(reasoning)
    cleaned = think_pattern.sub("", text).strip()
    return reasoning, cleaned


def extract_answer_tags(text: str) -> str:
    """Extract content inside <answer>...</answer>; if absent, return original text."""
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# LLM rate limit guard — shared across all LLM clients
# ---------------------------------------------------------------------------

_llm_rate_limit_event: asyncio.Event | None = None
_llm_rate_limit_lock: asyncio.Lock | None = None

# Defaults — overridden by CLI options via configure_llm_guard()
_llm_rate_limit_sleep: float = 60
_llm_server_error_sleep: float = 30
_llm_max_retries: int = 3


def configure_llm_guard(
    rate_limit_sleep: float = 60,
    server_error_sleep: float = 30,
    max_retries: int = 3,
) -> None:
    """Set LLM guard parameters from CLI options. Call before any LLM calls."""
    global _llm_rate_limit_sleep, _llm_server_error_sleep, _llm_max_retries
    _llm_rate_limit_sleep = rate_limit_sleep
    _llm_server_error_sleep = server_error_sleep
    _llm_max_retries = max_retries


def _get_llm_rate_limit_primitives() -> tuple[asyncio.Event, asyncio.Lock]:
    """Lazily create the shared rate limit primitives (must be called inside event loop)."""
    global _llm_rate_limit_event, _llm_rate_limit_lock
    if _llm_rate_limit_event is None:
        _llm_rate_limit_event = asyncio.Event()
        _llm_rate_limit_event.set()
    if _llm_rate_limit_lock is None:
        _llm_rate_limit_lock = asyncio.Lock()
    return _llm_rate_limit_event, _llm_rate_limit_lock


def _is_llm_rate_limit(exc: Exception) -> bool:
    """Check if an exception is an LLM API rate limit error."""
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def _is_llm_server_error(exc: Exception) -> bool:
    """Check if an exception is an LLM API server error (500/overload)."""
    msg = str(exc).lower()
    return (
        "500" in msg
        or "internal server error" in msg
        or "overloaded" in msg
        or "529" in msg
    )


def _is_llm_retryable(exc: Exception) -> bool:
    """Check if an LLM error should trigger a global pause + retry."""
    return _is_llm_rate_limit(exc) or _is_llm_server_error(exc)


async def _llm_rate_limit_guard(exc: Exception) -> None:
    """On rate limit or server error, pause all LLM calls globally, sleep, then resume."""
    if not _is_llm_retryable(exc):
        return
    sleep_time = (
        _llm_rate_limit_sleep if _is_llm_rate_limit(exc) else _llm_server_error_sleep
    )
    label = "RATE LIMIT" if _is_llm_rate_limit(exc) else "SERVER ERROR"
    event, lock = _get_llm_rate_limit_primitives()
    async with lock:
        if event.is_set():
            event.clear()
            print(f"  [LLM {label}] Pausing all LLM calls for {sleep_time}s...")
            await asyncio.sleep(sleep_time)
            event.set()
            print(f"  [LLM {label}] Resuming LLM calls.")


# ---------------------------------------------------------------------------
# LLM client — unified via LiteLLM
# ---------------------------------------------------------------------------


def _is_openai_model(model: str) -> bool:
    """Check if a model name refers to a native OpenAI model (gpt-*, o1-*, etc.).

    Models routed through OpenAI-compatible endpoints (e.g. HuggingFace router
    via ``openai/``) are excluded — they use LiteLLM with ``api_base`` instead.
    """
    # openai/ prefix with a slash in the remaining name indicates a third-party
    # model served via an OpenAI-compatible endpoint (e.g. openai/Qwen/Qwen3.5-27B:novita)
    if model.startswith("openai/") and "/" in model[len("openai/"):]:
        return False
    prefixes = ("gpt-", "o1-", "o3-", "o4-", "openai/")
    return any(model.startswith(p) for p in prefixes)


class LiteLLMClient:
    """LLM client using LiteLLM for multi-provider support.

    For OpenAI models, uses the Responses API directly (supports >128 tools).
    For other providers, uses LiteLLM which handles format conversion internally.
    """

    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self._logfire = get_logfire()
        self._logfire.instrument_litellm()
        self._use_responses_api = _is_openai_model(model)
        if self._use_responses_api:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI()

    # -- Responses API helpers (for OpenAI models with >128 tools) --

    @staticmethod
    def _tools_to_responses_format(tools: list[dict]) -> list[dict]:
        """Convert Chat Completions tool format to Responses API format."""
        resp_tools = []
        for t in tools:
            func = t.get("function", t)
            resp_tools.append(
                {
                    "type": "function",
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return resp_tools

    @staticmethod
    def _messages_to_responses_input(messages: list[dict]) -> list[dict]:
        """Convert Chat Completions message format to Responses API input format."""
        input_items: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role in ("system", "user"):
                input_items.append({"role": role, "content": msg["content"]})
            elif role == "assistant":
                if msg.get("content"):
                    input_items.append(
                        {"role": "assistant", "content": msg["content"]}
                    )
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", tc)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", tc.get("call_id", "")),
                            "name": func["name"],
                            "arguments": func.get("arguments", "{}"),
                        }
                    )
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg["tool_call_id"],
                        "output": msg["content"],
                    }
                )
        return input_items

    async def _chat_completion_responses_api(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Call OpenAI Responses API (supports >128 tools) and return normalized response."""
        resp_tools = self._tools_to_responses_format(tools) if tools else []
        input_items = self._messages_to_responses_input(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if resp_tools:
            kwargs["tools"] = resp_tools
            kwargs["tool_choice"] = tool_choice

        response = await self._openai_client.responses.create(**kwargs)

        # Parse response output items
        tool_calls = []
        text_parts = []
        for item in response.output:
            if item.type == "message":
                if getattr(item, "content", None):
                    for block in item.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
            elif item.type == "function_call":
                tool_calls.append(
                    {
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                )

        assistant_text = "".join(text_parts)

        # Build raw_message in Chat Completions format for conversation history
        raw_msg: dict = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            raw_msg["tool_calls"] = [
                {
                    "id": tc["call_id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]

        return {
            "tool_calls": tool_calls,
            "assistant_message": assistant_text,
            "raw_message": raw_msg,
        }

    # -- LiteLLM path (for non-OpenAI models) --

    async def _chat_completion_litellm(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Call LLM via LiteLLM and return a normalized response."""
        import litellm

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = await litellm.acompletion(**kwargs)

        # Normalize response to our standard format
        choice = response.choices[0]  # type: ignore
        msg = choice.message

        assistant_text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        raw_msg: dict = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            raw_msg["tool_calls"] = [
                {
                    "id": tc["call_id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]

        return {
            "tool_calls": tool_calls,
            "assistant_message": assistant_text,
            "raw_message": raw_msg,
        }

    # -- Public interface --

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Call LLM and return a normalized response."""
        retries = 0
        while True:
            event, _ = _get_llm_rate_limit_primitives()
            await event.wait()

            with self._logfire.span(
                "llm_chat_completion",
                model=self.model,
                tool_choice=tool_choice,
                num_tools=len(tools),
            ):
                try:
                    if self._use_responses_api:
                        return await self._chat_completion_responses_api(
                            messages, tools, tool_choice
                        )
                    else:
                        return await self._chat_completion_litellm(
                            messages, tools, tool_choice
                        )
                except Exception as exc:
                    if _is_llm_retryable(exc):
                        retries += 1
                        if retries > _llm_max_retries:
                            raise
                        await _llm_rate_limit_guard(exc)
                        continue
                    raise


# ---------------------------------------------------------------------------
# MCP infrastructure — single long-lived session with concurrent calls
# ---------------------------------------------------------------------------


class MCPConnection:
    """Single long-lived MCP session supporting concurrent tool calls.

    The MCP protocol multiplexes concurrent requests via JSON-RPC message IDs,
    so a single session can handle multiple in-flight call_tool() coroutines.
    On failure, the session is automatically reconnected.
    """

    # Error messages that indicate tool access is denied (paid tier) — don't retry these
    _ACCESS_DENIED_KEYWORDS = [
        "access denied",
        "unauthorized",
        "forbidden",
        "403",
        "not available",
        "permission",
    ]

    def __init__(
        self,
        mcp_url: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        rate_limit_sleep: float = 120.0,
    ):
        self.mcp_url = mcp_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_sleep = rate_limit_sleep
        self._session: Any = None  # mcp.ClientSession (deferred import)
        self._http_ctx: Any = None
        self._session_ctx: Any = None
        self._tools: list[dict] = []
        self._raw_mcp_tools: list = []
        self._logfire = get_logfire()
        # Global rate limit pause: all call_tool() coroutines wait on this event
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # starts open (not rate limited)
        self._rate_limit_lock = asyncio.Lock()

    async def start(self) -> list[dict]:
        """Open the MCP connection and return OpenAI-formatted tool definitions."""
        await self._connect()
        all_tools = []
        cursor = None
        while True:
            tools_result = await self._session.list_tools(cursor=cursor)
            all_tools.extend(tools_result.tools)
            if not tools_result.nextCursor:
                break
            cursor = tools_result.nextCursor
        self._tools = [mcp_tool_to_openai(t) for t in all_tools]
        self._raw_mcp_tools = all_tools
        print(f"MCP connection started. {len(self._tools)} tools available.")
        return self._tools

    def get_tools(self) -> list[dict]:
        """Return cached OpenAI-formatted tool definitions."""
        return self._tools

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit error (429 or similar)."""
        msg = str(exc).lower()
        return any(kw in msg for kw in ["rate limit", "429", "too many requests"])

    def _is_access_denied_error(self, exc: Exception) -> bool:
        """Check if an exception indicates the tool is not accessible (paid tier)."""
        msg = str(exc).lower()
        return any(kw in msg for kw in self._ACCESS_DENIED_KEYWORDS)

    async def _handle_rate_limit(self) -> None:
        """Pause all coroutines globally, sleep for rate_limit_sleep, then resume."""

        async with self._rate_limit_lock:
            if self._rate_limit_event.is_set():
                # First coroutine to detect rate limit — trigger the pause
                self._rate_limit_event.clear()
                print(
                    f"  [RATE LIMIT] Pausing all MCP calls for {self.rate_limit_sleep}s..."
                )
                await asyncio.sleep(self.rate_limit_sleep)
                self._rate_limit_event.set()
                print("  [RATE LIMIT] Resuming MCP calls.")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP tool with retry, rate limit handling, and auto-reconnection."""

        clean_args = {k: v for k, v in arguments.items() if v is not None and v != ""}

        for attempt in range(1, self.max_retries + 1):
            # Block if a global rate limit pause is active
            await self._rate_limit_event.wait()

            if self._session is None:
                await self._reconnect()

            with self._logfire.span(
                "mcp_call_tool",
                tool_name=name,
                attempt=attempt,
            ):
                try:
                    result = await self._session.call_tool(name, clean_args)
                    parts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                        else:
                            parts.append(str(block))
                    result_str = "\n".join(parts)
                    if len(result_str) > 4000:
                        result_str = result_str[:4000] + "...[truncated]"
                    return result_str

                except Exception as exc:
                    # Access denied — don't retry, return error for LLM to see
                    if self._is_access_denied_error(exc):
                        print(f"  [ACCESS DENIED] {name}: {exc}")
                        return json.dumps(
                            {"error": f"Tool '{name}' is not accessible: {exc}"}
                        )

                    # Rate limit — pause globally, don't count as failed attempt
                    if self._is_rate_limit_error(exc):
                        await self._handle_rate_limit()
                        continue  # retry without incrementing attempt

                    # Other errors — reconnect and retry
                    print(
                        f"  [MCP RETRY {attempt}/{self.max_retries}] {name} failed: {exc}"
                    )
                    self._session = None
                    if attempt < self.max_retries:
                        await self._reconnect()
                        await asyncio.sleep(self.retry_delay)
                    else:
                        raise RuntimeError(
                            f"MCP call {name} failed after {self.max_retries} retries"
                        ) from exc

        raise RuntimeError(f"MCP call {name} failed: no retries configured")

    async def close(self):
        """Tear down the session and transport gracefully."""

        for ctx in [self._session_ctx, self._http_ctx]:
            if ctx is not None:
                try:
                    await asyncio.wait_for(ctx.__aexit__(None, None, None), timeout=5)
                except BaseException:
                    pass
        self._session = None
        self._session_ctx = None
        self._http_ctx = None

    async def _connect(self):
        """Open transport + session + initialize."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        self._http_ctx = streamable_http_client(url=self.mcp_url)
        read_stream, write_stream, _ = await self._http_ctx.__aenter__()
        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _reconnect(self):
        """Close the current session (if any) and open a fresh one."""
        with self._logfire.span("mcp_reconnect"):
            await self.close()
            await self._connect()
            print("MCP session reconnected.")


# ---------------------------------------------------------------------------
# Trajectory generation core
# ---------------------------------------------------------------------------


async def execute_tool_calls_parallel(
    conn: MCPConnection,
    tool_calls: list[dict],
    semaphore: "asyncio.Semaphore",
) -> list[dict]:
    """Execute multiple tool calls in parallel on a single MCP session.

    Returns a list of tool message dicts for chat completion history.
    """

    async def _execute_one(tc: dict) -> dict:
        async with semaphore:
            try:
                args = json.loads(tc["arguments"])
                result = await conn.call_tool(tc["name"], args)
            except Exception as exc:
                result = json.dumps({"error": str(exc)}, ensure_ascii=False)
            return {
                "role": "tool",
                "tool_call_id": tc["call_id"],
                "content": result,
            }

    results = await asyncio.gather(*[_execute_one(tc) for tc in tool_calls])
    return list(results)


def trajectory_to_messages(
    trajectory: list[dict],
    system_prompt: str,
    user_content: str,
) -> list[dict]:
    """Convert trajectory steps into a standard OpenAI chat-format messages list.

    Output follows the OpenAI Chat Completions message schema:
        system -> user -> assistant(tool_calls) -> tool(result) -> ... -> assistant(final_answer)
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for step in trajectory:
        tool_calls_in_step = [
            item for item in step.get("output", []) if item["type"] == "function_call"
        ]
        tool_outputs_in_step = [
            item
            for item in step.get("output", [])
            if item["type"] == "function_call_output"
        ]
        text_messages = [
            item
            for item in step.get("output", [])
            if item["type"] == "message" and item.get("content")
        ]

        if tool_calls_in_step:
            assistant_msg: dict = {
                "role": "assistant",
                "content": step.get("reasoning") or None,
                "tool_calls": [
                    {
                        "id": tc["call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_in_step
                ],
            }
            messages.append(assistant_msg)

            output_by_id = {o["call_id"]: o["output"] for o in tool_outputs_in_step}
            for tc in tool_calls_in_step:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["call_id"],
                        "content": output_by_id.get(tc["call_id"], ""),
                    }
                )

        for msg in text_messages:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg["content"],
                }
            )

    return messages


async def generate_trajectory(
    query: str,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LiteLLMClient,
    semaphore: "asyncio.Semaphore",
    max_turns: int = 10,
    max_tool_calls: int = 30,
) -> dict:
    """Run an iterative tool-calling conversation for a single query.

    Returns a dict with keys:
        - "trajectory": list of step dicts
        - "messages": list of chat-format messages (OpenAI Chat Completions schema)
    """
    logfire = get_logfire()
    user_content = USER_PROMPT_TEMPLATE.format(query=query)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    trajectory: list[dict] = []
    has_tool_call = False
    total_tool_calls = 0

    with logfire.span("generate_trajectory", query=query[:100], max_turns=max_turns):
        for turn in range(max_turns):
            response = await llm_client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            raw_text = response["assistant_message"]
            step_reasoning, cleaned_text = extract_think_tags(raw_text)

            step: dict = {
                "turn": turn,
                "reasoning": step_reasoning,
                "endpoints_called": [],
                "output": [],
            }

            has_tool_call = bool(response["tool_calls"])
            pending_tool_calls = []

            for tc in response["tool_calls"]:
                fc_item = {
                    "type": "function_call",
                    "call_id": tc["call_id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                step["output"].append(fc_item)
                step["endpoints_called"].append(tc["name"])
                pending_tool_calls.append(tc)

            # Execute tool calls in parallel
            if pending_tool_calls:
                total_tool_calls += len(pending_tool_calls)
                tool_call_outputs = await execute_tool_calls_parallel(
                    conn,
                    pending_tool_calls,
                    semaphore,
                )
                for fc_output in tool_call_outputs:
                    step["output"].append(
                        {
                            "type": "function_call_output",
                            "call_id": fc_output["tool_call_id"],
                            "output": fc_output["content"],
                        }
                    )
            else:
                tool_call_outputs = []

            if cleaned_text:
                step["output"].append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": cleaned_text,
                    }
                )

            trajectory.append(step)

            if not has_tool_call:
                break

            # Check max tool calls limit
            if total_tool_calls >= max_tool_calls:
                print(
                    f"  [MAX TOOL CALLS] Reached {total_tool_calls}/{max_tool_calls}, forcing stop."
                )
                break

            messages.append(response["raw_message"])
            messages.extend(tool_call_outputs)

        # If last turn ended on tool calls, force a final summary
        if has_tool_call:
            response = await llm_client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="none",
            )

            raw_text = response["assistant_message"]
            step_reasoning, cleaned_text = extract_think_tags(raw_text)

            final_step: dict = {
                "turn": len(trajectory),
                "reasoning": step_reasoning,
                "output": [],
            }
            if cleaned_text:
                final_step["output"].append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": cleaned_text,
                    }
                )
            trajectory.append(final_step)

    chat_messages = trajectory_to_messages(trajectory, SYSTEM_PROMPT, user_content)
    return {"trajectory": trajectory, "messages": chat_messages}


# ---------------------------------------------------------------------------
# Orchestration & saving
# ---------------------------------------------------------------------------


def _build_record(
    query_item: dict,
    trajectory: list[dict],
    chat_messages: list[dict],
) -> dict:
    """Extract results from a trajectory and build a record dict for saving."""
    final_answer = ""
    final_reasoning = ""
    if trajectory:
        for out_item in reversed(trajectory[-1]["output"]):
            if out_item.get("type") == "message" and out_item.get("content"):
                final_answer = extract_answer_tags(out_item["content"])
                break
        final_reasoning = trajectory[-1].get("reasoning", "")

    all_endpoints_called = []
    for step in trajectory:
        all_endpoints_called.extend(step.get("endpoints_called", []))
    unique_endpoints_called = list(dict.fromkeys(all_endpoints_called))

    return {
        "id": query_item["id"],
        "source_query": query_item["source_query"],
        "task_type": query_item.get("task_type", ""),
        "resource": query_item.get("resource", ""),
        "source": query_item.get("source", ""),
        "difficulty": query_item.get("difficulty", ""),
        "reference_answer": query_item.get("answer", ""),
        "endpoints_called": unique_endpoints_called,
        "trajectory": trajectory,
        "messages": chat_messages,
        "final_answer": final_answer,
        "reasoning": final_reasoning,
    }


async def _save_record(
    record: dict,
    all_records: list[dict],
    save_lock: asyncio.Lock,
    output_path: Any,
) -> None:
    """Append record to all_records and write the sorted list to disk."""
    async with save_lock:
        all_records.append(record)
        sorted_records = sorted(all_records, key=lambda r: r["id"])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_records, f, indent=2, ensure_ascii=False)


async def process_single_query(
    query_item: dict,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LiteLLMClient,
    mcp_semaphore: asyncio.Semaphore,
    concurrency_semaphore: asyncio.Semaphore,
    max_turns: int,
    max_tool_calls: int,
    max_retries: int,
    retry_delay: float,
    client_error_sleep: float,
    all_records: list[dict],
    save_lock: asyncio.Lock,
    output_path: Any,
    pbar: Any,
) -> None:
    """Process a single query with retry logic and incremental saving."""
    import traceback

    logfire = get_logfire()
    idx = query_item["id"]

    try:
        async with concurrency_semaphore:
            query = query_item["source_query"]

            with logfire.span("process_query", query_id=idx, query=query[:100]):
                for attempt in range(1, max_retries + 1):
                    try:
                        gen_result = await generate_trajectory(
                            query,
                            conn,
                            tools,
                            llm_client,
                            mcp_semaphore,
                            max_turns=max_turns,
                            max_tool_calls=max_tool_calls,
                        )
                        record = _build_record(
                            query_item, gen_result["trajectory"], gen_result["messages"]
                        )
                        await _save_record(record, all_records, save_lock, output_path)
                        pbar.update(1)
                        pbar.set_postfix(saved=len(all_records))
                        return

                    except Exception as exc:
                        print(
                            f"[ERROR] query {idx} attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()

                        if is_client_request_error(exc):
                            print(
                                f"  [CLIENT ERROR] query {idx}: sleeping {client_error_sleep}s..."
                            )
                            await asyncio.sleep(client_error_sleep)
                            try:
                                gen_result = await generate_trajectory(
                                    query,
                                    conn,
                                    tools,
                                    llm_client,
                                    mcp_semaphore,
                                    max_turns=max_turns,
                                    max_tool_calls=max_tool_calls,
                                )
                                record = _build_record(
                                    query_item,
                                    gen_result["trajectory"],
                                    gen_result["messages"],
                                )
                                await _save_record(
                                    record, all_records, save_lock, output_path
                                )
                                pbar.update(1)
                                pbar.set_postfix(saved=len(all_records))
                            except Exception as retry_exc:
                                print(
                                    f"  [SKIP] query {idx} still failed: {type(retry_exc).__name__}: {retry_exc}"
                                )
                            return

                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                        else:
                            print(
                                f"  query {idx} failed after {max_retries} retries — skipping"
                            )
                            pbar.update(1)

    except (asyncio.CancelledError, _BaseExceptionGroup) as exc:  # type: ignore
        print(f"  [WARN] query {idx} cancelled: {type(exc).__name__}")
        pbar.update(1)


async def process_testset(
    input_path: str,
    output_dir: str,
    model_name: str,
    num_queries: int,
    max_turns: int,
    max_tool_calls: int,
    concurrency: int,
    mcp_concurrency: int,
    max_retries: int,
    retry_delay: int,
    client_error_sleep: int,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    """Load queries, generate trajectories, and save results."""
    import os
    from pathlib import Path

    from tqdm import tqdm

    logfire = get_logfire()

    fmp_api_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_api_key:
        raise ValueError("Set the FMP_API_KEY environment variable before running.")

    mcp_url = f"{FMP_MCP_URL}?apikey={fmp_api_key}"

    # Create LLM client
    llm_client = LiteLLMClient(model=model_name, api_base=api_base, api_key=api_key)
    print(f"Using model: {model_name}")

    # Load queries
    with open(input_path, "r", encoding="utf-8") as f:
        query_items: list[dict] = json.load(f)
    if num_queries > 0:
        query_items = query_items[:num_queries]
    total = len(query_items)
    print(f"Loaded {total} queries from {input_path}")

    # Output path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_stem = Path(input_path).stem
    model_slug = model_name.replace("/", "_")
    output_name = f"{input_stem}_{model_slug}_trajectories.json"
    output_path = out / output_name

    # Resume: load existing records
    all_records: list[dict] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
            print(
                f"Resumed: loaded {len(all_records)} existing records from {output_path}"
            )
        except (json.JSONDecodeError, Exception) as exc:
            print(f"Warning: could not load existing records ({exc}), starting fresh.")
            all_records = []

    # Filter already-processed queries
    processed_ids = {r["id"] for r in all_records}
    remaining = [item for item in query_items if item["id"] not in processed_ids]
    if not remaining:
        print("All queries already processed.")
        return

    print(f"{len(remaining)} queries remaining.")

    # Semaphores
    concurrency_semaphore = asyncio.Semaphore(concurrency)
    mcp_semaphore = asyncio.Semaphore(mcp_concurrency)
    save_lock = asyncio.Lock()

    # Connect to MCP
    conn = MCPConnection(
        mcp_url=mcp_url,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_sleep=client_error_sleep,
    )

    try:
        tools = await conn.start()

        with logfire.span(
            "process_testset", model=model_name, total=total, remaining=len(remaining)
        ):
            with tqdm(total=len(remaining), desc="Generating trajectories") as pbar:
                tasks = [
                    process_single_query(
                        item,
                        conn,
                        tools,
                        llm_client,
                        mcp_semaphore,
                        concurrency_semaphore,
                        max_turns,
                        max_tool_calls,
                        max_retries,
                        retry_delay,
                        client_error_sleep,
                        all_records,
                        save_lock,
                        output_path,
                        pbar,
                    )
                    for item in remaining
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

        # Save tool definitions
        tools_path = out / "tools.json"
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=2, ensure_ascii=False)
        print(f"Tool definitions saved to {tools_path}")

    finally:
        await conn.close()

    print(f"\nDone. {len(all_records)}/{total} trajectories saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Tool-calling trajectory generation and evaluation pipeline.")


@app.command()
def generate(
    input: str = typer.Option(
        "data/testset/selected_800.json",
        "--input",
        "-i",
        help="Path to the JSON test set file.",
    ),
    output: str = typer.Option(
        "data/trajectory/testset_trajectory",
        "--output",
        "-o",
        help="Directory to save trajectory results.",
    ),
    model: str = typer.Option(
        "gpt-5-mini",
        "--model",
        "-m",
        help="Model name (e.g. gpt-5-mini, claude-opus-4-6, openai/Qwen/Qwen3.5-27B:novita).",
    ),
    api_base: str = typer.Option(
        None,
        "--api-base",
        help="Custom API base URL for OpenAI-compatible endpoints (e.g. https://router.huggingface.co/v1).",
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        help="API key for custom endpoint. Can also use env vars (e.g. HF_TOKEN).",
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all)."
    ),
    max_turns: int = typer.Option(
        6, "--max-turns", help="Max agentic loop turns per query."
    ),
    max_tool_calls: int = typer.Option(
        30,
        "--max-tool-calls",
        help="Max total tool calls per query (prevents confused LLMs from burning rate limit).",
    ),
    concurrency: int = typer.Option(
        30, "--concurrency", help="Max concurrent query tasks."
    ),
    mcp_concurrency: int = typer.Option(
        60,
        "--mcp-concurrency",
        help="Max concurrent in-flight MCP calls (rate limit control).",
    ),
    max_retries: int = typer.Option(
        2, "--max-retries", help="Max retries per query on failure."
    ),
    retry_delay: int = typer.Option(
        10, "--retry-delay", help="Seconds to wait between retries."
    ),
    client_error_sleep: int = typer.Option(
        120,
        "--client-error-sleep",
        help="Seconds to sleep on client-side errors (4xx/rate limit) before final retry.",
    ),
    llm_rate_limit_sleep: int = typer.Option(
        60,
        "--llm-rate-limit-sleep",
        help="Seconds to pause all LLM calls on 429 rate limit.",
    ),
    llm_server_error_sleep: int = typer.Option(
        30,
        "--llm-server-error-sleep",
        help="Seconds to pause all LLM calls on 500/overload.",
    ),
    llm_max_retries: int = typer.Option(
        3,
        "--llm-max-retries",
        help="Max retries per LLM call on rate limit/server errors.",
    ),
) -> None:
    """Generate tool-calling trajectories for a test set using a given LLM."""
    configure_llm_guard(
        rate_limit_sleep=llm_rate_limit_sleep,
        server_error_sleep=llm_server_error_sleep,
        max_retries=llm_max_retries,
    )
    asyncio.run(
        process_testset(
            input_path=input,
            output_dir=output,
            model_name=model,
            num_queries=num_queries,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            concurrency=concurrency,
            mcp_concurrency=mcp_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
            client_error_sleep=client_error_sleep,
            api_base=api_base,
            api_key=api_key,
        )
    )


@app.command()
def extract_schemas(
    output_dir: str = typer.Option(
        "mcp_tool_call_schema_extract",
        "--output-dir",
        "-o",
        help="Directory to save schema JSON files.",
    ),
) -> None:
    """Connect to the FMP MCP server and export tool schemas (raw, OpenAI, Claude formats)."""
    import os
    from pathlib import Path

    async def _extract():
        fmp_api_key = os.environ.get("FMP_API_KEY", "")
        if not fmp_api_key:
            raise ValueError("Set the FMP_API_KEY environment variable before running.")

        mcp_url = f"{FMP_MCP_URL}?apikey={fmp_api_key}"
        conn = MCPConnection(mcp_url=mcp_url)
        try:
            await conn.start()
            raw_tools = conn._raw_mcp_tools

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            raw_schemas = [mcp_tool_to_raw(t) for t in raw_tools]
            openai_schemas = [mcp_tool_to_openai(t) for t in raw_tools]
            claude_schemas = [mcp_tool_to_claude(t) for t in raw_tools]

            for filename, data in [
                ("tool_call_schemas.json", raw_schemas),
                ("tool_call_schemas_openai.json", openai_schemas),
                ("tool_call_schemas_claude.json", claude_schemas),
            ]:
                path = out / filename
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                print(f"Saved {filename} ({len(data)} tools)")
        finally:
            await conn.close()

    asyncio.run(_extract())


if __name__ == "__main__":
    app()
