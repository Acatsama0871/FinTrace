"""
Standalone trajectory generation for models via OpenAI-compatible endpoints.

Supports:
  - Qwen3.5-27B via HuggingFace Router (default)
  - DeepSeek-R1/V3 via DeepSeek API (captures reasoning_content)
  - Any OpenAI-compatible endpoint with tool calling support

Usage:
    # Qwen via HuggingFace Router
    python benchmark/generate_openai_compatible.py generate
    
    # DeepSeek via DeepSeek API
    python benchmark/generate_openai_compatible.py generate --model deepseek-reasoner --api-base https://api.deepseek.com
"""

import asyncio
import json
import os
import re
import sys
from typing import Any

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Logfire — lazy-loaded to avoid segfault on Python 3.14a5
# ---------------------------------------------------------------------------

_logfire = None


def get_logfire():
    """Lazy-load and configure logfire. Safe to call multiple times."""
    global _logfire
    if _logfire is None:
        import logfire  # type: ignore[import-untyped]

        logfire.configure()
        _logfire = logfire
    return _logfire


# Handle BaseExceptionGroup for Python < 3.11
if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup  # noqa: F821
else:
    _BaseExceptionGroup = Exception  # type: ignore

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MODEL = "qwen/qwen3.5-27b"
API_BASE = "https://api.novita.ai/openai"
FMP_MCP_URL = "https://financialmodelingprep.com/mcp"

SYSTEM_PROMPT = (
    "You are a financial assistant operating in an interactive environment with access to external tools.\n"
    "Your goal is to answer user queries accurately and efficiently.\n"
)

USER_PROMPT_TEMPLATE = """\
Please answer the following financial question using the available FMP API tools when needed.

Question:
{query}
"""

# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema
            if tool.inputSchema
            else {"type": "object", "properties": {}},
        },
    }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_step_goal(text: str) -> str:
    return re.sub(r"(?i)step[_ ]goal\s*:\s*", "", text).strip()


def extract_think_tags(text: str) -> tuple[str, str]:
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(text)
    reasoning = "\n".join(m.strip() for m in matches) if matches else ""
    reasoning = clean_step_goal(reasoning)
    cleaned = think_pattern.sub("", text).strip()
    return reasoning, cleaned


def extract_answer_tags(text: str) -> str:
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# LLM Client — Qwen via HuggingFace Router (OpenAI-compatible)
# ---------------------------------------------------------------------------

# Rate limit guard
_rate_limit_event: asyncio.Event | None = None
_rate_limit_lock: asyncio.Lock | None = None
_rate_limit_sleep: float = 60
_server_error_sleep: float = 30
_max_retries: int = 3


def _get_rate_limit_primitives() -> tuple[asyncio.Event, asyncio.Lock]:
    global _rate_limit_event, _rate_limit_lock
    if _rate_limit_event is None:
        _rate_limit_event = asyncio.Event()
        _rate_limit_event.set()
    if _rate_limit_lock is None:
        _rate_limit_lock = asyncio.Lock()
    return _rate_limit_event, _rate_limit_lock


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def _is_server_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "500" in msg or "internal server error" in msg or "overloaded" in msg or "529" in msg


def _is_retryable(exc: Exception) -> bool:
    return _is_rate_limit(exc) or _is_server_error(exc)


async def _rate_limit_guard(exc: Exception) -> None:
    if not _is_retryable(exc):
        return
    sleep_time = _rate_limit_sleep if _is_rate_limit(exc) else _server_error_sleep
    label = "RATE LIMIT" if _is_rate_limit(exc) else "SERVER ERROR"
    event, lock = _get_rate_limit_primitives()
    async with lock:
        if event.is_set():
            event.clear()
            print(f"  [LLM {label}] Pausing all LLM calls for {sleep_time}s...")
            await asyncio.sleep(sleep_time)
            event.set()
            print(f"  [LLM {label}] Resuming LLM calls.")


class LLMClient:
    """LLM client for OpenAI-compatible endpoints (Qwen, DeepSeek, etc.).

    Captures reasoning_content from DeepSeek-style responses when available.
    """

    def __init__(self, model: str = MODEL, api_base: str = API_BASE, api_key: str | None = None):
        self.model = model
        self._logfire = get_logfire()
        # Auto-detect API key from api_base if not explicitly provided
        if api_key is None:
            if "deepseek" in api_base:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            elif "novita" in api_base:
                api_key = os.environ.get("NOVITA_API_KEY", "")
            else:
                api_key = os.environ.get("HF_TOKEN", "")
        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Call LLM via OpenAI-compatible Chat Completions API.

        Returns dict with keys:
            - tool_calls: list of {call_id, name, arguments}
            - assistant_message: text content from the response
            - reasoning_content: reasoning/thinking content (DeepSeek-style), or ""
            - raw_message: dict suitable for appending to messages history
        """
        retries = 0
        while True:
            event, _ = _get_rate_limit_primitives()
            await event.wait()

            try:
                with self._logfire.span(
                    "llm_chat_completion",
                    model=self.model,
                    tool_choice=tool_choice,
                    num_tools=len(tools),
                ):
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 8192,
                    }
                    if tools and tool_choice != "none":
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = tool_choice

                    response = await self._client.chat.completions.create(**kwargs)

                    choice = response.choices[0]
                    msg = choice.message

                    assistant_text = msg.content or ""

                    # Capture reasoning_content (DeepSeek-R1 style)
                    reasoning_content = getattr(msg, "reasoning_content", None) or ""

                    tool_calls = []
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append({
                                "call_id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            })

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
                        "reasoning_content": reasoning_content,
                        "raw_message": raw_msg,
                    }

            except Exception as exc:
                if _is_retryable(exc):
                    retries += 1
                    if retries > _max_retries:
                        raise
                    await _rate_limit_guard(exc)
                    continue
                raise


# ---------------------------------------------------------------------------
# MCP Connection
# ---------------------------------------------------------------------------

class MCPConnection:
    """Single long-lived MCP session supporting concurrent tool calls."""

    _ACCESS_DENIED_KEYWORDS = [
        "access denied", "unauthorized", "forbidden", "403",
        "not available", "permission",
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
        self._session: Any = None
        self._http_ctx: Any = None
        self._session_ctx: Any = None
        self._tools: list[dict] = []
        self._raw_mcp_tools: list = []
        self._logfire = get_logfire()
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()
        self._rate_limit_lock = asyncio.Lock()

    async def start(self) -> list[dict]:
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
        return self._tools

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(kw in msg for kw in ["rate limit", "429", "too many requests"])

    def _is_access_denied_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(kw in msg for kw in self._ACCESS_DENIED_KEYWORDS)

    async def _handle_rate_limit(self) -> None:
        async with self._rate_limit_lock:
            if self._rate_limit_event.is_set():
                self._rate_limit_event.clear()
                print(f"  [RATE LIMIT] Pausing all MCP calls for {self.rate_limit_sleep}s...")
                await asyncio.sleep(self.rate_limit_sleep)
                self._rate_limit_event.set()
                print("  [RATE LIMIT] Resuming MCP calls.")

    async def call_tool(self, name: str, arguments: dict) -> str:
        clean_args = {k: v for k, v in arguments.items() if v is not None and v != ""}

        for attempt in range(1, self.max_retries + 1):
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
                    if self._is_access_denied_error(exc):
                        print(f"  [ACCESS DENIED] {name}: {exc}")
                        return json.dumps({"error": f"Tool '{name}' is not accessible: {exc}"})

                    if self._is_rate_limit_error(exc):
                        await self._handle_rate_limit()
                        continue

                    print(f"  [MCP RETRY {attempt}/{self.max_retries}] {name} failed: {exc}")
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
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        self._http_ctx = streamable_http_client(url=self.mcp_url)
        read_stream, write_stream, _ = await self._http_ctx.__aenter__()
        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _reconnect(self):
        with self._logfire.span("mcp_reconnect"):
            await self.close()
            await self._connect()
            print("MCP session reconnected.")


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


async def execute_tool_calls_parallel(
    conn: MCPConnection,
    tool_calls: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
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
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for step in trajectory:
        tool_calls_in_step = [
            item for item in step.get("output", []) if item["type"] == "function_call"
        ]
        tool_outputs_in_step = [
            item for item in step.get("output", []) if item["type"] == "function_call_output"
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["call_id"],
                    "content": output_by_id.get(tc["call_id"], ""),
                })

        for msg in text_messages:
            messages.append({"role": "assistant", "content": msg["content"]})

    return messages


async def generate_trajectory(
    query: str,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LLMClient,
    semaphore: asyncio.Semaphore,
    max_turns: int = 10,
    max_tool_calls: int = 30,
) -> dict:
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

            # DeepSeek-style reasoning_content takes priority if present
            ds_reasoning = response.get("reasoning_content", "")
            if ds_reasoning:
                step_reasoning = ds_reasoning

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

            if pending_tool_calls:
                total_tool_calls += len(pending_tool_calls)
                tool_call_outputs = await execute_tool_calls_parallel(
                    conn, pending_tool_calls, semaphore,
                )
                for fc_output in tool_call_outputs:
                    step["output"].append({
                        "type": "function_call_output",
                        "call_id": fc_output["tool_call_id"],
                        "output": fc_output["content"],
                    })
            else:
                tool_call_outputs = []

            if cleaned_text:
                step["output"].append({
                    "type": "message",
                    "role": "assistant",
                    "content": cleaned_text,
                })

            trajectory.append(step)

            if not has_tool_call:
                break

            if total_tool_calls >= max_tool_calls:
                print(f"  [MAX TOOL CALLS] Reached {total_tool_calls}/{max_tool_calls}, forcing stop.")
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

            ds_reasoning = response.get("reasoning_content", "")
            if ds_reasoning:
                step_reasoning = ds_reasoning

            final_step: dict = {
                "turn": len(trajectory),
                "reasoning": step_reasoning,
                "output": [],
            }
            if cleaned_text:
                final_step["output"].append({
                    "type": "message",
                    "role": "assistant",
                    "content": cleaned_text,
                })
            trajectory.append(final_step)

    chat_messages = trajectory_to_messages(trajectory, SYSTEM_PROMPT, user_content)
    return {"trajectory": trajectory, "messages": chat_messages}


# ---------------------------------------------------------------------------
# Orchestration & saving
# ---------------------------------------------------------------------------


def _build_record(query_item: dict, trajectory: list[dict], chat_messages: list[dict]) -> dict:
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
    async with save_lock:
        all_records.append(record)
        sorted_records = sorted(all_records, key=lambda r: r["id"])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_records, f, indent=2, ensure_ascii=False)


def _is_client_request_error(exc: Exception) -> bool:
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if isinstance(exc, APIStatusError) and 400 <= exc.status_code < 500:
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    error_msg = str(exc).lower()
    return any(
        keyword in error_msg
        for keyword in ["rate limit", "429", "400", "401", "403", "timeout", "connection"]
    )


async def process_single_query(
    query_item: dict,
    conn: MCPConnection,
    tools: list[dict],
    llm_client: LLMClient,
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
                            query, conn, tools, llm_client, mcp_semaphore,
                            max_turns=max_turns, max_tool_calls=max_tool_calls,
                        )
                        record = _build_record(
                            query_item, gen_result["trajectory"], gen_result["messages"]
                        )
                        await _save_record(record, all_records, save_lock, output_path)
                        pbar.update(1)
                        pbar.set_postfix(saved=len(all_records))
                        return

                    except Exception as exc:
                        print(f"[ERROR] query {idx} attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}")
                        traceback.print_exc()

                        if _is_client_request_error(exc):
                            print(f"  [CLIENT ERROR] query {idx}: sleeping {client_error_sleep}s...")
                            await asyncio.sleep(client_error_sleep)
                            try:
                                gen_result = await generate_trajectory(
                                    query, conn, tools, llm_client, mcp_semaphore,
                                    max_turns=max_turns, max_tool_calls=max_tool_calls,
                                )
                                record = _build_record(
                                    query_item, gen_result["trajectory"], gen_result["messages"]
                                )
                                await _save_record(record, all_records, save_lock, output_path)
                                pbar.update(1)
                                pbar.set_postfix(saved=len(all_records))
                            except Exception as retry_exc:
                                print(f"  [SKIP] query {idx} still failed: {type(retry_exc).__name__}: {retry_exc}")
                            return

                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                        else:
                            print(f"  query {idx} failed after {max_retries} retries — skipping")
                            pbar.update(1)

    except (asyncio.CancelledError, _BaseExceptionGroup) as exc:  # type: ignore
        print(f"  [WARN] query {idx} cancelled: {type(exc).__name__}")
        pbar.update(1)


async def process_testset(
    input_path: str,
    output_dir: str,
    model_name: str,
    api_base: str,
    api_key: str | None,
    num_queries: int,
    max_turns: int,
    max_tool_calls: int,
    concurrency: int,
    mcp_concurrency: int,
    max_retries: int,
    retry_delay: int,
    client_error_sleep: int,
) -> None:
    from pathlib import Path

    from tqdm import tqdm

    fmp_api_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_api_key:
        raise ValueError("Set the FMP_API_KEY environment variable before running.")

    mcp_url = f"{FMP_MCP_URL}?apikey={fmp_api_key}"

    llm_client = LLMClient(model=model_name, api_base=api_base, api_key=api_key or None)
    print(f"Using model: {model_name} via {api_base}")

    with open(input_path, "r", encoding="utf-8") as f:
        query_items: list[dict] = json.load(f)
    if num_queries > 0:
        query_items = query_items[:num_queries]
    total = len(query_items)
    print(f"Loaded {total} queries from {input_path}")

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
            print(f"Resumed: loaded {len(all_records)} existing records from {output_path}")
        except (json.JSONDecodeError, Exception) as exc:
            print(f"Warning: could not load existing records ({exc}), starting fresh.")
            all_records = []

    processed_ids = {r["id"] for r in all_records}
    remaining = [item for item in query_items if item["id"] not in processed_ids]
    if not remaining:
        print("All queries already processed.")
        return

    print(f"{len(remaining)} queries remaining.")

    concurrency_semaphore = asyncio.Semaphore(concurrency)
    mcp_semaphore = asyncio.Semaphore(mcp_concurrency)
    save_lock = asyncio.Lock()

    conn = MCPConnection(
        mcp_url=mcp_url,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_sleep=client_error_sleep,
    )

    logfire = get_logfire()

    try:
        tools = await conn.start()

        with logfire.span(
            "process_testset", model=model_name, total=total, remaining=len(remaining)
        ), tqdm(total=len(remaining), desc="Generating trajectories") as pbar:
            tasks = [
                process_single_query(
                    item, conn, tools, llm_client, mcp_semaphore,
                    concurrency_semaphore, max_turns, max_tool_calls,
                    max_retries, retry_delay, client_error_sleep,
                    all_records, save_lock, output_path, pbar,
                )
                for item in remaining
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

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

app = typer.Typer(help="Trajectory generation via OpenAI-compatible endpoints (Qwen, DeepSeek, etc.).")


@app.command("generate")
def generate(
    input: str = typer.Option(
        "testset/selected_800.json", "--input", "-i",
        help="Path to the JSON test set file.",
    ),
    output: str = typer.Option(
        "data/trajectory/testset_trajectory", "--output", "-o",
        help="Directory to save trajectory results.",
    ),
    model: str = typer.Option(
        MODEL, "--model", "-m",
        help="Model name on HuggingFace Router.",
    ),
    api_base: str = typer.Option(
        API_BASE, "--api-base",
        help="OpenAI-compatible API base URL.",
    ),
    api_key: str = typer.Option(
        None, "--api-key",
        help="API key (defaults to HF_TOKEN or DEEPSEEK_API_KEY env var).",
    ),
    num_queries: int = typer.Option(
        0, "--num-queries", "-n", help="Limit to first N queries (0 = all).",
    ),
    max_turns: int = typer.Option(
        6, "--max-turns", help="Max agentic loop turns per query.",
    ),
    max_tool_calls: int = typer.Option(
        30, "--max-tool-calls",
        help="Max total tool calls per query.",
    ),
    concurrency: int = typer.Option(
        30, "--concurrency", help="Max concurrent query tasks.",
    ),
    mcp_concurrency: int = typer.Option(
        60, "--mcp-concurrency",
        help="Max concurrent in-flight MCP calls.",
    ),
    max_retries: int = typer.Option(
        2, "--max-retries", help="Max retries per query on failure.",
    ),
    retry_delay: int = typer.Option(
        10, "--retry-delay", help="Seconds to wait between retries.",
    ),
    client_error_sleep: int = typer.Option(
        120, "--client-error-sleep",
        help="Seconds to sleep on client-side errors before final retry.",
    ),
) -> None:
    """Generate tool-calling trajectories for a test set."""
    asyncio.run(
        process_testset(
            input_path=input,
            output_dir=output,
            model_name=model,
            api_base=api_base,
            api_key=api_key,
            num_queries=num_queries,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            concurrency=concurrency,
            mcp_concurrency=mcp_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
            client_error_sleep=client_error_sleep,
        )
    )


@app.command("extract-schemas")
def extract_schemas(
    output_dir: str = typer.Option(
        "mcp_tool_call_schema_extract", "--output-dir", "-o",
        help="Directory to save schema JSON files.",
    ),
) -> None:
    """Connect to the FMP MCP server and export tool schemas."""
    from pathlib import Path

    async def _extract():
        fmp_api_key = os.environ.get("FMP_API_KEY", "")
        if not fmp_api_key:
            raise ValueError("Set the FMP_API_KEY environment variable before running.")

        mcp_url = f"{FMP_MCP_URL}?apikey={fmp_api_key}"
        conn = MCPConnection(mcp_url=mcp_url)
        try:
            await conn.start()
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            tools_path = out / "tool_call_schemas_openai.json"
            tools_path.write_text(json.dumps(conn.get_tools(), indent=2, ensure_ascii=False))
            print(f"Saved {len(conn.get_tools())} tool schemas to {tools_path}")
        finally:
            await conn.close()

    asyncio.run(_extract())


if __name__ == "__main__":
    app()
